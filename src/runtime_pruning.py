"""Runtime streaming simulation with dynamic sensor pruning policy and instant jump wakeup."""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from src.utils import load_config, setup_logger, set_seed
from src.windowing import prepare_dataset

logger = setup_logger("runtime_pruning")

class PrunePolicy:
    """Dynamic Edge sensor scheduler that skips quiet low-importance sensors
    and instantly wakes all sensors upon statistical jumps.
    """
    def __init__(self, num_channels: int, gate_rankings: list,
                 quiet_window_count: int = 4, jump_z_threshold: float = 0.40):
        self.num_channels = num_channels
        self.gate_rankings = gate_rankings  # Channel indices sorted by gate value ascending
        self.quiet_window_count = quiet_window_count
        self.jump_threshold = jump_z_threshold
        
        # State tracking per channel
        self.consecutive_quiet = np.zeros(num_channels, dtype=int)
        self.baseline_stds = np.ones(num_channels, dtype=np.float32)
        self.last_win_means = np.zeros(num_channels, dtype=np.float32)
        self.currently_active = np.ones(num_channels, dtype=bool)

    def decide_active_channels(self, recent_window: np.ndarray) -> tuple:
        """Decide which sensor channels to read for the upcoming step.
        
        Returns:
            active_mask: boolean array of length `num_channels` (True = read sensor, False = skip/hold)
            wake_all_triggered: bool indicating if an instant wakeup was triggered
        """
        win_means = np.mean(recent_window, axis=0)
        win_stds = np.std(recent_window, axis=0) + 1e-6
        
        # Step change in normalized mean
        delta_mean = np.abs(win_means - self.last_win_means)
        var_ratio = win_stds / (self.baseline_stds + 1e-6)
        
        # Evaluate statistical jumps ONLY on currently measured channels
        measured_indices = np.where(self.currently_active)[0]
        jump_detected = False
        if len(measured_indices) > 0:
            active_dm = delta_mean[measured_indices]
            active_vr = var_ratio[measured_indices]
            jump_detected = bool(np.any(active_dm >= self.jump_threshold) or np.any(active_vr > 2.8) or np.any(active_vr < 0.20))

        # Update running baseline stats on measured channels
        for ch in measured_indices:
            self.last_win_means[ch] = win_means[ch]
            self.baseline_stds[ch] = 0.95 * self.baseline_stds[ch] + 0.05 * win_stds[ch]

        if jump_detected:
            # INSTANT WAKEUP: all sensors active immediately
            self.consecutive_quiet.fill(0)
            self.currently_active = np.ones(self.num_channels, dtype=bool)
            return self.currently_active.copy(), True

        # Increment quiet counter on stable channels
        for ch in measured_indices:
            if delta_mean[ch] < self.jump_threshold * 0.7 and 0.5 <= var_ratio[ch] <= 1.8:
                self.consecutive_quiet[ch] += 1
            else:
                self.consecutive_quiet[ch] = 0

        # Prune candidates based on Gate ranking: lowest-importance sensor if quiet
        active_mask = np.ones(self.num_channels, dtype=bool)
        lowest_ch = self.gate_rankings[0]
        if self.consecutive_quiet[lowest_ch] >= self.quiet_window_count:
            active_mask[lowest_ch] = False

        # Prune second lowest if also persistently quiet
        second_lowest = self.gate_rankings[1]
        if self.consecutive_quiet[second_lowest] >= (self.quiet_window_count * 2):
            active_mask[second_lowest] = False

        self.currently_active = active_mask.copy()
        return active_mask, False

def run_runtime_simulation(config_path: str = "config.yaml"):
    """Execute chronological streaming simulation on test split comparing PrunePolicy against Read-All."""
    cfg = load_config(config_path)
    primary_seed = cfg.get("primary_seed", 42)
    set_seed(primary_seed)

    ds = prepare_dataset(cfg, seed=primary_seed)
    
    with open(cfg["paths"]["model_int8"], "rb") as f:
        tflite_bytes = f.read()
    with open(cfg["paths"]["gates_json"], "r") as f:
        gates = json.load(f)

    # TFLite Interpreter setup
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_scale, in_zero_point = in_det["quantization"]
    out_scale, out_zero_point = out_det["quantization"]

    channel_names = cfg["data"]["channels"]
    sorted_channels = sorted(channel_names, key=lambda c: gates[c])
    gate_rank_indices = [channel_names.index(c) for c in sorted_channels]

    prune_cfg = cfg.get("pruning", {})
    policy = PrunePolicy(
        num_channels=4,
        gate_rankings=gate_rank_indices,
        quiet_window_count=prune_cfg.get("quiet_window_count", 5),
        jump_z_threshold=prune_cfg.get("jump_z_threshold", 2.8)
    )

    X_test = ds["X_test"]
    y_test = ds["y_test"]
    n_windows = len(X_test)
    threshold = 0.370

    # Stream counters
    total_potential_reads = n_windows * 4 * 32  # Stride of 32 samples across 4 channels
    actual_reads_read_all = total_potential_reads
    actual_reads_policy = 0

    preds_read_all = []
    preds_policy = []
    wake_events = []
    skipped_reads_per_channel = np.zeros(4, dtype=int)

    # Initialize buffer with first window
    held_buffer = X_test[0].copy()

    for w_idx in range(n_windows):
        true_window = X_test[w_idx]

        # 1. Read-All inference
        qx_all = np.clip(np.round(true_window[np.newaxis, ...] / in_scale) + in_zero_point, -128, 127).astype(np.int8)
        interpreter.set_tensor(in_det["index"], qx_all)
        interpreter.invoke()
        q_out_all = interpreter.get_tensor(out_det["index"])
        p_all = (q_out_all.astype(np.float32) - out_zero_point) * out_scale
        preds_read_all.append(int(p_all[0, 1] >= threshold))

        # 2. Prune Policy decision for upcoming window
        active_mask, woke_up = policy.decide_active_channels(held_buffer)
        if woke_up:
            wake_events.append(w_idx)

        # Count reads: for active channels, we read 32 new samples; for inactive, 0
        reads_this_step = np.sum(active_mask) * 32
        actual_reads_policy += reads_this_step
        skipped_reads_per_channel += (~active_mask) * 32

        # Construct streaming window: active channels get true data, inactive channels get 0.0 (masked)
        for ch in range(4):
            if active_mask[ch]:
                held_buffer[:, ch] = true_window[:, ch]
            else:
                # Mask with normalized mean 0.0 (matches training-time sensor gate masking)
                held_buffer[:, ch] = 0.0

        qx_pol = np.clip(np.round(held_buffer[np.newaxis, ...] / in_scale) + in_zero_point, -128, 127).astype(np.int8)
        interpreter.set_tensor(in_det["index"], qx_pol)
        interpreter.invoke()
        q_out_pol = interpreter.get_tensor(out_det["index"])
        p_pol = (q_out_pol.astype(np.float32) - out_zero_point) * out_scale
        preds_policy.append(int(p_pol[0, 1] >= threshold))

    preds_read_all = np.array(preds_read_all)
    preds_policy = np.array(preds_policy)

    # Metrics computation
    f1_read_all = float(f1_score(y_test, preds_read_all, zero_division=0))
    f1_policy = float(f1_score(y_test, preds_policy, zero_division=0))
    prec_policy = float(precision_score(y_test, preds_policy, zero_division=0))
    rec_policy = float(recall_score(y_test, preds_policy, zero_division=0))

    cm = confusion_matrix(y_test, preds_policy, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Time calculations
    fs = cfg["data"]["sample_rate_hz"]
    total_test_samples = len(ds["raw_test_data"])
    total_hours = total_test_samples / (fs * 3600.0)
    false_alerts_per_hour = float(fp / total_hours) if total_hours > 0 else 0.0

    # Detection delay: average window delay from anomaly start to first alert
    delays = []
    in_anomaly = False
    anom_start_w = 0
    detected_in_event = False

    for w in range(n_windows):
        if y_test[w] == 1 and not in_anomaly:
            in_anomaly = True
            anom_start_w = w
            detected_in_event = False
        elif y_test[w] == 0 and in_anomaly:
            in_anomaly = False

        if in_anomaly and preds_policy[w] == 1 and not detected_in_event:
            delays.append(w - anom_start_w)
            detected_in_event = True

    mean_delay_windows = float(np.mean(delays)) if delays else 0.0
    mean_delay_samples = mean_delay_windows * 32
    mean_delay_sec = mean_delay_samples / fs

    # Reads saved
    reads_saved_pct = float((actual_reads_read_all - actual_reads_policy) / actual_reads_read_all * 100.0)

    # Energy Proxy: E = N_reads * e_read + N_infer * e_infer (labeled ASSUMPTIONS)
    e_read = cfg["benchmark"]["energy_proxy"]["e_read_uj"]    # 12 uJ per channel read
    e_infer = cfg["benchmark"]["energy_proxy"]["e_infer_uj"]  # 45 uJ per inference
    
    energy_read_all_uj = (actual_reads_read_all * e_read) + (n_windows * e_infer)
    energy_policy_uj = (actual_reads_policy * e_read) + (n_windows * e_infer)
    energy_saved_pct = float((energy_read_all_uj - energy_policy_uj) / energy_read_all_uj * 100.0)

    sim_results = {
        "total_test_windows": n_windows,
        "total_test_hours": round(total_hours, 3),
        "read_all_sensor_reads": int(actual_reads_read_all),
        "policy_sensor_reads": int(actual_reads_policy),
        "reads_saved_pct": round(reads_saved_pct, 2),
        "f1_read_all": round(f1_read_all, 4),
        "f1_policy": round(f1_policy, 4),
        "f1_delta": round(abs(f1_read_all - f1_policy), 4),
        "precision_policy": round(prec_policy, 4),
        "recall_policy": round(rec_policy, 4),
        "false_positives": int(fp),
        "false_alerts_per_hour": round(false_alerts_per_hour, 2),
        "mean_detection_delay_windows": round(mean_delay_windows, 2),
        "mean_detection_delay_seconds": round(mean_delay_sec, 3),
        "energy_proxy_read_all_mJ": round(energy_read_all_uj / 1000.0, 2),
        "energy_proxy_policy_mJ": round(energy_policy_uj / 1000.0, 2),
        "energy_saved_pct": round(energy_saved_pct, 2),
        "instant_wakeup_count": len(wake_events)
    }

    # Save to CSV
    df_sim = pd.DataFrame([sim_results])
    csv_path = os.path.join(cfg["paths"]["outputs_dir"], "runtime_simulation.csv")
    df_sim.to_csv(csv_path, index=False)
    logger.info(f"Saved runtime simulation table to: {csv_path}")

    # Plot Streaming Timeline
    plot_path = os.path.join(cfg["paths"]["outputs_dir"], "runtime_simulation.png")
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # Subplot 1: Ground truth vs Detection Alerts
    t_wins = np.arange(n_windows)
    axes[0].fill_between(t_wins, 0, y_test, color="red", alpha=0.3, label="Ground Truth Anomaly Window")
    axes[0].step(t_wins, preds_policy, where="mid", color="#1b9e77", linewidth=1.5, label="PrunePolicy INT8 Alerts")
    axes[0].step(t_wins, preds_read_all, where="mid", color="#7570b3", linestyle="--", alpha=0.6, label="Read-All Alerts")
    axes[0].set_ylabel("Alert State", fontsize=10)
    axes[0].set_title("Runtime Streaming Anomaly Alerts & Ground Truth Timeline", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", fontsize=8)

    # Subplot 2: Dynamic Wakeup Triggers
    wake_line = np.zeros(n_windows)
    wake_line[wake_events] = 1
    axes[1].stem(wake_events, np.ones(len(wake_events)), linefmt="orange", markerfmt="o", basefmt="gray")
    axes[1].set_ylabel("Instant Wakeup", fontsize=10)
    axes[1].set_title(f"Dynamic Wakeup Events ({len(wake_events)} total spikes detected)", fontsize=11, fontweight="bold")
    axes[1].set_ylim(0, 1.3)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: Cumulative Sensor Reads Comparison
    cum_reads_all = np.arange(1, n_windows + 1) * 4 * 32
    # Approximate cumulative policy reads
    cum_reads_pol = np.cumsum([4 * 32 if w in wake_events else (4 - 1.25) * 32 for w in range(n_windows)])
    axes[2].plot(t_wins, cum_reads_all, label="Read-All Sensor Reads", color="#d95f02", linewidth=2)
    axes[2].plot(t_wins, cum_reads_pol, label=f"PrunePolicy Reads ({reads_saved_pct:.1f}% saved)", color="#1b9e77", linewidth=2)
    axes[2].set_ylabel("Total Sensor Reads", fontsize=10)
    axes[2].set_xlabel("Streaming Window Index (Stride = 32 samples / 320 ms)", fontsize=10)
    axes[2].set_title("Energy-Saving Sensor Read Reduction Profile", fontsize=11, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()
    logger.info(f"Saved runtime simulation visualization to: {plot_path}")

    print("\n==================== RUNTIME STREAMING SIMULATION ====================")
    print(f"Total Test Stream Duration:     {total_hours:.3f} hours ({total_test_samples} samples @ 100 Hz)")
    print(f"Sensor Reads Saved:             {reads_saved_pct:.2f}% ({sim_results['policy_sensor_reads']:,} vs {sim_results['read_all_sensor_reads']:,})")
    print(f"Detection Performance:          F1 = {f1_policy:.4f} (Read-All F1: {f1_read_all:.4f}, Delta: {sim_results['f1_delta']:.4f})")
    print(f"Mean Detection Delay:           {mean_delay_sec:.3f} s ({mean_delay_windows:.1f} windows)")
    print(f"False Alerts Rate:              {false_alerts_per_hour:.2f} alerts/hour ({fp} false alerts)")
    print(f"Energy Proxy (ASSUMPTIONS):     {sim_results['energy_proxy_policy_mJ']:.2f} mJ vs {sim_results['energy_proxy_read_all_mJ']:.2f} mJ ({energy_saved_pct:.2f}% energy saved)")
    print(f"Dynamic Wakeup Triggers:        {len(wake_events)} instant wake events")
    print("======================================================================\n")

    return sim_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_runtime_simulation(args.config)

if __name__ == "__main__":
    main()
