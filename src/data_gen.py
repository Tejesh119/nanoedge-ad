"""Synthetic multi-sensor IoT time-series generator with subtle anomalies.

NOTICE: All generated data is SYNTHETIC benchmark telemetry for TinyML anomaly detection research.
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import load_config, set_seed, setup_logger

logger = setup_logger("data_gen")

def generate_synthetic_stream(config: dict, seed: int = 42) -> tuple:
    """Generate 4-channel synthetic time-series with subtle multi-type anomalies.
    
    Returns:
        timeseries: np.ndarray of shape (N, 4)
        anomaly_mask: np.ndarray of shape (N,) boolean
        anomaly_types: list of str of length N (name of anomaly or 'normal')
        metadata: dict containing generation parameters and statistics
    """
    set_seed(seed)
    data_cfg = config["data"]
    n_samples = data_cfg.get("total_samples", 220000)
    fs = data_cfg.get("sample_rate_hz", 100)
    t = np.arange(n_samples) / fs

    # Baseline signals: 1-2 sinusoids + slow trend + mild cross-channel correlation
    # Latent shared process to induce cross-correlation
    latent_process = 0.5 * np.sin(2 * np.pi * 0.15 * t) + 0.3 * np.cos(2 * np.pi * 0.03 * t)

    # Channel 0: Temperature (slow thermal cycle + trend + noise)
    temp_base = 25.0 + 3.0 * np.sin(2 * np.pi * 0.01 * t) + 0.8 * np.cos(2 * np.pi * 0.05 * t)
    temp_trend = 0.00002 * t
    temp_noise = np.random.normal(0, 0.25, n_samples)
    temperature = temp_base + temp_trend + 0.2 * latent_process + temp_noise

    # Channel 1: Vibration X (mechanical rotation ~15 Hz & ~35 Hz + noise)
    vib_x_base = 1.2 * np.sin(2 * np.pi * 15.0 * t) + 0.6 * np.sin(2 * np.pi * 35.0 * t)
    vib_x_noise = np.random.normal(0, 0.4, n_samples)
    vibration_x = vib_x_base + 0.5 * latent_process + vib_x_noise

    # Channel 2: Vibration Y (correlated with X, phase-shifted + ~20 Hz + noise)
    vib_y_base = 1.0 * np.cos(2 * np.pi * 15.0 * t + np.pi / 4) + 0.5 * np.sin(2 * np.pi * 20.0 * t)
    vib_y_noise = np.random.normal(0, 0.4, n_samples)
    vibration_y = vib_y_base + 0.5 * latent_process + vib_y_noise

    # Channel 3: Pressure (slow pneumatic cycle + micro-pulsations + noise)
    press_base = 101.3 + 2.5 * np.sin(2 * np.pi * 0.05 * t) + 0.7 * np.sin(2 * np.pi * 0.8 * t)
    press_noise = np.random.normal(0, 0.3, n_samples)
    pressure = press_base + 0.3 * latent_process + press_noise

    # Combine channels: shape (N, 4)
    data = np.stack([temperature, vibration_x, vibration_y, pressure], axis=1)
    
    # Compute channel-wise standard deviations for subtle injection scaling
    channel_stds = np.std(data, axis=0)

    anomaly_mask = np.zeros(n_samples, dtype=bool)
    anomaly_type_labels = ["normal"] * n_samples

    anomaly_types = data_cfg.get("anomaly_types", ["spike", "drift", "stuck_at", "noise_burst", "dropout"])
    target_ratio = data_cfg.get("anomaly_ratio_target", 0.12)
    
    # Ensure every split (0-60%, 60-80%, 80-100%) gets all anomaly types
    # Partition each split into sub-segments for each anomaly type
    split_bounds = [
        (0, int(n_samples * 0.60)),
        (int(n_samples * 0.60), int(n_samples * 0.80)),
        (int(n_samples * 0.80), n_samples)
    ]
    
    event_counts = {atype: 0 for atype in anomaly_types}
    rng = np.random.RandomState(seed)

    for split_idx, (s_start, s_end) in enumerate(split_bounds):
        split_len = s_end - s_start
        # At least 2 events per anomaly type in each split
        reps_per_type = max(2, int(np.round((split_len * target_ratio) / (5 * 60))))
        
        # Build list of anomaly assignments for this split
        split_types = anomaly_types * reps_per_type
        rng.shuffle(split_types)
        
        chunk_len = split_len // len(split_types)
        for i, atype in enumerate(split_types):
            c_start = s_start + i * chunk_len + 5
            c_end = min(s_end - 5, s_start + (i + 1) * chunk_len - 5)
            if c_end - c_start < 25:
                continue

            # Select target channel(s): single channel or pair
            if rng.rand() < 0.75:
                target_channels = [rng.choice(4)]
            else:
                target_channels = rng.choice(4, size=2, replace=False).tolist()

            if atype == "spike":
                # Duration: 8-15 samples so >= 10% of window (>= 7 samples) is guaranteed
                dur = rng.randint(8, min(16, c_end - c_start))
                if c_end - dur <= c_start:
                    continue
                start = rng.randint(c_start, c_end - dur)
                sign = rng.choice([-1, 1])
                for ch in target_channels:
                    amp = sign * rng.uniform(2.4, 3.4) * channel_stds[ch]
                    data[start:start+dur, ch] += amp
                    
            elif atype == "drift":
                # Gradual ramp over 45-85 samples
                dur = rng.randint(45, min(85, max(46, (c_end - c_start) - 5)))
                if c_end - dur <= c_start:
                    continue
                start = rng.randint(c_start, c_end - dur)
                sign = rng.choice([-1, 1])
                for ch in target_channels:
                    ramp = np.linspace(0, sign * rng.uniform(2.0, 3.0) * channel_stds[ch], dur)
                    data[start:start+dur, ch] += ramp

            elif atype == "stuck_at":
                # Value freezes at a constant value for 35-70 samples
                dur = rng.randint(35, min(70, max(36, (c_end - c_start) - 5)))
                if c_end - dur <= c_start:
                    continue
                start = rng.randint(c_start, c_end - dur)
                for ch in target_channels:
                    freeze_val = data[start, ch]
                    data[start:start+dur, ch] = freeze_val

            elif atype == "noise_burst":
                # Localized high-frequency variance increase (2.2 to 3.2x noise) for 30-65 samples
                dur = rng.randint(30, min(65, max(31, (c_end - c_start) - 5)))
                if c_end - dur <= c_start:
                    continue
                start = rng.randint(c_start, c_end - dur)
                for ch in target_channels:
                    noise_inject = rng.normal(0, rng.uniform(2.2, 3.2) * channel_stds[ch], dur)
                    data[start:start+dur, ch] += noise_inject

            elif atype == "dropout":
                # Sensor drops to 0 for 25-55 samples
                dur = rng.randint(25, min(55, max(26, (c_end - c_start) - 5)))
                if c_end - dur <= c_start:
                    continue
                start = rng.randint(c_start, c_end - dur)
                for ch in target_channels:
                    data[start:start+dur, ch] = 0.0

            anomaly_mask[start:start+dur] = True
            for idx in range(start, start+dur):
                anomaly_type_labels[idx] = atype
            event_counts[atype] += 1

        anomaly_mask[start:start+dur] = True
        for idx in range(start, start+dur):
            anomaly_type_labels[idx] = atype
        event_counts[atype] += 1

    actual_ratio = np.mean(anomaly_mask)
    logger.info(f"Generated {n_samples} samples across 4 channels. Anomaly sample ratio: {actual_ratio:.4f}")
    logger.info(f"Injected anomaly events per type: {event_counts}")

    metadata = {
        "n_samples": n_samples,
        "sample_rate_hz": fs,
        "channels": data_cfg["channels"],
        "anomaly_sample_ratio": float(actual_ratio),
        "event_counts": event_counts,
        "synthetic": True,
        "seed": seed
    }

    return data, anomaly_mask, np.array(anomaly_type_labels), metadata

def plot_sample_anomalies(data: np.ndarray, anomaly_mask: np.ndarray, anomaly_types: np.ndarray,
                          channels: list, output_path: str = "outputs/sample_windows.png"):
    """Plot sample sections displaying normal signals and each of the 5 anomaly types."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    unique_types = ["normal", "spike", "drift", "stuck_at", "noise_burst", "dropout"]
    fig, axes = plt.subplots(len(unique_types), 1, figsize=(12, 14), sharex=False)
    
    for i, atype in enumerate(unique_types):
        ax = axes[i]
        if atype == "normal":
            # Find a 150-sample contiguous clean normal stretch
            clean_indices = np.where(~anomaly_mask)[0]
            # Pick a stretch
            start = clean_indices[1000]
        else:
            match_indices = np.where(anomaly_types == atype)[0]
            if len(match_indices) > 0:
                center = match_indices[len(match_indices) // 2]
                start = max(0, center - 60)
            else:
                start = 0
        end = min(len(data), start + 140)
        
        t_slice = np.arange(start, end)
        for ch_idx, ch_name in enumerate(channels):
            ax.plot(t_slice, data[start:end, ch_idx], label=ch_name, alpha=0.8, linewidth=1.2)
            
        # Highlight anomalous region
        is_anom = anomaly_mask[start:end]
        if np.any(is_anom):
            ax.axvspan(t_slice[np.where(is_anom)[0][0]], t_slice[np.where(is_anom)[0][-1]],
                       color="red", alpha=0.2, label="Injected Anomaly")
            
        ax.set_title(f"Telemetry Segment: {atype.upper()} (Synthetic)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        if i == 0:
            ax.legend(loc="upper right", ncol=5, fontsize=8)

    axes[-1].set_xlabel("Time Step (Sample Index @ 100 Hz)", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    logger.info(f"Saved sample windows plot to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic IoT stream for NanoEdge-AD")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("primary_seed", 42)
    data, mask, types, meta = generate_synthetic_stream(cfg, seed=seed)

    plot_path = os.path.join(cfg["paths"]["outputs_dir"], "sample_windows.png")
    plot_sample_anomalies(data, mask, types, cfg["data"]["channels"], plot_path)
    print("Class balance on raw point stream:")
    print(f"  Normal points: {np.sum(~mask):,} ({(1.0 - meta['anomaly_sample_ratio'])*100:.2f}%)")
    print(f"  Anomalous points: {np.sum(mask):,} ({meta['anomaly_sample_ratio']*100:.2f}%)")
    print(f"Sample window visualization written to: {plot_path}")

if __name__ == "__main__":
    main()
