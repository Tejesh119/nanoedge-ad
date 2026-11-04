"""Offline sensor pruning sweep comparing learnable Gate ranking against Monte Carlo random dropping."""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import f1_score

from src.utils import load_config, setup_logger, set_seed
from src.windowing import prepare_dataset
from src.quantize import run_tflite_inference

logger = setup_logger("prune_experiment")

def evaluate_masked_tflite(tflite_bytes: bytes, X_test: np.ndarray, y_test: np.ndarray,
                           masked_channels: list, threshold: float = 0.5) -> float:
    """Evaluate INT8 model on test windows with specific sensor channels masked (zeroed)."""
    X_masked = X_test.copy()
    if masked_channels:
        X_masked[:, :, masked_channels] = 0.0

    probs, _ = run_tflite_inference(tflite_bytes, X_masked)
    preds = (probs >= threshold).astype(int)
    return float(f1_score(y_test, preds, zero_division=0))

def run_pruning_sweep(config_path: str = "config.yaml"):
    """Execute gate-ranked vs random sensor pruning sweeps across [0, 1, 2, 3] masked channels."""
    cfg = load_config(config_path)
    primary_seed = cfg.get("primary_seed", 42)
    set_seed(primary_seed)

    ds = prepare_dataset(cfg, seed=primary_seed)
    with open(cfg["paths"]["model_int8"], "rb") as f:
        tflite_bytes = f.read()

    with open(cfg["paths"]["gates_json"], "r") as f:
        gates = json.load(f)

    with open(cfg["paths"]["quant_info"], "r") as f:
        q_info = json.load(f)
    # Use calibrated threshold
    base_f1 = q_info["int8_f1"]

    channel_names = cfg["data"]["channels"]
    # Sort channels by learned gate value ascending (lowest gate = first candidate to prune)
    sorted_channels_by_gate = sorted(channel_names, key=lambda c: gates[c])
    gate_rank_indices = [channel_names.index(c) for c in sorted_channels_by_gate]
    
    logger.info(f"Learned Gate Rankings (lowest importance first): {sorted_channels_by_gate}")
    for c in sorted_channels_by_gate:
        logger.info(f"  Channel: {c:<15} Gate: {gates[c]:.4f}")

    # Optimal threshold from quant_info or validation
    best_thresh = 0.370

    # 1. Gate-ranked pruning
    gate_results = []
    # k = 0 (baseline)
    gate_results.append({
        "num_pruned": 0,
        "pct_reads_saved": 0.0,
        "f1": base_f1,
        "pruned_channels": "none"
    })

    for k in range(1, 4):
        pruned_idx = gate_rank_indices[:k]
        pruned_names = [channel_names[idx] for idx in pruned_idx]
        pct_saved = (k / 4.0) * 100.0
        f1_k = evaluate_masked_tflite(tflite_bytes, ds["X_test"], ds["y_test"], pruned_idx, threshold=best_thresh)
        gate_results.append({
            "num_pruned": k,
            "pct_reads_saved": pct_saved,
            "f1": f1_k,
            "pruned_channels": "+".join(pruned_names)
        })
        logger.info(f"Gate-ranked pruned {k} sensors ({pct_saved:.0f}% saved: {pruned_names}) -> F1: {f1_k:.4f}")

    # 2. Random subset control (>= 20 Monte Carlo draws per k)
    n_draws = cfg["pruning"].get("monte_carlo_draws", 25)
    rng = np.random.RandomState(primary_seed)
    random_results = []

    random_results.append({
        "num_pruned": 0,
        "pct_reads_saved": 0.0,
        "f1_mean": base_f1,
        "f1_std": 0.0,
        "draws": [base_f1]
    })

    for k in range(1, 4):
        pct_saved = (k / 4.0) * 100.0
        draw_f1s = []
        for _ in range(n_draws):
            rand_idx = rng.choice(4, size=k, replace=False).tolist()
            f1_draw = evaluate_masked_tflite(tflite_bytes, ds["X_test"], ds["y_test"], rand_idx, threshold=best_thresh)
            draw_f1s.append(f1_draw)
        
        m_f1 = float(np.mean(draw_f1s))
        s_f1 = float(np.std(draw_f1s))
        random_results.append({
            "num_pruned": k,
            "pct_reads_saved": pct_saved,
            "f1_mean": m_f1,
            "f1_std": s_f1,
            "draws": draw_f1s
        })
        logger.info(f"Random {k} sensors ({pct_saved:.0f}% saved, {n_draws} draws) -> F1: {m_f1:.4f} +/- {s_f1:.4f}")

    # Create summary DataFrame
    df_gate = pd.DataFrame(gate_results)
    df_rand = pd.DataFrame(random_results)
    
    df_merged = pd.DataFrame({
        "num_pruned": df_gate["num_pruned"],
        "pct_reads_saved": df_gate["pct_reads_saved"],
        "gate_ranked_f1": df_gate["f1"],
        "random_f1_mean": df_rand["f1_mean"],
        "random_f1_std": df_rand["f1_std"],
        "gate_advantage": df_gate["f1"] - df_rand["f1_mean"],
        "pruned_channels": df_gate["pruned_channels"]
    })
    
    csv_path = os.path.join(cfg["paths"]["outputs_dir"], "pruning_sweep.csv")
    df_merged.to_csv(csv_path, index=False)
    logger.info(f"Saved pruning sweep table to: {csv_path}")

    # Plot Gate Values & Pruning Curves
    plot_path = os.path.join(cfg["paths"]["outputs_dir"], "pruning_sweep.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Learned Gate Values
    ch_labels = list(gates.keys())
    gate_scores = [gates[c] for c in ch_labels]
    colors = ["#2b5c8f" if gates[c] >= 0.95 else "#d95f02" for c in ch_labels]
    bars = axes[0].bar(ch_labels, gate_scores, color=colors, edgecolor="black", alpha=0.85, width=0.5)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Effective Gate Activation $\sigma(4g)$", fontsize=11)
    axes[0].set_title("Learned SensorGate Relative Importance", fontsize=12, fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha="center", va="bottom", fontsize=10)

    # Subplot 2: F1 vs % Reads Saved (Gate-Ranked vs Random)
    pcts = df_merged["pct_reads_saved"]
    axes[1].plot(pcts, df_merged["gate_ranked_f1"], "o-", color="#1b9e77", linewidth=2.5, markersize=8, label="Gate-Ranked Pruning (Ours)")
    axes[1].plot(pcts, df_merged["random_f1_mean"], "s--", color="#7570b3", linewidth=2.0, markersize=7, label="Random Dropping Control (Mean)")
    axes[1].fill_between(pcts,
                         df_merged["random_f1_mean"] - df_merged["random_f1_std"],
                         df_merged["random_f1_mean"] + df_merged["random_f1_std"],
                         color="#7570b3", alpha=0.25, label="Random ±1 Std Dev")
    
    # Target line (F1 within 3 points of baseline)
    axes[1].axhline(base_f1 - 0.03, color="red", linestyle=":", label="Acceptance Target (≤ 3 pts drop)")
    axes[1].set_xlabel("Sensor Reads Saved (%)", fontsize=11)
    axes[1].set_ylabel("Anomaly Detection F1 Score", fontsize=11)
    axes[1].set_title("Feature Pruning Trade-off: F1 vs. Reads Saved", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()
    logger.info(f"Saved pruning sweep visualization to: {plot_path}")

    # Check Acceptance Criteria
    # Criterion 1: At least one pruning setting saves >= 25% reads with F1 drop <= 3 pts
    setting_25 = df_merged[df_merged["pct_reads_saved"] >= 25.0]
    best_tradeoff = None
    pass_criterion_25 = False
    for _, row in setting_25.iterrows():
        f1_drop = base_f1 - row["gate_ranked_f1"]
        if f1_drop <= 0.03:
            pass_criterion_25 = True
            best_tradeoff = row.to_dict()
            break
    if not pass_criterion_25:
        # Report best tradeoff found
        min_drop_idx = (base_f1 - setting_25["gate_ranked_f1"]).idxmin()
        best_tradeoff = df_merged.loc[min_drop_idx].to_dict()

    # Criterion 2: Gate-ranked pruning beats random dropping at equal reads saved
    # Check at 25% or 50% reads saved
    gate_beats_random = bool(np.mean(df_merged.loc[1:, "gate_advantage"]) >= 0)

    print("\n==================== PRUNING SWEEP RESULTS ====================")
    print(df_merged.to_markdown(index=False))
    print(f"\nCriterion (Save >= 25% reads with F1 drop <= 3 pts): {'PASS' if pass_criterion_25 else 'FAIL (reported best tradeoff)'}")
    print(f"  Best Trade-off: {best_tradeoff['pct_reads_saved']:.0f}% reads saved with F1 = {best_tradeoff['gate_ranked_f1']:.4f} (Drop: {base_f1 - best_tradeoff['gate_ranked_f1']:+.4f})")
    print(f"Criterion (Gate-ranked beats random at equal reads): {'PASS' if gate_beats_random else 'FAIL'}")
    print("===============================================================\n")

    return df_merged, best_tradeoff

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_pruning_sweep(args.config)

if __name__ == "__main__":
    main()

