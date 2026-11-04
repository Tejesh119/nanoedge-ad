"""Comprehensive multi-metric evaluation, per-anomaly recall, and robustness sweeps."""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score, f1_score
import tensorflow as tf

from src.utils import load_config, setup_logger, set_seed
from src.windowing import prepare_dataset
from src.models import (
    SensorGate,
    extract_statistical_features,
    tune_decision_tree
)
from src.quantize import run_tflite_inference

logger = setup_logger("evaluate")

def plot_confusion_matrices(cms: dict, output_path: str = "outputs/confusion_matrices.png"):
    """Plot confusion matrices for all evaluated models."""
    n_models = len(cms)
    cols = 3
    rows = int(np.ceil(n_models / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i, (m_name, cm) in enumerate(cms.items()):
        ax = axes[i]
        cax = ax.matshow(cm, cmap="Blues", alpha=0.8)
        for r in range(2):
            for c in range(2):
                val = cm[r, c]
                ax.text(c, r, f"{val:,}", ha="center", va="center", fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Anomaly"])
        ax.set_yticklabels(["Normal", "Anomaly"])
        ax.set_title(m_name, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=9)
        ax.set_ylabel("True Label", fontsize=9)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    logger.info(f"Saved confusion matrices to: {output_path}")

def evaluate_per_anomaly_recall(ds: dict, tflite_bytes: bytes, dt_model, config: dict, threshold: float = 0.370) -> pd.DataFrame:
    """Compute per-anomaly-type recall for TCN INT8 and Decision Tree baseline."""
    types_test = np.array(ds["types_test"])
    y_test = ds["y_test"]
    X_test = ds["X_test"]

    # TFLite predictions
    tcn_probs, _ = run_tflite_inference(tflite_bytes, X_test)
    tcn_preds = (tcn_probs >= threshold).astype(int)

    # DT predictions
    X_test_feat = extract_statistical_features(X_test)
    dt_preds = dt_model.predict(X_test_feat)

    anomaly_types = ["spike", "drift", "stuck_at", "noise_burst", "dropout"]
    recall_rows = []

    for atype in anomaly_types:
        idx = np.where((types_test == atype) & (y_test == 1))[0]
        if len(idx) == 0:
            continue
        tcn_rec = float(recall_score(y_test[idx], tcn_preds[idx], zero_division=0))
        dt_rec = float(recall_score(y_test[idx], dt_preds[idx], zero_division=0))
        recall_rows.append({
            "anomaly_type": atype,
            "sample_count": len(idx),
            "TCN_INT8_Recall": tcn_rec,
            "DecisionTree_Recall": dt_rec,
            "Advantage": tcn_rec - dt_rec
        })

    df_rec = pd.DataFrame(recall_rows)
    return df_rec

def run_robustness_sweeps(ds: dict, tflite_bytes: bytes, dt_model, config: dict, threshold: float = 0.370) -> dict:
    """Execute robustness experiments against additive noise, sensor dropout, and sensor drift."""
    X_test = ds["X_test"]
    y_test = ds["y_test"]
    rng = np.random.RandomState(42)

    # 1. Additive Noise Sweep (SNR / Noise scale)
    noise_scales = [0.0, 0.2, 0.5, 0.8, 1.2, 1.6]
    tcn_noise_f1 = []
    dt_noise_f1 = []

    for sigma in noise_scales:
        noise = rng.normal(0, sigma, X_test.shape).astype(np.float32)
        X_noisy = X_test + noise

        # TCN
        p, _ = run_tflite_inference(tflite_bytes, X_noisy)
        tcn_noise_f1.append(float(f1_score(y_test, (p >= threshold).astype(int), zero_division=0)))

        # DT
        dt_preds = dt_model.predict(extract_statistical_features(X_noisy))
        dt_noise_f1.append(float(f1_score(y_test, dt_preds, zero_division=0)))

    # 2. Sensor Dropout Rate Sweep
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    tcn_drop_f1 = []
    dt_drop_f1 = []

    for rate in dropout_rates:
        X_dropped = X_test.copy()
        if rate > 0:
            mask = rng.rand(*X_dropped.shape) < rate
            X_dropped[mask] = 0.0

        p, _ = run_tflite_inference(tflite_bytes, X_dropped)
        tcn_drop_f1.append(float(f1_score(y_test, (p >= threshold).astype(int), zero_division=0)))

        dt_preds = dt_model.predict(extract_statistical_features(X_dropped))
        dt_drop_f1.append(float(f1_score(y_test, dt_preds, zero_division=0)))

    # 3. Sensor Drift Magnitude Sweep
    drift_magnitudes = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
    tcn_drift_f1 = []
    dt_drift_f1 = []

    for d_mag in drift_magnitudes:
        X_drifted = X_test.copy()
        if d_mag > 0:
            # Linear ramp across window
            ramp = np.linspace(0, d_mag, 64)[:, np.newaxis]
            X_drifted = X_drifted + ramp

        p, _ = run_tflite_inference(tflite_bytes, X_drifted)
        tcn_drift_f1.append(float(f1_score(y_test, (p >= threshold).astype(int), zero_division=0)))

        dt_preds = dt_model.predict(extract_statistical_features(X_drifted))
        dt_drift_f1.append(float(f1_score(y_test, dt_preds, zero_division=0)))

    return {
        "noise": {"scales": noise_scales, "tcn": tcn_noise_f1, "dt": dt_noise_f1},
        "dropout": {"rates": dropout_rates, "tcn": tcn_drop_f1, "dt": dt_drop_f1},
        "drift": {"magnitudes": drift_magnitudes, "tcn": tcn_drift_f1, "dt": dt_drift_f1}
    }

def run_evaluation(config_path: str = "config.yaml"):
    """Execute all evaluation experiments and save tables and plots."""
    cfg = load_config(config_path)
    primary_seed = cfg.get("primary_seed", 42)
    set_seed(primary_seed)

    ds = prepare_dataset(cfg, seed=primary_seed)
    with open(cfg["paths"]["model_int8"], "rb") as f:
        tflite_bytes = f.read()

    # Train primary DT for direct comparison
    X_tr_feat = extract_statistical_features(ds["X_train"])
    X_val_feat = extract_statistical_features(ds["X_val"])
    dt_model = tune_decision_tree(X_tr_feat, ds["y_train"], X_val_feat, ds["y_val"], seed=primary_seed)

    threshold = 0.370
    X_test = ds["X_test"]
    y_test = ds["y_test"]

    # Evaluate confusion matrices
    tcn_probs, _ = run_tflite_inference(tflite_bytes, X_test)
    tcn_preds = (tcn_probs >= threshold).astype(int)
    cm_tcn = confusion_matrix(y_test, tcn_preds, labels=[0, 1])

    X_test_feat = extract_statistical_features(X_test)
    dt_preds = dt_model.predict(X_test_feat)
    cm_dt = confusion_matrix(y_test, dt_preds, labels=[0, 1])

    # Plot confusion matrices
    cms = {
        "NanoEdge-AD TCN (INT8)": cm_tcn,
        "Baseline B1: Decision Tree": cm_dt
    }
    plot_confusion_matrices(cms, os.path.join(cfg["paths"]["outputs_dir"], "confusion_matrices.png"))

    # E2: Per-anomaly recall
    df_recall = evaluate_per_anomaly_recall(ds, tflite_bytes, dt_model, cfg, threshold=threshold)
    recall_csv = os.path.join(cfg["paths"]["outputs_dir"], "per_anomaly_recall.csv")
    df_recall.to_csv(recall_csv, index=False)
    logger.info(f"Saved per-anomaly recall table to: {recall_csv}")

    # Plot per-anomaly recall
    plot_recall_path = os.path.join(cfg["paths"]["outputs_dir"], "per_anomaly_recall.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_recall))
    width = 0.35
    ax.bar(x - width/2, df_recall["TCN_INT8_Recall"], width, label="NanoEdge-AD TCN (INT8)", color="#1b9e77", edgecolor="black")
    ax.bar(x + width/2, df_recall["DecisionTree_Recall"], width, label="Baseline B1: Decision Tree", color="#7570b3", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", " ").title() for t in df_recall["anomaly_type"]], fontsize=10, fontweight="bold")
    ax.set_ylabel("Anomaly Detection Recall", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Anomaly-Type Detection Recall (Test Split)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_recall_path, dpi=180)
    plt.close()
    logger.info(f"Saved per-anomaly recall plot to: {plot_recall_path}")

    # E7: Robustness curves
    rob = run_robustness_sweeps(ds, tflite_bytes, dt_model, cfg, threshold=threshold)
    plot_rob_path = os.path.join(cfg["paths"]["outputs_dir"], "robustness_curves.png")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Noise
    axes[0].plot(rob["noise"]["scales"], rob["noise"]["tcn"], "o-", color="#1b9e77", linewidth=2, label="TCN (INT8)")
    axes[0].plot(rob["noise"]["scales"], rob["noise"]["dt"], "s--", color="#7570b3", linewidth=2, label="Decision Tree")
    axes[0].set_xlabel("Noise Std Dev ($\sigma$)", fontsize=10)
    axes[0].set_ylabel("Detection F1 Score", fontsize=10)
    axes[0].set_title("Robustness to Additive Noise", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=9)

    # Dropout
    axes[1].plot(rob["dropout"]["rates"], rob["dropout"]["tcn"], "o-", color="#1b9e77", linewidth=2, label="TCN (INT8)")
    axes[1].plot(rob["dropout"]["rates"], rob["dropout"]["dt"], "s--", color="#7570b3", linewidth=2, label="Decision Tree")
    axes[1].set_xlabel("Sensor Dropout Probability", fontsize=10)
    axes[1].set_ylabel("Detection F1 Score", fontsize=10)
    axes[1].set_title("Robustness to Sensor Dropout", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=9)

    # Drift
    axes[2].plot(rob["drift"]["magnitudes"], rob["drift"]["tcn"], "o-", color="#1b9e77", linewidth=2, label="TCN (INT8)")
    axes[2].plot(rob["drift"]["magnitudes"], rob["drift"]["dt"], "s--", color="#7570b3", linewidth=2, label="Decision Tree")
    axes[2].set_xlabel("Slow Drift Offset Magnitude", fontsize=10)
    axes[2].set_ylabel("Detection F1 Score", fontsize=10)
    axes[2].set_title("Robustness to Sensor Drift", fontsize=11, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(plot_rob_path, dpi=180)
    plt.close()
    logger.info(f"Saved robustness curves to: {plot_rob_path}")

    print("\n==================== PER-ANOMALY RECALL TABLE ====================")
    print(df_recall.to_markdown(index=False))
    print("==================================================================\n")

    return df_recall, rob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_evaluation(args.config)

if __name__ == "__main__":
    main()

