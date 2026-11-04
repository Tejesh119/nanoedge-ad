"""Component ablation study evaluating incremental contributions of Gate, INT8, and Pruning."""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import f1_score

from src.utils import load_config, setup_logger, set_seed, compute_tcn_macs
from src.windowing import prepare_dataset
from src.models import build_tcn_model, SensorGate
from src.quantize import run_tflite_inference
from src.runtime_pruning import PrunePolicy

logger = setup_logger("ablation")

def run_ablation_study(config_path: str = "config.yaml"):
    """Evaluate incremental architectural and systems components."""
    cfg = load_config(config_path)
    primary_seed = cfg.get("primary_seed", 42)
    set_seed(primary_seed)

    ds = prepare_dataset(cfg, seed=primary_seed)
    X_train = ds["X_train"]
    y_train = ds["y_train"]
    X_val = ds["X_val"]
    y_val = ds["y_val"]
    X_test = ds["X_test"]
    y_test = ds["y_test"]

    threshold = 0.370
    e_read = cfg["benchmark"]["energy_proxy"]["e_read_uj"]
    e_infer = cfg["benchmark"]["energy_proxy"]["e_infer_uj"]
    n_windows = len(X_test)
    total_reads_baseline = n_windows * 4 * 32

    results = []

    # 1. Config A: Vanilla TCN (no SensorGate, no channel masking)
    logger.info("Evaluating Ablation Config A: Vanilla TCN...")
    m_a = build_tcn_model(use_gate=False)
    m_a.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="categorical_crossentropy")
    m_a.fit(X_train, tf.keras.utils.to_categorical(y_train, 2),
            validation_data=(X_val, tf.keras.utils.to_categorical(y_val, 2)),
            epochs=15, batch_size=64, verbose=0)
    
    probs_a = m_a.predict(X_test, verbose=0)[:, 1]
    f1_a = float(f1_score(y_test, (probs_a >= threshold).astype(int), zero_division=0))
    params_a = m_a.count_params()
    size_a_kb = round((params_a * 4.0) / 1024.0, 2)
    macs_a = compute_tcn_macs()["total_macs"] - 256  # without 256 gate MACs
    energy_a = ((total_reads_baseline * e_read) + (n_windows * e_infer * 1.5)) / 1000.0  # FP32 inference takes ~1.5x energy

    results.append({
        "Configuration": "A: Vanilla TCN (FP32)",
        "F1_Score": round(f1_a, 4),
        "Parameters": params_a,
        "Flash_Size_KB": size_a_kb,
        "Analytic_MACs": macs_a,
        "Reads_Saved_Pct": 0.0,
        "Energy_Proxy_mJ": round(energy_a, 2)
    })

    # 2. Config B: Vanilla TCN + INT8 Quantization
    logger.info("Evaluating Ablation Config B: Vanilla TCN + INT8...")
    def rep_data_a():
        for i in range(500):
            yield [X_train[i:i+1].astype(np.float32)]
    conv_b = tf.lite.TFLiteConverter.from_keras_model(m_a)
    conv_b.optimizations = [tf.lite.Optimize.DEFAULT]
    conv_b.representative_dataset = rep_data_a
    conv_b.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    conv_b.inference_input_type = tf.int8
    conv_b.inference_output_type = tf.int8
    bytes_b = conv_b.convert()

    probs_b, _ = run_tflite_inference(bytes_b, X_test)
    f1_b = float(f1_score(y_test, (probs_b >= threshold).astype(int), zero_division=0))
    size_b_kb = round(len(bytes_b) / 1024.0, 2)
    energy_b = ((total_reads_baseline * e_read) + (n_windows * e_infer)) / 1000.0

    results.append({
        "Configuration": "B: TCN + INT8",
        "F1_Score": round(f1_b, 4),
        "Parameters": params_a,
        "Flash_Size_KB": size_b_kb,
        "Analytic_MACs": macs_a,
        "Reads_Saved_Pct": 0.0,
        "Energy_Proxy_mJ": round(energy_b, 2)
    })

    # 3. Config C: TCN + SensorGate & Masking Augmentation (FP32)
    logger.info("Evaluating Ablation Config C: TCN + SensorGate & Masking...")
    # Load primary trained FP32 model
    m_c = tf.keras.models.load_model(cfg["paths"]["model_fp32"], custom_objects={"SensorGate": SensorGate})
    probs_c = m_c.predict(X_test, verbose=0)[:, 1]
    f1_c = float(f1_score(y_test, (probs_c >= threshold).astype(int), zero_division=0))
    params_c = m_c.count_params()
    size_c_kb = round((params_c * 4.0) / 1024.0, 2)
    macs_c = compute_tcn_macs()["total_macs"]
    energy_c = ((total_reads_baseline * e_read) + (n_windows * e_infer * 1.5)) / 1000.0

    results.append({
        "Configuration": "C: TCN + SensorGate (FP32)",
        "F1_Score": round(f1_c, 4),
        "Parameters": params_c,
        "Flash_Size_KB": size_c_kb,
        "Analytic_MACs": macs_c,
        "Reads_Saved_Pct": 0.0,
        "Energy_Proxy_mJ": round(energy_c, 2)
    })

    # 4. Config D: TCN + SensorGate + INT8 Quantization
    logger.info("Evaluating Ablation Config D: TCN + SensorGate + INT8...")
    with open(cfg["paths"]["model_int8"], "rb") as f:
        bytes_d = f.read()
    probs_d, _ = run_tflite_inference(bytes_d, X_test)
    f1_d = float(f1_score(y_test, (probs_d >= threshold).astype(int), zero_division=0))
    size_d_kb = round(len(bytes_d) / 1024.0, 2)
    energy_d = ((total_reads_baseline * e_read) + (n_windows * e_infer)) / 1000.0

    results.append({
        "Configuration": "D: TCN + SensorGate + INT8",
        "F1_Score": round(f1_d, 4),
        "Parameters": params_c,
        "Flash_Size_KB": size_d_kb,
        "Analytic_MACs": macs_c,
        "Reads_Saved_Pct": 0.0,
        "Energy_Proxy_mJ": round(energy_d, 2)
    })

    # 5. Config E: Full NanoEdge-AD (TCN + SensorGate + INT8 + Runtime Pruning Policy)
    logger.info("Evaluating Ablation Config E: Full NanoEdge-AD...")
    # Load simulation results
    sim_df = pd.read_csv("outputs/runtime_simulation.csv")
    f1_e = float(sim_df["f1_policy"].iloc[0])
    reads_saved_e = float(sim_df["reads_saved_pct"].iloc[0])
    energy_e = float(sim_df["energy_proxy_policy_mJ"].iloc[0])

    results.append({
        "Configuration": "E: NanoEdge-AD (Full System)",
        "F1_Score": round(f1_e, 4),
        "Parameters": params_c,
        "Flash_Size_KB": size_d_kb,
        "Analytic_MACs": macs_c,
        "Reads_Saved_Pct": round(reads_saved_e, 2),
        "Energy_Proxy_mJ": round(energy_e, 2)
    })

    df_ablation = pd.DataFrame(results)
    csv_path = os.path.join(cfg["paths"]["outputs_dir"], "ablation.csv")
    df_ablation.to_csv(csv_path, index=False)
    logger.info(f"Saved ablation study table to: {csv_path}")

    # Plot Ablation Comparison (F1, Memory Footprint, Energy)
    plot_path = os.path.join(cfg["paths"]["outputs_dir"], "ablation_waterfall.png")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    configs = ["A (Base)", "B (+INT8)", "C (+Gate)", "D (+Gate+INT8)", "E (Full System)"]

    # F1 Score
    axes[0].bar(configs, df_ablation["F1_Score"], color="#1b9e77", edgecolor="black", alpha=0.85)
    axes[0].set_ylabel("Anomaly Detection F1 Score", fontsize=10)
    axes[0].set_title("Detection Performance", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, 0.8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(df_ablation["F1_Score"]):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    # Flash Footprint (KB)
    axes[1].bar(configs, df_ablation["Flash_Size_KB"], color="#d95f02", edgecolor="black", alpha=0.85)
    axes[1].set_ylabel("Model Storage Footprint (KB)", fontsize=10)
    axes[1].set_title("Flash Memory Footprint Reduction", fontsize=11, fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(df_ablation["Flash_Size_KB"]):
        axes[1].text(i, v + 1.0, f"{v:.1f} KB", ha="center", fontsize=9, fontweight="bold")

    # Energy Proxy (mJ)
    axes[2].bar(configs, df_ablation["Energy_Proxy_mJ"], color="#7570b3", edgecolor="black", alpha=0.85)
    axes[2].set_ylabel("Energy Proxy per Test Stream (mJ)", fontsize=10)
    axes[2].set_title("Edge Energy Consumption", fontsize=11, fontweight="bold")
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(df_ablation["Energy_Proxy_mJ"]):
        axes[2].text(i, v + 30.0, f"{v:.0f} mJ", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()
    logger.info(f"Saved ablation visualization to: {plot_path}")

    print("\n==================== COMPONENT ABLATION SUMMARY ====================")
    print(df_ablation.to_markdown(index=False))
    print("====================================================================\n")

    return df_ablation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_ablation_study(args.config)

if __name__ == "__main__":
    main()

