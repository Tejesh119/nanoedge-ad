"""Diagnostic experiments investigating why TCN trails the baselines.

Evaluates pooling methods, masking probabilities, gate regularization and initialization,
and capacity/receptive field across all 5 seeds.
Saves outputs/aggregate/diagnostics.csv and LaTeX table table_diagnostics.tex.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, Model, regularizers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from src.utils import load_config, set_seed, setup_logger
from src.windowing import prepare_dataset
from src.models import (
    SensorGate,
    residual_block_1d,
    extract_statistical_features,
    tune_decision_tree
)
from src.train import (
    apply_channel_masking_augmentation,
    find_best_threshold,
    compute_classification_metrics
)
from src.quantize import run_tflite_inference
from src.multiseed import quantize_model_to_int8, evaluate_int8_model, eval_tflite_probs

logger = setup_logger("diagnostics")

def build_diagnostic_tcn(
    input_shape: tuple = (64, 4),
    blocks_cfg: list = None,
    num_classes: int = 2,
    gate_l1: float = 0.001,
    gate_init_val: float = 1.0,
    pooling_type: str = "avg",  # "avg", "max", "concat"
    use_gate: bool = True
) -> Model:
    """Build parameterized TCN for diagnostic ablations."""
    if blocks_cfg is None:
        blocks_cfg = [
            {"filters": 16, "kernel_size": 3, "dilation_rate": 1},
            {"filters": 16, "kernel_size": 3, "dilation_rate": 2},
            {"filters": 24, "kernel_size": 3, "dilation_rate": 4}
        ]

    inputs = layers.Input(shape=input_shape, name="diag_sensor_input")
    if use_gate:
        # Custom gate with configurable initializer
        class CustomGate(SensorGate):
            def build(self, inp_shape):
                self.gate_weights = self.add_weight(
                    name="custom_gate_weights",
                    shape=(self.num_channels,),
                    initializer=tf.keras.initializers.Constant(gate_init_val),
                    regularizer=regularizers.l1(self.l1_reg),
                    trainable=True
                )
                self.built = True
        x = CustomGate(num_channels=input_shape[-1], l1_reg=gate_l1, name="sensor_gate")(inputs)
    else:
        x = inputs

    for i, b in enumerate(blocks_cfg):
        x = residual_block_1d(
            x,
            filters=b["filters"],
            kernel_size=b["kernel_size"],
            dilation_rate=b["dilation_rate"],
            name=f"diag_block_{i+1}"
        )

    if pooling_type == "max":
        x_pool = layers.GlobalMaxPooling1D(name="global_max_pool")(x)
    elif pooling_type == "concat":
        avg_p = layers.GlobalAveragePooling1D(name="concat_avg_pool")(x)
        max_p = layers.GlobalMaxPooling1D(name="concat_max_pool")(x)
        x_pool = layers.Concatenate(name="concat_pooling")([avg_p, max_p])
    else:  # avg
        x_pool = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="diag_head")(x_pool)
    return Model(inputs=inputs, outputs=outputs, name=f"TCN_Diag_{pooling_type}")

def run_diagnostic_variant(variant_name: str, config: dict, seeds: list,
                           blocks_cfg=None, pooling="avg", gate_l1=0.001,
                           gate_init=1.0, mask_prob=0.25) -> list:
    """Train and evaluate a specific diagnostic variant across all seeds in FP32 and INT8."""
    logger.info(f"--- Running Diagnostic Variant: {variant_name} across {len(seeds)} seeds ---")
    results = []

    for seed in seeds:
        set_seed(seed)
        ds = prepare_dataset(config, seed=seed)
        X_train, y_train = ds["X_train"], ds["y_train"]
        X_val, y_val = ds["X_val"], ds["y_val"]
        X_test, y_test = ds["X_test"], ds["y_test"]

        # Augmentation
        X_train_aug = apply_channel_masking_augmentation(X_train, y_train, prob=mask_prob, seed=seed)

        cls_w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
        cw_dict = {0: float(cls_w[0]), 1: float(cls_w[1])}

        model = build_diagnostic_tcn(
            input_shape=(64, 4),
            blocks_cfg=blocks_cfg,
            num_classes=2,
            gate_l1=gate_l1,
            gate_init_val=gate_init,
            pooling_type=pooling,
            use_gate=True
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="categorical_crossentropy")
        cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=0)]
        model.fit(
            X_train_aug, tf.keras.utils.to_categorical(y_train, 2),
            validation_data=(X_val, tf.keras.utils.to_categorical(y_val, 2)),
            batch_size=64, epochs=25, class_weight=cw_dict, callbacks=cb, verbose=0
        )

        # Gate values
        gate_layer = model.get_layer("sensor_gate")
        raw_w = gate_layer.gate_weights.numpy()
        eff_gates = 1.0 / (1.0 + np.exp(-4.0 * raw_w))

        # Threshold on val
        val_p = model.predict(X_val, verbose=0)[:, 1]
        best_t = find_best_threshold(y_val, val_p)

        # FP32 evaluation
        test_p = model.predict(X_test, verbose=0)[:, 1]
        test_preds_fp32 = (test_p >= best_t).astype(int)
        f1_fp32 = float(f1_score(y_test, test_preds_fp32, zero_division=0))
        rec_fp32 = float(recall_score(y_test, test_preds_fp32, zero_division=0))
        cm = confusion_matrix(y_test, test_preds_fp32, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr_fp32 = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        # Quantize to INT8
        int8_bytes = quantize_model_to_int8(model, X_train, calib_n=600)
        res_int8 = evaluate_int8_model(int8_bytes, X_val, y_val, X_test, y_test, threshold=best_t)

        results.append({
            "variant": variant_name,
            "seed": seed,
            "fp32_f1": f1_fp32,
            "fp32_recall": rec_fp32,
            "fp32_fpr": fpr_fp32,
            "int8_f1": res_int8["f1"],
            "int8_precision": res_int8["precision"],
            "int8_recall": res_int8["recall"],
            "int8_fpr": res_int8["fpr"],
            "gate_values": eff_gates.tolist(),
            "best_threshold": best_t,
            "int8_bytes": int8_bytes,
            "model_fp32": model,
            "ds": ds
        })

    return results

def run_all_diagnostics(config_path: str = "config.yaml", seeds: list = None):
    """Execute all diagnostic experiments and assemble summary tables."""
    if seeds is None:
        seeds = [42, 101, 777, 1234, 2024]
    cfg = load_config(config_path)
    os.makedirs("outputs/aggregate", exist_ok=True)

    default_blocks = [
        {"filters": 16, "kernel_size": 3, "dilation_rate": 1},
        {"filters": 16, "kernel_size": 3, "dilation_rate": 2},
        {"filters": 24, "kernel_size": 3, "dilation_rate": 4}
    ]

    all_variant_runs = {}

    # a) Pooling Variants
    all_variant_runs["Pooling: Global Max"] = run_diagnostic_variant(
        "Pooling: Global Max", cfg, seeds, pooling="max"
    )
    all_variant_runs["Pooling: Concat Avg+Max"] = run_diagnostic_variant(
        "Pooling: Concat Avg+Max", cfg, seeds, pooling="concat"
    )

    # b) Masking Probability Sweep
    all_variant_runs["Masking Prob: 0.0"] = run_diagnostic_variant(
        "Masking Prob: 0.0", cfg, seeds, mask_prob=0.0
    )
    all_variant_runs["Masking Prob: 0.10"] = run_diagnostic_variant(
        "Masking Prob: 0.10", cfg, seeds, mask_prob=0.10
    )
    all_variant_runs["Masking Prob: 0.25 (Current)"] = run_diagnostic_variant(
        "Masking Prob: 0.25 (Current)", cfg, seeds, mask_prob=0.25
    )

    # c) Gate Regularization and Initialization Sweep
    all_variant_runs["Gate L1: 0.0"] = run_diagnostic_variant(
        "Gate L1: 0.0", cfg, seeds, gate_l1=0.0
    )
    all_variant_runs["Gate L1: 1e-4"] = run_diagnostic_variant(
        "Gate L1: 1e-4", cfg, seeds, gate_l1=1e-4
    )
    all_variant_runs["Gate L1: 1e-3 (Current)"] = run_diagnostic_variant(
        "Gate L1: 1e-3 (Current)", cfg, seeds, gate_l1=1e-3
    )
    all_variant_runs["Gate L1: 1e-2"] = run_diagnostic_variant(
        "Gate L1: 1e-2", cfg, seeds, gate_l1=1e-2
    )
    all_variant_runs["Gate Init: 0.0 (Unbiased)"] = run_diagnostic_variant(
        "Gate Init: 0.0 (Unbiased)", cfg, seeds, gate_init=0.0
    )

    # d) Capacity & Receptive Field Variants
    wider_blocks = [
        {"filters": 32, "kernel_size": 3, "dilation_rate": 1},
        {"filters": 32, "kernel_size": 3, "dilation_rate": 2},
        {"filters": 48, "kernel_size": 3, "dilation_rate": 4}
    ]
    all_variant_runs["Capacity: Wider TCN (32-48 filters)"] = run_diagnostic_variant(
        "Capacity: Wider TCN (32-48 filters)", cfg, seeds, blocks_cfg=wider_blocks
    )

    full_rf_blocks = [
        {"filters": 16, "kernel_size": 3, "dilation_rate": 1},
        {"filters": 16, "kernel_size": 3, "dilation_rate": 2},
        {"filters": 24, "kernel_size": 3, "dilation_rate": 4},
        {"filters": 24, "kernel_size": 3, "dilation_rate": 8},
        {"filters": 24, "kernel_size": 3, "dilation_rate": 16}
    ]
    all_variant_runs["Receptive Field: Full-Window (d=1,2,4,8,16)"] = run_diagnostic_variant(
        "Receptive Field: Full-Window (d=1,2,4,8,16)", cfg, seeds, blocks_cfg=full_rf_blocks
    )

    # e) Quantization-Aware Training Check
    qat_status = "Skipped: tensorflow_model_optimization is not supported on Python 3.13 / TensorFlow 2.21"
    logger.info(f"QAT Status: {qat_status}")

    # Compile diagnostic records
    diag_summary = []
    flat_rows = []
    for vname, runs in all_variant_runs.items():
        fp32_f1s = [r["fp32_f1"] for r in runs]
        int8_f1s = [r["int8_f1"] for r in runs]
        int8_recs = [r["int8_recall"] for r in runs]
        int8_fprs = [r["int8_fpr"] for r in runs]

        m_f1_fp32 = np.mean(fp32_f1s)
        s_f1_fp32 = np.std(fp32_f1s, ddof=1) if len(fp32_f1s) > 1 else 0.0

        m_f1_int8 = np.mean(int8_f1s)
        s_f1_int8 = np.std(int8_f1s, ddof=1) if len(int8_f1s) > 1 else 0.0

        m_rec_int8 = np.mean(int8_recs)
        s_rec_int8 = np.std(int8_recs, ddof=1) if len(int8_recs) > 1 else 0.0

        m_fpr_int8 = np.mean(int8_fprs)
        s_fpr_int8 = np.std(int8_fprs, ddof=1) if len(int8_fprs) > 1 else 0.0

        diag_summary.append({
            "Variant": vname,
            "FP32_F1_Mean": round(float(m_f1_fp32), 4),
            "FP32_F1_Std": round(float(s_f1_fp32), 4),
            "INT8_F1_Mean": round(float(m_f1_int8), 4),
            "INT8_F1_Std": round(float(s_f1_int8), 4),
            "INT8_Recall_Mean": round(float(m_rec_int8), 4),
            "INT8_Recall_Std": round(float(s_rec_int8), 4),
            "INT8_FPR_Mean": round(float(m_fpr_int8), 4),
            "INT8_FPR_Std": round(float(s_fpr_int8), 4),
        })

        for r in runs:
            flat_rows.append({
                "variant": vname,
                "seed": r["seed"],
                "fp32_f1": r["fp32_f1"],
                "int8_f1": r["int8_f1"],
                "int8_recall": r["int8_recall"],
                "int8_fpr": r["int8_fpr"]
            })

    df_summary = pd.DataFrame(diag_summary)
    df_summary.sort_values(by="INT8_F1_Mean", ascending=False, inplace=True)
    df_summary.to_csv("outputs/aggregate/diagnostics.csv", index=False)
    pd.DataFrame(flat_rows).to_csv("outputs/aggregate/diagnostics_raw_seeds.csv", index=False)

    # Generate LaTeX table
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Diagnostic Experiments: Evaluating Architectural and Training Hypotheses (5 Seeds, Mean $\pm$ Std)}",
        r"\label{tab:diagnostics}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Diagnostic Variant} & \textbf{FP32 F1} & \textbf{INT8 F1} & \textbf{INT8 Recall} & \textbf{INT8 FPR} \\",
        r"\midrule"
    ]
    for _, row in df_summary.iterrows():
        v = row["Variant"]
        f_fp = f"{row['FP32_F1_Mean']:.4f} $\\pm$ {row['FP32_F1_Std']:.4f}"
        f_in = f"{row['INT8_F1_Mean']:.4f} $\\pm$ {row['INT8_F1_Std']:.4f}"
        r_in = f"{row['INT8_Recall_Mean']:.4f} $\\pm$ {row['INT8_Recall_Std']:.4f}"
        fpr_in = f"{row['INT8_FPR_Mean']:.4f} $\\pm$ {row['INT8_FPR_Std']:.4f}"
        tex_lines.append(f"{v} & {f_fp} & {f_in} & {r_in} & {fpr_in} \\\\")
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    with open("outputs/aggregate/table_diagnostics.tex", "w") as f:
        f.write("\n".join(tex_lines))

    # f) Per-type recall for best variant vs original TCN vs Decision Tree
    best_variant_name = df_summary.iloc[0]["Variant"]
    logger.info(f"Top-performing Diagnostic Variant: {best_variant_name}")
    best_runs = all_variant_runs[best_variant_name]

    # Evaluate per-anomaly-type recall on the top variant
    anomaly_types = ["spike", "drift", "stuck_at", "noise_burst", "dropout"]
    best_type_recalls = {atype: [] for atype in anomaly_types}

    for r in best_runs:
        ds = r["ds"]
        types_test = np.array(ds["types_test"])
        y_test = ds["y_test"]
        X_test = ds["X_test"]
        interp_best = tf.lite.Interpreter(model_content=r["int8_bytes"])
        interp_best.allocate_tensors()
        probs = eval_tflite_probs(interp_best, interp_best.get_input_details()[0], interp_best.get_output_details()[0], X_test)
        preds = (probs >= r["best_threshold"]).astype(int)

        for atype in anomaly_types:
            idx = np.where((types_test == atype) & (y_test == 1))[0]
            if len(idx) > 0:
                rec_val = float(recall_score(y_test[idx], preds[idx], zero_division=0))
                best_type_recalls[atype].append(rec_val)

    best_type_summary = []
    for atype in anomaly_types:
        vals = best_type_recalls[atype]
        best_type_summary.append({
            "anomaly_type": atype,
            "best_variant_name": best_variant_name,
            "recall_mean": float(np.mean(vals)),
            "recall_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        })
    pd.DataFrame(best_type_summary).to_csv("outputs/aggregate/best_variant_per_type_recall.csv", index=False)

    logger.info("Diagnostics completed successfully! Results written to outputs/aggregate/diagnostics.csv and LaTeX table.")
    return df_summary, best_variant_name

def main():
    parser = argparse.ArgumentParser(description="Diagnostics Experiments")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 777, 1234, 2024])
    args = parser.parse_args()

    run_all_diagnostics(args.config, args.seeds)

if __name__ == "__main__":
    main()

