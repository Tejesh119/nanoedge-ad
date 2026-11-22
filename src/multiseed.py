"""Multi-seed pipeline execution for NanoEdge-AD across 5 seeds.

Runs all baselines, FP32 & INT8 TCN, B2/B4 INT8 quantization, full ablation (A-E),
quantization fidelity, gate-ranked pruning vs 25 random subsets, streaming PrunePolicy,
per-anomaly-type recall, and robustness sweeps for every seed.
Outputs are saved cleanly under outputs/seed_<N>/.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from src.utils import (
    load_config,
    set_seed,
    setup_logger,
    compute_tcn_macs,
    compute_cnn_macs,
    compute_ae_macs,
    compute_tree_feature_ops
)
from src.windowing import prepare_dataset
from src.models import (
    SensorGate,
    build_tcn_model,
    build_baseline_cnn,
    build_dense_autoencoder,
    extract_statistical_features,
    tune_decision_tree
)
from src.quantize import run_tflite_inference
from src.runtime_pruning import PrunePolicy
from src.train import (
    apply_channel_masking_augmentation,
    compute_classification_metrics,
    find_best_threshold,
    train_baseline_static_pruned_cnn
)

logger = setup_logger("multiseed")

def quantize_model_to_int8(keras_model, X_train, calib_n: int = 600) -> bytes:
    """Perform post-training integer quantization with representative calibration dataset."""
    def rep_dataset():
        for i in range(min(calib_n, len(X_train))):
            yield [X_train[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()

def eval_tflite_probs(interp, in_det, out_det, X_data: np.ndarray) -> np.ndarray:
    """Fast batch inference using an existing allocated TFLite Interpreter."""
    in_scale, in_zp = in_det["quantization"]
    out_scale, out_zp = out_det["quantization"]
    probs = []
    for i in range(len(X_data)):
        samp = X_data[i:i+1].astype(np.float32)
        if in_det["dtype"] == np.int8 and in_scale > 0:
            qs = np.clip(np.round(samp / in_scale) + in_zp, -128, 127).astype(np.int8)
        else:
            qs = samp
        interp.set_tensor(in_det["index"], qs)
        interp.invoke()
        out = interp.get_tensor(out_det["index"])
        if out_det["dtype"] == np.int8 and out_scale > 0:
            deq = (out.astype(np.float32) - out_zp) * out_scale
        else:
            deq = out.astype(np.float32)
        probs.append(float(deq[0, 1]))
    return np.array(probs, dtype=np.float32)

def evaluate_int8_model(tflite_bytes: bytes, X_val: np.ndarray, y_val: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        threshold: float = None) -> dict:
    """Evaluate INT8 TFLite model on validation and test sets using fast pre-allocated interpreter."""
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    val_probs = eval_tflite_probs(interp, in_det, out_det, X_val)
    if threshold is None:
        best_t = find_best_threshold(y_val, val_probs)
    else:
        best_t = threshold

    test_probs = eval_tflite_probs(interp, in_det, out_det, X_test)
    test_preds = (test_probs >= best_t).astype(int)

    metrics = compute_classification_metrics(y_test, test_preds, test_probs)
    metrics["best_threshold"] = best_t
    metrics["val_f1"] = float(f1_score(y_val, (val_probs >= best_t).astype(int), zero_division=0))
    metrics["quant_params"] = {
        "input_scale": float(in_det["quantization"][0]),
        "input_zero_point": int(in_det["quantization"][1]),
        "output_scale": float(out_det["quantization"][0]),
        "output_zero_point": int(out_det["quantization"][1])
    }
    metrics["test_probs"] = test_probs
    metrics["test_preds"] = test_preds
    return metrics

def run_single_seed_pipeline(seed: int, config: dict, out_base_dir: str = "outputs") -> dict:
    """Execute complete experimental pipeline for a single random seed."""
    logger.info(f"==================================================")
    logger.info(f"   STARTING RUN FOR SEED: {seed}")
    logger.info(f"==================================================")

    seed_dir = os.path.join(out_base_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    set_seed(seed)
    channel_names = config["data"]["channels"]
    train_cfg = config["training"]
    model_cfg = config["model"]
    prune_cfg = config.get("pruning", {})
    e_read = config["benchmark"]["energy_proxy"]["e_read_uj"]
    e_infer = config["benchmark"]["energy_proxy"]["e_infer_uj"]

    # 1. Dataset generation and chronological 60/20/20 split
    ds = prepare_dataset(config, seed=seed)
    X_train, y_train = ds["X_train"], ds["y_train"]
    X_val, y_val = ds["X_val"], ds["y_val"]
    X_test, y_test = ds["X_test"], ds["y_test"]
    n_test_windows = len(X_test)
    baseline_reads_total = n_test_windows * 4 * 32

    seed_records = []

    # ----------------------------------------------------
    # Model 1: Baseline B1 - Decision Tree
    # ----------------------------------------------------
    logger.info(f"[Seed {seed}] Training Baseline B1: Decision Tree...")
    X_tr_feat = extract_statistical_features(X_train)
    X_va_feat = extract_statistical_features(X_val)
    X_te_feat = extract_statistical_features(X_test)

    dt_model = tune_decision_tree(X_tr_feat, y_train, X_va_feat, y_val, seed=seed)
    dt_val_probs = dt_model.predict_proba(X_va_feat)[:, 1]
    dt_thresh = find_best_threshold(y_val, dt_val_probs)
    dt_test_probs = dt_model.predict_proba(X_te_feat)[:, 1]
    dt_test_preds = (dt_test_probs >= dt_thresh).astype(int)

    dt_metrics = compute_classification_metrics(y_test, dt_test_preds, dt_test_probs)
    dt_metrics["best_threshold"] = dt_thresh
    dt_nodes = int(dt_model.tree_.node_count)
    dt_depth = int(dt_model.tree_.max_depth)
    dt_size_kb = round((dt_nodes * 16) / 1024.0, 2)  # ~16 bytes per decision node in C
    dt_feat_ops = compute_tree_feature_ops(64, 4)["total_feature_ops"]

    dt_rec = {
        "model": "B1_DecisionTree",
        "seed": seed,
        "nodes": dt_nodes,
        "depth": dt_depth,
        "size_kb": dt_size_kb,
        "feature_ops": dt_feat_ops,
        **{k: dt_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "best_threshold"]}
    }
    seed_records.append(dt_rec)

    # ----------------------------------------------------
    # Model 2: Baseline B2 - Vanilla 1D-CNN (FP32 & INT8)
    # ----------------------------------------------------
    logger.info(f"[Seed {seed}] Training Baseline B2: Vanilla 1D-CNN...")
    set_seed(seed)
    cls_w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    cw_dict = {0: float(cls_w[0]), 1: float(cls_w[1])}

    b2_model = build_baseline_cnn(input_shape=(64, 4), num_classes=2)
    b2_model.compile(
        optimizer=tf.keras.optimizers.Adam(train_cfg["learning_rate"]),
        loss="categorical_crossentropy"
    )
    b2_cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=train_cfg["early_stopping_patience"], restore_best_weights=True, verbose=0)]
    b2_model.fit(
        X_train, tf.keras.utils.to_categorical(y_train, 2),
        validation_data=(X_val, tf.keras.utils.to_categorical(y_val, 2)),
        batch_size=train_cfg["batch_size"],
        epochs=train_cfg["epochs"],
        class_weight=cw_dict,
        callbacks=b2_cb,
        verbose=0
    )
    b2_val_probs = b2_model.predict(X_val, verbose=0)[:, 1]
    b2_thresh = find_best_threshold(y_val, b2_val_probs)
    b2_test_probs = b2_model.predict(X_test, verbose=0)[:, 1]
    b2_test_preds = (b2_test_probs >= b2_thresh).astype(int)

    b2_fp32_metrics = compute_classification_metrics(y_test, b2_test_preds, b2_test_probs)
    b2_fp32_metrics["best_threshold"] = b2_thresh
    b2_params = b2_model.count_params()
    b2_macs = compute_cnn_macs()["total_macs"]
    b2_fp32_size_kb = round((b2_params * 4) / 1024.0, 2)

    seed_records.append({
        "model": "B2_VanillaCNN (FP32)",
        "seed": seed,
        "parameters": b2_params,
        "size_kb": b2_fp32_size_kb,
        "macs": b2_macs,
        **{k: b2_fp32_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "best_threshold"]}
    })

    # Quantize B2 to INT8
    logger.info(f"[Seed {seed}] Quantizing Baseline B2 to INT8...")
    b2_int8_bytes = quantize_model_to_int8(b2_model, X_train, calib_n=600)
    with open(os.path.join(seed_dir, "b2_int8.tflite"), "wb") as f:
        f.write(b2_int8_bytes)
    b2_int8_size_kb = round(len(b2_int8_bytes) / 1024.0, 2)
    b2_int8_res = evaluate_int8_model(b2_int8_bytes, X_val, y_val, X_test, y_test, threshold=b2_thresh)

    seed_records.append({
        "model": "B2_VanillaCNN (INT8)",
        "seed": seed,
        "parameters": b2_params,
        "size_kb": b2_int8_size_kb,
        "macs": b2_macs,
        **{k: b2_int8_res[k] for k in ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "best_threshold"]}
    })

    # ----------------------------------------------------
    # Model 3: Baseline B3 - Dense Autoencoder
    # ----------------------------------------------------
    logger.info(f"[Seed {seed}] Training Baseline B3: Dense Autoencoder...")
    set_seed(seed)
    normal_idx = np.where(y_train == 0)[0]
    X_tr_norm = X_train[normal_idx]

    b3_model = build_dense_autoencoder(input_shape=(64, 4))
    b3_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse")
    b3_cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)]
    b3_model.fit(
        X_tr_norm, X_tr_norm,
        validation_data=(X_val, X_val),
        batch_size=64,
        epochs=20,
        callbacks=b3_cb,
        verbose=0
    )
    val_recon = b3_model.predict(X_val, verbose=0)
    val_mse = np.mean(np.square(X_val - val_recon), axis=(1, 2))
    # Threshold on val
    b3_thresh = float(np.percentile(val_mse, 88.0))
    best_b3_f1 = -1.0
    for p in np.linspace(80, 98, 37):
        th = float(np.percentile(val_mse, p))
        f1_tmp = f1_score(y_val, (val_mse >= th).astype(int), zero_division=0)
        if f1_tmp > best_b3_f1:
            best_b3_f1 = f1_tmp
            b3_thresh = th

    test_recon = b3_model.predict(X_test, verbose=0)
    test_mse = np.mean(np.square(X_test - test_recon), axis=(1, 2))
    b3_test_preds = (test_mse >= b3_thresh).astype(int)
    norm_scores = (test_mse - np.min(test_mse)) / (np.max(test_mse) - np.min(test_mse) + 1e-8)

    b3_metrics = compute_classification_metrics(y_test, b3_test_preds, norm_scores)
    b3_metrics["best_threshold"] = float(b3_thresh)
    b3_params = b3_model.count_params()
    b3_macs = compute_ae_macs()["total_macs"]
    b3_size_kb = round((b3_params * 4) / 1024.0, 2)

    seed_records.append({
        "model": "B3_DenseAutoencoder",
        "seed": seed,
        "parameters": b3_params,
        "size_kb": b3_size_kb,
        "macs": b3_macs,
        **{k: b3_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "best_threshold"]}
    })

    # ----------------------------------------------------
    # Model 4: Baseline B4 - Static Pruned CNN (FP32 & INT8)
    # ----------------------------------------------------
    logger.info(f"[Seed {seed}] Creating Baseline B4: Static Pruned CNN...")
    b4_model, b4_fp32_metrics = train_baseline_static_pruned_cnn(b2_model, ds, prune_fraction=0.30)
    b4_macs = int(b2_macs * 0.70)
    b4_params = b4_model.count_params()
    b4_fp32_size_kb = b2_fp32_size_kb

    seed_records.append({
        "model": "B4_StaticPrunedCNN",
        "seed": seed,
        "parameters": b4_params,
        "size_kb": b4_fp32_size_kb,
        "macs": b4_macs,
        **{k: b4_fp32_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "best_threshold"]}
    })

    # Quantize B4 to INT8
    logger.info(f"[Seed {seed}] Quantizing Baseline B4 to INT8...")
    b4_int8_bytes = quantize_model_to_int8(b4_model, X_train, calib_n=600)
    with open(os.path.join(seed_dir, "b4_int8.tflite"), "wb") as f:
        f.write(b4_int8_bytes)
    b4_int8_size_kb = round(len(b4_int8_bytes) / 1024.0, 2)
    b4_int8_res = evaluate_int8_model(b4_int8_bytes, X_val, y_val, X_test, y_test, threshold=b4_fp32_metrics["best_threshold"])

    seed_records.append({
        "model": "B4_StaticPrunedCNN (INT8)",
        "seed": seed,
        "parameters": b4_params,
        "size_kb": b4_int8_size_kb,
        "macs": b4_macs,
        **{k: b4_int8_res[k] for k in ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "best_threshold"]}
    })

    # ----------------------------------------------------
    # Model 5: Proposed TCN + SensorGate (FP32)
    # ----------------------------------------------------
    logger.info(f"[Seed {seed}] Training TCN with SensorGate & Masking Augmentation...")
    set_seed(seed)
    X_train_aug = apply_channel_masking_augmentation(
        X_train, y_train,
        prob=train_cfg.get("channel_masking_prob", 0.25),
        seed=seed
    )
    tcn_model = build_tcn_model(
        input_shape=(64, 4),
        blocks_cfg=model_cfg["blocks"],
        num_classes=2,
        gate_l1=model_cfg.get("gate_l1", 0.001),
        use_gate=True
    )
    tcn_model.compile(
        optimizer=tf.keras.optimizers.Adam(train_cfg["learning_rate"]),
        loss="categorical_crossentropy"
    )
    tcn_cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=train_cfg["early_stopping_patience"], restore_best_weights=True, verbose=0)]
    tcn_model.fit(
        X_train_aug, tf.keras.utils.to_categorical(y_train, 2),
        validation_data=(X_val, tf.keras.utils.to_categorical(y_val, 2)),
        batch_size=train_cfg["batch_size"],
        epochs=train_cfg["epochs"],
        class_weight=cw_dict,
        callbacks=tcn_cb,
        verbose=0
    )

    # Save FP32 model
    tcn_model.save(os.path.join(seed_dir, "model_fp32.keras"))

    # Extract learned gate values
    gate_layer = tcn_model.get_layer("sensor_gate")
    gate_vals = gate_layer.get_gate_values()
    gate_dict = {channel_names[i]: float(gate_vals[i]) for i in range(len(channel_names))}
    with open(os.path.join(seed_dir, "gates.json"), "w") as f:
        json.dump(gate_dict, f, indent=2)

    # Threshold on val
    tcn_val_probs = tcn_model.predict(X_val, verbose=0)[:, 1]
    tcn_thresh = find_best_threshold(y_val, tcn_val_probs)
    tcn_test_probs = tcn_model.predict(X_test, verbose=0)[:, 1]
    tcn_test_preds = (tcn_test_probs >= tcn_thresh).astype(int)

    tcn_fp32_metrics = compute_classification_metrics(y_test, tcn_test_preds, tcn_test_probs)
    tcn_fp32_metrics["best_threshold"] = tcn_thresh
    tcn_params = tcn_model.count_params()
    tcn_macs = compute_tcn_macs()["total_macs"]
    tcn_fp32_size_kb = round((tcn_params * 4) / 1024.0, 2)

    seed_records.append({
        "model": "TCN_SensorGate (FP32)",
        "seed": seed,
        "parameters": tcn_params,
        "size_kb": tcn_fp32_size_kb,
        "macs": tcn_macs,
        **{k: tcn_fp32_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "best_threshold"]}
    })

    # ----------------------------------------------------
    # Model 6: Proposed NanoEdge-AD TCN (INT8)
    # ----------------------------------------------------
    logger.info(f"[Seed {seed}] Quantizing TCN with SensorGate to INT8...")
    tcn_int8_bytes = quantize_model_to_int8(tcn_model, X_train, calib_n=600)
    with open(os.path.join(seed_dir, "model_int8.tflite"), "wb") as f:
        f.write(tcn_int8_bytes)
    tcn_int8_size_kb = round(len(tcn_int8_bytes) / 1024.0, 2)

    tcn_int8_res = evaluate_int8_model(tcn_int8_bytes, X_val, y_val, X_test, y_test, threshold=tcn_thresh)
    agreement = float(np.mean(tcn_int8_res["test_preds"] == tcn_test_preds) * 100.0)
    f1_drop = float(tcn_fp32_metrics["f1"] - tcn_int8_res["f1"])
    acc_drop = float(tcn_fp32_metrics["accuracy"] - tcn_int8_res["accuracy"])

    seed_records.append({
        "model": "NanoEdge-AD TCN (INT8)",
        "seed": seed,
        "parameters": tcn_params,
        "size_kb": tcn_int8_size_kb,
        "macs": tcn_macs,
        **{k: tcn_int8_res[k] for k in ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "best_threshold"]}
    })

    # Quantization fidelity record
    quant_info = {
        "seed": seed,
        "fp32_f1": tcn_fp32_metrics["f1"],
        "int8_f1": tcn_int8_res["f1"],
        "f1_drop": f1_drop,
        "fp32_accuracy": tcn_fp32_metrics["accuracy"],
        "int8_accuracy": tcn_int8_res["accuracy"],
        "accuracy_drop": acc_drop,
        "agreement_pct": agreement,
        "flatbuffer_size_kb": tcn_int8_size_kb,
        "quant_params": tcn_int8_res["quant_params"]
    }
    with open(os.path.join(seed_dir, "quant_info.json"), "w") as f:
        json.dump(quant_info, f, indent=2)

    # ----------------------------------------------------
    # Full Ablation Study (Configs A to E)
    # ----------------------------------------------------
    logger.info(f"[Seed {seed}] Running Full Ablation Study (Configs A to E)...")
    # Config A: Vanilla TCN FP32 (no gate, no masking)
    set_seed(seed)
    m_vanilla = build_tcn_model(input_shape=(64, 4), use_gate=False)
    m_vanilla.compile(optimizer=tf.keras.optimizers.Adam(train_cfg["learning_rate"]), loss="categorical_crossentropy")
    m_vanilla.fit(
        X_train, tf.keras.utils.to_categorical(y_train, 2),
        validation_data=(X_val, tf.keras.utils.to_categorical(y_val, 2)),
        batch_size=train_cfg["batch_size"], epochs=15, verbose=0
    )
    v_val_p = m_vanilla.predict(X_val, verbose=0)[:, 1]
    v_th = find_best_threshold(y_val, v_val_p)
    v_te_p = m_vanilla.predict(X_test, verbose=0)[:, 1]
    v_te_preds = (v_te_p >= v_th).astype(int)
    f1_a = float(f1_score(y_test, v_te_preds, zero_division=0))
    params_a = m_vanilla.count_params()
    size_a_kb = round((params_a * 4) / 1024.0, 2)
    macs_a = tcn_macs - 256
    energy_a = ((baseline_reads_total * e_read) + (n_test_windows * e_infer * 1.5)) / 1000.0

    # Config B: Vanilla TCN INT8
    bytes_b = quantize_model_to_int8(m_vanilla, X_train, calib_n=500)
    res_b = evaluate_int8_model(bytes_b, X_val, y_val, X_test, y_test, threshold=v_th)
    f1_b = res_b["f1"]
    size_b_kb = round(len(bytes_b) / 1024.0, 2)
    energy_b = ((baseline_reads_total * e_read) + (n_test_windows * e_infer)) / 1000.0

    # Config C: TCN + SensorGate & Masking FP32
    f1_c = tcn_fp32_metrics["f1"]
    params_c = tcn_params
    size_c_kb = tcn_fp32_size_kb
    macs_c = tcn_macs
    energy_c = energy_a

    # Config D: TCN + SensorGate & Masking INT8
    f1_d = tcn_int8_res["f1"]
    size_d_kb = tcn_int8_size_kb
    macs_d = tcn_macs
    energy_d = energy_b

    # Config E: Config D + Runtime PrunePolicy
    # (Simulated below, filled in ablation table)
    # ----------------------------------------------------
    # Offline Gate-Ranked Pruning vs 25 Random Subsets
    # ----------------------------------------------------
    logger.info(f"[Seed {seed}] Running Offline Pruning Sweep vs 25 Random Subsets...")
    sorted_channels_by_gate = sorted(channel_names, key=lambda c: gate_dict[c])
    gate_rank_indices = [channel_names.index(c) for c in sorted_channels_by_gate]

    # Pre-allocate TFLite Interpreter for fast batch inference
    prune_interp = tf.lite.Interpreter(model_content=tcn_int8_bytes)
    prune_interp.allocate_tensors()
    p_in_det = prune_interp.get_input_details()[0]
    p_out_det = prune_interp.get_output_details()[0]
    p_in_scale, p_in_zp = p_in_det["quantization"]
    p_out_scale, p_out_zp = p_out_det["quantization"]

    def eval_masked(ch_list):
        if not ch_list:
            X_m = X_test
        else:
            X_m = X_test.copy()
            X_m[:, :, ch_list] = 0.0
        probs_m = eval_tflite_probs(prune_interp, p_in_det, p_out_det, X_m)
        return float(f1_score(y_test, (probs_m >= tcn_thresh).astype(int), zero_division=0))

    pruning_rows = []
    # k=0
    pruning_rows.append({
        "seed": seed,
        "num_pruned": 0,
        "pct_reads_saved": 0.0,
        "gate_ranked_f1": f1_d,
        "random_f1_mean": f1_d,
        "random_f1_std": 0.0,
        "gate_advantage": 0.0,
        "random_std_advantage": 0.0,
        "pruned_channels": "none"
    })

    rng_mc = np.random.RandomState(seed)
    for k in range(1, 4):
        p_idx = gate_rank_indices[:k]
        p_names = [channel_names[idx] for idx in p_idx]
        pct_s = (k / 4.0) * 100.0
        f1_gate_k = eval_masked(p_idx)

        # 25 Monte Carlo draws
        draws = []
        for _ in range(25):
            r_idx = rng_mc.choice(4, size=k, replace=False).tolist()
            draws.append(eval_masked(r_idx))
        r_mean = float(np.mean(draws))
        r_std = float(np.std(draws, ddof=1)) if len(draws) > 1 else 0.0
        adv = f1_gate_k - r_mean
        adv_std = float(adv / r_std) if r_std > 1e-6 else 0.0

        pruning_rows.append({
            "seed": seed,
            "num_pruned": k,
            "pct_reads_saved": pct_s,
            "gate_ranked_f1": f1_gate_k,
            "random_f1_mean": r_mean,
            "random_f1_std": r_std,
            "gate_advantage": adv,
            "random_std_advantage": adv_std,
            "pruned_channels": "+".join(p_names)
        })

    df_prune = pd.DataFrame(pruning_rows)
    df_prune.to_csv(os.path.join(seed_dir, "pruning_sweep.csv"), index=False)

    # ----------------------------------------------------
    # Streaming Simulation with PrunePolicy
    # ----------------------------------------------------
    logger.info(f"[Seed {seed}] Running Streaming Simulation with PrunePolicy...")
    policy = PrunePolicy(
        num_channels=4,
        gate_rankings=gate_rank_indices,
        quiet_window_count=prune_cfg.get("quiet_window_count", 5),
        jump_z_threshold=prune_cfg.get("jump_z_threshold", 2.8)
    )

    stride = config["windowing"].get("stride", 32)
    sample_rate = config["data"].get("sample_rate_hz", 100)
    stride_sec = stride / sample_rate

    sim_preds = []
    reads_taken = 0
    wake_triggers = 0
    active_masks = []
    detection_delays = []
    ongoing_anomaly = False
    anomaly_start_win = 0

    held_window = np.zeros((64, 4), dtype=np.float32)

    for i in range(n_test_windows):
        true_win = X_test[i]
        active_mask, woke = policy.decide_active_channels(true_win)
        active_masks.append(active_mask)
        if woke:
            wake_triggers += 1

        active_count = int(np.sum(active_mask))
        reads_taken += active_count * stride

        # Feed held values for skipped sensors
        for ch in range(4):
            if active_mask[ch]:
                held_window[:, ch] = true_win[:, ch]
            else:
                held_window[:, ch] = 0.0  # Zero fill matches sensor masking augmentation

        # Fast inference with pre-allocated interpreter
        samp_stream = held_window[np.newaxis, ...].astype(np.float32)
        if p_in_det["dtype"] == np.int8 and p_in_scale > 0:
            qs = np.clip(np.round(samp_stream / p_in_scale) + p_in_zp, -128, 127).astype(np.int8)
        else:
            qs = samp_stream
        prune_interp.set_tensor(p_in_det["index"], qs)
        prune_interp.invoke()
        out = prune_interp.get_tensor(p_out_det["index"])
        if p_out_det["dtype"] == np.int8 and p_out_scale > 0:
            prob_sim = (out.astype(np.float32) - p_out_zp) * p_out_scale
        else:
            prob_sim = out.astype(np.float32)

        pred_sim = int(prob_sim[0, 1] >= tcn_thresh)
        sim_preds.append(pred_sim)

        # Track detection delay
        if y_test[i] == 1:
            if not ongoing_anomaly:
                ongoing_anomaly = True
                anomaly_start_win = i
            if pred_sim == 1 and ongoing_anomaly:
                delay = (i - anomaly_start_win) * stride_sec
                detection_delays.append(delay)
                ongoing_anomaly = False
        else:
            ongoing_anomaly = False

    sim_preds = np.array(sim_preds)
    f1_sim = float(f1_score(y_test, sim_preds, zero_division=0))
    prec_sim = float(precision_score(y_test, sim_preds, zero_division=0))
    rec_sim = float(recall_score(y_test, sim_preds, zero_division=0))
    cm_sim = confusion_matrix(y_test, sim_preds, labels=[0, 1])
    tn_s, fp_s, fn_s, tp_s = cm_sim.ravel()
    fpr_sim = float(fp_s / (fp_s + tn_s)) if (fp_s + tn_s) > 0 else 0.0

    pct_reads_saved = float((1.0 - (reads_taken / baseline_reads_total)) * 100.0)
    total_time_hours = (n_test_windows * stride_sec) / 3600.0
    false_alerts_per_hr = float(fp_s / total_time_hours) if total_time_hours > 0 else 0.0
    mean_delay = float(np.mean(detection_delays)) if detection_delays else 0.0
    sim_energy = ((reads_taken * e_read) + (n_test_windows * e_infer)) / 1000.0
    energy_saved_pct = float((1.0 - (sim_energy / energy_d)) * 100.0)

    sim_result = {
        "seed": seed,
        "total_test_windows": n_test_windows,
        "baseline_reads": baseline_reads_total,
        "actual_reads": reads_taken,
        "pct_reads_saved": pct_reads_saved,
        "streaming_f1": f1_sim,
        "streaming_precision": prec_sim,
        "streaming_recall": rec_sim,
        "streaming_fpr": fpr_sim,
        "false_alerts_per_hour": false_alerts_per_hr,
        "wake_up_count": wake_triggers,
        "mean_detection_delay_sec": mean_delay,
        "mean_detection_delay_windows": mean_delay / stride_sec,
        "modelled_energy_mj": sim_energy,
        "baseline_energy_mj": energy_d,
        "pct_energy_saved": energy_saved_pct
    }
    with open(os.path.join(seed_dir, "streaming_simulation.json"), "w") as f:
        json.dump(sim_result, f, indent=2)
    pd.DataFrame([sim_result]).to_csv(os.path.join(seed_dir, "streaming_simulation.csv"), index=False)

    # Config E in ablation
    f1_e = f1_sim
    size_e_kb = tcn_int8_size_kb
    macs_e = tcn_macs
    reads_saved_e = pct_reads_saved
    energy_e = sim_energy

    ablation_rows = [
        {"seed": seed, "config": "A: Vanilla TCN (FP32)", "f1": f1_a, "parameters": params_a, "flash_size_kb": size_a_kb, "analytic_macs": macs_a, "reads_saved_pct": 0.0, "energy_proxy_mj": energy_a},
        {"seed": seed, "config": "B: TCN + INT8", "f1": f1_b, "parameters": params_a, "flash_size_kb": size_b_kb, "analytic_macs": macs_a, "reads_saved_pct": 0.0, "energy_proxy_mj": energy_b},
        {"seed": seed, "config": "C: TCN + Gate & Mask (FP32)", "f1": f1_c, "parameters": params_c, "flash_size_kb": size_c_kb, "analytic_macs": macs_c, "reads_saved_pct": 0.0, "energy_proxy_mj": energy_c},
        {"seed": seed, "config": "D: TCN + Gate & Mask (INT8)", "f1": f1_d, "parameters": params_c, "flash_size_kb": size_d_kb, "analytic_macs": macs_d, "reads_saved_pct": 0.0, "energy_proxy_mj": energy_d},
        {"seed": seed, "config": "E: Config D + PrunePolicy", "f1": f1_e, "parameters": params_c, "flash_size_kb": size_e_kb, "analytic_macs": macs_e, "reads_saved_pct": reads_saved_e, "energy_proxy_mj": energy_e}
    ]
    pd.DataFrame(ablation_rows).to_csv(os.path.join(seed_dir, "ablation.csv"), index=False)

    # ----------------------------------------------------
    # Per-Anomaly-Type Recall (INT8 TCN vs Decision Tree)
    # ----------------------------------------------------
    logger.info(f"[Seed {seed}] Evaluating Per-Anomaly-Type Recall...")
    types_test = np.array(ds["types_test"])
    anomaly_types = ["spike", "drift", "stuck_at", "noise_burst", "dropout"]
    recall_rows = []
    for atype in anomaly_types:
        idx = np.where((types_test == atype) & (y_test == 1))[0]
        if len(idx) == 0:
            continue
        tcn_rec = float(recall_score(y_test[idx], tcn_int8_res["test_preds"][idx], zero_division=0))
        dt_rec = float(recall_score(y_test[idx], dt_test_preds[idx], zero_division=0))
        recall_rows.append({
            "seed": seed,
            "anomaly_type": atype,
            "sample_count": len(idx),
            "TCN_INT8_Recall": tcn_rec,
            "DecisionTree_Recall": dt_rec,
            "Advantage": tcn_rec - dt_rec
        })
    pd.DataFrame(recall_rows).to_csv(os.path.join(seed_dir, "per_anomaly_recall.csv"), index=False)

    # ----------------------------------------------------
    # Robustness Sweeps (INT8 TCN vs Decision Tree)
    # ----------------------------------------------------
    logger.info(f"[Seed {seed}] Executing Robustness Sweeps...")
    rng_rob = np.random.RandomState(seed)

    # 1. Noise
    noise_scales = [0.0, 0.2, 0.5, 0.8, 1.2, 1.6]
    noise_res = []
    for s in noise_scales:
        noise = rng_rob.normal(0, s, X_test.shape).astype(np.float32)
        X_n = X_test + noise
        p_tcn = eval_tflite_probs(prune_interp, p_in_det, p_out_det, X_n)
        f1_tcn_n = float(f1_score(y_test, (p_tcn >= tcn_thresh).astype(int), zero_division=0))
        f_dt = extract_statistical_features(X_n)
        p_dt = (dt_model.predict_proba(f_dt)[:, 1] >= dt_thresh).astype(int)
        f1_dt_n = float(f1_score(y_test, p_dt, zero_division=0))
        noise_res.append({"scale": s, "TCN_INT8_F1": f1_tcn_n, "DecisionTree_F1": f1_dt_n})

    # 2. Dropout
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    drop_res = []
    for r in dropout_rates:
        mask = rng_rob.binomial(1, 1.0 - r, X_test.shape).astype(np.float32)
        X_d = X_test * mask
        p_tcn = eval_tflite_probs(prune_interp, p_in_det, p_out_det, X_d)
        f1_tcn_d = float(f1_score(y_test, (p_tcn >= tcn_thresh).astype(int), zero_division=0))
        f_dt = extract_statistical_features(X_d)
        p_dt = (dt_model.predict_proba(f_dt)[:, 1] >= dt_thresh).astype(int)
        f1_dt_d = float(f1_score(y_test, p_dt, zero_division=0))
        drop_res.append({"rate": r, "TCN_INT8_F1": f1_tcn_d, "DecisionTree_F1": f1_dt_d})

    # 3. Drift
    drift_offsets = [0.0, 0.5, 1.0, 1.5, 2.0]
    drift_res = []
    for off in drift_offsets:
        ramp = np.linspace(0, off, 64)[np.newaxis, :, np.newaxis].astype(np.float32)
        X_dr = X_test + ramp
        p_tcn = eval_tflite_probs(prune_interp, p_in_det, p_out_det, X_dr)
        f1_tcn_dr = float(f1_score(y_test, (p_tcn >= tcn_thresh).astype(int), zero_division=0))
        f_dt = extract_statistical_features(X_dr)
        p_dt = (dt_model.predict_proba(f_dt)[:, 1] >= dt_thresh).astype(int)
        f1_dt_dr = float(f1_score(y_test, p_dt, zero_division=0))
        drift_res.append({"offset": off, "TCN_INT8_F1": f1_tcn_dr, "DecisionTree_F1": f1_dt_dr})

    robustness_dict = {
        "seed": seed,
        "noise": noise_res,
        "dropout": drop_res,
        "drift": drift_res
    }
    with open(os.path.join(seed_dir, "robustness.json"), "w") as f:
        json.dump(robustness_dict, f, indent=2)

    # Save all seed model records
    df_seed_models = pd.DataFrame(seed_records)
    df_seed_models.to_csv(os.path.join(seed_dir, "raw_metrics.csv"), index=False)
    with open(os.path.join(seed_dir, "metrics.json"), "w") as f:
        json.dump(seed_records, f, indent=2)

    logger.info(f"Finished seed {seed}. Saved all outputs to: {seed_dir}")
    return {
        "seed": seed,
        "models": seed_records,
        "pruning": pruning_rows,
        "simulation": sim_result,
        "ablation": ablation_rows,
        "recall": recall_rows,
        "quant_info": quant_info
    }

def main():
    parser = argparse.ArgumentParser(description="Multi-seed NanoEdge-AD Pipeline")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 777, 1234, 2024],
                        help="List of integer seeds to evaluate (default: 42 101 777 1234 2024)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config YAML file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info(f"Running multi-seed pipeline across seeds: {args.seeds}")

    all_seed_results = []
    for s in args.seeds:
        res = run_single_seed_pipeline(s, cfg)
        all_seed_results.append(res)

    logger.info("All seeds completed successfully!")

if __name__ == "__main__":
    main()

