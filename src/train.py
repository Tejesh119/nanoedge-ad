"""Training pipeline for NanoEdge-AD TCN and Baselines across multiple seeds."""
import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from src.utils import load_config, set_seed, setup_logger
from src.windowing import prepare_dataset
from src.models import (
    build_tcn_model,
    build_baseline_cnn,
    build_dense_autoencoder,
    extract_statistical_features,
    tune_decision_tree
)

logger = setup_logger("train")

def apply_channel_masking_augmentation(X: np.ndarray, y: np.ndarray, prob: float = 0.25, seed: int = 42) -> np.ndarray:
    """Randomly mask sensor channels in windows while preserving class label semantics.
    
    For normal windows (y == 0): randomly zeroes 1-2 channels (never all 4).
    For anomalous windows (y == 1): randomly zeroes 1 background channel not driving the anomaly.
    """
    rng = np.random.RandomState(seed)
    X_aug = X.copy()
    num_windows, win_len, num_channels = X.shape

    for i in range(num_windows):
        if rng.rand() < prob:
            if y[i] == 0:
                num_drop = rng.randint(1, min(3, num_channels))
                drop_indices = rng.choice(num_channels, size=num_drop, replace=False)
                X_aug[i, :, drop_indices] = 0.0
            else:
                # Mask a background (least active) channel so anomaly remains present
                ch_deviations = np.max(np.abs(X_aug[i]), axis=0)
                background_chs = np.argsort(ch_deviations)[:2]
                drop_ch = rng.choice(background_chs)
                X_aug[i, :, drop_ch] = 0.0
            
    return X_aug

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray = None) -> dict:
    """Calculate accuracy, precision, recall, f1, fpr, and roc-auc."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    auc = 0.5
    if y_scores is not None and len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_scores))
        except Exception:
            auc = 0.5

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "roc_auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

def find_best_threshold(y_val: np.ndarray, scores_val: np.ndarray) -> float:
    """Tune decision threshold strictly on validation predictions to maximize F1."""
    best_t = 0.5
    best_f1 = -1.0
    thresholds = np.linspace(0.05, 0.95, 91)
    for t in thresholds:
        preds = (scores_val >= t).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t

def train_tcn_seed(ds: dict, config: dict, seed: int = 42, save_artifacts: bool = False):
    """Train TCN with SensorGate and channel masking on one seed."""
    set_seed(seed)
    train_cfg = config["training"]
    model_cfg = config["model"]

    # Augmentation on train windows only
    X_train_aug = apply_channel_masking_augmentation(
        ds["X_train"],
        ds["y_train"],
        prob=train_cfg.get("channel_masking_prob", 0.25),
        seed=seed
    )

    y_train_cat = tf.keras.utils.to_categorical(ds["y_train"], num_classes=2)
    y_val_cat = tf.keras.utils.to_categorical(ds["y_val"], num_classes=2)

    # Class weights to handle class imbalance
    cls_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=ds["y_train"])
    class_weight_dict = {0: float(cls_weights[0]), 1: float(cls_weights[1])}

    model = build_tcn_model(
        input_shape=tuple(model_cfg["input_shape"]),
        blocks_cfg=model_cfg["blocks"],
        num_classes=model_cfg["num_classes"],
        gate_l1=model_cfg.get("gate_l1", 0.001),
        use_gate=True
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=train_cfg["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=train_cfg["early_stopping_patience"],
            restore_best_weights=True,
            verbose=0
        )
    ]

    model.fit(
        X_train_aug, y_train_cat,
        validation_data=(ds["X_val"], y_val_cat),
        batch_size=train_cfg["batch_size"],
        epochs=train_cfg["epochs"],
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=0
    )

    # Val predictions for threshold search
    val_probs = model.predict(ds["X_val"], verbose=0)[:, 1]
    best_thresh = find_best_threshold(ds["y_val"], val_probs)

    # Test predictions
    test_probs = model.predict(ds["X_test"], verbose=0)[:, 1]
    test_preds = (test_probs >= best_thresh).astype(int)

    metrics = compute_classification_metrics(ds["y_test"], test_preds, test_probs)
    metrics["best_threshold"] = best_thresh

    # Extract learned gates
    gate_layer = model.get_layer("sensor_gate")
    gate_values = gate_layer.get_gate_values()
    channel_names = config["data"]["channels"]
    gate_dict = {ch: float(val) for ch, val in zip(channel_names, gate_values)}

    if save_artifacts:
        os.makedirs(config["paths"]["outputs_dir"], exist_ok=True)
        model.save(config["paths"]["model_fp32"])
        with open(config["paths"]["gates_json"], "w") as f:
            json.dump(gate_dict, f, indent=2)
        # Save a few hundred test windows for INT8 validation & future hardware deployment
        np.savez_compressed(
            config["paths"]["test_windows"],
            X_test=ds["X_test"][:500],
            y_test=ds["y_test"][:500],
            test_probs=test_probs[:500],
            test_preds=test_preds[:500],
            types_test=ds["types_test"][:500],
            channels=channel_names
        )
        logger.info(f"Saved primary FP32 model to {config['paths']['model_fp32']}")
        logger.info(f"Learned sensor gate values: {gate_dict}")

    return model, metrics, gate_dict

def train_baseline_dt_seed(ds: dict, seed: int = 42):
    """Train Baseline B1: Decision Tree on 24 per-channel statistical features."""
    X_train_feat = extract_statistical_features(ds["X_train"])
    X_val_feat = extract_statistical_features(ds["X_val"])
    X_test_feat = extract_statistical_features(ds["X_test"])

    dt = tune_decision_tree(X_train_feat, ds["y_train"], X_val_feat, ds["y_val"], seed=seed)
    test_probs = dt.predict_proba(X_test_feat)[:, 1]
    test_preds = dt.predict(X_test_feat)

    metrics = compute_classification_metrics(ds["y_test"], test_preds, test_probs)
    return dt, metrics

def train_baseline_cnn_seed(ds: dict, config: dict, seed: int = 42):
    """Train Baseline B2: Standard 1D-CNN (no gate, no channel masking)."""
    set_seed(seed)
    train_cfg = config["training"]
    model_cfg = config["model"]

    y_train_cat = tf.keras.utils.to_categorical(ds["y_train"], num_classes=2)
    y_val_cat = tf.keras.utils.to_categorical(ds["y_val"], num_classes=2)

    cls_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=ds["y_train"])
    class_weight_dict = {0: float(cls_weights[0]), 1: float(cls_weights[1])}

    model = build_baseline_cnn(input_shape=tuple(model_cfg["input_shape"]), num_classes=2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=train_cfg["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [EarlyStopping(monitor="val_loss", patience=train_cfg["early_stopping_patience"], restore_best_weights=True, verbose=0)]
    model.fit(
        ds["X_train"], y_train_cat,
        validation_data=(ds["X_val"], y_val_cat),
        batch_size=train_cfg["batch_size"],
        epochs=train_cfg["epochs"],
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=0
    )

    val_probs = model.predict(ds["X_val"], verbose=0)[:, 1]
    best_thresh = find_best_threshold(ds["y_val"], val_probs)

    test_probs = model.predict(ds["X_test"], verbose=0)[:, 1]
    test_preds = (test_probs >= best_thresh).astype(int)

    metrics = compute_classification_metrics(ds["y_test"], test_preds, test_probs)
    metrics["best_threshold"] = best_thresh
    return model, metrics

def train_baseline_ae_seed(ds: dict, config: dict, seed: int = 42):
    """Train Baseline B3: Dense Autoencoder on normal train windows only."""
    set_seed(seed)
    train_cfg = config["training"]
    model_cfg = config["model"]

    # Filter normal windows only
    normal_idx = np.where(ds["y_train"] == 0)[0]
    X_train_normal = ds["X_train"][normal_idx]

    model = build_dense_autoencoder(input_shape=tuple(model_cfg["input_shape"]))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)]
    model.fit(
        X_train_normal, X_train_normal,
        validation_data=(ds["X_val"], ds["X_val"]),
        batch_size=train_cfg["batch_size"],
        epochs=20,
        callbacks=callbacks,
        verbose=0
    )

    # Compute reconstruction errors on val to select optimal threshold
    val_recon = model.predict(ds["X_val"], verbose=0)
    val_mse = np.mean(np.square(ds["X_val"] - val_recon), axis=(1, 2))

    # Search threshold on val MSE
    best_t = float(np.percentile(val_mse, 80))
    best_f1 = -1.0
    for pct in np.linspace(50, 99, 50):
        t = np.percentile(val_mse, pct)
        preds = (val_mse >= t).astype(int)
        score = f1_score(ds["y_val"], preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)

    # Test reconstruction error
    test_recon = model.predict(ds["X_test"], verbose=0)
    test_mse = np.mean(np.square(ds["X_test"] - test_recon), axis=(1, 2))
    test_preds = (test_mse >= best_t).astype(int)

    # Normalize test_mse into [0, 1] pseudo-scores for AUC
    scores = (test_mse - np.min(test_mse)) / (np.max(test_mse) - np.min(test_mse) + 1e-8)
    metrics = compute_classification_metrics(ds["y_test"], test_preds, scores)
    metrics["best_threshold"] = best_t
    return model, metrics

def train_baseline_static_pruned_cnn(cnn_model, ds: dict, prune_fraction: float = 0.30):
    """Baseline B4: Static magnitude-pruned CNN (zero out lowest 30% weights)."""
    # Clone weights
    weights = cnn_model.get_weights()
    new_weights = []
    for w in weights:
        if w.ndim >= 2:  # Conv or Dense kernel weights
            abs_w = np.abs(w)
            cutoff = np.percentile(abs_w, prune_fraction * 100)
            mask = abs_w > cutoff
            new_weights.append(w * mask)
        else:
            new_weights.append(w)
    
    pruned_model = tf.keras.models.clone_model(cnn_model)
    pruned_model.set_weights(new_weights)

    val_probs = pruned_model.predict(ds["X_val"], verbose=0)[:, 1]
    best_thresh = find_best_threshold(ds["y_val"], val_probs)

    test_probs = pruned_model.predict(ds["X_test"], verbose=0)[:, 1]
    test_preds = (test_probs >= best_thresh).astype(int)

    metrics = compute_classification_metrics(ds["y_test"], test_preds, test_probs)
    metrics["best_threshold"] = best_thresh
    return pruned_model, metrics

def run_training_campaign(config_path: str = "config.yaml"):
    """Execute complete training across 3 seeds for TCN and all baselines."""
    cfg = load_config(config_path)
    seeds = cfg.get("seeds", [42, 101, 777])
    primary_seed = cfg.get("primary_seed", 42)

    logger.info(f"Starting training campaign over seeds: {seeds}")
    
    results = []

    for seed in seeds:
        logger.info(f"--- Running Seed {seed} ---")
        ds = prepare_dataset(cfg, seed=seed)

        # 1. Proposed TCN with SensorGate
        is_primary = (seed == primary_seed)
        tcn_m, tcn_met, gates = train_tcn_seed(ds, cfg, seed=seed, save_artifacts=is_primary)
        tcn_met.update({"model": "TCN_SensorGate (FP32)", "seed": seed})
        results.append(tcn_met)

        # 2. Baseline B1: Decision Tree
        _, dt_met = train_baseline_dt_seed(ds, seed=seed)
        dt_met.update({"model": "B1_DecisionTree", "seed": seed})
        results.append(dt_met)

        # 3. Baseline B2: Standard 1D-CNN
        cnn_m, cnn_met = train_baseline_cnn_seed(ds, cfg, seed=seed)
        cnn_met.update({"model": "B2_VanillaCNN (FP32)", "seed": seed})
        results.append(cnn_met)

        # 4. Baseline B3: Dense Autoencoder
        _, ae_met = train_baseline_ae_seed(ds, cfg, seed=seed)
        ae_met.update({"model": "B3_DenseAutoencoder", "seed": seed})
        results.append(ae_met)

        # 5. Baseline B4: Static Magnitude-Pruned CNN
        _, b4_met = train_baseline_static_pruned_cnn(cnn_m, ds, prune_fraction=0.30)
        b4_met.update({"model": "B4_StaticPrunedCNN", "seed": seed})
        results.append(b4_met)

    df_res = pd.DataFrame(results)
    
    # Save seed-level metrics
    os.makedirs(cfg["paths"]["outputs_dir"], exist_ok=True)
    df_res.to_csv("outputs/raw_seed_metrics.csv", index=False)

    # Compute summary table (mean +/- std)
    metric_cols = ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc"]
    summary = df_res.groupby("model")[metric_cols].agg(["mean", "std"])

    print("\n==================== TRAINING CAMPAIGN RESULTS (3 SEEDS MEAN +/- STD) ====================")
    summary_formatted = pd.DataFrame()
    for col in metric_cols:
        summary_formatted[col] = summary[col].apply(lambda row: f"{row['mean']:.4f} +/- {row['std']:.4f}", axis=1)
    
    print(summary_formatted.to_markdown())
    summary_formatted.to_csv("outputs/train_summary.csv")

    # Gate check: Verify F1 is not trivially ~1.0
    tcn_mean_f1 = summary["f1"]["mean"]["TCN_SensorGate (FP32)"]
    print(f"\nTCN Mean F1 across 3 seeds: {tcn_mean_f1:.4f}")
    if tcn_mean_f1 > 0.99:
        logger.warning("F1 is trivially ~1.0! Anomalies might be too easy.")
    else:
        logger.info(f"GATE PASSED: Non-trivial anomaly detection task (F1 = {tcn_mean_f1:.4f}).")

    return df_res, summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_training_campaign(args.config)

if __name__ == "__main__":
    main()

