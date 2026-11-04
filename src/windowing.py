"""Time-series windowing with strict chronological splitting and zero-leakage normalization."""
import os
import argparse
import numpy as np
from collections import Counter
from src.utils import load_config, setup_logger
from src.data_gen import generate_synthetic_stream

logger = setup_logger("windowing")

def create_windows(
    data: np.ndarray,
    anomaly_mask: np.ndarray,
    anomaly_types: np.ndarray,
    window_size: int = 64,
    stride: int = 32,
    anomaly_overlap_ratio: float = 0.10
) -> tuple:
    """Create sliding windows and label them based on anomaly overlap.
    
    A window is labeled anomalous (1) if at least `anomaly_overlap_ratio` * `window_size`
    samples are anomalous. Otherwise labeled normal (0).
    The window's anomaly type is assigned as the dominant anomaly type in that window.
    """
    n_samples = len(data)
    min_anom_samples = int(np.ceil(window_size * anomaly_overlap_ratio))
    
    windows = []
    labels = []
    window_types = []
    window_indices = []

    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        win_data = data[start:end]
        win_mask = anomaly_mask[start:end]
        anom_count = np.sum(win_mask)
        
        is_anom = 1 if anom_count >= min_anom_samples else 0
        if is_anom:
            anom_sample_types = [t for t in anomaly_types[start:end] if t != "normal"]
            if anom_sample_types:
                dom_type = Counter(anom_sample_types).most_common(1)[0][0]
            else:
                dom_type = "anomaly"
        else:
            dom_type = "normal"
            
        windows.append(win_data)
        labels.append(is_anom)
        window_types.append(dom_type)
        window_indices.append((start, end))

    return (
        np.array(windows, dtype=np.float32),
        np.array(labels, dtype=np.int32),
        np.array(window_types, dtype=object),
        np.array(window_indices, dtype=np.int32)
    )

def prepare_dataset(config: dict, seed: int = 42) -> dict:
    """Generate stream, perform chronological split, normalize strictly on train, and return datasets."""
    data, mask, types, raw_meta = generate_synthetic_stream(config, seed=seed)
    n_samples = len(data)
    
    splits = config["windowing"]["splits"]
    train_ratio = splits.get("train", 0.60)
    val_ratio = splits.get("val", 0.20)
    
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * (train_ratio + val_ratio))
    
    # Split raw time-series chronologically BEFORE windowing or normalization to guarantee zero leakage
    train_raw = data[:n_train]
    train_mask = mask[:n_train]
    train_types = types[:n_train]

    val_raw = data[n_train:n_val]
    val_mask = mask[n_train:n_val]
    val_types = types[n_train:n_val]

    test_raw = data[n_val:]
    test_mask = mask[n_val:]
    test_types = types[n_val:]

    # Zero-leakage: Compute normalization statistics SOLELY on train_raw
    train_mean = np.mean(train_raw, axis=0, keepdims=True)
    train_std = np.std(train_raw, axis=0, keepdims=True) + 1e-7

    # Normalize each split
    train_norm = (train_raw - train_mean) / train_std
    val_norm = (val_raw - train_mean) / train_std
    test_norm = (test_raw - train_mean) / train_std

    win_size = config["windowing"]["window_size"]
    stride = config["windowing"]["stride"]
    anom_thresh = config["windowing"]["anomaly_overlap_ratio"]

    X_train, y_train, types_train, idx_train = create_windows(train_norm, train_mask, train_types, win_size, stride, anom_thresh)
    X_val, y_val, types_val, idx_val = create_windows(val_norm, val_mask, val_types, win_size, stride, anom_thresh)
    X_test, y_test, types_test, idx_test = create_windows(test_norm, test_mask, test_types, win_size, stride, anom_thresh)

    # Adjust indices to absolute stream indices for inspection/debugging
    idx_val += n_train
    idx_test += n_val

    dataset = {
        "X_train": X_train, "y_train": y_train, "types_train": types_train, "idx_train": idx_train,
        "X_val": X_val,     "y_val": y_val,     "types_val": types_val,     "idx_val": idx_val,
        "X_test": X_test,   "y_test": y_test,   "types_test": types_test,   "idx_test": idx_test,
        "train_mean": train_mean.squeeze(),
        "train_std": train_std.squeeze(),
        "raw_test_data": test_raw,
        "channels": config["data"]["channels"],
        "metadata": {
            "synthetic": True,
            "seed": seed,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "train_anom_ratio": float(np.mean(y_train)),
            "val_anom_ratio": float(np.mean(y_val)),
            "test_anom_ratio": float(np.mean(y_test)),
            "train_types_distribution": dict(Counter(types_train)),
            "val_types_distribution": dict(Counter(types_val)),
            "test_types_distribution": dict(Counter(types_test))
        }
    }
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Prepare windowed dataset for NanoEdge-AD")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("primary_seed", 42)
    ds = prepare_dataset(cfg, seed=seed)

    print("Dataset Windowing Summary:")
    print(f"  Train windows: {ds['X_train'].shape} | Anomaly ratio: {ds['metadata']['train_anom_ratio']*100:.2f}%")
    print(f"    Types: {ds['metadata']['train_types_distribution']}")
    print(f"  Val windows:   {ds['X_val'].shape} | Anomaly ratio: {ds['metadata']['val_anom_ratio']*100:.2f}%")
    print(f"    Types: {ds['metadata']['val_types_distribution']}")
    print(f"  Test windows:  {ds['X_test'].shape} | Anomaly ratio: {ds['metadata']['test_anom_ratio']*100:.2f}%")
    print(f"    Types: {ds['metadata']['test_types_distribution']}")
    print(f"  Normalization Mean (Train): {np.round(ds['train_mean'], 3)}")
    print(f"  Normalization Std  (Train): {np.round(ds['train_std'], 3)}")

if __name__ == "__main__":
    main()
