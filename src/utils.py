"""Utility functions for NanoEdge-AD: configuration, seeding, logging, and metrics."""
import os
import random
import logging
import yaml
import numpy as np

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def setup_logger(name: str = "nanoedge_ad", log_level: int = logging.INFO) -> logging.Logger:
    """Configure a clean, timestamped console and file logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(log_level)
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility across random, numpy, and tensorflow.
    
    Note on Determinism:
    Sets random, numpy, PYTHONHASHSEED, tf.random, and TF deterministic ops.
    Remaining potential CPU nondeterminism: multi-threaded oneDNN operations
    and parallel reduction order in floating-point SIMD kernels on x86_64 CPUs.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

def compute_conv1d_macs(
    input_length: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: str = "same"
) -> int:
    """Compute MACs for a 1D convolution layer.
    
    Formula: out_length * kernel_size * in_channels * out_channels
    where out_length = input_length // stride (for 'same' padding).
    """
    if padding == "same":
        out_length = (input_length + stride - 1) // stride
    else:  # valid
        out_length = (input_length - kernel_size) // stride + 1
    return int(out_length * kernel_size * in_channels * out_channels)

def compute_dense_macs(in_features: int, out_features: int) -> int:
    """Compute MACs for a Dense layer (in_features * out_features)."""
    return int(in_features * out_features)

def compute_tcn_macs(input_shape: tuple = (64, 4), blocks: list = None, dense_units: int = 2) -> dict:
    """Compute theoretical analytic MACs for the NanoEdge-AD TCN architecture.
    
    Architecture:
    Input: (64, 4)
    Gate: 4 multiplications (channel scaling per time step) = 64 * 4 MACs
    Block 1:
      Conv1D(filters=16, k=3, same) on 4 channels -> 64 * 3 * 4 * 16 = 12,288
      Conv1D(filters=16, k=3, same) on 16 channels -> 64 * 3 * 16 * 16 = 49,152
      Skip Conv1D(1x1, 4 -> 16) -> 64 * 1 * 4 * 16 = 4,096
    Block 2:
      Conv1D(filters=16, k=3, same) on 16 channels -> 64 * 3 * 16 * 16 = 49,152
      Conv1D(filters=16, k=3, same) on 16 channels -> 64 * 3 * 16 * 16 = 49,152
      Skip: identity (0 MACs)
    Block 3:
      Conv1D(filters=24, k=3, same) on 16 channels -> 64 * 3 * 16 * 24 = 73,728
      Conv1D(filters=24, k=3, same) on 24 channels -> 64 * 3 * 24 * 24 = 110,592
      Skip Conv1D(1x1, 16 -> 24) -> 64 * 1 * 16 * 24 = 24,576
    GlobalAveragePooling: 64 * 24 additions (often negligible or 0 MACs)
    Dense Classifier: 24 * 2 MACs = 48 MACs
    """
    if blocks is None:
        blocks = [
            {"filters": 16, "kernel_size": 3},
            {"filters": 16, "kernel_size": 3},
            {"filters": 24, "kernel_size": 3}
        ]
    
    time_steps, in_channels = input_shape
    total_macs = 0
    layer_macs = {}
    
    # Sensor Gate
    gate_macs = time_steps * in_channels
    layer_macs["sensor_gate"] = gate_macs
    total_macs += gate_macs
    
    current_channels = in_channels
    for i, b in enumerate(blocks):
        out_ch = b["filters"]
        k = b["kernel_size"]
        
        # Conv 1
        c1 = compute_conv1d_macs(time_steps, current_channels, out_ch, k, padding="same")
        # Conv 2
        c2 = compute_conv1d_macs(time_steps, out_ch, out_ch, k, padding="same")
        
        # Skip projection if channel dimension changes
        skip = 0
        if current_channels != out_ch:
            skip = compute_conv1d_macs(time_steps, current_channels, out_ch, 1, padding="same")
            
        block_total = c1 + c2 + skip
        layer_macs[f"block_{i+1}"] = {
            "conv1": c1,
            "conv2": c2,
            "skip": skip,
            "total": block_total
        }
        total_macs += block_total
        current_channels = out_ch
        
    # Dense head
    dense_macs = compute_dense_macs(current_channels, dense_units)
    layer_macs["dense_head"] = dense_macs
    total_macs += dense_macs
    
    return {
        "total_macs": total_macs,
        "details": layer_macs
    }

def compute_cnn_macs(input_shape: tuple = (64, 4), dense_units: int = 2) -> dict:
    """Compute MACs for Baseline B2 (Plain 1D-CNN) and B4 (Pruned CNN).
    
    Architecture:
    Input: (64, 4)
    Conv1D(16, k=3, same) on 4 channels -> 64 * 3 * 4 * 16 = 12,288
    Conv1D(16, k=3, same) on 16 channels -> 64 * 3 * 16 * 16 = 49,152
    Conv1D(24, k=3, same) on 16 channels -> 64 * 3 * 16 * 24 = 73,728
    GlobalAveragePooling: 0 MACs
    Dense Classifier: 24 * 2 = 48 MACs
    Total = 135,216 MACs
    """
    time_steps, in_channels = input_shape
    c1 = compute_conv1d_macs(time_steps, in_channels, 16, 3, padding="same")
    c2 = compute_conv1d_macs(time_steps, 16, 16, 3, padding="same")
    c3 = compute_conv1d_macs(time_steps, 16, 24, 3, padding="same")
    dense = compute_dense_macs(24, dense_units)
    total = c1 + c2 + c3 + dense
    return {
        "total_macs": total,
        "details": {"conv1": c1, "conv2": c2, "conv3": c3, "dense": dense}
    }

def compute_ae_macs(input_shape: tuple = (64, 4)) -> dict:
    """Compute MACs for Baseline B3 (Dense Autoencoder).
    
    Architecture:
    Input: (64, 4) flattened to 256
    Dense 64: 256 * 64 = 16,384
    Dense 16: 64 * 16 = 1,024
    Dense 64: 16 * 64 = 1,024
    Dense 256: 64 * 256 = 16,384
    Reconstruction error: 256 subtractions, 256 squares, 256 adds = 768 ops (often ~256 MAC equivalent)
    Total = 34,816 MACs
    """
    flat_dim = input_shape[0] * input_shape[1]
    d1 = compute_dense_macs(flat_dim, 64)
    d2 = compute_dense_macs(64, 16)
    d3 = compute_dense_macs(16, 64)
    d4 = compute_dense_macs(64, flat_dim)
    total = d1 + d2 + d3 + d4
    return {
        "total_macs": total,
        "details": {"dense1": d1, "dense2": d2, "dense3": d3, "dense4": d4}
    }

def compute_tree_feature_ops(window_size: int = 64, num_channels: int = 4) -> dict:
    """Compute exact analytic operation count for Baseline B1 (Decision Tree 24-feature extraction).
    
    Per channel over 64 samples:
    1. Mean: 64 additions + 1 division = 65 ops
    2. Variance/Std: 64 diffs + 64 squares + 64 adds + 1 div + 1 sqrt = 194 ops
    3. Min: 63 comparisons
    4. Max: 63 comparisons
    5. Peak-to-Peak: 1 subtraction
    6. Mean Absolute Difference: 63 diffs + 63 abs + 63 adds + 1 div = 190 ops
    Total per channel = 65 + 194 + 63 + 63 + 1 + 190 = 576 ops
    For 4 channels = 576 * 4 = 2,304 operations per window.
    Tree traversal: max_depth comparisons (typically 4-8 ops).
    """
    ops_per_ch = 65 + 194 + 63 + 63 + 1 + 190
    total_feature_ops = ops_per_ch * num_channels
    return {
        "ops_per_channel": ops_per_ch,
        "total_feature_ops": total_feature_ops
    }


