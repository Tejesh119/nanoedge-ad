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
    """Set seeds for reproducibility across random, numpy, and tensorflow."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
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
