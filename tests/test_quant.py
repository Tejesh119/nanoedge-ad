"""Tests for SensorGate layer, masking logic, and INT8 quantization criteria."""
import json
import os
import numpy as np
import pytest
import tensorflow as tf

from src.models import SensorGate, build_tcn_model
from src.train import apply_channel_masking_augmentation
from src.quantize import run_tflite_inference
from src.utils import load_config

def test_sensor_gate_layer_output_range():
    """Verify SensorGate outputs effective weights bounded in [0, 1]."""
    gate = SensorGate(num_channels=4, l1_reg=0.001)
    # Build with dummy input
    dummy_input = tf.zeros((2, 64, 4))
    _ = gate(dummy_input)
    
    gate_vals = gate.get_gate_values()
    assert gate_vals.shape == (4,)
    assert np.all(gate_vals >= 0.0) and np.all(gate_vals <= 1.0), "Gate values must be within [0, 1]"

def test_channel_masking_never_zeroes_all_channels():
    """Verify data augmentation never zeroes out all 4 channels in any window."""
    dummy_X = np.ones((100, 64, 4), dtype=np.float32)
    dummy_y = np.random.choice([0, 1], size=100)
    
    X_aug = apply_channel_masking_augmentation(dummy_X, dummy_y, prob=1.0, seed=42)
    # Check that in every window, at least 1 channel has non-zero values
    channel_energy = np.sum(np.abs(X_aug), axis=(1, 2))
    assert np.all(channel_energy > 0), "Augmentation must never zero all channels in a window"

def test_int8_quantization_acceptance_criteria():
    """Verify saved INT8 model meets file size, F1 tolerance, and agreement requirements."""
    cfg = load_config()
    quant_info_path = cfg["paths"]["quant_info"]
    assert os.path.exists(quant_info_path), f"Quant info file {quant_info_path} must exist"

    with open(quant_info_path, "r") as f:
        q_info = json.load(f)

    crit = q_info["acceptance_criteria"]
    assert crit["param_count_under_15k"] is True, f"Params: {q_info['fp32_params']}"
    assert crit["int8_file_under_60kb"] is True, f"Size: {q_info['model_size_kb']} KB"
    assert crit["f1_within_3_pts"] is True, f"F1 Delta: {q_info['f1_delta']}"
    assert crit["agreement_gte_95_pct"] is True, f"Agreement: {q_info['prediction_agreement_pct']}%"

