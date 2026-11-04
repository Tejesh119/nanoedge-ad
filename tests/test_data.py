"""Unit tests for synthetic data generation, windowing, and leakage prevention."""
import numpy as np
import pytest
from src.utils import load_config, compute_conv1d_macs, compute_dense_macs, compute_tcn_macs
from src.data_gen import generate_synthetic_stream
from src.windowing import prepare_dataset, create_windows

def test_analytic_macs_calculation():
    """Verify MACs analytical formulas match hand-computed arithmetic."""
    # Hand-computed Conv1D: input_length=10, in_channels=2, out_channels=4, kernel_size=3, padding='same'
    # MACs = 10 * 3 * 2 * 4 = 240
    c_macs = compute_conv1d_macs(input_length=10, in_channels=2, out_channels=4, kernel_size=3, padding="same")
    assert c_macs == 240, f"Expected 240 MACs, got {c_macs}"

    # Hand-computed Dense: 16 inputs * 2 outputs = 32 MACs
    d_macs = compute_dense_macs(16, 2)
    assert d_macs == 32, f"Expected 32 MACs, got {d_macs}"

    # Hand-computed TCN overall MACs
    cfg = load_config()
    res = compute_tcn_macs(input_shape=(64, 4), blocks=cfg["model"]["blocks"], dense_units=2)
    assert res["total_macs"] > 100000, "TCN MACs should be realistic for length 64"
    assert "sensor_gate" in res["details"]
    assert res["details"]["sensor_gate"] == 64 * 4  # 256 MACs

def test_data_generation_shapes_and_types():
    """Verify synthetic stream shapes, types, and channel count."""
    cfg = load_config()
    # Use smaller total samples for quick test
    test_cfg = dict(cfg)
    test_cfg["data"] = dict(cfg["data"])
    test_cfg["data"]["total_samples"] = 5000

    data, mask, types, meta = generate_synthetic_stream(test_cfg, seed=42)
    assert data.shape == (5000, 4)
    assert mask.shape == (5000,)
    assert len(types) == 5000
    assert meta["synthetic"] is True
    # Check all 5 anomaly types are present
    unique_types = set(types)
    for expected in ["spike", "drift", "stuck_at", "noise_burst", "dropout"]:
        assert expected in unique_types, f"Anomaly type {expected} should be present in stream"

def test_windowing_no_train_test_overlap_and_zero_leakage():
    """Verify zero overlap between train and test windows and strict train-only normalization."""
    cfg = load_config()
    test_cfg = dict(cfg)
    test_cfg["data"] = dict(cfg["data"])
    test_cfg["data"]["total_samples"] = 10000
    
    ds = prepare_dataset(test_cfg, seed=42)
    
    X_train, y_train, idx_train = ds["X_train"], ds["y_train"], ds["idx_train"]
    X_test, y_test, idx_test = ds["X_test"], ds["y_test"], ds["idx_test"]
    X_val, y_val, idx_val = ds["X_val"], ds["y_val"], ds["idx_val"]

    # Window shapes
    assert X_train.shape[1:] == (64, 4)
    assert X_val.shape[1:] == (64, 4)
    assert X_test.shape[1:] == (64, 4)

    # Label shapes
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)

    # Overlap check: train windows must strictly end before val starts; val must strictly end before test starts
    train_max_idx = np.max(idx_train[:, 1])
    val_min_idx = np.min(idx_val[:, 0])
    val_max_idx = np.max(idx_val[:, 1])
    test_min_idx = np.min(idx_test[:, 0])

    assert train_max_idx <= val_min_idx, f"Train ({train_max_idx}) overlaps with Val ({val_min_idx})"
    assert val_max_idx <= test_min_idx, f"Val ({val_max_idx}) overlaps with Test ({test_min_idx})"

    # Normalization check: Train split normalized data must have approximately zero mean and unit std
    train_channel_means = np.mean(X_train, axis=(0, 1))
    train_channel_stds = np.std(X_train, axis=(0, 1))
    np.testing.assert_allclose(train_channel_means, 0.0, atol=0.15)
    np.testing.assert_allclose(train_channel_stds, 1.0, atol=0.15)

    # Anomaly distribution across all splits
    for split_name, types_split in [("train", ds["types_train"]), ("val", ds["types_val"]), ("test", ds["types_test"])]:
        unique = set(types_split)
        for atype in ["spike", "drift", "stuck_at", "noise_burst", "dropout"]:
            assert atype in unique, f"Split {split_name} must contain anomaly type {atype}"

