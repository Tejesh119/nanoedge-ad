"""Unit tests for PrunePolicy dynamic wakeup and quiet sensor skipping."""
import numpy as np
import pytest
from src.runtime_pruning import PrunePolicy

def test_prune_policy_skips_quiet_low_gate_sensor():
    """Verify PrunePolicy skips the lowest-ranked gate sensor after quiet consecutive windows."""
    policy = PrunePolicy(
        num_channels=4,
        gate_rankings=[3, 1, 2, 0],  # Channel 3 is lowest gate
        quiet_window_count=3,
        jump_z_threshold=0.40
    )

    # Simulate quiet normal windows with typical baseline variance (not collapse to zero)
    rng = np.random.RandomState(42)

    # First window: all sensors active
    win0 = rng.normal(0, 0.8, (64, 4)).astype(np.float32)
    active_mask, woke = policy.decide_active_channels(win0)
    assert np.all(active_mask), "All sensors should be active initially"

    # Step through quiet windows
    for _ in range(6):
        win_quiet = rng.normal(0, 0.8, (64, 4)).astype(np.float32)
        active_mask, woke = policy.decide_active_channels(win_quiet)

    # Channel 3 (lowest gate) should now be skipped
    assert active_mask[3] is False or active_mask[3] == 0, "Lowest-ranked sensor should be skipped when persistently quiet"
    assert woke is False, "No wakeup should trigger on quiet data"

def test_prune_policy_instant_wakeup_on_jump():
    """Verify PrunePolicy instantly wakes ALL sensors when a statistical jump is injected."""
    policy = PrunePolicy(
        num_channels=4,
        gate_rankings=[3, 1, 2, 0],
        quiet_window_count=2,
        jump_z_threshold=0.40
    )

    rng = np.random.RandomState(42)
    # Put policy into quiet/pruned state
    for _ in range(5):
        win_quiet = rng.normal(0, 0.8, (64, 4)).astype(np.float32)
        policy.decide_active_channels(win_quiet)

    # Inject an abrupt jump on active channel 0
    jump_win = rng.normal(0, 0.8, (64, 4)).astype(np.float32)
    jump_win[:, 0] += 3.5  # Large deviation jump

    active_mask, woke = policy.decide_active_channels(jump_win)
    assert woke is True, "Instant wakeup must trigger upon jump"
    assert np.all(active_mask), "All sensors must be woken up immediately upon jump"
