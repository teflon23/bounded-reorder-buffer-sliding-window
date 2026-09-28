"""Tests for the bounded reorder buffer sliding window aggregation project."""

from __future__ import annotations

import numpy as np
import pytest

from app import run_experiment
from data import make_dataset


def test_dataset_invalid_n_samples_raises_value_error() -> None:
    """make_dataset must raise ValueError when n_samples < 32."""
    with pytest.raises(ValueError, match="n_samples must be >= 32"):
        make_dataset(seed=1, n_samples=31)


def test_dataset_different_seed_changes_x() -> None:
    """Different seeds must produce different X arrays."""
    X1, _ = make_dataset(seed=42, n_samples=64)
    X2, _ = make_dataset(seed=99, n_samples=64)
    assert not np.array_equal(X1, X2)


def test_dataset_respects_n_samples() -> None:
    """X and y must have length equal to n_samples."""
    for n in (32, 64, 128, 256):
        X, y = make_dataset(seed=7, n_samples=n)
        assert len(X) == n
        assert len(y) == n
        assert X.shape == (n, 2)
        assert y.shape == (n,)


def test_run_experiment_deterministic() -> None:
    """Same inputs must give exactly the same output."""
    result1 = run_experiment(seed=42, n_samples=256)
    result2 = run_experiment(seed=42, n_samples=256)
    assert result1 == result2


def test_run_experiment_respects_n_samples() -> None:
    """Different n_samples must be reflected in the output."""
    for n in (32, 64, 128):
        result = run_experiment(seed=10, n_samples=n)
        assert result["n_samples"] == n


def test_reorder_buffer_beats_baseline_on_mae() -> None:
    """The reorder buffer must achieve strictly lower window_sum_mae than the baseline."""
    result = run_experiment(seed=42, n_samples=256)
    rb_mae = result["metrics"]["window_sum_mae"]
    fc_mae = result["baseline_metrics"]["window_sum_mae"]
    assert rb_mae < fc_mae, (
        f"Reorder buffer MAE ({rb_mae}) should be strictly less than "
        f"baseline MAE ({fc_mae})"
    )


def test_reorder_buffer_beats_baseline_on_retention() -> None:
    """The reorder buffer must achieve strictly higher late_retention_rate than the baseline."""
    result = run_experiment(seed=42, n_samples=256)
    rb_ret = result["metrics"]["late_retention_rate"]
    fc_ret = result["baseline_metrics"]["late_retention_rate"]
    assert rb_ret > fc_ret, (
        f"Reorder buffer retention ({rb_ret}) should be strictly greater than "
        f"baseline retention ({fc_ret})"
    )


def test_metrics_are_finite() -> None:
    """All metric values must be finite numbers."""
    result = run_experiment(seed=42, n_samples=256)
    for key, value in result["metrics"].items():
        assert isinstance(value, (int, float)), f"Metric {key} is not a number"
        assert np.isfinite(value), f"Metric {key} is not finite: {value}"
    for key, value in result["baseline_metrics"].items():
        assert isinstance(value, (int, float)), f"Baseline metric {key} is not a number"
        assert np.isfinite(value), f"Baseline metric {key} is not finite: {value}"


def test_edge_case_minimal_n_samples() -> None:
    """run_experiment must work correctly at the minimum n_samples=32."""
    result = run_experiment(seed=42, n_samples=32)
    assert result["n_samples"] == 32
    assert len(result["metrics"]) > 0
    assert len(result["baseline_metrics"]) > 0
    # Reorder buffer should still beat baseline at minimal size
    assert result["metrics"]["window_sum_mae"] < result["baseline_metrics"]["window_sum_mae"]
    assert result["metrics"]["late_retention_rate"] > result["baseline_metrics"]["late_retention_rate"]


def test_max_abs_error_non_negative() -> None:
    """max_abs_error must be non-negative for both algorithm and baseline."""
    result = run_experiment(seed=42, n_samples=256)
    assert result["metrics"]["max_abs_error"] >= 0.0
    assert result["baseline_metrics"]["max_abs_error"] >= 0.0


def test_window_sum_mae_non_negative() -> None:
    """window_sum_mae must be non-negative for both algorithm and baseline."""
    result = run_experiment(seed=42, n_samples=256)
    assert result["metrics"]["window_sum_mae"] >= 0.0
    assert result["baseline_metrics"]["window_sum_mae"] >= 0.0


def test_retention_rate_in_valid_range() -> None:
    """late_retention_rate must be in [0, 1] for both algorithm and baseline."""
    result = run_experiment(seed=42, n_samples=256)
    assert 0.0 <= result["metrics"]["late_retention_rate"] <= 1.0
    assert 0.0 <= result["baseline_metrics"]["late_retention_rate"] <= 1.0
