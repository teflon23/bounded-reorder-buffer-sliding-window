"""Bounded reorder buffer vs strict forward cursor for sliding window aggregation."""

from __future__ import annotations

import numpy as np

from data import make_dataset


def _reorder_buffer_window_sums(
    X: np.ndarray,
    reorder_delay: int,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Process events through a bounded reorder buffer.

    Parameters
    ----------
    X : np.ndarray
        Shape (n, 2). Column 0 is event_time, column 1 is value. Rows in arrival order.
    reorder_delay : int
        An event is dropped only if event_time < max_seen_time - reorder_delay.
    window_size : int
        Sliding window width W.

    Returns
    -------
    window_sums : np.ndarray
        Shape (n,). The sliding window sum computed at each event's processing step.
    late_retention_rate : float
        Fraction of out-of-order events (event_time < max_seen_time at arrival)
        that were NOT dropped.
    """
    n = len(X)
    stored: dict[int, float] = {}
    max_seen_time = -1
    window_sums = np.zeros(n, dtype=np.float64)
    out_of_order_count = 0
    retained_count = 0

    for i in range(n):
        event_time = int(X[i, 0])
        value = float(X[i, 1])

        # Track whether this event is out-of-order relative to max seen so far
        is_late = event_time < max_seen_time

        # Update max_seen_time
        if event_time > max_seen_time:
            max_seen_time = event_time

        # Decide whether to drop or retain
        drop_threshold = max_seen_time - reorder_delay
        if event_time < drop_threshold:
            # Drop the event
            if is_late:
                out_of_order_count += 1
                # not retained
        else:
            # Retain the event in the buffer
            stored[event_time] = value
            if is_late:
                out_of_order_count += 1
                retained_count += 1

        # Compute sliding window sum: sum of stored values in [event_time - W, event_time]
        lo = event_time - window_size
        hi = event_time
        total = 0.0
        for t in range(lo, hi + 1):
            if t in stored:
                total += stored[t]
        window_sums[i] = total

    if out_of_order_count > 0:
        late_retention_rate = retained_count / out_of_order_count
    else:
        late_retention_rate = 1.0

    return window_sums, window_sums, late_retention_rate


def _forward_cursor_window_sums(
    X: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, float]:
    """Process events through a strict forward-cursor baseline.

    Parameters
    ----------
    X : np.ndarray
        Shape (n, 2). Column 0 is event_time, column 1 is value. Rows in arrival order.
    window_size : int
        Sliding window width W.

    Returns
    -------
    window_sums : np.ndarray
        Shape (n,). The sliding window sum computed at each event's processing step.
    late_retention_rate : float
        Fraction of out-of-order events that were NOT dropped (always 0 for this baseline).
    """
    n = len(X)
    stored: dict[int, float] = {}
    processing_time = -1
    window_sums = np.zeros(n, dtype=np.float64)
    out_of_order_count = 0
    retained_count = 0

    for i in range(n):
        event_time = int(X[i, 0])
        value = float(X[i, 1])

        is_late = event_time < processing_time

        # Ratchet processing_time forward
        if event_time > processing_time:
            processing_time = event_time

        # Strict forward cursor: drop if event_time < processing_time
        if event_time < processing_time:
            # Permanently dropped
            if is_late:
                out_of_order_count += 1
                # not retained
        else:
            # event_time == processing_time (just advanced), retain
            stored[event_time] = value
            if is_late:
                out_of_order_count += 1
                retained_count += 1

        # Compute sliding window sum: sum of stored values in [event_time - W, event_time]
        lo = event_time - window_size
        hi = event_time
        total = 0.0
        for t in range(lo, hi + 1):
            if t in stored:
                total += stored[t]
        window_sums[i] = total

    if out_of_order_count > 0:
        late_retention_rate = retained_count / out_of_order_count
    else:
        late_retention_rate = 1.0

    return window_sums, late_retention_rate


def run_experiment(seed: int = 42, n_samples: int = 256) -> dict:
    """Run the bounded reorder buffer experiment and compare against baseline.

    Parameters
    ----------
    seed : int
        Seed for dataset generation.
    n_samples : int
        Number of events. Must be at least 32.

    Returns
    -------
    dict
        JSON-serializable dictionary with n_samples, metrics, baseline_metrics,
        and explanation.
    """
    X, y = make_dataset(seed=seed, n_samples=n_samples)
    n = len(X)
    W = n // 8
    reorder_delay = n // 8

    # Bounded reorder buffer
    rb_sums, _, rb_retention = _reorder_buffer_window_sums(X, reorder_delay, W)

    # Strict forward cursor baseline
    fc_sums, fc_retention = _forward_cursor_window_sums(X, W)

    # Metrics for reorder buffer
    rb_mae = float(np.mean(np.abs(rb_sums - y)))
    rb_max_abs_error = float(np.max(np.abs(rb_sums - y)))

    # Metrics for forward cursor baseline
    fc_mae = float(np.mean(np.abs(fc_sums - y)))
    fc_max_abs_error = float(np.max(np.abs(fc_sums - y)))

    metrics = {
        "window_sum_mae": rb_mae,
        "late_retention_rate": float(rb_retention),
        "max_abs_error": rb_max_abs_error,
    }
    baseline_metrics = {
        "window_sum_mae": fc_mae,
        "late_retention_rate": float(fc_retention),
        "max_abs_error": fc_max_abs_error,
    }

    explanation = (
        f"Sliding window size W={W}, reorder_delay={reorder_delay}. "
        f"Reorder buffer retains late events within {reorder_delay} time units "
        f"of the max seen time, so they contribute to future window sums. "
        f"Forward cursor permanently drops events once the cursor passes them. "
        f"Reorder buffer MAE={rb_mae:.4f} vs baseline MAE={fc_mae:.4f}. "
        f"Late retention: buffer={rb_retention:.4f}, baseline={fc_retention:.4f}."
    )

    return {
        "n_samples": int(n),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "explanation": explanation,
    }
