"""Synthetic out-of-order event stream for bounded reorder buffer experiments."""

from __future__ import annotations

import numpy as np


def make_dataset(seed: int = 42, n_samples: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic event stream with out-of-order arrivals.

    Parameters
    ----------
    seed : int
        Seed for the random number generator.
    n_samples : int
        Number of events. Must be at least 32.

    Returns
    -------
    X : np.ndarray
        Shape (n_samples, 2). Column 0 is event_time (int), column 1 is value (float).
        Rows are in arrival order (perturbed from sorted order).
    y : np.ndarray
        Shape (n_samples,). Oracle sliding-window sum for each event, computed using
        ALL events (regardless of arrival order) with event_time in [t - W, t],
        where W = n_samples // 8.

    Raises
    ------
    ValueError
        If n_samples < 32.
    """
    if n_samples < 32:
        raise ValueError(f"n_samples must be >= 32, got {n_samples}")

    rng = np.random.default_rng(seed)

    # Event times: 0, 1, 2, ..., n-1 (one event per integer time slot)
    event_times = np.arange(n_samples, dtype=np.int64)

    # Values: uniform in [0, 10]
    values = rng.uniform(0.0, 10.0, size=n_samples)

    # Perturbation: bounded integer displacement of at most n//8 positions.
    # We create a permutation where each element moves at most max_shift positions.
    max_shift = n_samples // 8

    # Build arrival order via a constrained shuffle:
    # Start with sorted order, then for each position, swap with a random
    # position within max_shift distance (if not already swapped).
    order = np.arange(n_samples, dtype=np.int64)
    for i in range(n_samples):
        lo = max(0, i - max_shift)
        hi = min(n_samples - 1, i + max_shift)
        j = rng.integers(lo, hi + 1)
        order[i], order[j] = order[j], order[i]

    # X in arrival order: row i is the event that arrives at step i.
    # order[i] is the original index of the event arriving at step i.
    X = np.column_stack((event_times[order], values[order]))

    # Oracle window sums: for each event with time t, sum values of ALL events
    # with event_time in [t - W, t].
    W = n_samples // 8
    # Use the original (sorted) arrays for oracle computation.
    y = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        t = event_times[i]
        lo = max(0, t - W)
        hi = min(n_samples - 1, t)
        y[i] = values[lo : hi + 1].sum()

    return X, y
