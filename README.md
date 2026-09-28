# Bounded Reorder Buffer for Out-of-Order Sliding Window Aggregation

An in-memory simulation comparing a bounded reorder buffer against a strict forward-cursor baseline for computing sliding window sums over an out-of-order event stream. The project demonstrates how retaining late events within a bounded delay window improves aggregation accuracy compared to permanently dropping events once the time cursor advances.

## Problem

In streaming data systems, events often arrive out of order due to network latency, partitioning, or processing delays. When computing sliding window aggregations (e.g., sum of values over the last W time units), out-of-order arrivals create a fundamental tension:

- **Strict forward cursor**: Process events in arrival order, ratcheting a time cursor forward. Any event with a timestamp older than the cursor is permanently dropped. This is simple and memory-bounded but loses data, causing cumulative drift in all future window computations.
- **Bounded reorder buffer**: Retain late events for a bounded delay period, allowing them to contribute to window sums computed after their arrival. This improves accuracy at the cost of bounded memory.

This project simulates both approaches on a synthetic event stream and quantifies the accuracy difference using mean absolute error against an oracle that has access to all events.

## Implementation

### Architecture

The project consists of four files:

| File | Purpose |
|------|---------|
| `data.py` | Generates a synthetic out-of-order event stream and oracle window sums |
| `app.py` | Implements the reorder buffer and forward cursor algorithms, computes metrics |
| `test_project.py` | Pytest test cases validating dataset properties and algorithm behavior |
| `main.py` | Host-provided entry point (not included in this repository) |

### Data Generation (`data.py`)

`make_dataset(seed, n_samples)` produces:

- **X**: A NumPy array of shape `(n_samples, 2)` where column 0 is `event_time` (integer, 0 to n-1) and column 1 is `value` (float, uniform in [0, 10]). Rows are in **arrival order**, which is a perturbation of the sorted time order. Each event's position is displaced by at most `n_samples // 8` positions from its sorted position.
- **y**: A NumPy array of shape `(n_samples,)` containing the oracle sliding window sum for each event. The oracle computes the sum of all values with `event_time` in `[t - W, t]` where `W = n_samples // 8`, using all events regardless of arrival order.

The perturbation is seeded and deterministic. A different seed produces a different arrival order. `n_samples < 32` raises `ValueError`.

### Algorithm: Bounded Reorder Buffer (`app.py`)

For each event in arrival order:

1. Track `max_seen_time` (the maximum event time observed so far).
2. Compute `drop_threshold = max_seen_time - reorder_delay`.
3. If `event_time < drop_threshold`, the event is **dropped** (too late).
4. Otherwise, the event is **retained** in an in-memory dictionary mapping `event_time` to `value`.
5. The sliding window sum at this step is the sum of all retained values with `event_time` in `[event_time - W, event_time]`.

Key property: A retained late event contributes to **all future** window computations where its time falls within the window range.

### Baseline: Strict Forward Cursor (`app.py`)

For each event in arrival order:

1. Track `processing_time` (ratchets forward to the maximum event time seen).
2. If `event_time < processing_time`, the event is **permanently dropped** and never contributes to any window sum.
3. If `event_time == processing_time`, the event is stored.
4. The sliding window sum is computed from stored values in `[event_time - W, event_time]`.

Key property: Once the cursor passes an event's time, that event is lost forever. Errors are **cumulative** in all subsequent windows.

### Metrics

| Metric | Direction | Description |
|--------|-----------|-------------|
| `window_sum_mae` | Lower is better | Mean absolute error between computed window sums and oracle sums |
| `late_retention_rate` | Higher is better | Fraction of out-of-order events that were not dropped |
| `max_abs_error` | Lower is better | Maximum absolute error across all window computations |

The reorder buffer is expected to achieve lower `window_sum_mae` and higher `late_retention_rate` than the forward cursor baseline, because it retains late events that the baseline permanently discards.

## Synthetic Dataset Assumptions

- **Event times**: One event per integer time slot from 0 to n-1. No gaps, no duplicates.
- **Values**: Independent uniform random variables in [0, 10].
- **Out-of-order perturbation**: Bounded displacement of at most `n // 8` positions from sorted order. This models mild reordering (e.g., small network jitter) rather than extreme shuffling.
- **Window size**: `W = n // 8`. The reorder delay equals the window size, meaning an event is retained if it arrives within one window of the latest seen time.
- **No train/test split**: This is a single-pass streaming simulation. All events are processed in arrival order and evaluated. There is no leakage concern because the oracle is computed independently from the full dataset.

### Limitations

- The perturbation is bounded and relatively mild (at most n/8 displacement). In production systems, out-of-order arrivals can be more severe.
- The reorder buffer uses an in-memory dictionary and does not model memory pressure, eviction policies, or backpressure.
- The simulation is single-threaded and in-memory; it does not model network latency, partitioning, or distributed state.
- The oracle assumes perfect knowledge of all events, which is not available in real streaming systems.
- Results are specific to the synthetic data distribution and perturbation model; they do not generalize to arbitrary event streams.

## Reproducibility

The project is deterministic for a given seed and n_samples. Running the same experiment twice with identical parameters produces identical results.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Run the Experiment

```bash
python main.py --seed 42 --n-samples 256 --output results.json
```

This produces a JSON file with the experiment results, including metrics for both the reorder buffer and the forward cursor baseline.

### Run Tests

```bash
python -m pytest -q
```

### Python Version

The project requires Python 3.11, 3.12, or 3.13. It uses `from __future__ import annotations` for forward-reference compatibility and relies on NumPy's `default_rng` for reproducible random number generation.

## Test Coverage

`test_project.py` includes test cases that verify:

- Dataset generation respects the `n_samples` parameter and raises `ValueError` for `n_samples < 32`.
- Different seeds produce different arrival orders.
- The reorder buffer achieves lower `window_sum_mae` than the forward cursor baseline.
- The reorder buffer achieves higher `late_retention_rate` than the forward cursor baseline.
- The oracle sums in `y` are consistent with the event times and values in `X`.

### Limits of Automated Tests

The tests validate structural properties and relative metric ordering on the synthetic dataset. They do not:

- Verify correctness against a real streaming system.
- Test performance under memory pressure or high event rates.
- Validate behavior under more severe out-of-order perturbations.
- Measure wall-clock time or throughput.

## Output Artifacts

- `results.json`: Experiment output from `main.py`, containing `n_samples`, `metrics`, `baseline_metrics`, and `explanation`.
- `validation_report.json`: Host-generated validation report (not included in this repository).
- `example_results.json`: Example output for reference (not included in this repository).

## Scope

This project is an educational simulation. It is not a production-ready streaming system. It does not include:

- Distributed state management
- Checkpointing or fault tolerance
- Network communication
- Backpressure or flow control
- Real-time processing guarantees

The goal is to illustrate the accuracy tradeoff between bounded reordering and strict forward processing in a controlled, reproducible setting.

## Recorded automated validation

Host contract tests and project tests passed (16 tests, 0 skipped). Demo completed on Python 3.13.15. See `validation_report.json` and `example_results.json`. These checks validate the execution contract, not scientific novelty or every algorithmic claim.
