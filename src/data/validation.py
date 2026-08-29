"""Training-target validation shared by dataset loading and run provenance."""

from __future__ import annotations

import numpy as np


def summarize_parameterized_targets(
    trajectories: np.ndarray,
    *,
    configured_control_points: int,
    source_has_movement: bool,
    decimals: int = 6,
) -> dict[str, object]:
    """Validate and summarize a fixed-width trajectory representation.

    A small number of stationary trajectories is valid.  A moving source dataset
    collapsing entirely to one repeated control point is not, and should stop the
    run before optimization starts.
    """
    trajectories = np.asarray(trajectories, dtype=np.float64)
    expected_shape = (int(configured_control_points), 2)
    if trajectories.ndim != 3 or tuple(trajectories.shape[1:]) != expected_shape:
        raise ValueError(
            "Parameterized targets must have shape "
            f"(N, {expected_shape[0]}, 2); got {trajectories.shape}"
        )
    if len(trajectories) == 0:
        raise ValueError("Parameterized target dataset is empty")
    if not np.isfinite(trajectories).all():
        raise ValueError("Parameterized targets contain NaN or infinite values")

    distinct_counts = np.asarray(
        [
            len(np.unique(np.round(trajectory, decimals=decimals), axis=0))
            for trajectory in trajectories
        ],
        dtype=np.int64,
    )
    if source_has_movement and int(distinct_counts.max()) <= 1:
        raise RuntimeError(
            "Parameterized targets collapsed a moving source dataset to repeated points"
        )
    unique, counts = np.unique(distinct_counts, return_counts=True)
    degenerate_count = int((distinct_counts <= 1).sum())
    return {
        "shape": list(trajectories.shape),
        "configured_control_points": int(configured_control_points),
        "finite": True,
        "source_has_movement": bool(source_has_movement),
        "distinct_control_points": {
            "min": int(distinct_counts.min()),
            "p10": float(np.quantile(distinct_counts, 0.10)),
            "p50": float(np.quantile(distinct_counts, 0.50)),
            "p90": float(np.quantile(distinct_counts, 0.90)),
            "max": int(distinct_counts.max()),
            "mean": float(distinct_counts.mean()),
        },
        "distinct_control_point_histogram": {
            str(int(value)): int(count)
            for value, count in zip(unique, counts, strict=True)
        },
        "repeated_single_point_trajectory_count": degenerate_count,
        "repeated_single_point_trajectory_rate": float(
            degenerate_count / len(distinct_counts)
        ),
    }
