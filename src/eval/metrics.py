"""Mobility metrics and controls for trajectory samples on a Cartesian grid."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _as_trajectories(values: np.ndarray) -> np.ndarray:
    trajectories = np.asarray(values, dtype=np.float64)
    if trajectories.ndim == 2 and trajectories.shape[1] % 2 == 0:
        trajectories = trajectories.reshape(trajectories.shape[0], -1, 2)
    if trajectories.ndim != 3 or trajectories.shape[-1] != 2:
        raise ValueError("Trajectories must have shape (N, L, 2) or (N, 2L)")
    if trajectories.shape[0] == 0 or trajectories.shape[1] < 2:
        raise ValueError("At least one trajectory with two points is required")
    if not np.isfinite(trajectories).all():
        raise ValueError("Trajectories contain NaN or infinite values")
    return trajectories


def density_jensen_shannon(
    generated: np.ndarray,
    reference: np.ndarray,
    *,
    width: int = 200,
    height: int = 200,
    coordinate_min: float = 1.0,
) -> float:
    """Jensen-Shannon divergence between point-density histograms (natural log)."""
    generated = _as_trajectories(generated)
    reference = _as_trajectories(reference)
    x_edges = np.arange(width + 1, dtype=np.float64) + coordinate_min - 0.5
    y_edges = np.arange(height + 1, dtype=np.float64) + coordinate_min - 0.5
    generated_hist, _, _ = np.histogram2d(
        generated[..., 0].ravel(), generated[..., 1].ravel(), bins=(x_edges, y_edges)
    )
    reference_hist, _, _ = np.histogram2d(
        reference[..., 0].ravel(), reference[..., 1].ravel(), bins=(x_edges, y_edges)
    )
    if generated_hist.sum() == 0 or reference_hist.sum() == 0:
        raise ValueError("No trajectory points fall inside the configured grid")
    p = generated_hist.ravel() / generated_hist.sum()
    q = reference_hist.ravel() / reference_hist.sum()
    midpoint = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log(left[mask] / right[mask])))

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def dynamic_time_warping(curve_a: np.ndarray, curve_b: np.ndarray) -> float:
    """Exact DTW distance with Euclidean point cost."""
    curve_a = np.asarray(curve_a, dtype=np.float64)
    curve_b = np.asarray(curve_b, dtype=np.float64)
    distances = np.linalg.norm(curve_a[:, None, :] - curve_b[None, :, :], axis=2)
    previous = np.full(curve_b.shape[0] + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    for row in distances:
        current = np.full_like(previous, np.inf)
        for column, point_distance in enumerate(row, start=1):
            current[column] = point_distance + min(
                current[column - 1], previous[column], previous[column - 1]
            )
        previous = current
    return float(previous[-1])


def _remove_consecutive_duplicates(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float64)
    keep = np.ones(len(curve), dtype=bool)
    keep[1:] = np.any(np.diff(curve, axis=0) != 0, axis=1)
    return curve[keep]


def continuous_frechet(curve_a: np.ndarray, curve_b: np.ndarray) -> float:
    """Continuous Fréchet distance using CurveSimilarities' Alt-Godau implementation."""
    curve_a = _remove_consecutive_duplicates(curve_a)
    curve_b = _remove_consecutive_duplicates(curve_b)
    if len(curve_a) == 1 and len(curve_b) == 1:
        return float(np.linalg.norm(curve_a[0] - curve_b[0]))
    if len(curve_a) == 1:
        return float(np.linalg.norm(curve_b - curve_a[0], axis=1).max())
    if len(curve_b) == 1:
        return float(np.linalg.norm(curve_a - curve_b[0], axis=1).max())
    try:
        from curvesimilarities.frechet import fd
    except ImportError as exc:
        raise RuntimeError(
            "Continuous Frechet evaluation requires curvesimilarities==0.3.0"
        ) from exc
    return float(fd(curve_a, curve_b, rel_tol=1e-6, abs_tol=1e-8))


def _paired_summary(
    generated: np.ndarray,
    reference: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    max_pairs: int,
) -> tuple[float, float, int]:
    pair_count = min(len(generated), len(reference), max_pairs)
    if pair_count <= 0:
        raise ValueError("max_pairs must be positive")
    values = np.asarray(
        [metric(generated[index], reference[index]) for index in range(pair_count)],
        dtype=np.float64,
    )
    return float(values.mean()), float(np.median(values)), pair_count


def origin_destination_straight_lines(reference: np.ndarray) -> np.ndarray:
    """Build the no-model control that linearly joins each trajectory's O/D.

    The YJMob generation path decodes the condition's origin and destination into
    the first and last reference points before evaluation.  Using those endpoints
    here therefore measures what the condition alone can achieve.
    """
    reference = _as_trajectories(reference)
    weights = np.linspace(0.0, 1.0, reference.shape[1], dtype=np.float64)
    return (
        reference[:, :1, :] * (1.0 - weights[None, :, None])
        + reference[:, -1:, :] * weights[None, :, None]
    )


def _bounds_summary(
    trajectories: np.ndarray,
    *,
    width: int,
    height: int,
    coordinate_min: float,
) -> dict[str, float | int]:
    trajectories = _as_trajectories(trajectories)
    x_min = coordinate_min - 0.5
    x_max = coordinate_min + width - 0.5
    y_min = coordinate_min - 0.5
    y_max = coordinate_min + height - 0.5
    in_bounds = (
        (trajectories[..., 0] >= x_min)
        & (trajectories[..., 0] < x_max)
        & (trajectories[..., 1] >= y_min)
        & (trajectories[..., 1] < y_max)
    )
    point_oob = ~in_bounds
    trajectory_oob = np.any(point_oob, axis=1)
    return {
        "out_of_bounds_point_count": int(point_oob.sum()),
        "out_of_bounds_point_rate": float(point_oob.mean()),
        "out_of_bounds_trajectory_count": int(trajectory_oob.sum()),
        "out_of_bounds_trajectory_rate": float(trajectory_oob.mean()),
    }


def compute_baseline_metrics(
    generated: np.ndarray,
    reference: np.ndarray,
    *,
    grid_metadata: dict | None = None,
    max_pairs: int = 200,
) -> dict[str, object]:
    """Compute density JSD plus paired DTW and continuous Fréchet distances."""
    generated = _as_trajectories(generated)
    reference = _as_trajectories(reference)
    metadata = grid_metadata or {}
    width = int(metadata.get("width", 200))
    height = int(metadata.get("height", 200))
    coordinate_min = float(metadata.get("coordinate_min", 1))
    cell_size_km = float(metadata.get("cell_size_m", 500.0)) / 1000.0

    dtw_mean, dtw_median, pair_count = _paired_summary(
        generated, reference, dynamic_time_warping, max_pairs
    )
    frechet_mean, frechet_median, _ = _paired_summary(
        generated, reference, continuous_frechet, max_pairs
    )
    jsd = density_jensen_shannon(
        generated,
        reference,
        width=width,
        height=height,
        coordinate_min=coordinate_min,
    )
    generated_bounds = _bounds_summary(
        generated,
        width=width,
        height=height,
        coordinate_min=coordinate_min,
    )
    reference_bounds = _bounds_summary(
        reference,
        width=width,
        height=height,
        coordinate_min=coordinate_min,
    )
    return {
        "generated_shape": list(generated.shape),
        "reference_shape": list(reference.shape),
        "density_js_divergence": jsd,
        "density_js_log_base": "e",
        "paired_curve_count": pair_count,
        "dtw_mean_grid_units": dtw_mean,
        "dtw_median_grid_units": dtw_median,
        "dtw_mean_km": dtw_mean * cell_size_km,
        "dtw_median_km": dtw_median * cell_size_km,
        "continuous_frechet_mean_grid_units": frechet_mean,
        "continuous_frechet_median_grid_units": frechet_median,
        "continuous_frechet_mean_km": frechet_mean * cell_size_km,
        "continuous_frechet_median_km": frechet_median * cell_size_km,
        # Keep the original key for compatibility; it is the point-level rate.
        "generated_out_of_bounds_rate": generated_bounds["out_of_bounds_point_rate"],
        "generated_out_of_bounds_point_count": generated_bounds["out_of_bounds_point_count"],
        "generated_out_of_bounds_trajectory_count": generated_bounds[
            "out_of_bounds_trajectory_count"
        ],
        "generated_out_of_bounds_trajectory_rate": generated_bounds[
            "out_of_bounds_trajectory_rate"
        ],
        "reference_out_of_bounds_point_rate": reference_bounds[
            "out_of_bounds_point_rate"
        ],
        "reference_out_of_bounds_trajectory_rate": reference_bounds[
            "out_of_bounds_trajectory_rate"
        ],
        "grid_cell_size_km": cell_size_km,
    }


def compute_control_metrics(
    generated: np.ndarray,
    reference: np.ndarray,
    real_control: np.ndarray,
    *,
    parameterized_reference: np.ndarray | None = None,
    grid_metadata: dict | None = None,
    max_pairs: int = 200,
) -> dict[str, object]:
    """Compare a model with independent-real and condition-only controls."""
    generated = _as_trajectories(generated)
    reference = _as_trajectories(reference)
    real_control = _as_trajectories(real_control)
    straight_line = origin_destination_straight_lines(reference)
    model_metrics = compute_baseline_metrics(
        generated,
        reference,
        grid_metadata=grid_metadata,
        max_pairs=max_pairs,
    )
    straight_metrics = compute_baseline_metrics(
        straight_line,
        reference,
        grid_metadata=grid_metadata,
        max_pairs=max_pairs,
    )

    metadata = grid_metadata or {}
    width = int(metadata.get("width", 200))
    height = int(metadata.get("height", 200))
    coordinate_min = float(metadata.get("coordinate_min", 1))
    real_control_bounds = _bounds_summary(
        real_control,
        width=width,
        height=height,
        coordinate_min=coordinate_min,
    )
    real_reference_bounds = _bounds_summary(
        reference,
        width=width,
        height=height,
        coordinate_min=coordinate_min,
    )
    real_density_control = {
        "reference_shape": list(reference.shape),
        "independent_real_shape": list(real_control.shape),
        "density_js_divergence": density_jensen_shannon(
            reference,
            real_control,
            width=width,
            height=height,
            coordinate_min=coordinate_min,
        ),
        "density_js_log_base": "e",
        "reference_out_of_bounds_point_rate": real_reference_bounds[
            "out_of_bounds_point_rate"
        ],
        "independent_real_out_of_bounds_point_rate": real_control_bounds[
            "out_of_bounds_point_rate"
        ],
    }
    comparison = {
        "positive_difference_means_model_is_better": True,
        "dtw_median_km_straight_minus_model": (
            straight_metrics["dtw_median_km"] - model_metrics["dtw_median_km"]
        ),
        "continuous_frechet_median_km_straight_minus_model": (
            straight_metrics["continuous_frechet_median_km"]
            - model_metrics["continuous_frechet_median_km"]
        ),
        "model_beats_straight_line_on_dtw_median": (
            model_metrics["dtw_median_km"] < straight_metrics["dtw_median_km"]
        ),
        "model_beats_straight_line_on_continuous_frechet_median": (
            model_metrics["continuous_frechet_median_km"]
            < straight_metrics["continuous_frechet_median_km"]
        ),
    }
    result: dict[str, object] = {
        "schema_version": "trajflow-control-evaluation-v1",
        "metric_semantics": {
            "primary_reference": "paired raw test trajectories",
            "real_density_control": "an independent, disjoint test sample",
            "straight_line_control": (
                "linear interpolation between condition-decoded origin and destination"
            ),
            "density_histogram_oob_policy": "out-of-grid points are excluded and reported separately",
            "dtw": "exact accumulated Euclidean point cost",
            "continuous_frechet": "Alt-Godau continuous curve distance",
        },
        "array_shapes": {
            "generated": list(generated.shape),
            "paired_raw_reference": list(reference.shape),
            "independent_real_control": list(real_control.shape),
            "od_straight_line_control": list(straight_line.shape),
        },
        "model_vs_paired_raw_test": model_metrics,
        "real_vs_real_density_control": real_density_control,
        "od_straight_line_vs_paired_raw_test": straight_metrics,
        "model_improvement_over_straight_line": comparison,
    }
    if parameterized_reference is not None:
        parameterized_reference = _as_trajectories(parameterized_reference)
        result["array_shapes"]["parameterized_training_target"] = list(
            parameterized_reference.shape
        )
        result["model_vs_parameterized_training_target_diagnostic"] = (
            compute_baseline_metrics(
                generated,
                parameterized_reference,
                grid_metadata=grid_metadata,
                max_pairs=max_pairs,
            )
        )
    return result
