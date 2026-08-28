"""Mobility metrics and controls for trajectory samples on a Cartesian grid."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

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
    bins: int | tuple[int, int] | None = None,
) -> float:
    """Jensen-Shannon divergence between point-density histograms (natural log)."""
    generated = _as_trajectories(generated)
    reference = _as_trajectories(reference)
    x_bins, y_bins = _normalize_density_bins(bins, width=width, height=height)
    x_edges = np.linspace(
        coordinate_min - 0.5,
        coordinate_min + width - 0.5,
        x_bins + 1,
        dtype=np.float64,
    )
    y_edges = np.linspace(
        coordinate_min - 0.5,
        coordinate_min + height - 0.5,
        y_bins + 1,
        dtype=np.float64,
    )
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


def _normalize_density_bins(
    bins: int | tuple[int, int] | None,
    *,
    width: int,
    height: int,
) -> tuple[int, int]:
    if bins is None:
        result = (int(width), int(height))
    elif isinstance(bins, (int, np.integer)):
        result = (int(bins), int(bins))
    else:
        if len(bins) != 2:
            raise ValueError("density bins must be an integer or a two-item tuple")
        result = (int(bins[0]), int(bins[1]))
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError("density bins must be positive")
    return result


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
        "point_count": int(point_oob.size),
        "in_bounds_point_count": int(in_bounds.sum()),
        "in_bounds_point_rate": float(in_bounds.mean()),
        "out_of_bounds_point_count": int(point_oob.sum()),
        "out_of_bounds_point_rate": float(point_oob.mean()),
        "trajectory_count": len(trajectories),
        "out_of_bounds_trajectory_count": int(trajectory_oob.sum()),
        "out_of_bounds_trajectory_rate": float(trajectory_oob.mean()),
    }


def compute_baseline_metrics(
    generated: np.ndarray,
    reference: np.ndarray,
    *,
    grid_metadata: dict | None = None,
    max_pairs: int = 200,
    density_bins: int | tuple[int, int] | None = None,
) -> dict[str, object]:
    """Compute density JSD plus paired DTW and continuous Fréchet distances."""
    generated = _as_trajectories(generated)
    reference = _as_trajectories(reference)
    metadata = grid_metadata or {}
    width = int(metadata.get("width", 200))
    height = int(metadata.get("height", 200))
    coordinate_min = float(metadata.get("coordinate_min", 1))
    cell_size_km = float(metadata.get("cell_size_m", 500.0)) / 1000.0
    normalized_density_bins = _normalize_density_bins(
        density_bins,
        width=width,
        height=height,
    )

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
        bins=normalized_density_bins,
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
        "density_grid_bins": list(normalized_density_bins),
        "density_histogram_points": {
            "generated_total": generated_bounds["point_count"],
            "generated_in_bounds": generated_bounds["in_bounds_point_count"],
            "generated_excluded_out_of_bounds": generated_bounds[
                "out_of_bounds_point_count"
            ],
            "generated_excluded_out_of_bounds_rate": generated_bounds[
                "out_of_bounds_point_rate"
            ],
            "reference_total": reference_bounds["point_count"],
            "reference_in_bounds": reference_bounds["in_bounds_point_count"],
            "reference_excluded_out_of_bounds": reference_bounds[
                "out_of_bounds_point_count"
            ],
            "reference_excluded_out_of_bounds_rate": reference_bounds[
                "out_of_bounds_point_rate"
            ],
        },
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
        "reference_out_of_bounds_point_count": reference_bounds[
            "out_of_bounds_point_count"
        ],
        "reference_out_of_bounds_trajectory_count": reference_bounds[
            "out_of_bounds_trajectory_count"
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
    density_bins: int | tuple[int, int] | None = None,
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
        density_bins=density_bins,
    )
    straight_metrics = compute_baseline_metrics(
        straight_line,
        reference,
        grid_metadata=grid_metadata,
        max_pairs=max_pairs,
        density_bins=density_bins,
    )

    metadata = grid_metadata or {}
    width = int(metadata.get("width", 200))
    height = int(metadata.get("height", 200))
    coordinate_min = float(metadata.get("coordinate_min", 1))
    normalized_density_bins = _normalize_density_bins(
        density_bins,
        width=width,
        height=height,
    )
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
            bins=normalized_density_bins,
        ),
        "density_js_log_base": "e",
        "density_grid_bins": list(normalized_density_bins),
        "density_histogram_points": {
            "reference_total": real_reference_bounds["point_count"],
            "reference_in_bounds": real_reference_bounds["in_bounds_point_count"],
            "reference_excluded_out_of_bounds": real_reference_bounds[
                "out_of_bounds_point_count"
            ],
            "reference_excluded_out_of_bounds_rate": real_reference_bounds[
                "out_of_bounds_point_rate"
            ],
            "independent_real_total": real_control_bounds["point_count"],
            "independent_real_in_bounds": real_control_bounds[
                "in_bounds_point_count"
            ],
            "independent_real_excluded_out_of_bounds": real_control_bounds[
                "out_of_bounds_point_count"
            ],
            "independent_real_excluded_out_of_bounds_rate": real_control_bounds[
                "out_of_bounds_point_rate"
            ],
        },
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
        "schema_version": "trajflow-control-evaluation-v2",
        "metric_semantics": {
            "primary_reference": "paired raw test trajectories",
            "real_density_control": "an independent, disjoint test sample",
            "straight_line_control": (
                "linear interpolation between condition-decoded origin and destination"
            ),
            "density_histogram_oob_policy": "out-of-grid points are excluded and reported separately",
            "density_grid_bins": list(normalized_density_bins),
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
        "baseline_acceptance_gate": {
            "criterion": (
                "model median DTW and continuous Frechet must both be strictly "
                "lower than the same-sample O-to-D straight-line control"
            ),
            "passed": bool(
                comparison["model_beats_straight_line_on_dtw_median"]
                and comparison[
                    "model_beats_straight_line_on_continuous_frechet_median"
                ]
            ),
        },
    }
    if parameterized_reference is not None:
        parameterized_reference = _as_trajectories(parameterized_reference)
        result["array_shapes"]["parameterized_training_target"] = list(
            parameterized_reference.shape
        )
        model_vs_parameterized = compute_baseline_metrics(
            generated,
            parameterized_reference,
            grid_metadata=grid_metadata,
            max_pairs=max_pairs,
            density_bins=density_bins,
        )
        representation_vs_raw = compute_baseline_metrics(
            parameterized_reference,
            reference,
            grid_metadata=grid_metadata,
            max_pairs=max_pairs,
            density_bins=density_bins,
        )
        result["model_vs_parameterized_training_target_diagnostic"] = (
            model_vs_parameterized
        )
        result["parameterized_representation_vs_paired_raw_test"] = (
            representation_vs_raw
        )
        result["representation_ceiling_comparison"] = {
            "positive_straight_minus_representation_means_representation_is_better": True,
            "dtw_median_km_straight_minus_representation": (
                straight_metrics["dtw_median_km"]
                - representation_vs_raw["dtw_median_km"]
            ),
            "continuous_frechet_median_km_straight_minus_representation": (
                straight_metrics["continuous_frechet_median_km"]
                - representation_vs_raw["continuous_frechet_median_km"]
            ),
            "model_gap_to_representation_dtw_median_km": (
                model_metrics["dtw_median_km"]
                - representation_vs_raw["dtw_median_km"]
            ),
            "model_gap_to_representation_continuous_frechet_median_km": (
                model_metrics["continuous_frechet_median_km"]
                - representation_vs_raw["continuous_frechet_median_km"]
            ),
        }
    return result


def _distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot summarize an empty distribution")
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def _curve_distance_pair(
    pair: tuple[np.ndarray, np.ndarray],
) -> tuple[float, float]:
    generated, reference = pair
    return (
        dynamic_time_warping(generated, reference),
        continuous_frechet(generated, reference),
    )


def _candidate_curve_distances(
    generated: np.ndarray,
    reference: np.ndarray,
    *,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    condition_count, samples_per_condition = generated.shape[:2]
    pairs = [
        (generated[condition_index, sample_index], reference[condition_index])
        for condition_index in range(condition_count)
        for sample_index in range(samples_per_condition)
    ]
    if workers <= 1:
        values = list(map(_curve_distance_pair, pairs))
    else:
        chunksize = max(1, len(pairs) // (workers * 8))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            values = list(
                executor.map(_curve_distance_pair, pairs, chunksize=chunksize)
            )
    distances = np.asarray(values, dtype=np.float64).reshape(
        condition_count,
        samples_per_condition,
        2,
    )
    return distances[..., 0], distances[..., 1]


def _as_candidate_trajectories(
    generated: np.ndarray,
    *,
    condition_count: int,
) -> np.ndarray:
    generated = np.asarray(generated, dtype=np.float64)
    if generated.ndim == 3 and generated.shape[-1] == 2:
        if len(generated) % condition_count:
            raise ValueError(
                "Flattened candidates must be divisible by the reference count"
            )
        generated = generated.reshape(
            condition_count,
            len(generated) // condition_count,
            generated.shape[1],
            2,
        )
    if generated.ndim != 4 or generated.shape[-1] != 2:
        raise ValueError(
            "Candidates must have shape (N, K, L, 2) or (N*K, L, 2)"
        )
    if generated.shape[0] != condition_count:
        raise ValueError("Candidate and reference condition counts differ")
    if generated.shape[1] <= 0 or generated.shape[2] < 2:
        raise ValueError("At least one candidate with two points is required")
    if not np.isfinite(generated).all():
        raise ValueError("Candidates contain NaN or infinite values")
    return generated


def compute_best_of_k_metrics(
    generated: np.ndarray,
    reference: np.ndarray,
    *,
    grid_metadata: dict | None = None,
    density_bins: int | tuple[int, int] | None = None,
    workers: int = 1,
) -> dict[str, object]:
    """Diagnose conditional coverage from K draws without defining a model gate.

    DTW and continuous Frechet are evaluated between every candidate and its
    paired raw reference.  Diversity is the mean Euclidean distance between
    aligned points for every unordered pair of candidates under one condition;
    it is deliberately cheaper than the two oracle metrics and is never mixed
    into them.
    """
    reference = _as_trajectories(reference)
    generated = _as_candidate_trajectories(
        generated,
        condition_count=len(reference),
    )
    workers = int(workers)
    if workers <= 0:
        raise ValueError("workers must be positive")

    condition_count, samples_per_condition, trajectory_length, _ = generated.shape
    metadata = grid_metadata or {}
    width = int(metadata.get("width", 200))
    height = int(metadata.get("height", 200))
    coordinate_min = float(metadata.get("coordinate_min", 1))
    cell_size_km = float(metadata.get("cell_size_m", 500.0)) / 1000.0
    normalized_density_bins = _normalize_density_bins(
        density_bins,
        width=width,
        height=height,
    )

    dtw_grid, frechet_grid = _candidate_curve_distances(
        generated,
        reference,
        workers=workers,
    )
    dtw_km = dtw_grid * cell_size_km
    frechet_km = frechet_grid * cell_size_km
    best_dtw_indices = np.argmin(dtw_km, axis=1)
    best_frechet_indices = np.argmin(frechet_km, axis=1)
    row_indices = np.arange(condition_count)
    best_dtw_km = dtw_km[row_indices, best_dtw_indices]
    best_frechet_km = frechet_km[row_indices, best_frechet_indices]

    x_min = coordinate_min - 0.5
    x_max = coordinate_min + width - 0.5
    y_min = coordinate_min - 0.5
    y_max = coordinate_min + height - 0.5
    point_in_bounds = (
        (generated[..., 0] >= x_min)
        & (generated[..., 0] < x_max)
        & (generated[..., 1] >= y_min)
        & (generated[..., 1] < y_max)
    )
    candidate_oob = np.any(~point_in_bounds, axis=2)
    pooled_bounds = _bounds_summary(
        generated.reshape(-1, trajectory_length, 2),
        width=width,
        height=height,
        coordinate_min=coordinate_min,
    )

    pair_left, pair_right = np.triu_indices(samples_per_condition, k=1)
    if len(pair_left):
        pairwise_diversity_km = np.empty(
            (condition_count, len(pair_left)),
            dtype=np.float64,
        )
        for condition_index in range(condition_count):
            point_distances = np.linalg.norm(
                generated[condition_index, pair_left]
                - generated[condition_index, pair_right],
                axis=2,
            )
            pairwise_diversity_km[condition_index] = (
                point_distances.mean(axis=1) * cell_size_km
            )
        condition_diversity_km = pairwise_diversity_km.mean(axis=1)
        diversity = {
            "unordered_pairs_per_condition": int(len(pair_left)),
            "all_pair_distances_km": _distribution_summary(
                pairwise_diversity_km.ravel()
            ),
            "condition_mean_pairwise_distance_km": _distribution_summary(
                condition_diversity_km
            ),
        }
    else:
        pairwise_diversity_km = np.empty((condition_count, 0), dtype=np.float64)
        condition_diversity_km = np.zeros(condition_count, dtype=np.float64)
        diversity = {
            "unordered_pairs_per_condition": 0,
            "all_pair_distances_km": None,
            "condition_mean_pairwise_distance_km": None,
        }

    straight_line = origin_destination_straight_lines(reference)
    straight_metrics = compute_baseline_metrics(
        straight_line,
        reference,
        grid_metadata=metadata,
        max_pairs=condition_count,
        density_bins=normalized_density_bins,
    )

    per_condition = []
    candidate_rows = []
    for condition_index in range(condition_count):
        per_condition.append({
            "condition_index": condition_index,
            "best_dtw_sample_index": int(best_dtw_indices[condition_index]),
            "best_dtw_km": float(best_dtw_km[condition_index]),
            "best_continuous_frechet_sample_index": int(
                best_frechet_indices[condition_index]
            ),
            "best_continuous_frechet_km": float(
                best_frechet_km[condition_index]
            ),
            "candidate_out_of_bounds_count": int(
                candidate_oob[condition_index].sum()
            ),
            "mean_pairwise_aligned_point_distance_km": (
                float(condition_diversity_km[condition_index])
                if len(pair_left) else None
            ),
        })
        for sample_index in range(samples_per_condition):
            candidate_rows.append({
                "condition_index": condition_index,
                "sample_index": sample_index,
                "dtw_km": float(dtw_km[condition_index, sample_index]),
                "continuous_frechet_km": float(
                    frechet_km[condition_index, sample_index]
                ),
                "out_of_bounds": bool(
                    candidate_oob[condition_index, sample_index]
                ),
            })

    result = {
        "schema_version": "trajflow-best-of-k-evaluation-v1",
        "diagnostic_only": True,
        "metric_semantics": {
            "primary_reference": "the paired raw test trajectory for each condition",
            "best_of_k": (
                "minimum distance among K stochastic draws; increasing K can only "
                "improve this oracle statistic and it is not a training or selection gate"
            ),
            "dtw": "exact accumulated Euclidean point cost",
            "continuous_frechet": "Alt-Godau continuous curve distance",
            "diversity": (
                "all unordered candidate pairs under each condition; each pair is "
                "the mean Euclidean distance between corresponding trajectory points"
            ),
            "straight_line_control": (
                "one deterministic O-to-D interpolation per condition; repeating it "
                "K times does not change its paired distance"
            ),
            "density_histogram_oob_policy": (
                "out-of-grid candidate points are excluded from density JSD and OOB "
                "is reported independently"
            ),
            "density_grid_bins": list(normalized_density_bins),
        },
        "condition_count": condition_count,
        "samples_per_condition": samples_per_condition,
        "array_shapes": {
            "generated_candidates": list(generated.shape),
            "paired_raw_reference": list(reference.shape),
            "od_straight_line_control": list(straight_line.shape),
        },
        "best_of_k_vs_paired_raw_test": {
            "dtw_km": _distribution_summary(best_dtw_km),
            "continuous_frechet_km": _distribution_summary(best_frechet_km),
        },
        "all_candidates_vs_paired_raw_test": {
            "dtw_km": _distribution_summary(dtw_km.ravel()),
            "continuous_frechet_km": _distribution_summary(frechet_km.ravel()),
        },
        "candidate_diversity": diversity,
        "pooled_candidate_density_js_divergence": density_jensen_shannon(
            generated.reshape(-1, trajectory_length, 2),
            reference,
            width=width,
            height=height,
            coordinate_min=coordinate_min,
            bins=normalized_density_bins,
        ),
        "out_of_bounds": {
            "pooled_candidates": pooled_bounds,
            "condition_with_any_oob_candidate_count": int(
                np.any(candidate_oob, axis=1).sum()
            ),
            "condition_with_any_oob_candidate_rate": float(
                np.any(candidate_oob, axis=1).mean()
            ),
            "condition_with_all_candidates_in_bounds_count": int(
                np.all(~candidate_oob, axis=1).sum()
            ),
            "condition_with_all_candidates_in_bounds_rate": float(
                np.all(~candidate_oob, axis=1).mean()
            ),
            "best_dtw_candidate_oob_count": int(
                candidate_oob[row_indices, best_dtw_indices].sum()
            ),
            "best_dtw_candidate_oob_rate": float(
                candidate_oob[row_indices, best_dtw_indices].mean()
            ),
            "best_continuous_frechet_candidate_oob_count": int(
                candidate_oob[row_indices, best_frechet_indices].sum()
            ),
            "best_continuous_frechet_candidate_oob_rate": float(
                candidate_oob[row_indices, best_frechet_indices].mean()
            ),
        },
        "od_straight_line_vs_paired_raw_test": straight_metrics,
        "best_of_k_improvement_over_straight_line": {
            "positive_straight_minus_best_of_k_means_oracle_is_closer": True,
            "dtw_median_km_straight_minus_best_of_k": float(
                straight_metrics["dtw_median_km"] - np.median(best_dtw_km)
            ),
            "continuous_frechet_median_km_straight_minus_best_of_k": float(
                straight_metrics["continuous_frechet_median_km"]
                - np.median(best_frechet_km)
            ),
        },
        "per_condition": per_condition,
        "candidate_metrics": candidate_rows,
        "grid_cell_size_km": cell_size_km,
        "metric_workers": workers,
    }
    return result
