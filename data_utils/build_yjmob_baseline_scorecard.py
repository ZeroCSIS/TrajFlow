"""Freeze the final YJMob baseline scorecard from immutable best-of-K artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.metrics import (
    continuous_frechet,
    dynamic_time_warping,
    origin_destination_straight_lines,
)


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: np.ndarray) -> dict[str, float | int] | None:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return None
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def _control_distance_pair(
    pair: tuple[np.ndarray, np.ndarray],
) -> tuple[float, float]:
    line, reference = pair
    return (
        dynamic_time_warping(line, reference),
        continuous_frechet(line, reference),
    )


def control_distances(
    reference: np.ndarray,
    *,
    cell_size_km: float,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    lines = origin_destination_straight_lines(reference)
    pairs = list(zip(lines, reference, strict=True))
    if workers <= 1:
        values = list(map(_control_distance_pair, pairs))
    else:
        chunksize = max(1, len(pairs) // (workers * 8))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            values = list(
                executor.map(_control_distance_pair, pairs, chunksize=chunksize)
            )
    distances = np.asarray(values, dtype=np.float64) * float(cell_size_km)
    return distances[:, 0], distances[:, 1]


def _candidate_metric_matrices(
    metrics: dict[str, object],
    condition_count: int,
    samples_per_condition: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dtw = np.full((condition_count, samples_per_condition), np.nan)
    frechet = np.full_like(dtw, np.nan)
    oob = np.zeros((condition_count, samples_per_condition), dtype=bool)
    seen = np.zeros((condition_count, samples_per_condition), dtype=bool)
    for row in metrics["candidate_metrics"]:
        condition_index = int(row["condition_index"])
        sample_index = int(row["sample_index"])
        if not (0 <= condition_index < condition_count):
            raise ValueError("candidate condition index is out of range")
        if not (0 <= sample_index < samples_per_condition):
            raise ValueError("candidate sample index is out of range")
        if seen[condition_index, sample_index]:
            raise ValueError("candidate metric row is duplicated")
        seen[condition_index, sample_index] = True
        dtw[condition_index, sample_index] = float(row["dtw_km"])
        frechet[condition_index, sample_index] = float(
            row["continuous_frechet_km"]
        )
        oob[condition_index, sample_index] = bool(row["out_of_bounds"])
    if not seen.all() or not np.isfinite(dtw).all() or not np.isfinite(frechet).all():
        raise ValueError("candidate metrics do not form a complete N-by-K matrix")
    return dtw, frechet, oob


def _assert_close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(actual, expected, rtol=1e-9, atol=1e-9):
        raise ValueError(f"{name} failed reconciliation: {actual} != {expected}")


def build_scorecard(
    metrics: dict[str, object],
    manifest: dict[str, object],
    generated: np.ndarray,
    reference: np.ndarray,
    selected_test_local_indices: np.ndarray,
    *,
    workers: int = 1,
    input_hashes: dict[str, str] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    generated = np.asarray(generated, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    selected_test_local_indices = np.asarray(
        selected_test_local_indices,
        dtype=np.int64,
    )
    condition_count = int(metrics["condition_count"])
    samples_per_condition = int(metrics["samples_per_condition"])
    if int(manifest["samples_per_condition"]) != samples_per_condition:
        raise ValueError("metric and generation-manifest K differ")
    if (
        reference.ndim != 3
        or reference.shape[0] != condition_count
        or reference.shape[-1] != 2
        or reference.shape[1] < 2
    ):
        raise ValueError("reference shape does not match the metric condition count")
    expected_generated_shape = (
        condition_count,
        samples_per_condition,
        reference.shape[1],
        2,
    )
    if generated.shape != expected_generated_shape:
        raise ValueError(
            f"candidate shape {generated.shape} != {expected_generated_shape}"
        )
    if selected_test_local_indices.shape != (condition_count,):
        raise ValueError("selected index count does not match the conditions")
    manifest_local_indices = np.asarray(
        manifest["selected_test_local_indices"],
        dtype=np.int64,
    )
    if not np.array_equal(selected_test_local_indices, manifest_local_indices):
        raise ValueError("candidate and generation-manifest indices differ")
    source_indices = np.asarray(
        manifest["selected_source_sample_indices"],
        dtype=np.int64,
    )
    if source_indices.shape != (condition_count,):
        raise ValueError("source index count does not match the conditions")

    dtw, frechet, oob = _candidate_metric_matrices(
        metrics,
        condition_count,
        samples_per_condition,
    )
    cell_size_km = float(metrics["grid_cell_size_km"])
    line_dtw, line_frechet = control_distances(
        reference,
        cell_size_km=cell_size_km,
        workers=int(workers),
    )
    _assert_close(
        float(np.median(line_dtw)),
        float(metrics["od_straight_line_vs_paired_raw_test"]["dtw_median_km"]),
        "straight-line DTW median",
    )
    _assert_close(
        float(np.median(line_frechet)),
        float(
            metrics["od_straight_line_vs_paired_raw_test"][
                "continuous_frechet_median_km"
            ]
        ),
        "straight-line Frechet median",
    )

    same_od = np.all(
        np.isclose(reference[:, 0], reference[:, -1], rtol=0.0, atol=1e-12),
        axis=1,
    )
    candidate_excursion = np.linalg.norm(
        generated - generated[:, :, :1, :],
        axis=3,
    ).max(axis=2)
    candidate_constant = candidate_excursion <= 1e-12
    candidate_origin_deviation = np.linalg.norm(
        generated - reference[:, None, :1, :],
        axis=3,
    ).max(axis=2)
    candidate_constant_at_origin = candidate_origin_deviation <= 1e-12
    all_candidates_constant = candidate_constant.all(axis=1)
    candidate_ranges = np.ptp(generated, axis=1)
    all_candidates_identical = np.max(np.abs(candidate_ranges), axis=(1, 2)) <= 1e-12
    reference_excursion = np.linalg.norm(
        reference - reference[:, :1, :],
        axis=2,
    ).max(axis=1) * cell_size_km
    reference_path_length = np.linalg.norm(
        np.diff(reference, axis=1),
        axis=2,
    ).sum(axis=1) * cell_size_km
    diversity = np.full(condition_count, np.nan, dtype=np.float64)
    for row in metrics["per_condition"]:
        condition_index = int(row["condition_index"])
        if not 0 <= condition_index < condition_count:
            raise ValueError("per-condition metric index is out of range")
        if np.isfinite(diversity[condition_index]):
            raise ValueError("per-condition metric row is duplicated")
        diversity[condition_index] = float(
            row["mean_pairwise_aligned_point_distance_km"] or 0.0
        )
    if not np.isfinite(diversity).all():
        raise ValueError("per-condition metrics are incomplete")

    prefix_rows = []
    for k in (1, 2, 5, 10, 20):
        if k > samples_per_condition:
            continue
        best_dtw = dtw[:, :k].min(axis=1)
        best_frechet = frechet[:, :k].min(axis=1)
        both_win = (best_dtw < line_dtw) & (best_frechet < line_frechet)
        prefix_oob = oob[:, :k]
        prefix_rows.append({
            "k": k,
            "best_dtw_median_km": float(np.median(best_dtw)),
            "best_continuous_frechet_median_km": float(
                np.median(best_frechet)
            ),
            "candidate_oob_trajectory_count": int(prefix_oob.sum()),
            "candidate_oob_trajectory_rate": float(prefix_oob.mean()),
            "condition_with_any_oob_candidate_count": int(
                np.any(prefix_oob, axis=1).sum()
            ),
            "condition_with_any_oob_candidate_rate": float(
                np.any(prefix_oob, axis=1).mean()
            ),
            "both_metrics_better_than_line_count": int(both_win.sum()),
            "both_metrics_better_than_line_rate": float(both_win.mean()),
            "non_same_od_both_metrics_better_than_line_count": int(
                (both_win & ~same_od).sum()
            ),
            "non_same_od_both_metrics_better_than_line_rate": (
                float(both_win[~same_od].mean()) if (~same_od).any() else None
            ),
        })

    final_best_dtw = dtw.min(axis=1)
    final_best_frechet = frechet.min(axis=1)
    final_both_win = (
        (final_best_dtw < line_dtw) & (final_best_frechet < line_frechet)
    )
    _assert_close(
        float(np.median(final_best_dtw)),
        float(metrics["best_of_k_vs_paired_raw_test"]["dtw_km"]["p50"]),
        "best-of-K DTW median",
    )
    _assert_close(
        float(np.median(final_best_frechet)),
        float(
            metrics["best_of_k_vs_paired_raw_test"][
                "continuous_frechet_km"
            ]["p50"]
        ),
        "best-of-K Frechet median",
    )
    expected_oob_count = int(
        metrics["out_of_bounds"]["pooled_candidates"][
            "out_of_bounds_trajectory_count"
        ]
    )
    if int(oob.sum()) != expected_oob_count:
        raise ValueError("candidate OOB count failed reconciliation")

    oob_counts = oob.sum(axis=1)
    diversity_oob_correlation = (
        float(np.corrcoef(diversity, oob_counts)[0, 1])
        if np.std(diversity) > 0 and np.std(oob_counts) > 0
        else None
    )

    od_rows: list[dict[str, object]] = []
    for condition_index in np.flatnonzero(same_od):
        od_rows.append({
            "condition_index": int(condition_index),
            "test_local_index": int(selected_test_local_indices[condition_index]),
            "source_sample_index": int(source_indices[condition_index]),
            "origin_x": float(reference[condition_index, 0, 0]),
            "origin_y": float(reference[condition_index, 0, 1]),
            "reference_max_excursion_km": float(
                reference_excursion[condition_index]
            ),
            "reference_path_length_km": float(
                reference_path_length[condition_index]
            ),
            "constant_candidate_count": int(
                candidate_constant[condition_index].sum()
            ),
            "all_candidates_constant": bool(
                all_candidates_constant[condition_index]
            ),
            "constant_at_origin_candidate_count": int(
                candidate_constant_at_origin[condition_index].sum()
            ),
            "all_candidates_constant_at_origin": bool(
                candidate_constant_at_origin[condition_index].all()
            ),
            "all_candidates_identical": bool(
                all_candidates_identical[condition_index]
            ),
            "mean_pairwise_aligned_point_distance_km": float(
                diversity[condition_index]
            ),
            "candidate_oob_count": int(oob_counts[condition_index]),
            "best_dtw_km": float(final_best_dtw[condition_index]),
            "line_dtw_km": float(line_dtw[condition_index]),
            "best_continuous_frechet_km": float(
                final_best_frechet[condition_index]
            ),
            "line_continuous_frechet_km": float(
                line_frechet[condition_index]
            ),
        })

    best_frechet_indices = np.argmin(frechet, axis=1)
    best_frechet_oob = oob[np.arange(condition_count), best_frechet_indices]
    scorecard = {
        "schema_version": "trajflow-yjmob-baseline-closeout-v1",
        "diagnostic_only": True,
        "model_or_representation_changed": False,
        "input_provenance": {
            "git_commit": manifest["git_commit"],
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "seed": manifest["seed"],
            "sampling_steps": manifest["sampling_steps"],
            "condition_count": condition_count,
            "samples_per_condition": samples_per_condition,
            "input_sha256": input_hashes or {},
        },
        "interpretation": {
            "best_of_k": (
                "oracle coverage diagnostic only; increasing K improves the minimum "
                "by construction and is not a training or model-selection gate"
            ),
            "oob": (
                "candidate-pool OOB is reported separately so broad invalid search "
                "cannot be mistaken for useful coverage"
            ),
            "old_representation_status": (
                "accepted as a reproducible baseline and frozen for further tuning"
            ),
        },
        "final_k_metrics": {
            "best_dtw_km": _distribution(final_best_dtw),
            "best_continuous_frechet_km": _distribution(final_best_frechet),
            "od_straight_line_dtw_km": _distribution(line_dtw),
            "od_straight_line_continuous_frechet_km": _distribution(
                line_frechet
            ),
            "both_metrics_better_than_line_count": int(final_both_win.sum()),
            "both_metrics_better_than_line_rate": float(final_both_win.mean()),
            "non_same_od_both_metrics_better_than_line_count": int(
                (final_both_win & ~same_od).sum()
            ),
            "non_same_od_both_metrics_better_than_line_rate": (
                float(final_both_win[~same_od].mean()) if (~same_od).any() else None
            ),
        },
        "k_prefix_diagnostics": prefix_rows,
        "candidate_pool_quality": {
            "candidate_oob_trajectory_count": int(oob.sum()),
            "candidate_oob_trajectory_rate": float(oob.mean()),
            "condition_with_any_oob_candidate_count": int(
                np.any(oob, axis=1).sum()
            ),
            "condition_with_any_oob_candidate_rate": float(
                np.any(oob, axis=1).mean()
            ),
            "best_frechet_candidate_oob_count": int(best_frechet_oob.sum()),
            "best_frechet_candidate_oob_rate": float(best_frechet_oob.mean()),
            "condition_mean_pairwise_distance_km": _distribution(diversity),
            "diversity_vs_oob_count_pearson": diversity_oob_correlation,
            "maximum_candidate_frechet_km": float(frechet.max()),
            "maximum_absolute_candidate_grid_coordinate": float(
                np.abs(generated).max()
            ),
        },
        "same_origin_destination_acceptance_set": {
            "condition_count": int(same_od.sum()),
            "condition_rate": float(same_od.mean()),
            "nonconstant_reference_condition_count": int(
                (same_od & (reference_excursion > 1e-12)).sum()
            ),
            "all_candidates_constant_condition_count": int(
                (same_od & all_candidates_constant).sum()
            ),
            "all_candidates_constant_at_origin_condition_count": int(
                (
                    same_od
                    & np.all(candidate_constant_at_origin, axis=1)
                ).sum()
            ),
            "all_candidates_identical_condition_count": int(
                (same_od & all_candidates_identical).sum()
            ),
            "zero_diversity_condition_count": int(
                (same_od & (diversity <= 1e-12)).sum()
            ),
            "zero_diversity_conditions_outside_set": int(
                ((~same_od) & (diversity <= 1e-12)).sum()
            ),
            "current_baseline_expected_failure_reproduced": bool(
                same_od.any()
                and np.all(candidate_constant_at_origin[same_od])
                and np.all(diversity[same_od] <= 1e-12)
            ),
            "future_use": (
                "evaluate every listed condition under any new representation and "
                "report nonconstant-candidate coverage, diversity, OOB, and paired "
                "distances; do not hide these cases inside an aggregate median"
            ),
            "hard_gate_policy": (
                "this set is a structural acceptance diagnostic, not a training gate; "
                "numeric thresholds require the future representation design contract"
            ),
        },
        "reconciliation": {
            "candidate_metric_matrix_complete": True,
            "candidate_indices_equal_generation_manifest": True,
            "aggregate_medians_match_frozen_metrics": True,
            "candidate_oob_count_matches_frozen_metrics": True,
        },
    }
    return scorecard, od_rows


def render_markdown(scorecard: dict[str, object]) -> str:
    final = scorecard["final_k_metrics"]
    quality = scorecard["candidate_pool_quality"]
    od = scorecard["same_origin_destination_acceptance_set"]
    prefix = scorecard["k_prefix_diagnostics"]
    lines = [
        "## YJMob100K TrajFlow baseline 最终成绩单",
        "",
        "本成绩单冻结旧表示上的结果：不再继续调参；best-of-K 只作为覆盖诊断，不是训练或选模门槛。",
        "",
        "| 指标 | 最终值 |",
        "|---|---:|",
        f"| best-of-20 DTW 中位数 | {final['best_dtw_km']['p50']:.3f} km |",
        f"| best-of-20 连续 Fréchet 中位数 | {final['best_continuous_frechet_km']['p50']:.3f} km |",
        f"| O→D 直线 DTW 中位数 | {final['od_straight_line_dtw_km']['p50']:.3f} km |",
        f"| O→D 直线连续 Fréchet 中位数 | {final['od_straight_line_continuous_frechet_km']['p50']:.3f} km |",
        f"| 两项同时优于直线 | {final['both_metrics_better_than_line_rate']:.1%} |",
        f"| 排除 O=D 后两项同时优于直线 | {final['non_same_od_both_metrics_better_than_line_rate']:.1%} |",
        f"| 候选轨迹 OOB | {quality['candidate_oob_trajectory_count']} / {scorecard['input_provenance']['condition_count'] * scorecard['input_provenance']['samples_per_condition']} ({quality['candidate_oob_trajectory_rate']:.2%}) |",
        f"| 至少含一个 OOB 候选的条件 | {quality['condition_with_any_oob_candidate_rate']:.1%} |",
        "",
        "### K 前缀诊断",
        "",
        "| K | best DTW 中位数 (km) | best Fréchet 中位数 (km) | 候选 OOB 率 | 条件含 OOB 率 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in prefix:
        lines.append(
            f"| {row['k']} | {row['best_dtw_median_km']:.3f} | "
            f"{row['best_continuous_frechet_median_km']:.3f} | "
            f"{row['candidate_oob_trajectory_rate']:.2%} | "
            f"{row['condition_with_any_oob_candidate_rate']:.1%} |"
        )
    lines.extend([
        "",
        "### O=D 结构性反例",
        "",
        f"- 验收集含 {od['condition_count']} 个条件，占 {od['condition_rate']:.1%}。",
        f"- 其中 {od['nonconstant_reference_condition_count']} 个真实参考都离开过起点，但当前表示在 {od['all_candidates_constant_condition_count']} 个条件上把全部候选压成常数轨迹。",
        f"- 零多样性条件在验收集内为 {od['zero_diversity_condition_count']} 个，在集合外为 {od['zero_diversity_conditions_outside_set']} 个。",
        "- 后续新表示必须单独报告这组条件的非恒定候选覆盖、多样性、越界和配对距离，不能只看总体中位数。",
        "",
        "### 收口边界",
        "",
        "旧表示可作为可重复 baseline 接受并冻结；O=D 退化、越界和 oracle 偏乐观均作为已知限制保留。两层新表示的定义与训练不属于本成绩单。",
        "",
    ])
    return "\n".join(lines)


def write_od_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("O=D acceptance set is empty")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    metrics_sha256 = sha256sum(args.metrics)
    if metrics_sha256 != manifest["metrics_sha256"]:
        raise ValueError("metrics SHA-256 does not match the generation manifest")
    with np.load(args.candidates) as arrays:
        generated = arrays["generated_candidates"]
        reference = arrays["paired_raw_reference"]
        selected = arrays["selected_test_local_indices"]
    scorecard, od_rows = build_scorecard(
        metrics,
        manifest,
        generated,
        reference,
        selected,
        workers=args.workers,
        input_hashes={
            "metrics": metrics_sha256,
            "manifest": sha256sum(args.manifest),
            "candidates": sha256sum(args.candidates),
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = args.output_dir / "baseline_closeout_scorecard.json"
    scorecard_path.write_text(
        json.dumps(scorecard, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "baseline_closeout_scorecard.md").write_text(
        render_markdown(scorecard),
        encoding="utf-8",
    )
    write_od_rows(args.output_dir / "same_od_acceptance_set.csv", od_rows)
    print(json.dumps({
        "scorecard": str(scorecard_path),
        "same_od_condition_count": len(od_rows),
        "reconciliation": scorecard["reconciliation"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
