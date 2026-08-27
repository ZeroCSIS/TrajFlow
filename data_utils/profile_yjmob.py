"""Create a compact quality profile for a prepared YJMob dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def distribution_summary(values: np.ndarray) -> dict[str, float | int]:
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


def distinct_control_point_counts(
    trajectories: np.ndarray,
    *,
    decimals: int = 6,
) -> np.ndarray:
    trajectories = np.asarray(trajectories)
    if trajectories.ndim != 3 or trajectories.shape[-1] != 2:
        raise ValueError("RDP trajectories must have shape (N, M, 2)")
    return np.asarray([
        len(np.unique(np.round(trajectory, decimals=decimals), axis=0))
        for trajectory in trajectories
    ], dtype=np.int64)


def _manifest_profile(rows: list[dict[str, str]]) -> dict[str, object]:
    result: dict[str, object] = {}
    split_names = ["all", "train", "val", "test"]
    for split in split_names:
        selected = rows if split == "all" else [
            row for row in rows if row["split"] == split
        ]
        if not selected:
            continue
        observation_counts = np.asarray([
            int(row["observation_count"]) for row in selected
        ])
        span_slots = np.asarray([
            int(row["last_timeslot"]) - int(row["first_timeslot"])
            for row in selected
        ])
        result[split] = {
            "sample_count": len(selected),
            "observation_count": distribution_summary(observation_counts),
            "observed_span_half_hour_slots": distribution_summary(span_slots),
            "observed_span_hours": distribution_summary(span_slots / 2.0),
            "two_observation_sample_count": int((observation_counts == 2).sum()),
            "two_observation_sample_rate": float((observation_counts == 2).mean()),
        }
    return result


def build_dataset_profile(
    processed_dir: Path,
    *,
    region: str,
    parameterization: str,
    control_points: int,
) -> dict[str, object]:
    processed_dir = Path(processed_dir)
    manifest_path = processed_dir / "manifest.csv"
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Dataset manifest is empty: {manifest_path}")

    rdp_profile: dict[str, object] = {}
    for split in ("train", "val", "test"):
        cache_path = processed_dir / (
            f"processed_coeffs_{region}_{parameterization}_{control_points}_{split}.npy"
        )
        if not cache_path.exists():
            continue
        trajectories = np.load(cache_path, mmap_mode="r")
        distinct_counts = distinct_control_point_counts(trajectories)
        unique, counts = np.unique(distinct_counts, return_counts=True)
        rdp_profile[split] = {
            "cache_file": cache_path.name,
            "trajectory_count": int(len(trajectories)),
            "configured_control_points": int(control_points),
            "distinct_control_points": distribution_summary(distinct_counts),
            "distinct_control_point_histogram": {
                str(int(value)): int(count)
                for value, count in zip(unique, counts, strict=True)
            },
        }

    dataset_summary_path = processed_dir / "dataset_summary.json"
    dataset_summary = None
    if dataset_summary_path.exists():
        dataset_summary = json.loads(dataset_summary_path.read_text(encoding="utf-8"))
    return {
        "schema_version": "yjmob-dataset-profile-v1",
        "processed_dir": str(processed_dir.resolve()),
        "dataset_summary": dataset_summary,
        "manifest_statistics": _manifest_profile(rows),
        "rdp_control_point_statistics": rdp_profile,
        "notes": {
            "observed_span": "last observed half-hour slot minus first observed slot",
            "distinct_control_points": (
                "unique RDP coordinates after rounding to 6 decimals; this detects "
                "duplicates, not geometric collinearity"
            ),
        },
    }


def write_dataset_profile(
    output_path: Path,
    processed_dir: Path,
    *,
    region: str,
    parameterization: str,
    control_points: int,
) -> dict[str, object]:
    profile = build_dataset_profile(
        processed_dir,
        region=region,
        parameterization=parameterization,
        control_points=control_points,
    )
    output_path = Path(output_path)
    output_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--parameterization", default="rdp_k")
    parser.add_argument("--control-points", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_dataset_profile(
        args.output,
        args.processed_dir,
        region=args.region,
        parameterization=args.parameterization,
        control_points=args.control_points,
    )


if __name__ == "__main__":
    main()
