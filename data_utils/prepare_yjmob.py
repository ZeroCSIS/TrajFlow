"""Convert the public YJMob100K Dataset1 file to TrajFlow's processed schema.

The converter deliberately treats one ``(uid, day)`` group as one sample.  The
uid and day are written only to ``manifest.csv``; neither value is exposed to
the model conditions.  Dataset1 is ordered by ``uid, d, t`` in the official
release, so the implementation streams groups without loading the raw CSV into
memory.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import pickle
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np

OFFICIAL_DATASET1_MD5 = "3781f6f03a118b5f639bdb4f94dcfdb8"
EXPECTED_COLUMNS = ("uid", "d", "t", "x", "y")
CONDITION_NAMES = ("total_dis", "total_time", "total_len", "avg_dis", "avg_speed")


@dataclass(frozen=True)
class Observation:
    uid: int
    day: int
    timeslot: int
    x: int
    y: int


@dataclass(frozen=True)
class Sample:
    uid: int
    day: int
    split: str
    timeslots: tuple[int, ...]
    points: np.ndarray


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_csv(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return path.open(mode="r", encoding="utf-8", newline="")


def iter_observations(path: Path) -> Iterator[Observation]:
    with _open_csv(path) as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or tuple(reader.fieldnames) != EXPECTED_COLUMNS:
            raise ValueError(
                f"Expected columns {EXPECTED_COLUMNS}, got {tuple(reader.fieldnames or ())}"
            )
        previous_key: tuple[int, int, int] | None = None
        for line_number, row in enumerate(reader, start=2):
            try:
                observation = Observation(
                    uid=int(row["uid"]),
                    day=int(row["d"]),
                    timeslot=int(row["t"]),
                    x=int(row["x"]),
                    y=int(row["y"]),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid integer value on line {line_number}") from exc

            if observation.uid < 0 or observation.day < 0:
                raise ValueError(f"Negative uid/day on line {line_number}")
            if not 0 <= observation.timeslot <= 47:
                raise ValueError(f"timeslot outside [0, 47] on line {line_number}")
            if not (1 <= observation.x <= 200 and 1 <= observation.y <= 200):
                raise ValueError(f"grid coordinate outside [1, 200] on line {line_number}")

            key = (observation.uid, observation.day, observation.timeslot)
            if previous_key is not None and key < previous_key:
                raise ValueError(
                    "Input must be ordered by uid, d, t as in the official Dataset1 release; "
                    f"order decreased on line {line_number}"
                )
            previous_key = key
            yield observation


def assign_user_split(
    uid: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> str:
    token = hashlib.sha256(f"{seed}:{uid}".encode("ascii")).digest()[:8]
    unit_value = int.from_bytes(token, byteorder="big") / float(2**64)
    if unit_value < train_ratio:
        return "train"
    if unit_value < train_ratio + val_ratio:
        return "val"
    return "test"


def group_samples(
    observations: Iterable[Observation],
    *,
    max_users: int | None,
    excluded_days: set[int],
    min_observations: int,
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[Sample], dict[str, int]]:
    samples: list[Sample] = []
    stats = {
        "rows_read": 0,
        "excluded_rows": 0,
        "duplicate_timeslot_rows": 0,
        "short_user_days": 0,
        "users": 0,
    }
    current_uid: int | None = None
    current_day: int | None = None
    current_points: dict[int, tuple[int, int]] = {}

    def flush() -> None:
        if current_uid is None or current_day is None:
            return
        if current_day in excluded_days:
            return
        if len(current_points) < min_observations:
            stats["short_user_days"] += 1
            return
        timeslots = tuple(sorted(current_points))
        points = np.asarray([current_points[t] for t in timeslots], dtype=np.float32)
        split = assign_user_split(
            current_uid,
            seed=split_seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        samples.append(Sample(current_uid, current_day, split, timeslots, points))

    for observation in observations:
        if current_uid is None:
            current_uid = observation.uid
            current_day = observation.day
            stats["users"] = 1
        elif observation.uid != current_uid:
            flush()
            if max_users is not None and stats["users"] >= max_users:
                break
            current_uid = observation.uid
            current_day = observation.day
            current_points = {}
            stats["users"] += 1
        elif observation.day != current_day:
            flush()
            current_day = observation.day
            current_points = {}

        stats["rows_read"] += 1
        if observation.day in excluded_days:
            stats["excluded_rows"] += 1
            continue
        if observation.timeslot in current_points:
            stats["duplicate_timeslot_rows"] += 1
            continue
        current_points[observation.timeslot] = (observation.x, observation.y)

    else:
        flush()

    if not samples:
        raise ValueError("No user-day samples remained after filtering")
    if not any(sample.split == "train" for sample in samples):
        raise ValueError("The selected users produced no training samples")
    return samples, stats


def resample_by_timeslot(sample: Sample, target_length: int) -> np.ndarray:
    source_t = np.asarray(sample.timeslots, dtype=np.float64)
    target_t = np.linspace(source_t[0], source_t[-1], target_length)
    output = np.empty((target_length, 2), dtype=np.float32)
    for axis in range(2):
        output[:, axis] = np.interp(target_t, source_t, sample.points[:, axis])
    return output


def grid_cell_id(x: float, y: float, height: int = 200) -> int:
    return (int(x) - 1) * height + (int(y) - 1)


def raw_condition(sample: Sample, cell_size_m: float) -> np.ndarray:
    deltas = np.diff(sample.points.astype(np.float64), axis=0)
    total_distance = float(np.linalg.norm(deltas, axis=1).sum() * cell_size_m)
    total_time = float(max(sample.timeslots[-1] - sample.timeslots[0], 1) * 30 * 60)
    observation_count = float(len(sample.timeslots))
    # Match TrajFlow's existing condition definition: total distance divided by
    # the number of observed points (``total_len``), not the number of legs.
    average_distance = total_distance / observation_count
    average_speed = total_distance / total_time
    departure_bucket = float(sample.timeslots[0] * 6)
    origin = float(grid_cell_id(*sample.points[0]))
    destination = float(grid_cell_id(*sample.points[-1]))
    return np.asarray(
        [
            departure_bucket,
            total_distance,
            total_time,
            observation_count,
            average_distance,
            average_speed,
            origin,
            destination,
            0.0,
        ],
        dtype=np.float64,
    )


def _safe_train_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds < 1e-8] = 1.0
    return means, stds


def _write_mean_std(path: Path, names: Iterable[str], means: np.ndarray, stds: np.ndarray) -> None:
    lines: list[str] = []
    for name, mean, std in zip(names, means, stds):
        lines.extend((f"{name}_mean: {float(mean):.10f}\n", f"{name}_std: {float(std):.10f}\n"))
    path.write_text("".join(lines), encoding="utf-8")


def prepare_dataset(
    input_path: Path,
    output_dir: Path,
    *,
    max_users: int | None = 1000,
    trajectory_length: int = 120,
    excluded_days: set[int] | None = None,
    min_observations: int = 2,
    split_seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    cell_size_m: float = 500.0,
    expected_md5: str | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Convert a bounded prefix of Dataset1 and return its summary."""
    excluded_days = {27} if excluded_days is None else set(excluded_days)
    if max_users is not None and max_users <= 0:
        raise ValueError("max_users must be positive or omitted")
    if trajectory_length < 2 or min_observations < 2:
        raise ValueError("trajectory_length and min_observations must both be at least 2")
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("split ratios must satisfy train > 0, val >= 0, train + val < 1")
    if expected_md5 is not None:
        actual_md5 = md5sum(input_path)
        if actual_md5.lower() != expected_md5.lower():
            raise ValueError(f"MD5 mismatch: expected {expected_md5}, got {actual_md5}")
    else:
        actual_md5 = None

    managed_files = (
        "traj_segments.pkl",
        "conditions.pkl",
        "mesh_mapping_dict.pkl",
        "traj_mean_std.txt",
        "conditions_mean_std.txt",
        "grid_meta.json",
        "manifest.csv",
        "split_indices.npz",
        "dataset_summary.json",
    )
    existing = [name for name in managed_files if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output directory already contains managed files: {', '.join(existing)}; use --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, scan_stats = group_samples(
        iter_observations(input_path),
        max_users=max_users,
        excluded_days=excluded_days,
        min_observations=min_observations,
        split_seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    trajectories = np.stack(
        [resample_by_timeslot(sample, trajectory_length) for sample in samples]
    ).astype(np.float32)
    raw_conditions = np.stack([raw_condition(sample, cell_size_m) for sample in samples])
    split_names = np.asarray([sample.split for sample in samples])
    train_mask = split_names == "train"
    cond_means, cond_stds = _safe_train_statistics(raw_conditions[train_mask, 1:6])
    conditions = raw_conditions.copy()
    conditions[:, 1:6] = (conditions[:, 1:6] - cond_means) / cond_stds
    conditions = conditions.astype(np.float32)

    train_trajectories = trajectories[train_mask]
    traj_means = train_trajectories.mean(axis=(0, 1))
    traj_stds = train_trajectories.std(axis=(0, 1))
    traj_stds[traj_stds < 1e-8] = 1.0

    with (output_dir / "traj_segments.pkl").open("wb") as stream:
        pickle.dump(trajectories, stream, pickle.HIGHEST_PROTOCOL)
    with (output_dir / "conditions.pkl").open("wb") as stream:
        pickle.dump(conditions, stream, pickle.HIGHEST_PROTOCOL)
    with (output_dir / "mesh_mapping_dict.pkl").open("wb") as stream:
        pickle.dump({cell: cell for cell in range(200 * 200)}, stream, pickle.HIGHEST_PROTOCOL)

    _write_mean_std(output_dir / "traj_mean_std.txt", ("x", "y"), traj_means, traj_stds)
    _write_mean_std(
        output_dir / "conditions_mean_std.txt",
        CONDITION_NAMES,
        cond_means,
        cond_stds,
    )

    split_indices = {
        name: np.flatnonzero(split_names == name).astype(np.int64)
        for name in ("train", "val", "test")
    }
    split_users = {
        name: {sample.uid for sample in samples if sample.split == name}
        for name in ("train", "val", "test")
    }
    split_user_overlaps = {
        "train_val": len(split_users["train"] & split_users["val"]),
        "train_test": len(split_users["train"] & split_users["test"]),
        "val_test": len(split_users["val"] & split_users["test"]),
    }
    np.savez(output_dir / "split_indices.npz", **split_indices)

    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "sample_index",
                "uid",
                "day",
                "split",
                "observation_count",
                "first_timeslot",
                "last_timeslot",
            ),
        )
        writer.writeheader()
        for index, sample in enumerate(samples):
            writer.writerow(
                {
                    "sample_index": index,
                    "uid": sample.uid,
                    "day": sample.day,
                    "split": sample.split,
                    "observation_count": len(sample.timeslots),
                    "first_timeslot": sample.timeslots[0],
                    "last_timeslot": sample.timeslots[-1],
                }
            )

    grid_metadata = {
        "encoding": "cartesian_grid",
        "axis_order": ["x", "y"],
        "width": 200,
        "height": 200,
        "coordinate_min": 1,
        "coordinates_are_raw": True,
        "cell_size_m": cell_size_m,
        "cell_id_formula": "(x - 1) * height + (y - 1)",
        "condition_ids_are_mapped": True,
        "fixed_location_dim": 200 * 200,
    }
    (output_dir / "grid_meta.json").write_text(
        json.dumps(grid_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary: dict[str, object] = {
        "source": "YJMob100K v3 Dataset1",
        "source_file": input_path.name,
        "source_md5": actual_md5,
        "sample_unit": "user-day",
        "identity_policy": "uid and day exist only in manifest.csv",
        "excluded_days": sorted(excluded_days),
        "max_users": max_users,
        "trajectory_length": trajectory_length,
        "min_observations": min_observations,
        "split_seed": split_seed,
        "split_policy": "SHA-256(seed:uid), user-disjoint",
        "split_counts": {name: len(indices) for name, indices in split_indices.items()},
        "split_user_counts": {name: len(users) for name, users in split_users.items()},
        "split_user_overlap_counts": split_user_overlaps,
        "sample_count": len(samples),
        "scan_stats": scan_stats,
        "condition_statistics_source": "train split only",
        "license": "CC BY 4.0; follow the YJMob100K ethical-use restrictions",
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Dataset1 CSV or CSV.GZ")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--trajectory-length", type=int, default=120)
    parser.add_argument("--exclude-day", type=int, action="append", default=None)
    parser.add_argument("--min-observations", type=int, default=2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--cell-size-m", type=float, default=500.0)
    parser.add_argument(
        "--expected-md5",
        default=OFFICIAL_DATASET1_MD5,
        help="Set to an empty string to skip checksum verification",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_dataset(
        args.input,
        args.output_dir,
        max_users=args.max_users,
        trajectory_length=args.trajectory_length,
        excluded_days=set(args.exclude_day or [27]),
        min_observations=args.min_observations,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        cell_size_m=args.cell_size_m,
        expected_md5=args.expected_md5 or None,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
