"""Generate a fully synthetic TrajFlow toy dataset.

The generated samples are not derived from BW, DiDi, or any other real
trajectory dataset. They only demonstrate the processed file format expected by
the released training and generation code.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np


BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def geohash_encode(lat: float, lon: float, precision: int = 6) -> str:
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    geohash = []
    bit = 0
    ch = 0
    even = True
    bits = [16, 8, 4, 2, 1]

    while len(geohash) < precision:
        if even:
            mid = sum(lon_interval) / 2
            if lon > mid:
                ch |= bits[bit]
                lon_interval[0] = mid
            else:
                lon_interval[1] = mid
        else:
            mid = sum(lat_interval) / 2
            if lat > mid:
                ch |= bits[bit]
                lat_interval[0] = mid
            else:
                lat_interval[1] = mid
        even = not even

        if bit < 4:
            bit += 1
        else:
            geohash.append(BASE32[ch])
            bit = 0
            ch = 0
    return "".join(geohash)


def geohash_to_int(geohash: str) -> int:
    value = 0
    for char in geohash:
        value = value * 32 + BASE32.index(char)
    return value


def safe_standardize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-8:
        std = 1.0
    return (values - mean) / std, mean, std


def make_curve(rng: np.random.Generator, mode: int, length: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, length)
    start = rng.uniform(-0.8, 0.8, size=2)

    if mode == 0:  # walk: short local curve
        scale = rng.uniform(0.015, 0.035)
        x = start[0] + scale * (t + 0.2 * np.sin(2 * math.pi * t))
        y = start[1] + scale * (0.6 * t + 0.15 * np.cos(3 * math.pi * t))
    elif mode == 4:  # bike: medium S-curve
        scale = rng.uniform(0.04, 0.08)
        x = start[0] + scale * t
        y = start[1] + scale * (0.35 * np.sin(2 * math.pi * t) + 0.3 * t)
    elif mode == 1:  # car: longer arc
        scale = rng.uniform(0.08, 0.16)
        angle = rng.uniform(-0.7, 0.7)
        x = start[0] + scale * (t * math.cos(angle) - 0.3 * t * (1 - t) * math.sin(angle))
        y = start[1] + scale * (t * math.sin(angle) + 0.4 * t * (1 - t) * math.cos(angle))
    else:  # train: long mostly straight trajectory
        scale = rng.uniform(0.18, 0.35)
        angle = rng.uniform(-math.pi, math.pi)
        x = start[0] + scale * t * math.cos(angle)
        y = start[1] + scale * t * math.sin(angle)

    noise = rng.normal(0.0, 0.002, size=(length, 2))
    return np.column_stack([x, y]).astype(np.float32) + noise.astype(np.float32)


def cumulative_distance(points: np.ndarray) -> float:
    deltas = np.diff(points, axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def select_coefficients(trajectory: np.ndarray, k: int) -> np.ndarray:
    indices = np.linspace(0, len(trajectory) - 1, k).round().astype(int)
    return trajectory[indices].astype(np.float32)


def write_mean_std(path: Path, names: list[str], means: list[float], stds: list[float]) -> None:
    lines = []
    for name, mean, std in zip(names, means, stds):
        lines.append(f"{name}_mean: {mean:.10f}\n")
        lines.append(f"{name}_std: {std:.10f}\n")
    path.write_text("".join(lines), encoding="utf-8")


def build_dataset(output_dir: Path, coeff_path: Path, sample_count: int, length: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    geohash_precision = 3
    output_dir.mkdir(parents=True, exist_ok=True)
    coeff_path.parent.mkdir(parents=True, exist_ok=True)

    modes = np.array([0, 1, 3, 4], dtype=np.int64)
    trajectories = np.zeros((sample_count, length, 2), dtype=np.float32)
    raw_conditions = np.zeros((sample_count, 9), dtype=np.float32)
    geohash_to_id: dict[int, int] = {}

    for i in range(sample_count):
        mode = int(modes[i % len(modes)])
        traj = make_curve(rng, mode, length)
        trajectories[i] = traj

        total_dis = cumulative_distance(traj) * 1000.0
        total_time = float(length * rng.integers(20, 90))
        avg_dis = total_dis / length
        avg_speed = total_dis / max(total_time, 1.0)
        departure = float((i * 17) % 288)

        start_gh = geohash_to_int(geohash_encode(float(traj[0, 0]), float(traj[0, 1]), geohash_precision))
        end_gh = geohash_to_int(geohash_encode(float(traj[-1, 0]), float(traj[-1, 1]), geohash_precision))
        for cell in (start_gh, end_gh):
            if cell not in geohash_to_id:
                geohash_to_id[cell] = len(geohash_to_id)

        raw_conditions[i] = [
            departure,
            total_dis,
            total_time,
            float(length),
            avg_dis,
            avg_speed,
            float(geohash_to_id[start_gh]),
            float(geohash_to_id[end_gh]),
            float(mode),
        ]

    conditions = raw_conditions.copy()
    means = []
    stds = []
    for col in range(1, 6):
        z_values, mean, std = safe_standardize(raw_conditions[:, col])
        conditions[:, col] = z_values
        means.append(mean)
        stds.append(std)

    with (output_dir / "traj_segments.pkl").open("wb") as f:
        pickle.dump(trajectories, f, pickle.HIGHEST_PROTOCOL)
    with (output_dir / "conditions.pkl").open("wb") as f:
        pickle.dump(conditions.astype(np.float32), f, pickle.HIGHEST_PROTOCOL)
    with (output_dir / "mesh_mapping_dict.pkl").open("wb") as f:
        pickle.dump(geohash_to_id, f, pickle.HIGHEST_PROTOCOL)

    write_mean_std(
        output_dir / "traj_mean_std.txt",
        ["lat", "lon"],
        [float(trajectories[:, :, 0].mean()), float(trajectories[:, :, 1].mean())],
        [float(trajectories[:, :, 0].std()), float(trajectories[:, :, 1].std())],
    )
    write_mean_std(
        output_dir / "conditions_mean_std.txt",
        ["total_dis", "total_time", "total_len", "avg_dis", "avg_speed"],
        means,
        stds,
    )

    grid_meta = {
        "encoding": "geohash",
        "geohash_precision": geohash_precision,
        "note": "Synthetic geohash cells generated from dummy coordinates.",
    }
    (output_dir / "grid_meta.json").write_text(json.dumps(grid_meta, indent=2) + "\n", encoding="utf-8")

    summary = (
        "Synthetic TrajFlow toy dataset\n"
        f"samples: {sample_count}\n"
        f"trajectory_length: {length}\n"
        "source: fully synthetic procedural curves; no BW or DiDi data used\n"
        "condition_columns: departure, total_dis_z, total_time_z, total_len_z, "
        "avg_dis_z, avg_speed_z, origin_id, destination_id, transport_mode\n"
    )
    (output_dir / "dataset_summary.txt").write_text(summary, encoding="utf-8")

    coeffs = np.stack([select_coefficients(traj, 10) for traj in trajectories]).astype(np.float32)
    np.save(coeff_path, coeffs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the synthetic TrajFlow toy dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/toy_data"))
    parser.add_argument("--coeff-path", type=Path, default=Path("data/processed_coeffs_Toy_rdp_k_10.npy"))
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--length", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(args.output_dir, args.coeff_path, args.sample_count, args.length, args.seed)
    print(f"Wrote synthetic toy data to {args.output_dir}")
    print(f"Wrote cached toy coefficients to {args.coeff_path}")


if __name__ == "__main__":
    main()
