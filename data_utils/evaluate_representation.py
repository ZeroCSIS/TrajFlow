"""Measure the fixed trajectory representation's error without model sampling."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

from generate import raw_dataset_trajectories, select_evaluation_indices
from src.data.dataset import FlowMatchingDataset
from src.data.transforms import para2point_batch
from src.eval.inference import FlowMatchingInference
from src.eval.metrics import compute_baseline_metrics, origin_destination_straight_lines


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def build_representation_report(
    parameterized_reference: np.ndarray,
    raw_reference: np.ndarray,
    *,
    grid_metadata: dict,
    max_pairs: int,
    density_bins: int,
) -> dict[str, object]:
    straight_line = origin_destination_straight_lines(raw_reference)
    metric_options = {
        "grid_metadata": grid_metadata,
        "max_pairs": max_pairs,
        "density_bins": density_bins,
    }
    representation_metrics = compute_baseline_metrics(
        parameterized_reference,
        raw_reference,
        **metric_options,
    )
    straight_metrics = compute_baseline_metrics(
        straight_line,
        raw_reference,
        **metric_options,
    )
    return {
        "schema_version": "trajflow-representation-evaluation-v1",
        "metric_semantics": {
            "representation": (
                "configured control points reconstructed to the raw reference length"
            ),
            "primary_reference": "paired raw test trajectories",
            "density_histogram_oob_policy": (
                "out-of-grid points are excluded and reported separately"
            ),
            "density_grid_bins": [int(density_bins), int(density_bins)],
            "dtw": "exact accumulated Euclidean point cost",
            "continuous_frechet": "Alt-Godau continuous curve distance",
        },
        "array_shapes": {
            "parameterized_reference": list(np.asarray(parameterized_reference).shape),
            "paired_raw_reference": list(np.asarray(raw_reference).shape),
            "od_straight_line_control": list(straight_line.shape),
        },
        "parameterized_representation_vs_paired_raw_test": representation_metrics,
        "od_straight_line_vs_paired_raw_test": straight_metrics,
        "representation_ceiling_comparison": {
            "positive_straight_minus_representation_means_representation_is_better": True,
            "dtw_median_km_straight_minus_representation": (
                straight_metrics["dtw_median_km"]
                - representation_metrics["dtw_median_km"]
            ),
            "continuous_frechet_median_km_straight_minus_representation": (
                straight_metrics["continuous_frechet_median_km"]
                - representation_metrics["continuous_frechet_median_km"]
            ),
            "representation_beats_straight_line_on_dtw_median": (
                representation_metrics["dtw_median_km"]
                < straight_metrics["dtw_median_km"]
            ),
            "representation_beats_straight_line_on_continuous_frechet_median": (
                representation_metrics["continuous_frechet_median_km"]
                < straight_metrics["continuous_frechet_median_km"]
            ),
        },
    }


def evaluate_representation(
    config_path: Path,
    output_path: Path,
    *,
    samples: int | None = None,
) -> dict[str, object]:
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not config["data"].get("parametrized", False):
        raise ValueError("Representation evaluation requires data.parametrized=true")
    if not config.get("condition", {}).get("enabled", False):
        raise ValueError("The YJMob representation evaluator requires conditions")

    dataset = FlowMatchingDataset(config, mode="test")
    evaluation_config = config.get("evaluation", {})
    requested_samples = int(
        samples
        if samples is not None
        else config.get("inference", {}).get("num_samples", 200)
    )
    seed = int(config.get("project", {}).get("seed", 42))
    selected_indices, _ = select_evaluation_indices(
        len(dataset),
        requested_samples,
        seed,
        0,
    )
    trajectory_length = int(config["data"]["trajectory_length"])
    method = config["data"].get(
        "parametrized_method",
        config["data"].get("para_method", "rdp_k"),
    )
    reconstructed = para2point_batch(
        np.asarray(dataset.traj_segments)[selected_indices],
        trajectory_length,
        method,
    )
    inference = FlowMatchingInference(
        config=config,
        model=torch.nn.Identity(),
        dataset=dataset,
        save_dir=str(output_path.parent),
        device=torch.device("cpu"),
    )
    conditions = np.asarray(dataset.conditions)[selected_indices]
    parameterized_reference = inference.denormalize_trajectories(
        [reconstructed],
        conditions,
        dataset,
    )[0].reshape(-1, trajectory_length, 2)
    raw_reference = raw_dataset_trajectories(
        dataset,
        selected_indices,
        config,
        dataset.traj_mean,
        dataset.traj_std,
    )
    report = build_representation_report(
        parameterized_reference,
        raw_reference,
        grid_metadata=dataset.grid_metadata,
        max_pairs=int(evaluation_config.get("max_curve_pairs", requested_samples)),
        density_bins=int(evaluation_config.get("density_bins", 40)),
    )
    source_indices = np.asarray(
        getattr(dataset, "sample_indices", np.arange(len(dataset))),
        dtype=np.int64,
    )
    report["provenance"] = {
        "execution_device": "cpu",
        "model_or_checkpoint_used": False,
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256_file(config_path),
        "git_commit": _git_commit(),
        "seed": seed,
        "test_split_samples": len(dataset),
        "selected_test_local_indices": selected_indices.tolist(),
        "selected_source_sample_indices": source_indices[selected_indices].tolist(),
        "training_target_preflight": getattr(
            dataset,
            "parameterized_target_summary",
            None,
        ),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int)
    args = parser.parse_args()
    report = evaluate_representation(
        args.config,
        args.output,
        samples=args.samples,
    )
    representation = report[
        "parameterized_representation_vs_paired_raw_test"
    ]
    print(
        "Representation evaluation saved: "
        f"DTW median={representation['dtw_median_km']:.6f} km, "
        f"continuous Frechet median="
        f"{representation['continuous_frechet_median_km']:.6f} km"
    )


if __name__ == "__main__":
    main()
