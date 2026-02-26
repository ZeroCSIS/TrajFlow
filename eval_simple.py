#!/usr/bin/env python3
import argparse
from types import SimpleNamespace

def main():
    parser = argparse.ArgumentParser(description="Simple evaluation entry point")
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Path to a single generation result folder")
    parser.add_argument("--max_trajs", type=int, default=3000,
                        help="Limit number of trajectories used for metrics; "
                             "set -1 to use all available.")
    args = parser.parse_args()
    from src.eval.evaluate import evaluate_single_folder

    eval_args = SimpleNamespace(
        division_type="JISMESH",
        mesh_size=1000,
        grid_num=16,
        top_n=20,
        DTW_sample_size=2000,
        max_trajs=args.max_trajs,
    )

    result = evaluate_single_folder(args.result_dir, eval_args)
    if result is None:
        print("No evaluation result produced.")
        return

    print("Evaluation results:")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
