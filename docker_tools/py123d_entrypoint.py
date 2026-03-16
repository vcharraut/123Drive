"""Simplified runtime wrapper for py123d dataset conversion."""

import json
import os

import py123d_config


def build_hydra_args(dataset, args):
    config = py123d_config.DATASET_CONFIGS[dataset]

    hydra_args = [
        f"dataset={dataset}",
        f"dataset_paths.py123d_data_root={args.output}",
        f"dataset_paths.{config['data_root_key']}={args.input}",
    ]

    if config.get("input_subpaths"):
        hydra_args.extend(
            f"dataset_paths.{key}={args.input}/{subpath}" for key, subpath in config["input_subpaths"].items()
        )

    hydra_args.extend(py123d_config.HYDRA_OVERRIDES)

    if args.splits:
        hydra_args.append(f"dataset.parser.splits={json.dumps(args.splits)}")

    hydra_args.append(f"execution={args.worker_type}_executor")
    if args.worker_type != "ray":
        hydra_args.append(f"execution.max_workers={args.workers}")

    return hydra_args


if __name__ == "__main__":
    import argparse
    import sys

    dataset = os.environ.get("DATASET")
    if not dataset or dataset not in py123d_config.DATASET_CONFIGS:
        print(f"Error: DATASET env var must be one of {list(py123d_config.DATASET_CONFIGS)}", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="py123d extraction entrypoint")
    parser.add_argument("--input", "-i", default=py123d_config.INPUT_MOUNT)
    parser.add_argument("--output", "-o", default=py123d_config.OUTPUT_MOUNT)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--worker_type", choices=["ray", "process_pool", "thread_pool"], default="ray")
    parser.add_argument("--workers", type=int, default=max(1, int((os.cpu_count() or 4) * 0.8)))
    args = parser.parse_args()

    cmd = ["py123d-conversion", *build_hydra_args(dataset, args)]
    print(f"$ {' '.join(cmd)}")
    os.execvp("py123d-conversion", cmd)
