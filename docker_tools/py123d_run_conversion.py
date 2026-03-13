"""Simplified runtime wrapper for py123d dataset conversion."""

import json
import os

try:
    from docker_tools.py123d_config import DATASET_CONFIGS, INPUT_MOUNT, OUTPUT_MOUNT
except ImportError:
    from py123d_config import DATASET_CONFIGS, INPUT_MOUNT, OUTPUT_MOUNT


def build_hydra_args(dataset, args):
    config = DATASET_CONFIGS[dataset]
    hydra_dataset = config["hydra_dataset"] or dataset

    hydra_args = [
        f"dataset={hydra_dataset}",
        f"dataset_paths.py123d_data_root={args.output}",
        f"dataset_paths.{config['data_root_key']}={args.input}",
    ]

    hydra_args.extend(
        f"dataset_paths.{key}={args.input}/{subpath}" for key, subpath in config["input_subpaths"].items()
    )

    hydra_args.extend(config["hydra_overrides"])

    splits = args.splits or config["default_splits"]
    if splits:
        hydra_args.append(f"dataset.parser.splits={json.dumps(splits)}")

    hydra_args.append(f"execution.max_workers={args.workers}")

    return hydra_args


if __name__ == "__main__":
    import argparse
    import sys

    dataset = os.environ.get("DATASET")
    if not dataset or dataset not in DATASET_CONFIGS:
        print(f"Error: DATASET env var must be one of {list(DATASET_CONFIGS)}", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="py123d extraction entrypoint")
    parser.add_argument("--input", "-i", default=INPUT_MOUNT)
    parser.add_argument("--output", "-o", default=OUTPUT_MOUNT)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=max(1, int((os.cpu_count() or 4) * 0.8)))
    args = parser.parse_args()

    cmd = ["py123d-conversion", *build_hydra_args(dataset, args)]
    print(f"$ {' '.join(cmd)}")
    os.execvp("py123d-conversion", cmd)
