"""py123d extraction entrypoint."""

import json
import os


DATASETS = {
    "nuplan": {"data_root_key": "nuplan_data_root", "input_subpaths": {"nuplan_maps_root": "maps"}},
    "nuplan-mini": {"data_root_key": "nuplan_data_root", "input_subpaths": {"nuplan_maps_root": "maps"}},
    "wod-motion": {"data_root_key": "wod_motion_data_root"},
    "av2-sensor": {"data_root_key": "av2_data_root"},
}

HYDRA_OVERRIDES = [
    "dataset.log_writer_config.force_log_conversion=true",
    "dataset.map_writer.force_map_conversion=true",
    "scene_filter.shuffle=false",
    "scene_filter.has_map=true",
    "+dataset.log_writer_config.exclude_modality_types=[camera,lidar,custom]",
]


def build_hydra_args(dataset, args):
    config = DATASETS[dataset]
    hydra_args = [
        f"dataset={dataset}",
        f"dataset_paths.py123d_data_root={args.output}",
        f"dataset_paths.{config['data_root_key']}={args.input}",
    ]

    hydra_args.extend(f"dataset_paths.{k}={args.input}/{v}" for k, v in config.get("input_subpaths", {}).items())
    hydra_args.extend(HYDRA_OVERRIDES)

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
    if not dataset or dataset not in DATASETS:
        print(f"Error: DATASET env var must be one of {list(DATASETS)}", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="py123d extraction entrypoint")
    parser.add_argument("--input", "-i", default="/input")
    parser.add_argument("--output", "-o", default="/output")
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--worker_type", choices=["ray", "process_pool", "thread_pool"], default="ray")
    parser.add_argument("--workers", type=int, default=max(1, int((os.cpu_count() or 4) * 0.8)))
    args = parser.parse_args()

    cmd = ["py123d-conversion", *build_hydra_args(dataset, args)]
    print(f"$ {' '.join(cmd)}")
    os.execvp("py123d-conversion", cmd)
