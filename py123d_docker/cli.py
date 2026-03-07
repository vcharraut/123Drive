import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from py123d_docker.configs import DATA_LAYOUT, DATASET_CONFIGS


def image_name(dataset):
    return f"py123d-docker-{dataset}"


def build_hydra_args(dataset, config, splits, workers, extra_overrides):
    args = [f"dataset={dataset}", "dataset_paths.py123d_data_root=/output"]
    args += [f"dataset_paths.{k}=/data/{v}" for k, v in config["path_keys"].items()]
    args += config["sensor_overrides"]
    active_splits = splits or config["default_splits"]
    args.append(f"dataset.parser.splits={json.dumps(active_splits, separators=(',', ':'))}")
    if workers and workers > 1:
        args += ["execution=process_pool_executor", f"execution.max_workers={workers}"]
    args += extra_overrides
    return args


def build_docker_run_cmd(dataset, data_root, output, hydra_args, shm_size, ipc_host, network_host):
    cmd = [
        "docker",
        "run",
        "--rm",
        "-e",
        "HYDRA_FULL_ERROR=1",
        "-e",
        "CUDA_VISIBLE_DEVICES=",
        "-e",
        "TF_CPP_MIN_LOG_LEVEL=3",
        "--ulimit",
        "nofile=65536:65536",
    ]
    cmd += ["--ipc=host"] if ipc_host else [f"--shm-size={shm_size}"]
    if network_host:
        cmd += ["--network=host"]
    cmd += ["-v", f"{data_root}:/data", "-v", f"{output}:/output", image_name(dataset), *hydra_args]
    return cmd


def build_docker_build_cmd(dataset, config, py123d_ref=None, no_cache=False):
    dockerfile_dir = Path(__file__).parent / "docker"
    cmd = ["docker", "build"]
    if no_cache:
        cmd += ["--no-cache"]
    python_version = config.get("python_version", "3.12")
    cmd += ["--build-arg", f"PYTHON_VERSION={python_version}"]
    cmd += ["--build-arg", f"EXTRAS={config['extras']}"]
    if py123d_ref:
        cmd += ["--build-arg", f"PY123D_REF={py123d_ref}"]
    cmd += ["-t", image_name(dataset), str(dockerfile_dir)]
    return cmd


def image_exists(dataset):
    result = subprocess.run(
        ["docker", "image", "inspect", image_name(dataset)],
        capture_output=True,
    )
    return result.returncode == 0


def run_cmd(cmd, dry_run, label=""):
    print(f"$ {' '.join(cmd)}")
    if not dry_run:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Error: {label} failed with exit code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)


def print_list():
    print("Available datasets:")
    for name, cfg in DATASET_CONFIGS.items():
        splits = ", ".join(cfg["default_splits"])
        print(f"  {name:<30} extras={cfg['extras']}  splits=[{splits}]")
    print("\nExpected data layout:")
    print(DATA_LAYOUT)


def main():
    parser = argparse.ArgumentParser(prog="py123d-docker")
    parser.add_argument("--dataset", help="Dataset name (see --list)")
    parser.add_argument("--data_root", help="Path to raw data root")
    parser.add_argument("--output", help="Path to output directory")
    parser.add_argument("--splits", nargs="+", help="Splits to convert (default: all)")
    parser.add_argument("--workers", type=int, help="Number of parallel workers")
    parser.add_argument("--shm_size", default="10g", help="Shared memory size, ignored if --ipc_host (default: 10g)")
    parser.add_argument(
        "--ipc_host",
        action="store_true",
        default=True,
        help="Use host IPC namespace for Ray (default: on)",
    )
    parser.add_argument(
        "--no_ipc_host",
        dest="ipc_host",
        action="store_false",
        help="Use --shm-size instead of --ipc=host",
    )
    parser.add_argument(
        "--network_host",
        action="store_true",
        default=True,
        help="Use host network for Ray (default: on)",
    )
    parser.add_argument("--no_network_host", dest="network_host", action="store_false")
    parser.add_argument(
        "--extra",
        nargs="+",
        default=[],
        metavar="OVERRIDE",
        help="Extra Hydra overrides (appended last)",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    parser.add_argument("--build_only", action="store_true", help="Build Docker image only")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild Docker image (--no-cache)")
    parser.add_argument("--py123d_ref", default=None, metavar="REF", help="py123d git ref to pin (branch/tag/commit)")
    parser.add_argument("--list", action="store_true", help="List datasets and data layout")
    args = parser.parse_args()

    if args.list:
        print_list()
        return

    if not args.dataset:
        parser.error("--dataset is required (or use --list)")

    if args.dataset not in DATASET_CONFIGS:
        print(f"Unknown dataset: {args.dataset!r}. Use --list to see available datasets.", file=sys.stderr)
        sys.exit(1)

    config = DATASET_CONFIGS[args.dataset]
    build_cmd = build_docker_build_cmd(args.dataset, config, args.py123d_ref, args.rebuild)

    if not image_exists(args.dataset) or args.build_only or args.rebuild:
        run_cmd(build_cmd, args.dry_run, label="docker build")

    if args.build_only:
        return

    if not args.data_root:
        parser.error("--data_root is required for conversion")

    output = args.output or os.environ.get("PY123D_DATA_ROOT")
    if not output:
        parser.error("--output is required for conversion (or set PY123D_DATA_ROOT environment variable)")

    data_root = str(Path(args.data_root).resolve())
    output = str(Path(output).resolve())

    hydra_args = build_hydra_args(
        args.dataset,
        config,
        args.splits,
        args.workers,
        args.extra,
    )
    run_cmd(
        build_docker_run_cmd(
            args.dataset,
            data_root,
            output,
            hydra_args,
            args.shm_size,
            args.ipc_host,
            args.network_host,
        ),
        args.dry_run,
        label="docker run",
    )


if __name__ == "__main__":
    main()
