"""CLI for building Docker images for the py123d/123Drive pipeline."""

import argparse
import subprocess
import sys
from pathlib import Path

from docker_tools.configs import DATASET_CONFIGS


DOCKERFILES_DIR = Path(__file__).parent / "dockerfiles"


def _docker_build_cmd(
    dockerfile: str,
    tag: str,
    build_args: dict[str, str] | None = None,
    no_cache: bool = False,
) -> list[str]:
    """Assemble a docker build command."""
    cmd = ["docker", "build"]

    if no_cache:
        cmd += ["--no-cache"]

    for k, v in (build_args or {}).items():
        cmd += ["--build-arg", f"{k}={v}"]

    cmd += ["-f", str(DOCKERFILES_DIR / dockerfile), "-t", tag, str(DOCKERFILES_DIR)]
    return cmd


def run_cmd(cmd: list[str], dry_run: bool, label: str = "") -> None:
    """Print and optionally execute a shell command."""
    print(f"$ {' '.join(cmd)}")
    if not dry_run:
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"Error: {label} failed with exit code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)


def cmd_list(_args: argparse.Namespace) -> None:
    """Print available datasets and build targets."""
    print("Available datasets from 123D:")
    for name, cfg in DATASET_CONFIGS.items():
        splits = ", ".join(cfg["default_splits"]) if cfg["default_splits"] else "default"
        print(f"  {name:<30} extras={cfg['extras']}  splits=[{splits}]")


def cmd_py123d(args: argparse.Namespace) -> None:
    """Build a py123d dataset Docker image."""
    if not args.dataset:
        print("Error: --dataset is required for 'build py123d'", file=sys.stderr)
        sys.exit(1)

    if args.dataset not in DATASET_CONFIGS:
        print(f"Unknown dataset: {args.dataset!r}. Use 'build list' to see available datasets.", file=sys.stderr)
        sys.exit(1)

    config = DATASET_CONFIGS[args.dataset]
    name = f"py123d-{args.dataset}"
    build_args = {"PYTHON_VERSION": config.get("python_version", "3.12"), "EXTRAS": config["extras"]}

    ref_py123d = getattr(args, "py123d_ref", None)
    if ref_py123d:
        build_args["PY123D_REF"] = ref_py123d

    cmd = _docker_build_cmd("py123d.Dockerfile", name, build_args, args.no_cache)
    run_cmd(cmd, args.dry_run, label="docker build")


def cmd_123drive(args: argparse.Namespace) -> None:
    """Build the 123Drive converter Docker image."""
    ref = args.drive123_ref or "main"
    name = f"123drive:{ref}"

    cmd = _docker_build_cmd("123drive.Dockerfile", name, {"DRIVE123_REF": ref}, args.no_cache)
    run_cmd(cmd, args.dry_run, label="docker build")


def main() -> None:
    """Entry point for the build CLI."""
    parser = argparse.ArgumentParser(prog="build", description="Build Docker images for py123d/123Drive pipeline")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List available datasets")

    py123d_parser = sub.add_parser("py123d", help="Build py123d dataset image")
    py123d_parser.add_argument("--dataset", help="Dataset name (required)")
    py123d_parser.add_argument("--py123d_ref", metavar="REF", help="py123d git ref (branch/tag/commit)")
    py123d_parser.add_argument("--no_cache", action="store_true", help="Build without Docker cache")
    py123d_parser.add_argument("--dry_run", action="store_true", help="Print commands without running")

    drive_parser = sub.add_parser("123drive", help="Build 123Drive converter image")
    drive_parser.add_argument("--drive123_ref", metavar="REF", help="123Drive git ref (default: main)")
    drive_parser.add_argument("--no_cache", action="store_true", help="Build without Docker cache")
    drive_parser.add_argument("--dry_run", action="store_true", help="Print commands without running")

    args = parser.parse_args()

    commands = {"list": cmd_list, "py123d": cmd_py123d, "123drive": cmd_123drive}
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
