"""Build CLI for py123d/123Drive Docker images."""

import argparse
import pathlib
import subprocess
import sys


DOCKERFILES = pathlib.Path(__file__).parent / "dockerfiles"
REPO_ROOT = DOCKERFILES.parent.parent
DEFAULT_PY_VERSION = "3.13"

DATASETS = {
    "nuplan": "nuplan",
    "nuplan-mini": "nuplan",
    "wod-motion": "waymo",
    "av2-sensor": "av2",
}


def push_image(tag: str, registry: str, dry_run: bool) -> None:
    remote = f"{registry}/{tag}"
    for cmd in (["docker", "tag", tag, remote], ["docker", "push", remote]):
        print(f"$ {' '.join(cmd)}")
        if not dry_run:
            result = subprocess.run(cmd)
            if result.returncode:
                sys.exit(result.returncode)


def cmd_py123d(args: argparse.Namespace) -> None:
    if not args.dataset or args.dataset not in DATASETS:
        print(f"Error: --dataset must be one of {list(DATASETS)}", file=sys.stderr)
        sys.exit(1)

    tag = f"py123d-{args.dataset}"
    cmd = [
        "docker",
        "build",
        *(["--no-cache"] if args.no_cache else []),
        "--build-arg",
        f"PYTHON_VERSION={DEFAULT_PY_VERSION}",
        "--build-arg",
        f"EXTRAS={DATASETS[args.dataset]}",
        "--build-arg",
        f"DATASET={args.dataset}",
        "-f",
        str(DOCKERFILES / "py123d.Dockerfile"),
        "-t",
        tag,
        str(DOCKERFILES.parent),
    ]

    print(f"$ {' '.join(cmd)}")
    if not args.dry_run:
        result = subprocess.run(cmd)
        if result.returncode:
            sys.exit(result.returncode)

    if args.push:
        push_image(tag, args.push, args.dry_run)


def cmd_123drive(args: argparse.Namespace) -> None:
    cmd = [
        "docker",
        "build",
        *(["--no-cache"] if args.no_cache else []),
        "--build-arg",
        f"PYTHON_VERSION={DEFAULT_PY_VERSION}",
        "-f",
        str(DOCKERFILES / "123drive.Dockerfile"),
        "-t",
        "123drive",
        str(REPO_ROOT),
    ]

    print(f"$ {' '.join(cmd)}")
    if not args.dry_run:
        result = subprocess.run(cmd)
        if result.returncode:
            sys.exit(result.returncode)

    if args.push:
        push_image("123drive", args.push, args.dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(prog="build", description="Build Docker images for py123d/123Drive pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    py123d_parser = sub.add_parser("py123d", help="Build py123d dataset image")
    py123d_parser.add_argument("--dataset", required=True, help="Dataset name")
    py123d_parser.add_argument("--no_cache", action="store_true", help="Build without Docker cache")
    py123d_parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    py123d_parser.add_argument("--push", help="Registry to tag + push to")

    drive_parser = sub.add_parser("123drive", help="Build 123Drive converter image")
    drive_parser.add_argument("--no_cache", action="store_true", help="Build without Docker cache")
    drive_parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    drive_parser.add_argument("--push", help="Registry to tag + push to")

    args = parser.parse_args()
    {"py123d": cmd_py123d, "123drive": cmd_123drive}[args.command](args)


if __name__ == "__main__":
    main()
