"""Build CLI for py123d/123Drive Docker images."""

import argparse
import pathlib
import subprocess
import sys


DOCKERFILES = pathlib.Path(__file__).parent / "dockerfiles"
REPO_ROOT = DOCKERFILES.parent.parent
PY123D_ROOT = REPO_ROOT / "external" / "py123d"  # same checkout as [tool.uv.sources] in pyproject.toml
DEFAULT_PY_VERSION = "3.13"

DATASETS = {
    "nuplan": "nuplan",
    "nuplan-mini": "nuplan",
    "wod-motion": "waymo",
    "av2-sensor": "av2",
    "nurec-2604": "nurec",
    "opendrive": "",
}


def run(cmd: list[str], dry_run: bool) -> None:
    print(f"$ {' '.join(cmd)}")
    if not dry_run and (returncode := subprocess.run(cmd).returncode):
        sys.exit(returncode)


def build_image(
    args: argparse.Namespace,
    tag: str,
    dockerfile: str,
    context: pathlib.Path,
    build_args: list[str],
    contexts: list[str],
) -> None:
    """Build ``tag`` from ``dockerfile`` with py123d as a build context, then tag and push it to ``--push``."""
    cmd = ["docker", "build", *(["--no-cache"] if args.no_cache else [])]
    cmd += ["--build-arg", f"PYTHON_VERSION={DEFAULT_PY_VERSION}", *build_args]
    cmd += ["--build-context", f"py123d={PY123D_ROOT}", *contexts]
    run([*cmd, "-f", str(DOCKERFILES / dockerfile), "-t", tag, str(context)], args.dry_run)
    if args.push:
        remote = f"{args.push}/{tag}"
        for push_cmd in (["docker", "tag", tag, remote], ["docker", "push", remote]):
            run(push_cmd, args.dry_run)


def cmd_py123d(args: argparse.Namespace) -> None:
    if not args.dataset or args.dataset not in DATASETS:
        print(f"Error: --dataset must be one of {list(DATASETS)}", file=sys.stderr)
        sys.exit(1)
    build_args = ["--build-arg", f"EXTRAS={DATASETS[args.dataset]}", "--build-arg", f"DATASET={args.dataset}"]
    contexts = ["--build-context", f"123drive={REPO_ROOT}"]
    build_image(args, f"py123d-{args.dataset}", "py123d.Dockerfile", DOCKERFILES.parent, build_args, contexts)


def cmd_123drive(args: argparse.Namespace) -> None:
    build_image(args, "123drive", "123drive.Dockerfile", REPO_ROOT, [], [])


def main() -> None:
    parser = argparse.ArgumentParser(prog="build", description="Build Docker images for py123d/123Drive pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    py123d_parser = sub.add_parser("py123d", help="Build py123d dataset image")
    py123d_parser.add_argument("--dataset", required=True, help="Dataset name")
    for command in (py123d_parser, sub.add_parser("123drive", help="Build 123Drive converter image")):
        command.add_argument("--no_cache", action="store_true", help="Build without Docker cache")
        command.add_argument("--dry_run", action="store_true", help="Print commands without running")
        command.add_argument("--push", help="Registry to tag + push to")
    args = parser.parse_args()
    {"py123d": cmd_py123d, "123drive": cmd_123drive}[args.command](args)


if __name__ == "__main__":
    main()
