import argparse
import collections
import dataclasses
import json
import os
import pathlib
import re
import sys
import tomllib
import warnings
from typing import Any

import joblib
import tqdm

from bin_factory import loader, serialize
from bin_factory.log_context import bind, log, unbind
from bin_factory.preparation import ConversionSettings, prepare_scenario


DEFAULTS = ConversionSettings()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert 123D datasets to PufferDrive binary format")

    parser.add_argument("--py123d_path", type=str, help="Path to py123d dataset containing logs/ and maps/")
    parser.add_argument("--output", default="./output", help="Directory to save binary files")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel worker count. 0 uses 80%% of CPU cores, 1 disables parallelism, -1 uses all cores.",
    )
    parser.add_argument("--num_scenes", type=int, default=None, help="Maximum number of scenes to process")
    parser.add_argument(
        "--preset",
        type=str,
        help="Apply a dataset preset from presets.toml such as nuplan or wod-motion. CLI flags override preset values.",
    )
    parser.add_argument("--datasets", nargs="+", help="Dataset names to include such as nuplan or wod-motion")
    parser.add_argument("--split_types", nargs="+", help="Split types to include such as train, val, or test")
    parser.add_argument("--split_names", nargs="+", help="Split names to include such as nuplan-mini_val")
    parser.add_argument("--log_names", nargs="+", help="Log names to include")
    parser.add_argument("--scene_uuids", nargs="+", help="Scene UUIDs to include when debugging specific scenarios")
    parser.add_argument(
        "--scenario_id_field",
        choices=["scene_uuid", "log_name", "location"],
        default=DEFAULTS.scenario_id_field,
        help="py123d attribute used as metadata.id and the readable filename component. Map-only uses 'location'.",
    )
    parser.add_argument("--duration_s", type=float, default=0.0, help="Duration of scenario in seconds")
    parser.add_argument("--dt", type=float, default=DEFAULTS.dt, help="Iteration timestep in seconds")
    parser.add_argument("--map_only", action="store_true", help="Load map-only scenarios without logs")
    parser.add_argument(
        "--chunk_target_scenes",
        type=int,
        default=10000,
        help="Target number of scenarios per conversion chunk",
    )
    parser.add_argument(
        "--validate_level",
        type=int,
        choices=[0, 1, 2],
        default=DEFAULTS.validate_level,
        help="0=off, 1=schema, 2=semantic",
    )
    parser.add_argument(
        "--max_segment_length",
        type=float,
        default=DEFAULTS.max_segment_length,
        help="Max segment length for interpolation",
    )
    parser.add_argument(
        "--edge_max_segment_length",
        type=float,
        default=DEFAULTS.edge_max_segment_length,
        help="Max segment length for road-edge interpolation. 0 disables interpolation.",
    )
    parser.add_argument(
        "--area_threshold",
        type=float,
        default=DEFAULTS.area_threshold,
        help="Tolerance for polyline simplification. 0 disables simplification.",
    )
    parser.add_argument(
        "--speed_limit_source",
        choices=["road", "signs"],
        default=DEFAULTS.speed_limit_source,
        help="Select the road's speed limit or the last speed sign's limit as CARLA applies it",
    )
    parser.add_argument(
        "--static_objects",
        choices=["all", "onroad"],
        default=DEFAULTS.static_objects,
        help="Keep all static map objects or only those on the road surface",
    )
    parser.add_argument(
        "--shoulders",
        choices=["drivable", "offroad"],
        default=DEFAULTS.shoulders,
        help="Treat shoulders outside junctions as drivable or place road edges along adjacent lane boundaries",
    )
    parser.add_argument(
        "--ramp_widening_m",
        type=float,
        default=DEFAULTS.ramp_widening_m,
        help="When --shoulders is offroad, shift shoulder edges outward by up to this distance on single-lane ramps "
        "while keeping 1.5 m to the far side of the shoulder",
    )
    parser.add_argument(
        "--reverse_road_edges",
        action="store_true",
        default=DEFAULTS.reverse_road_edges,
        help="Reverse road-edge polyline order to match the Waymo convention",
    )
    parser.add_argument(
        "--infer_traffic_lights",
        action="store_true",
        default=DEFAULTS.infer_traffic_lights,
        help="Impute and correct traffic-light states from vehicle trajectories (arXiv 2506.07150)",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Root logging level",
    )
    return parser


def _apply_preset(parser: argparse.ArgumentParser, argv: list[str]) -> None:
    # Make preset values argparse defaults so explicit CLI flags still override them.
    # Store_true flags set by a preset cannot be turned back off from the CLI.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset")
    name = pre.parse_known_args(argv)[0].preset
    if not name:
        return
    presets = tomllib.loads((pathlib.Path(__file__).parent / "presets.toml").read_text())
    if name not in presets:
        parser.error(f"Unknown preset '{name}'. Available: {', '.join(sorted(presets))}")
    known_dests = {action.dest for action in parser._actions}
    unknown = sorted(set(presets[name]) - known_dests)
    if unknown:
        parser.error(f"Preset '{name}' has unknown keys: {', '.join(unknown)}")
    parser.set_defaults(**presets[name])


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, ConversionSettings]:
    """Resolve convert's command line: preset values first, explicit flags over them, then dataset rules."""
    parser = build_parser()
    _apply_preset(parser, argv)
    args = _validate_args(parser.parse_args(argv), parser)
    settings = ConversionSettings(**{field.name: getattr(args, field.name) for field in dataclasses.fields(DEFAULTS)})
    return args, settings


def _build_output_path(py123d_data: Any, output_dir: pathlib.Path, field: str) -> pathlib.Path:
    """Build a filename from the dataset, selected identity, and scene UUID.

    Map-only filenames use the dataset and location.
    """

    def sanitize(v: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", v.strip()).strip("._-") or "scenario"

    dataset = getattr(py123d_data, "dataset", "")
    if not dataset:
        raise ValueError("py123d_data is missing 'dataset' attribute, which is required for output filename")
    if not hasattr(py123d_data, "scene_uuid"):
        stem = f"{sanitize(dataset)}__{sanitize(loader.scenario_id(py123d_data, 'location'))}"
    else:
        scene_uuid = loader.scenario_id(py123d_data, "scene_uuid")
        source_id = loader.scenario_id(py123d_data, field)
        parts = [dataset, scene_uuid] if field == "scene_uuid" else [dataset, source_id, scene_uuid]
        stem = "__".join(sanitize(part) for part in parts)

    return output_dir / f"{stem}.bin"


def _worker_fn(py123d_data: Any, output_dir: pathlib.Path, settings: ConversionSettings, log_level: str) -> dict:
    log.setLevel(log_level)
    dataset = getattr(py123d_data, "dataset", "unknown")
    log_name = getattr(py123d_data, "log_name", "")
    identity = {"dataset": dataset, "log_name": log_name, "scenario_id": "unknown"}
    tokens = bind(dataset=dataset)

    try:
        identity["scenario_id"] = loader.scenario_id(py123d_data, settings.scenario_id_field)
        bind(dataset=dataset, scenario=identity["scenario_id"])
        _convert_one(py123d_data, output_dir, settings)
        return {"ok": True, **identity, "error": ""}
    except loader.ValidationError as ve:
        log.error("validation error: %s", ve)
        return {"ok": False, **identity, "error": str(ve)}
    except Exception as e:
        log.exception("scenario failed")
        return {"ok": False, **identity, "error": str(e)}
    finally:
        unbind(tokens)


def _convert_one(py123d_data: Any, output_dir: pathlib.Path, settings: ConversionSettings) -> None:
    scenario = prepare_scenario(py123d_data, settings)
    binary_data = serialize.scenario_to_binary(scenario)
    output_path = _build_output_path(py123d_data, output_dir, settings.scenario_id_field)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary_path.write_bytes(binary_data)
    temporary_path.replace(output_path)


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    for attr in ("datasets", "split_types", "split_names", "log_names", "scene_uuids"):
        values = getattr(args, attr)
        cleaned = [v.strip() for v in (values or []) if v and v.strip()] or None
        setattr(args, attr, cleaned)

    if args.num_scenes is not None and args.num_scenes <= 0:
        parser.error("--num_scenes must be > 0")
    if args.chunk_target_scenes <= 0:
        parser.error("--chunk_target_scenes must be > 0")
    if args.workers < -1:
        parser.error("--workers must be -1, 0, or a positive integer")
    if not 0.0 <= args.ramp_widening_m <= 5.0:
        parser.error("--ramp_widening_m must be between 0 and 5")

    cpu_count = os.cpu_count() or 1
    if args.workers == -1:
        args.workers = cpu_count
    elif args.workers == 0:
        args.workers = max(1, int(cpu_count * 0.8))

    if args.num_scenes is not None and args.num_scenes < args.workers:
        log.warning("num_scenes (%d) < workers (%d). Reducing workers.", args.num_scenes, args.workers)
        args.workers = args.num_scenes

    if "opendrive" in (args.datasets or []) and not args.map_only:
        log.warning("Dataset 'opendrive' selected with --map_only=False. Forcing --map_only=True.")
        args.map_only = True

    if "nuplan" in (args.datasets or []) and args.duration_s == 0.0 and not args.map_only:
        log.warning(
            "Converting nuPlan without --duration_s: raw nuPlan logs span minutes and can exhaust RAM. "
            "Set --duration_s (e.g. 20) or use --preset nuplan, which pins it for you."
        )

    args.py123d_path = args.py123d_path or os.environ.get("PY123D_DATA_ROOT")
    if not args.py123d_path:
        parser.error("--py123d_path is required (or set PY123D_DATA_ROOT environment variable)")
    return args


def main() -> int:
    args, settings = parse_args(sys.argv[1:])

    log.setLevel(args.log_level)

    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)

    log.info("123Drive: %s -> %s", args.py123d_path, args.output)
    log.info(
        "Filters - datasets: %s, split_types: %s, split_names: %s, log_names: %s, duration_s: %s, map_only: %s",
        args.datasets,
        args.split_types,
        args.split_names,
        args.log_names,
        args.duration_s,
        args.map_only,
    )

    scenes = loader.discover_scenes(
        py123d_data_root=args.py123d_path,
        workers=args.workers,
        num_scenes=args.num_scenes,
        datasets=args.datasets,
        split_types=args.split_types,
        split_names=args.split_names,
        log_names=args.log_names,
        scene_uuids=args.scene_uuids,
        duration_s=None if args.duration_s == 0.0 else args.duration_s,
        dt=args.dt,
        map_only=args.map_only,
    )
    if not scenes:
        log.info("No scenarios to process.")
        return 0

    output_dir = pathlib.Path(args.output)
    path_counts = collections.Counter(
        _build_output_path(scene, output_dir, settings.scenario_id_field) for scene in scenes
    )
    duplicates = sorted(str(path) for path, count in path_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Multiple scenarios resolve to the same output path: {', '.join(duplicates)}")

    log.info("Discovered %d scenarios. Starting conversion with %d workers.", len(scenes), args.workers)

    # Suppress the loky pool teardown warning that fires on normal completion.
    warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given")

    failed = 0
    with (
        (output_dir / "failures.jsonl").open("w", encoding="utf-8") as failure_handle,
        tqdm.tqdm(total=len(scenes)) as pbar,
        joblib.Parallel(
            n_jobs=args.workers, backend="loky", batch_size="auto", return_as="generator_unordered"
        ) as parallel,
    ):
        for start in range(0, len(scenes), args.chunk_target_scenes):
            chunk = scenes[start : start + args.chunk_target_scenes]
            for result in parallel(
                joblib.delayed(_worker_fn)(data, output_dir=output_dir, settings=settings, log_level=args.log_level)
                for data in chunk
            ):
                if not result["ok"]:
                    failed += 1
                    failure_handle.write(json.dumps(result) + "\n")
                    failure_handle.flush()
                pbar.update(1)

    succeeded = len(scenes) - failed
    log.info("Conversion complete. %d/%d succeeded, %d failed.", succeeded, len(scenes), failed)
    return 0 if succeeded else 1  # tolerate partial failures and fail only if nothing converted


if __name__ == "__main__":
    raise SystemExit(main())
