import argparse
import json
import os
import pathlib
import re
import warnings

import joblib
import tqdm

from bin_factory import loader, serialize, transforms
from bin_factory.log_context import bind, log, unbind


def build_parser():
    parser = argparse.ArgumentParser(description="Convert 123D datasets to PufferDrive binary format")

    parser.add_argument("--py123d_path", type=str, help="Path to py123d dataset (logs/ and maps/)")
    parser.add_argument("--output", default="./output", help="Directory to save binary files")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Number of parallel workers - "
            "0: 80%% of CPU cores (default), 1: no parallelism, -1: all CPU cores, 2-n: use n CPU cores"
        ),
    )
    parser.add_argument("--num_scenes", type=int, default=None, help="Maximum number of scenes to process")
    parser.add_argument("--datasets", nargs="+", help="Dataset names to include (e.g. nuplan, wod-motion)")
    parser.add_argument("--split_types", nargs="+", help="Split types to include (e.g. train, val, test)")
    parser.add_argument("--split_names", nargs="+", help="Split names to include (e.g. nuplan-mini_val)")
    parser.add_argument("--log_names", nargs="+", help="Log names to include")
    parser.add_argument("--scene_uuids", nargs="+", help="Scene UUIDs to include (for debugging specific scenarios)")
    parser.add_argument("--duration_s", type=float, default=0.0, help="Duration of scenario in seconds")
    parser.add_argument("--map_only", action="store_true", help="Load map-only scenarios (no logs)")
    parser.add_argument(
        "--chunk_target_scenes",
        type=int,
        default=10000,
        help="Target number of scenarios per conversion chunk",
    )
    parser.add_argument("--validate_level", type=int, choices=[0, 1, 2], default=1, help="0=off, 1=schema, 2=semantic")
    parser.add_argument("--max_segment_length", type=float, default=10.0, help="Max segment length for interpolation")
    parser.add_argument(
        "--area_threshold",
        type=float,
        default=0.1,
        help="Tolerance for polyline simplification (0=disabled)",
    )
    parser.add_argument(
        "--min_route_valid_points",
        type=float,
        default=0.0,
        help="Min valid trajectory percentage for route computation (0-100)",
    )
    parser.add_argument(
        "--route_check_timestep",
        type=int,
        default=0,
        help="Timestep at which agent must be valid for route computation",
    )
    parser.add_argument(
        "--no_reindex", action="store_true", help="Skip reindexing element IDs to contiguous range(0, n)"
    )
    parser.add_argument(
        "--impute_tl",
        action="store_true",
        help="Impute/correct traffic light states from vehicle trajectories (Yan et al. 2025)",
    )
    parser.add_argument(
        "--invalid_agent_overlap",
        action="store_true",
        help="Zero out log-only agent trajectories that overlap with active agents during replay",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Root logging level (default: INFO)",
    )
    return parser


def _scenario_identity(py123d_data) -> str | None:
    dataset = str(getattr(py123d_data, "dataset", "") or "")
    if dataset.startswith("nuplan"):
        preferred_attr = "scene_uuid"
    elif dataset.startswith("opendrive"):
        preferred_attr = "location"
    else:
        preferred_attr = "log_name"

    scenario_id = str(getattr(py123d_data, preferred_attr, ""))
    if scenario_id == "":
        raise ValueError(
            f"Cannot derive scenario id from py123d_data attributes: missing {preferred_attr} for dataset {dataset}"
        )

    return scenario_id


def _build_output_path(py123d_data, output_dir):
    """Derive output path for a given scenario based on its metadata attributes.

    Arguments:
        py123d_data: py123d scenario data object
        output_dir: Base directory to save the output file

    Returns:
        A pathlib.Path object representing the output file path, e.g. <dataset>__<source_id>.bin
    """

    def sanitize(v):
        return re.sub(r"[^A-Za-z0-9._-]+", "_", v.strip()).strip("._-") or "scenario"

    source_id = _scenario_identity(py123d_data)
    dataset = getattr(py123d_data, "dataset", "") or ""

    if not dataset:
        raise ValueError("py123d_data is missing 'dataset' attribute, which is required for output filename")

    stem = f"{sanitize(dataset)}__{sanitize(source_id)}"

    return output_dir / f"{stem}.bin"


def _worker_fn(py123d_data, output_dir, config):
    log.setLevel(config.log_level)
    scenario_id = _scenario_identity(py123d_data) or "unknown"
    dataset = getattr(py123d_data, "dataset", "unknown")
    log_name = getattr(py123d_data, "log_name", "")
    identity = {"dataset": dataset, "log_name": log_name, "scenario_id": scenario_id}
    tokens = bind(dataset=dataset, scenario=scenario_id)

    try:
        _convert_one(py123d_data, output_dir, config)
        return {"ok": True, **identity, "error": ""}
    except loader.ValidationError as ve:
        log.error("validation error: %s", ve)
        return {"ok": False, **identity, "error": str(ve)}
    except Exception as e:
        log.exception("scenario failed")
        return {"ok": False, **identity, "error": str(e)}
    finally:
        unbind(tokens)


def _convert_one(py123d_data, output_dir, config) -> None:
    # 1. Load and convert 123D scenario to PufferDrive format
    scenario, extras = loader.extract_scenario(py123d_data)

    # 2. Validate scenario
    if config.validate_level > 0:
        errors = loader.validate_scenario(scenario, extras=extras, level=config.validate_level)
        scenario_id = scenario.metadata.id
        for error in errors:
            log.error(f"{scenario_id}: {error}")
        if errors:
            raise loader.ValidationError(f"Validation failed for scenario {scenario_id} with {len(errors)} errors")

    # 3. Process scenario
    if config.impute_tl:
        transforms.impute_traffic_lights(scenario, extras)  # Yan et al. 2025
    transforms.process_polylines(scenario, config.max_segment_length, config.area_threshold)
    transforms.interpolate_all_polygons(scenario)
    transforms.prune_invalid_map_elements(scenario, extras)
    transforms.process_traffic_controls(scenario, extras)
    transforms.process_agent_routes(scenario, config.min_route_valid_points, config.route_check_timestep)
    if config.invalid_agent_overlap:
        transforms.invalid_agent_overlap(scenario)
    transforms.compute_lane_lengths(scenario)
    scenario.lane_graph = transforms.build_lane_distance_matrix(scenario.map)
    if not config.no_reindex:
        transforms.reindex_scenario(scenario)

    # 4. Serialize to binary and save
    binary_data = serialize.scenario_to_binary(scenario)
    output_path = _build_output_path(py123d_data, output_dir)
    with output_path.open("wb") as f:
        f.write(binary_data)


def _validate_args(args, parser):
    for attr in ("datasets", "split_types", "split_names", "log_names", "scene_uuids"):
        values = getattr(args, attr)
        cleaned = [v.strip() for v in (values or []) if v and v.strip()] or None
        setattr(args, attr, cleaned)

    if args.num_scenes is not None and args.num_scenes < 0:
        parser.error("--num_scenes must be >= 0")
    if args.chunk_target_scenes <= 0:
        parser.error("--chunk_target_scenes must be > 0")
    if args.route_check_timestep < 0:
        parser.error("--route_check_timestep must be >= 0")
    if not 0 <= args.min_route_valid_points <= 100:
        parser.error("--min_route_valid_points must be between 0 and 100")
    if args.workers < -1:
        parser.error("--workers must be -1, 0, or a positive integer")

    cpu_count = os.cpu_count() or 1
    if args.workers == -1:
        args.workers = cpu_count
    elif args.workers == 0:
        args.workers = max(1, int(cpu_count * 0.8))

    if args.num_scenes not in (None, 0) and args.num_scenes < args.workers:
        log.warning("num_scenes (%d) < workers (%d). Reducing workers.", args.num_scenes, args.workers)
        args.workers = args.num_scenes

    if "opendrive" in (args.datasets or []) and not args.map_only:
        log.warning("Dataset 'opendrive' selected with --map_only=False. Forcing --map_only=True.")
        args.map_only = True

    py123d_data_root = args.py123d_path or os.environ.get("PY123D_DATA_ROOT")
    if not py123d_data_root:
        parser.error("--py123d_path is required (or set PY123D_DATA_ROOT environment variable)")
    return args, py123d_data_root


def main() -> int:
    parser = build_parser()
    args, py123d_data_root = _validate_args(parser.parse_args(), parser)

    log.setLevel(args.log_level)

    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)

    log.info("123Drive: %s -> %s", py123d_data_root, args.output)
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
        py123d_data_root=py123d_data_root,
        workers=args.workers,
        num_scenes=args.num_scenes,
        datasets=args.datasets,
        split_types=args.split_types,
        split_names=args.split_names,
        log_names=args.log_names,
        scene_uuids=args.scene_uuids,
        duration_s=None if args.duration_s == 0.0 else args.duration_s,
        map_only=args.map_only,
    )
    if not scenes:
        log.info("No scenarios to process.")
        return 0

    output_dir = pathlib.Path(args.output)
    failures_path = output_dir / "failures.jsonl"

    log.info("Discovered %d scenarios. Starting conversion with %d workers.", len(scenes), args.workers)

    # Suppress the loky pool teardown warning that fires on normal completion.
    warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given")

    succeeded = 0
    failed = 0

    with (
        failures_path.open("w", encoding="utf-8") as failure_handle,
        tqdm.tqdm(total=len(scenes)) as pbar,
        joblib.Parallel(
            n_jobs=args.workers,
            backend="loky",
            batch_size="auto",
            return_as="generator_unordered",
        ) as parallel,
    ):
        for start in range(0, len(scenes), args.chunk_target_scenes):
            chunk = scenes[start : start + args.chunk_target_scenes]
            for result in parallel(
                joblib.delayed(_worker_fn)(data, output_dir=output_dir, config=args) for data in chunk
            ):
                if result["ok"]:
                    succeeded += 1
                else:
                    failed += 1
                    failure_handle.write(json.dumps(result) + "\n")
                    failure_handle.flush()
                pbar.update(1)

    log.info("Conversion complete. %d/%d succeeded, %d failed.", succeeded, succeeded + failed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
