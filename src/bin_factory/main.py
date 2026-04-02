import argparse
import logging
import os
import pathlib
import re
import warnings
from typing import NamedTuple

import joblib
import tqdm

from bin_factory import loader, serialize, transforms


class ConvertConfig(NamedTuple):
    max_segment_length: float
    area_threshold: float
    min_route_valid_points: int
    route_check_timestep: int
    no_reindex: bool
    impute_tl: bool
    validate_level: int


logger = logging.getLogger(__name__)


def build_parser():
    parser = argparse.ArgumentParser(description="Convert 123D datasets to PufferDrive binary format")

    parser.add_argument("--py123d_path", type=str, help="Path to py123d dataset (logs/ and maps/)")
    parser.add_argument("--output", default="./output", help="Directory to save binary files")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel workers - "
            "0: 80%% of CPU cores, 1: no parallelism, -1: all CPU cores, 2-n: use n CPU cores"
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
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--fail_fast", action="store_true", help="Stop on first error")
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
        type=int,
        default=0,
        help="Min valid trajectory points for route computation",
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
    return parser


def _build_output_path(py123d_data, output_dir):
    """Derive output path for a given scenario based on its metadata attributes.

    Arguments:
        py123d_data: py123d scenaio data object
        output_dir: Base directory to save the output file

    Returns:
        A pathlib.Path object representing the output file path, e.g. <dataset>__<source_id>.bin"
    """

    def sanitize(v):
        return re.sub(r"[^A-Za-z0-9._-]+", "_", v.strip()).strip("._-") or "scenario"

    attrs = ("scene_uuid", "scenario_id", "log_name", "location", "map_id")

    source_id = ""
    for attr in attrs:
        if value := getattr(py123d_data, attr, None):
            source_id = str(value)
            break

    if not source_id:
        raise ValueError("Cannot derive scenario identity from py123d_data attributes: " + ", ".join(attrs))

    dataset = getattr(py123d_data, "dataset", "") or ""

    if not dataset:
        raise ValueError("py123d_data is missing 'dataset' attribute, which is required for output filename")

    stem = f"{sanitize(dataset)}__{sanitize(source_id)}"

    return output_dir / f"{stem}.bin"


def _convert_one(py123d_data, output_dir, config) -> None:
    # 1. Load and convert 123D scenario to PufferDrive format
    scenario, extras = loader.extract_scenario(py123d_data)

    # 2. Validate scenario
    if config.validate_level > 0:
        errors = loader.validate_scenario(scenario, extras=extras, level=config.validate_level)
        scenario_id = scenario.metadata.id
        for error in errors:
            logger.error(f"{scenario_id}: {error}")
        if errors:
            raise loader.ValidationError(f"Validation failed for scenario {scenario_id} with {len(errors)} errors")

    # 3. Process scenario
    transforms.rotate_body_to_global_velocity(scenario)  # body-frame → global (dataset-specific)
    if config.impute_tl:
        transforms.impute_traffic_lights(scenario, extras)  # Yan et al. 2025
    transforms.process_polylines(scenario, config.max_segment_length, config.area_threshold)
    transforms.interpolate_all_polygons(scenario)
    transforms.reverse_road_edges(scenario)
    transforms.process_traffic_controls(scenario, extras)
    transforms.process_agent_routes(scenario, config.min_route_valid_points, config.route_check_timestep)
    scenario.lane_graph = transforms.build_lane_distance_matrix(scenario.map)
    if not config.no_reindex:
        transforms.reindex_scenario(scenario)

    # 4. Serialize to binary and save
    binary_data = serialize.scenario_to_binary(scenario)
    output_path = _build_output_path(py123d_data, output_dir)
    with output_path.open("wb") as f:
        f.write(binary_data)


def _convert_one_safe(py123d_data, output_dir, config):
    scenario_id = getattr(py123d_data, "scenario_id", None) or getattr(py123d_data, "log_name", "unknown")
    dataset = getattr(py123d_data, "dataset", "unknown")
    try:
        _convert_one(py123d_data, output_dir, config)
        return {"ok": True, "scenario_id": scenario_id, "error": ""}
    except loader.ValidationError as ve:
        logger.error("[%s][%s] Validation error: %s", dataset, scenario_id, ve)
        return {"ok": False, "scenario_id": scenario_id, "error": str(ve)}
    except Exception as e:
        logger.exception("[%s][%s] Scenario failed", dataset, scenario_id)
        return {"ok": False, "scenario_id": scenario_id, "error": str(e)}


def _validate_args(args, parser):
    for attr in ("datasets", "split_types", "split_names", "log_names", "scene_uuids"):
        values = getattr(args, attr)
        cleaned = [v.strip() for v in (values or []) if v and v.strip()] or None
        setattr(args, attr, cleaned)

    if args.num_scenes is not None and args.num_scenes < 0:
        parser.error("--num_scenes must be >= 0")
    if args.route_check_timestep < 0:
        parser.error("--route_check_timestep must be >= 0")
    if args.workers < -1:
        parser.error("--workers must be -1, 0, or a positive integer")

    cpu_count = os.cpu_count() or 1
    if args.workers == -1:
        args.workers = cpu_count
    elif args.workers == 0:
        args.workers = max(1, int(cpu_count * 0.8))

    if args.num_scenes not in (None, 0) and args.num_scenes < args.workers:
        logger.warning("num_scenes (%d) < workers (%d). Reducing workers.", args.num_scenes, args.workers)
        args.workers = args.num_scenes

    if "opendrive" in (args.datasets or []) and not args.map_only:
        logger.warning("Dataset 'opendrive' selected with --map_only=False. Forcing --map_only=True.")
        args.map_only = True

    py123d_data_root = args.py123d_path or os.environ.get("PY123D_DATA_ROOT")
    if not py123d_data_root:
        parser.error("--py123d_path is required (or set PY123D_DATA_ROOT environment variable)")
    return args, py123d_data_root


def main() -> int:
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=logging.INFO)

    parser = build_parser()
    args, py123d_data_root = _validate_args(parser.parse_args(), parser)

    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)

    logger.info("123Drive: %s -> %s", py123d_data_root, args.output)
    logger.info(
        "Filters - datasets: %s, split_types: %s, split_names: %s, log_names: %s, duration_s: %s, map_only: %s",
        args.datasets,
        args.split_types,
        args.split_names,
        args.log_names,
        args.duration_s,
        args.map_only,
    )

    py123d_data = list(
        loader.load_scenes(
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
        ),
    )

    if not py123d_data:
        logger.info("No scenarios to process.")
        return 0

    output_dir = pathlib.Path(args.output)

    config = ConvertConfig(
        max_segment_length=args.max_segment_length,
        area_threshold=args.area_threshold,
        min_route_valid_points=args.min_route_valid_points,
        route_check_timestep=args.route_check_timestep,
        no_reindex=args.no_reindex,
        impute_tl=args.impute_tl,
        validate_level=args.validate_level,
    )

    logger.info(
        "Loaded %d scenarios after filtering. Starting conversion with %d workers",
        len(py123d_data),
        args.workers,
    )
    worker = _convert_one if args.fail_fast else _convert_one_safe

    warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given")
    with joblib.Parallel(n_jobs=args.workers) as parallel:
        results = list(
            parallel(
                joblib.delayed(worker)(data, output_dir=output_dir, config=config)
                for data in tqdm.tqdm(py123d_data, total=len(py123d_data))
            ),
        )

    if args.fail_fast:
        logger.info("Conversion complete. %d/%d succeeded.", len(py123d_data), len(py123d_data))
        return 0

    failed = [r for r in results if not r["ok"]]
    logger.info(
        "Conversion complete. %d/%d succeeded, %d failed.",
        len(results) - len(failed),
        len(results),
        len(failed),
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
