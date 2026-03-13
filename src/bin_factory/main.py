import argparse
import os
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm

from bin_factory import logger_utils
from bin_factory.convert.pufferdrive import convert_to_puffer_dict
from bin_factory.loader.extractor import convert_py123d_data
from bin_factory.loader.load import get_py123d_data
from bin_factory.serialize import puffer_dict_to_binary
from bin_factory.transforms.polyline import process_polylines
from bin_factory.transforms.validation import ValidationError, validate_puffer_dict


logger = logger_utils.get_logger(__name__)


def _worker_fn(
    py123d_data,
    map_id,
    output,
    validate_level=0,
    max_segment_length=2.0,
    area_threshold=0.1,
    dist_threshold=10.0,
    min_route_valid_points=0,
    route_check_timestep=0,
    reindex_id=False,
):
    py123d_dict = convert_py123d_data(py123d_data)
    py123d_dict = process_polylines(
        py123d_dict,
        max_segment_length=max_segment_length,
        area_threshold=area_threshold,
        dist_threshold=dist_threshold,
    )
    puffer_dict = convert_to_puffer_dict(
        py123d_dict,
        min_route_valid_points=min_route_valid_points,
        route_check_timestep=route_check_timestep,
        reindex_id=reindex_id,
    )
    errors = []
    if validate_level > 0:
        validation_mode = {0: "off", 1: "schema", 2: "semantic"}[validate_level]
        errors = validate_puffer_dict(puffer_dict, validation_mode=validation_mode)
        scenario_id = puffer_dict.get("scenario_id", "unknown")
        for error in errors:
            logger.error(f"{scenario_id}: {error}")

    if errors:
        scenario_id = puffer_dict.get("scenario_id", "unknown")
        raise ValidationError(f"Validation failed for scenario {scenario_id} with {len(errors)} errors")

    binary_data = puffer_dict_to_binary(puffer_dict, map_id=map_id)
    output_path = Path(output) / f"map_{map_id:03d}.bin"
    with output_path.open("wb") as f:
        f.write(binary_data)


def _safe_process(py123d_data, **kwargs):
    scenario_id = getattr(py123d_data, "scenario_id", None) or getattr(py123d_data, "log_name", "unknown")
    try:
        return _worker_fn(py123d_data, **kwargs)
    except ValidationError as ve:
        logger.error(f"[{scenario_id}] Validation error: {ve}")
        return None
    except Exception as e:
        logger.exception(f"[{scenario_id}] Scenario failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert 123D datasets to PufferDrive binary format")

    # Core arguments
    parser.add_argument("--py123d_path", type=str, help="Path to py123d dataset (logs/ and maps/)")
    parser.add_argument("--output", default="./output", help="Directory to save binary files")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers - 0: 80%% of CPU cores, 1: no parallelism, -1: all CPU cores, 2-n: use n CPU cores",  # noqa: E501
    )

    # Dataset filtering
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=None,
        help="Maximum number of scenes to process",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Dataset names to include (e.g. nuplan, wod-motion)",
    )
    parser.add_argument(
        "--split_types",
        nargs="+",
        help="Split types to include (e.g. train, val, test)",
    )
    parser.add_argument(
        "--split_names",
        nargs="+",
        help="Split names to include (e.g. nuplan-mini_val)",
    )
    parser.add_argument(
        "--log_names",
        nargs="+",
        help="Log names to include",
    )
    parser.add_argument(
        "--duration_s",
        type=float,
        default=0.0,
        help="Duration of scenario in seconds",
    )
    parser.add_argument(
        "--history_s",
        type=float,
        default=0.0,
        help="History duration in seconds",
    )
    parser.add_argument(
        "--map_only",
        action="store_true",
        help="Load map-only scenarios (no logs)",
    )

    parser.add_argument("--fail_fast", action="store_true", help="Stop on first error")

    parser.add_argument(
        "--validate_level",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Validation level (0 = off, 1 = schema, 2 = semantic)",
    )

    # Configuration parameters
    parser.add_argument(
        "--max_segment_length",
        type=float,
        default=2.0,
        help="Max segment length for interpolation",
    )
    parser.add_argument(
        "--area_threshold",
        type=float,
        default=0.1,
        help="Area threshold for polyline simplification (0 = disabled)",
    )
    parser.add_argument(
        "--dist_threshold",
        type=float,
        default=10.0,
        help="Distance threshold for road graph",
    )
    parser.add_argument(
        "--min_route_valid_points",
        type=int,
        default=0,
        help="Min valid trajectory points for route computation (0 = no filter)",
    )
    parser.add_argument(
        "--route_check_timestep",
        type=int,
        default=0,
        help="Timestep at which agent must be valid for route computation",
    )
    parser.add_argument(
        "--reindex_id",
        action="store_true",
        help="Reindex all element IDs to contiguous range(0, n)",
    )

    args = parser.parse_args()

    args.datasets = _normalize_optional_list(args.datasets)
    args.split_types = _normalize_optional_list(args.split_types)
    args.split_names = _normalize_optional_list(args.split_names)
    args.log_names = _normalize_optional_list(args.log_names)

    py123d_data_root = args.py123d_path or os.environ.get("PY123D_DATA_ROOT")
    if not py123d_data_root:
        parser.error("--py123d_path is required (or set PY123D_DATA_ROOT environment variable)")

    logger_utils.setup_logger()
    Path(args.output).mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting conversion: {py123d_data_root} -> {args.output}")

    if "opendrive" in args.datasets and not args.map_only:
        logger.warning("Dataset 'opendrive' selected with --map_only=False. Forcing --map_only=True.")
        args.map_only = True

    logger.info(
        f"Loading scenarios with filters - datasets: {args.datasets}, split_types: {args.split_types}, "
        f"split_names: {args.split_names}, log_names: {args.log_names}, "
        f"duration_s: {args.duration_s}, history_s: {args.history_s}, map_only: {args.map_only}"
    )

    py123d_data = get_py123d_data(
        py123d_data_root=py123d_data_root,
        num_scenes=args.num_scenes,
        datasets=args.datasets,
        split_types=args.split_types,
        split_names=args.split_names,
        log_names=args.log_names,
        history_s=args.history_s,
        duration_s=None if args.duration_s == 0.0 else args.duration_s,
        map_only=args.map_only,
    )

    logger.info(f"Loaded {len(py123d_data)} scenarios after filtering")

    py123d_data = list(py123d_data)
    func = _worker_fn if args.fail_fast else _safe_process

    # Handle workers=0 as 80% of CPU cores
    if args.workers == 0:
        args.workers = max(1, int(os.cpu_count() * 0.8))

    # Warn if number of scenes is less than workers and adjust workers accordingly
    if args.num_scenes < args.workers:
        logger.warning(
            f"Number of scenes ({args.num_scenes}) is less than number of workers "
            f"({args.workers}). Reducing workers to {args.num_scenes}."
        )
        args.workers = args.num_scenes

    logger.info(f"Processing {len(py123d_data)} scenarios with {args.workers} workers")

    with Parallel(n_jobs=args.workers) as parallel:
        parallel(
            delayed(func)(
                py123d_data=data,
                map_id=i,
                output=args.output,
                validate_level=args.validate_level,
                max_segment_length=args.max_segment_length,
                area_threshold=args.area_threshold,
                dist_threshold=args.dist_threshold,
                min_route_valid_points=args.min_route_valid_points,
                route_check_timestep=args.route_check_timestep,
                reindex_id=args.reindex_id,
            )
            for i, data in tqdm(enumerate(py123d_data), total=len(py123d_data))
        )

    logger.info("Conversion complete.")


def _normalize_optional_list(values: list[str] | None) -> list[str] | None:
    """Normalize optional list arguments by stripping whitespace and filtering out empty values.

    Used for debugging command-line arguments.

    Args:
        values: List of strings or None.

    Returns:
        Cleaned list of strings or None if input was None or empty after cleaning.

    """
    if not values:
        return None
    cleaned = [v.strip() for v in values if v and v.strip()]
    return cleaned or None


if __name__ == "__main__":
    main()
