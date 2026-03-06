import argparse
import os

from joblib import Parallel, delayed
from tqdm import tqdm

from src.bin_factory import logger_utils
from src.bin_factory.convert.pufferdrive import convert_to_puffer_dict, puffer_dict_to_binary
from src.bin_factory.loader.extractor import convert_py123d_scenario
from src.bin_factory.loader.load import get_py123d_scenarios
from src.bin_factory.transforms.polyline import process_polylines
from src.bin_factory.transforms.traffic_lights.processor import add_traffic_lights_to_scenario
from src.bin_factory.transforms.validation import validate_puffer_dict


logger = logger_utils.get_logger(__name__)


def process_one_scenario(
    raw_scenario,
    map_id,
    output_dir,
    traffic_lights=False,
    validate_level=0,
    max_segment_length=2.0,
    area_threshold=0.1,
    dist_threshold=10.0,
    min_route_valid_points=0,
    route_check_timestep=0,
):
    # try:
    # 1. Convert Raw -> Intermediate
    scenario = convert_py123d_scenario(raw_scenario)

    # 2. Apply transforms to Intermediate representation
    scenario = process_polylines(
        scenario,
        max_segment_length=max_segment_length,
        area_threshold=area_threshold,
        dist_threshold=dist_threshold,
    )

    if traffic_lights:
        scenario = add_traffic_lights_to_scenario(scenario)

    # 3. Convert Intermediate -> Puffer Dict
    puffer_dict = convert_to_puffer_dict(
        scenario,
        min_route_valid_points=min_route_valid_points,
        route_check_timestep=route_check_timestep,
    )

    # 4. Validate Puffer Dict
    errors = []
    warnings = []
    if validate_level > 0:
        errors, warnings = validate_puffer_dict(puffer_dict, validation_level=validate_level)
        for warning in warnings:
            logger.warning(f"map_{map_id}: {warning}")
        for error in errors:
            logger.error(f"map_{map_id}: {error}")

    if len(errors) > 0:
        scenario_id = puffer_dict.get('scenario_id', 'unknown')
        raise ValueError(
            f"Validation failed for scenario {scenario_id} "
            f"with {len(errors)} errors and {len(warnings)} warnings",
        )

    # 5. Convert Puffer Dict -> Binary
    binary_data = puffer_dict_to_binary(puffer_dict, map_id=map_id)

    # 6. Write to file
    output_path = os.path.join(output_dir, f"map_{map_id:03d}.bin")
    with open(output_path, "wb") as f:
        f.write(binary_data)
    # except Exception as e:
    #     logger.error(f"Error processing scenario map_{map_id}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert py123d datasets to PufferDrive binary format")

    # Core arguments
    parser.add_argument("--dataset_path", required=True, help="Path to py123d dataset (logs/ and maps/)")
    parser.add_argument("--output_dir", default="./output", help="Directory to save binary files")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")

    # Dataset filtering
    parser.add_argument(
        "--max_scenarios",
        type=int,
        default=None,
        help="Maximum number of scenarios to process",
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

    # Processor flags
    parser.add_argument(
        "--traffic_lights",
        action="store_true",
        help="Generate synthetic traffic lights",
    )
    parser.add_argument(
        "--validate_level",
        type=int,
        default=1,
        help=(
            "Validation level "
            "(0 = off, 1 = mandatory schema, 2 = physics warnings, 3 = reject hard physics, 4 = reject all physics)"
        ),
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

    args = parser.parse_args()

    logger_utils.setup_logger()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Starting conversion: {args.dataset_path} -> {args.output_dir}")

    scenarios = get_py123d_scenarios(
        dataset_path=args.dataset_path,
        max_scenarios=args.max_scenarios,
        datasets=args.datasets,
        split_types=args.split_types,
        split_names=args.split_names,
        log_names=args.log_names,
        history_s=args.history_s,
        duration_s=None if args.duration_s == 0.0 else args.duration_s,
        map_only=args.map_only,
    )

    scenarios = list(scenarios)

    with Parallel(n_jobs=args.num_workers) as parallel:
        parallel(
            delayed(process_one_scenario)(
                raw_scenario=scenario,
                map_id=i,
                output_dir=args.output_dir,
                traffic_lights=args.traffic_lights,
                validate_level=args.validate_level,
                max_segment_length=args.max_segment_length,
                area_threshold=args.area_threshold,
                dist_threshold=args.dist_threshold,
                min_route_valid_points=args.min_route_valid_points,
                route_check_timestep=args.route_check_timestep,
            )
            for i, scenario in tqdm(enumerate(scenarios), total=len(scenarios))
        )

    logger.info("Conversion complete.")


if __name__ == "__main__":
    main()
