import argparse
import os
import sys

from joblib import Parallel, delayed
from tqdm import tqdm

from src import logger_utils
from src.encoder.pufferdrive import convert_to_puffer_dict, puffer_dict_to_binary
from src.loader.extractor import convert_py123d_scenario
from src.loader.load import get_py123d_scenarios
from src.processors.polyline_interpolation.processor import interpolate_polylines
from src.processors.traffic_lights.processor import add_traffic_lights_to_scenario


logger = logger_utils.get_logger(__name__)


def process_one_scenario(
    raw_scenario,
    map_id,
    output_dir,
    interpolate=False,
    traffic_lights=False,
    max_segment_length=2.0,
    polyline_reduction_threshold=0.1,
    dist_threshold=10.0,
    min_route_valid_points=0,
    route_check_timestep=0,
):
    try:
        # 1. Convert Raw -> Intermediate
        scenario = convert_py123d_scenario(raw_scenario)

        # 2. Apply Processors
        if interpolate:
            scenario = interpolate_polylines(scenario, max_segment_length=max_segment_length)

        if traffic_lights:
            scenario = add_traffic_lights_to_scenario(scenario)

        # 3. Convert Intermediate -> Puffer Dict
        puffer_dict = convert_to_puffer_dict(
            scenario,
            polyline_reduction_threshold=polyline_reduction_threshold,
            dist_threshold=dist_threshold,
            min_route_valid_points=min_route_valid_points,
            route_check_timestep=route_check_timestep,
        )

        # 4. Convert Puffer Dict -> Binary
        binary_data = puffer_dict_to_binary(puffer_dict, map_id=map_id)

        # 5. Write to file
        output_path = os.path.join(output_dir, f"map_{map_id:03d}.bin")
        with open(output_path, "wb") as f:
            f.write(binary_data)

        return {"status": "ok", "map_id": map_id}

    except Exception as e:
        return {"status": "error", "map_id": map_id, "error": str(e)}


def chunk_iterator(iterator, chunk_size):
    chunk = []
    for item in iterator:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def main():
    parser = argparse.ArgumentParser(description="Convert py123d datasets to PufferDrive binary format")

    # Core arguments
    parser.add_argument("--dataset_path", required=True, help="Path to py123d dataset (logs/ and maps/)")
    parser.add_argument("--output_dir", default="./output", help="Directory to save binary files")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--max_scenarios", type=int, default=None, help="Maximum number of scenarios to process")

    # Dataset filtering
    parser.add_argument("--map_only", action="store_true", help="Load map-only scenarios (no logs)")
    parser.add_argument("--history_s", type=float, default=0.0, help="History duration in seconds")

    # Processor flags
    parser.add_argument("--interpolate", action="store_true", help="Enable polyline interpolation")
    parser.add_argument("--traffic_lights", action="store_true", help="Generate synthetic traffic lights")

    # Configuration parameters
    parser.add_argument("--max_segment_length", type=float, default=2.0, help="Max segment length for interpolation")
    parser.add_argument("--polyline_reduction_threshold", type=float, default=0.1, help="Polyline reduction threshold")
    parser.add_argument("--dist_threshold", type=float, default=10.0, help="Distance threshold for road graph")

    args = parser.parse_args()

    logger_utils.setup_logger()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Starting conversion: {args.dataset_path} -> {args.output_dir}")

    # Get scenarios iterator
    try:
        scenarios_iter = iter(get_py123d_scenarios(args.dataset_path, map_only=args.map_only, history_s=args.history_s))
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    map_id = 0
    errors = 0
    processed = 0

    # Process in batches
    for batch in tqdm(chunk_iterator(scenarios_iter, args.batch_size), desc="Processing batches"):
        if args.max_scenarios and map_id >= args.max_scenarios:
            break

        tasks = []
        for raw_scenario in batch:
            if args.max_scenarios and map_id >= args.max_scenarios:
                break

            tasks.append((raw_scenario, map_id))
            map_id += 1

        results = Parallel(n_jobs=args.num_workers)(
            delayed(process_one_scenario)(
                raw,
                mid,
                args.output_dir,
                interpolate=args.interpolate,
                traffic_lights=args.traffic_lights,
                max_segment_length=args.max_segment_length,
                polyline_reduction_threshold=args.polyline_reduction_threshold,
                dist_threshold=args.dist_threshold,
            )
            for raw, mid in tasks
        )

        for res in results:
            if res["status"] == "error":
                errors += 1
                logger.error(f"Error processing map_{res['map_id']}: {res['error']}")
            else:
                processed += 1

    logger.info(f"Conversion complete. Processed: {processed}, Errors: {errors}")


if __name__ == "__main__":
    main()
