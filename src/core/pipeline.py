import os

from joblib import Parallel, delayed
from tqdm import tqdm

from src import logger_utils
from src.core.json_utils import to_jsonable
from src.encoder.pufferdrive import puffer_dict_to_binary, unified_to_puffer_dict
from src.loader import convert_py123d_scenario, get_py123d_scenarios
from src.processors import apply_processors
from src.processors.validation.processor import validate_puffer_scenario


logger = logger_utils.get_logger(__name__)


def _chunk(iterable, chunk_size: int):
    chunk = []
    for x in iterable:
        chunk.append(x)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _process_one(
    raw,
    map_id: int,
    convert_func,
    processor_names,
    processor_cfgs,
    puffer_cfg,
    output_dir: str,
    output_format: str,
    json_indent,
    run_validation: bool,
    validation_cfg: dict,
):
    try:
        scenario = convert_func(raw)

        # Run non-validation processors on unified format
        non_validation_processors = [p for p in processor_names if p != "validation"]
        if non_validation_processors:
            scenario = apply_processors(scenario, non_validation_processors, processor_cfgs)

        # Convert to puffer format
        puffer_dict = unified_to_puffer_dict(
            scenario,
            polyline_reduction_threshold=float(puffer_cfg.polyline_reduction_threshold),
            dist_threshold=float(puffer_cfg.dist_threshold),
            min_route_valid_points=int(puffer_cfg.min_route_valid_points),
            route_check_timestep=int(puffer_cfg.route_check_timestep),
        )

        # Run validation on puffer_dict if requested
        if run_validation:
            validate_puffer_scenario(puffer_dict, **validation_cfg)

        if output_format in ("bin", "both"):
            out = os.path.join(output_dir, f"map_{map_id:03d}.bin")
            binary = puffer_dict_to_binary(puffer_dict, map_id=map_id)
            with open(out, "wb") as f:
                f.write(binary)

        if output_format in ("json", "both"):
            import json

            out = os.path.join(output_dir, f"map_{map_id:03d}.json")
            payload = {"map_id": map_id, **puffer_dict}
            with open(out, "w") as f:
                json.dump(to_jsonable(payload), f, indent=json_indent)

        return {"map_id": map_id, "status": "ok"}
    except Exception as e:
        return {"map_id": map_id, "status": "error", "error": f"{type(e).__name__}: {e}"}


def run(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)

    processor_names = list(cfg.processors) if cfg.processors else []
    processor_cfgs = {name: dict(cfg.get(name) or {}) for name in processor_names}

    # Check if validation is requested
    run_validation = "validation" in processor_names
    validation_cfg = dict(cfg.get("validation") or {}) if run_validation else {}

    puffer_cfg = cfg.pufferdrive

    output_format = (cfg.output.format or "bin").lower()
    if output_format not in ("bin", "json", "both"):
        raise ValueError(f"Unsupported output.format: {output_format} (supported: bin|json|both)")
    json_indent = cfg.output.json_indent

    max_scenarios = cfg.get("max_scenarios", None)
    max_scenarios = int(max_scenarios) if max_scenarios is not None else None

    num_files = getattr(cfg, "num_files", None)
    py123d_cfg = cfg.get("py123d", None)
    py123d_args = dict(py123d_cfg) if py123d_cfg else {}
    if num_files is not None:
        py123d_args.setdefault("num_files", num_files)
    raw_iter = iter(get_py123d_scenarios(cfg.dataset_path, **py123d_args))
    convert_func = convert_py123d_scenario

    map_id = 0
    errors = 0
    filtered = 0

    for batch in tqdm(_chunk(raw_iter, int(cfg.batch_size)), desc="scenarios"):
        args = []
        for raw in batch:
            if max_scenarios is not None and map_id >= max_scenarios:
                break
            args.append((raw, map_id))
            map_id += 1

        if not args:
            break

        results = Parallel(n_jobs=int(cfg.num_workers))(
            delayed(_process_one)(
                raw=raw,
                map_id=mid,
                convert_func=convert_func,
                processor_names=processor_names,
                processor_cfgs=processor_cfgs,
                puffer_cfg=puffer_cfg,
                output_dir=str(cfg.output_dir),
                output_format=output_format,
                json_indent=json_indent,
                run_validation=run_validation,
                validation_cfg=validation_cfg,
            )
            for raw, mid in args
        )

        for r in results:
            if r["status"] == "filtered":
                filtered += 1
                continue
            if r["status"] == "error":
                errors += 1
                continue

    logger.info(f"done: wrote {map_id - errors - filtered} items, errors={errors}, filtered={filtered}")
