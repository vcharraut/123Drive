import glob
import json
import os

from joblib import Parallel, delayed
from tqdm import tqdm

from src import logger_utils
from src.encoder.pufferdrive import puffer_dict_to_binary


logger = logger_utils.get_logger(__name__)


def _stem(path: str) -> str:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    return stem


def _convert_one(json_path: str, output_dir: str):
    with open(json_path) as f:
        puffer_dict = json.load(f)

    map_id = puffer_dict.get("map_id", None)
    if map_id is None:
        # fallback: parse map_### from filename if present
        stem = _stem(json_path)
        if stem.startswith("map_"):
            try:
                map_id = int(stem.split("map_")[-1])
            except Exception:
                map_id = 0
        else:
            map_id = 0

    binary = puffer_dict_to_binary(puffer_dict, map_id=int(map_id))

    out_path = os.path.join(output_dir, f"{_stem(json_path)}.bin")
    with open(out_path, "wb") as f:
        f.write(binary)

    return out_path


def run(cfg):
    input_dir = cfg.json_to_bin.input_dir
    output_dir = cfg.json_to_bin.output_dir
    pattern = cfg.json_to_bin.glob

    os.makedirs(output_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not paths:
        raise ValueError(f"No JSON files matched: {os.path.join(input_dir, pattern)}")

    results = Parallel(n_jobs=int(cfg.num_workers))(
        delayed(_convert_one)(p, output_dir) for p in tqdm(paths, desc="json", unit="file")
    )

    logger.info(f"done: wrote {len(results)} bins to {output_dir}")
    return 0
