"""Shared dataset config for the py123d Docker tooling."""

from importlib import resources
from typing import NotRequired, TypedDict

import yaml


INPUT_MOUNT = "/input"
OUTPUT_MOUNT = "/output"

HYDRA_OVERRIDES = [
    "dataset.log_writer_config.force_log_conversion=true",
    "dataset.map_writer.force_map_conversion=true",
    "scene_filter.shuffle=false",
    "+dataset.log_writer_config.exclude_modality_types=[camera,lidar,custom]",  # NOTE: We shouldn't make the +
]

class DatasetConfig(TypedDict):
    extras: str
    data_root_key: str
    input_subpaths: NotRequired[dict[str, str]]
    python_version: NotRequired[str]


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "nuplan": {
        "extras": "nuplan",
        "data_root_key": "nuplan_data_root",
        "input_subpaths": {
            "nuplan_maps_root": "maps",
        },
    },
    "nuplan-mini": {
        "extras": "nuplan",
        "data_root_key": "nuplan_data_root",
        "input_subpaths": {
            "nuplan_maps_root": "maps",
        },
    },
    "wod-motion": {
        "extras": "waymo",
        "data_root_key": "wod_motion_data_root",
    },
    "av2-sensor": {
        "extras": "av2",
        "data_root_key": "av2_data_root",
    },
}


def get_default_splits(dataset: str) -> list[str]:
    cfg_path = resources.files("py123d.script.config.conversion.dataset").joinpath(f"{dataset}.yaml")
    if not cfg_path.is_file():
        return []
    raw_config = yaml.safe_load(cfg_path.read_text())
    if not isinstance(raw_config, dict):
        return []

    parser_config = raw_config.get("parser")
    if not isinstance(parser_config, dict):
        return []

    splits = parser_config.get("splits")
    if not isinstance(splits, list):
        return []

    return [str(split) for split in splits]
