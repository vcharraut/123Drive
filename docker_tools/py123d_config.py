"""Shared dataset config for the py123d Docker tooling."""

from importlib import resources

import yaml


INPUT_MOUNT = "/input"
OUTPUT_MOUNT = "/output"

HYDRA_OVERRIDES = [
    "force_map_conversion=true",
    "force_log_conversion=true",
    "scene_filter.shuffle=false",
    "dataset.dataset_converter_config.include_pinhole_cameras=false",
    "dataset.dataset_converter_config.include_lidars=false",
    "dataset.dataset_converter_config.include_fisheye_mei_cameras=false",
    "dataset.dataset_converter_config.include_route=false",
    "dataset.dataset_converter_config.include_scenario_tags=false",
]

DATASET_CONFIGS = {
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


def get_default_splits(dataset):
    cfg_path = resources.files("py123d.script.config.conversion.dataset").joinpath(f"{dataset}.yaml")
    if not cfg_path.is_file():
        return []
    return yaml.safe_load(cfg_path.read_text()).get("parser", {}).get("splits", [])
