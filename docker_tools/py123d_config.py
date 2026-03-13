"""Shared dataset config for the py123d Docker tooling."""

INPUT_MOUNT = "/mnt/input"
OUTPUT_MOUNT = "/mnt/output"

_NO_SENSOR_OVERRIDES = [
    "force_map_conversion=true",
    "force_log_conversion=true",
    "execution=process_pool_executor",
    "scene_filter.shuffle=false",
    "dataset.dataset_converter_config.include_pinhole_cameras=false",
    "dataset.dataset_converter_config.include_lidars=false",
    "dataset.dataset_converter_config.include_fisheye_mei_cameras=false",
    "dataset.dataset_converter_config.include_route=false",
    "dataset.dataset_converter_config.include_scenario_tags=false",
]


def _cfg(extras, data_root_key, default_splits):
    return {
        "extras": extras,
        "hydra_dataset": None,
        "data_root_key": data_root_key,
        "input_subpaths": {},
        "default_splits": default_splits,
        "hydra_overrides": list(_NO_SENSOR_OVERRIDES),
    }


DATASET_CONFIGS = {
    "nuplan": {
        **_cfg(
            "nuplan",
            "nuplan_data_root",
            ["nuplan_train", "nuplan_val", "nuplan_test"],
        ),
        "input_subpaths": {
            "nuplan_maps_root": "maps",
            "nuplan_sensor_root": "nuplan-v1.1/sensor_blobs",
        },
        "hydra_dataset": "nuplan-mini",
    },
    "nuplan-mini": {
        **_cfg(
            "nuplan",
            "nuplan_data_root",
            ["nuplan-mini_train", "nuplan-mini_val", "nuplan-mini_test"],
        ),
        "input_subpaths": {
            "nuplan_maps_root": "maps",
            "nuplan_sensor_root": "nuplan-v1.1/sensor_blobs",
        },
    },
    "nuscenes": _cfg(
        "nuscenes",
        "nuscenes_data_root",
        ["nuscenes_train", "nuscenes_val"],
    ),
    "nuscenes-mini": _cfg(
        "nuscenes",
        "nuscenes_data_root",
        ["nuscenes-mini_train", "nuscenes-mini_val"],
    ),
    "nuscenes-interpolated": _cfg(
        "nuscenes",
        "nuscenes_data_root",
        ["nuscenes-interpolated_train", "nuscenes-interpolated_val"],
    ),
    "nuscenes-interpolated-mini": _cfg(
        "nuscenes",
        "nuscenes_data_root",
        ["nuscenes-interpolated-mini_train", "nuscenes-interpolated-mini_val"],
    ),
    "wod-motion": _cfg(
        "waymo",
        "wod_motion_data_root",
        ["wod-motion_train", "wod-motion_val"],
    ),
    "wod-perception": _cfg(
        "waymo",
        "wod_perception_data_root",
        ["wod-perception_val"],
    ),
    "av2-sensor": _cfg(
        "av2",
        "av2_data_root",
        ["av2-sensor_train"],
    ),
    "kitti360": _cfg(
        "kitti360",
        "kitti360_data_root",
        ["kitti360_train", "kitti360_val", "kitti360_test"],
    ),
    "pandaset": _cfg(
        "pandaset",
        "pandaset_data_root",
        ["pandaset_train", "pandaset_val", "pandaset_test"],
    ),
}
