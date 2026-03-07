def _sensor_overrides(include_fisheye=True):
    overrides = [
        "dataset.dataset_converter_config.include_pinhole_cameras=false",
        "dataset.dataset_converter_config.include_lidars=false",
    ]
    if include_fisheye:
        overrides.append("dataset.dataset_converter_config.include_fisheye_mei_cameras=false")
    return overrides


def _dataset_config(extras: str, path_keys: dict, default_splits: list[str], include_fisheye=True):
    return {
        "extras": extras,
        "path_keys": path_keys,
        "sensor_overrides": _sensor_overrides(include_fisheye),
        "default_splits": default_splits,
    }


DATASET_CONFIGS = {
    "nuplan-mini": _dataset_config(
        "nuplan",
        {
            "nuplan_data_root": "nuplan/",
            "nuplan_maps_root": "nuplan/maps",
            "nuplan_sensor_root": "nuplan/sensor_blobs",
        },
        ["nuplan-mini_train", "nuplan-mini_val", "nuplan-mini_test"],
    ),
    "nuplan": _dataset_config(
        "nuplan",
        {
            "nuplan_data_root": "nuplan/",
            "nuplan_maps_root": "nuplan/maps",
            "nuplan_sensor_root": "nuplan/sensor_blobs",
        },
        ["nuplan_train", "nuplan_val", "nuplan_test"],
    ),
    "wod-motion": _dataset_config(
        "waymo",
        {
            "wod_motion_data_root": "waymo_open_motion",
        },
        ["wod-motion_train", "wod-motion_val", "wod-motion_test"],
        include_fisheye=False,
    ),
    "wod-perception": _dataset_config(
        "waymo",
        {
            "wod_perception_data_root": "waymo_open_perception",
        },
        ["wod-perception_train", "wod-perception_val", "wod-perception_test"],
        include_fisheye=False,
    ),
    "nuscenes": _dataset_config(
        "nuscenes",
        {
            "nuscenes_data_root": "nuscenes",
        },
        ["nuscenes_train", "nuscenes_val", "nuscenes_test"],
    ),
    "nuscenes-mini": _dataset_config(
        "nuscenes",
        {
            "nuscenes_data_root": "nuscenes",
        },
        ["nuscenes-mini_train", "nuscenes-mini_val"],
    ),
    "nuscenes-interpolated": _dataset_config(
        "nuscenes",
        {
            "nuscenes_data_root": "nuscenes",
        },
        ["nuscenes-interpolated_train", "nuscenes-interpolated_val", "nuscenes-interpolated_test"],
    ),
    "nuscenes-interpolated-mini": _dataset_config(
        "nuscenes",
        {
            "nuscenes_data_root": "nuscenes",
        },
        ["nuscenes-interpolated-mini_train", "nuscenes-interpolated-mini_val"],
    ),
    "av2-sensor": _dataset_config(
        "av2",
        {
            "av2_data_root": "av2",
        },
        ["av2-sensor_train", "av2-sensor_val", "av2-sensor_test"],
    ),
    "kitti360": _dataset_config(
        "kitti360",
        {
            "kitti360_data_root": "kitti360",
        },
        ["kitti360_train", "kitti360_val", "kitti360_test"],
    ),
    "pandaset": _dataset_config(
        "pandaset",
        {
            "pandaset_data_root": "pandaset",
        },
        ["pandaset_train", "pandaset_val", "pandaset_test"],
    ),
}

DATA_LAYOUT = """
data_root/
├── nuplan/dataset/
├── nuplan/maps/
├── nuplan/sensor_blobs/
├── waymo_open_motion/
├── waymo_open_perception/
├── nuscenes/
├── av2/
├── kitti360/
└── pandaset/
"""
