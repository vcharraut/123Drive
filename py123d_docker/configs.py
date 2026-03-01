def _SENSOR_DISABLE(ds, fisheye=True):
    overrides = [
        f"datasets.{ds}.dataset_converter_config.include_pinhole_cameras=false",
        f"datasets.{ds}.dataset_converter_config.include_lidars=false",
    ]
    if fisheye:
        overrides.append(f"datasets.{ds}.dataset_converter_config.include_fisheye_mei_cameras=false")
    return overrides

_NUPLAN_DEVKIT = "nuplan-devkit @ git+https://github.com/motional/nuplan-devkit/@nuplan-devkit-v1.2"

DATASET_CONFIGS = {
    "nuplan-mini": {
        "extras": "nuplan",
        "devkit": _NUPLAN_DEVKIT,
        "python_version": "3.10",
        "path_keys": {
            "nuplan_data_root": "nuplan/",
            "nuplan_maps_root": "nuplan/maps",
            "nuplan_sensor_root": "nuplan/sensor_blobs",
        },
        "sensor_overrides": _SENSOR_DISABLE("nuplan-mini"),
        "default_splits": ["nuplan-mini_train", "nuplan-mini_val", "nuplan-mini_test"],
    },
    "nuplan": {
        "extras": "nuplan",
        "devkit": _NUPLAN_DEVKIT,
        "python_version": "3.10",
        "path_keys": {
            "nuplan_data_root": "nuplan/",
            "nuplan_maps_root": "nuplan/maps",
            "nuplan_sensor_root": "nuplan/sensor_blobs",
        },
        "sensor_overrides": _SENSOR_DISABLE("nuplan"),
        "default_splits": ["nuplan_train", "nuplan_val", "nuplan_test"],
    },
    "wod-motion": {
        "extras": "waymo",
        "devkit": None,
        "path_keys": {
            "wod_motion_data_root": "waymo_open_motion",
        },
        "sensor_overrides": _SENSOR_DISABLE("wod-motion", fisheye=False),
        "default_splits": ["wod-motion_train", "wod-motion_val", "wod-motion_test"],
    },
    "wod-perception": {
        "extras": "waymo",
        "devkit": None,
        "path_keys": {
            "wod_perception_data_root": "waymo_open_perception",
        },
        "sensor_overrides": _SENSOR_DISABLE("wod-perception", fisheye=False),
        "default_splits": ["wod-perception_train", "wod-perception_val", "wod-perception_test"],
    },
    "nuscenes": {
        "extras": "nuscenes",
        "devkit": None,
        "path_keys": {
            "nuscenes_data_root": "nuscenes",
        },
        "sensor_overrides": _SENSOR_DISABLE("nuscenes"),
        "default_splits": ["nuscenes_train", "nuscenes_val", "nuscenes_test"],
    },
    "nuscenes-mini": {
        "extras": "nuscenes",
        "devkit": None,
        "path_keys": {
            "nuscenes_data_root": "nuscenes",
        },
        "sensor_overrides": _SENSOR_DISABLE("nuscenes-mini"),
        "default_splits": ["nuscenes-mini_train", "nuscenes-mini_val"],
    },
    "nuscenes-interpolated": {
        "extras": "nuscenes",
        "devkit": None,
        "path_keys": {
            "nuscenes_data_root": "nuscenes",
        },
        "sensor_overrides": _SENSOR_DISABLE("nuscenes-interpolated"),
        "default_splits": ["nuscenes-interpolated_train", "nuscenes-interpolated_val", "nuscenes-interpolated_test"],
    },
    "nuscenes-interpolated-mini": {
        "extras": "nuscenes",
        "devkit": None,
        "path_keys": {
            "nuscenes_data_root": "nuscenes",
        },
        "sensor_overrides": _SENSOR_DISABLE("nuscenes-interpolated-mini"),
        "default_splits": ["nuscenes-interpolated-mini_train", "nuscenes-interpolated-mini_val"],
    },
    "av2-sensor": {
        "extras": "av2",
        "devkit": None,
        "path_keys": {
            "av2_data_root": "av2",
        },
        "sensor_overrides": _SENSOR_DISABLE("av2-sensor"),
        "default_splits": ["av2-sensor_train", "av2-sensor_val", "av2-sensor_test"],
    },
    "kitti360": {
        "extras": "kitti360",
        "devkit": None,
        "path_keys": {
            "kitti360_data_root": "kitti360",
        },
        "sensor_overrides": _SENSOR_DISABLE("kitti360"),
        "default_splits": ["kitti360_train", "kitti360_val", "kitti360_test"],
    },
    "pandaset": {
        "extras": "pandaset",
        "devkit": None,
        "path_keys": {
            "pandaset_data_root": "pandaset",
        },
        "sensor_overrides": _SENSOR_DISABLE("pandaset"),
        "default_splits": ["pandaset_train", "pandaset_val", "pandaset_test"],
    },
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
