def _sensor_overrides(include_fisheye=True):
    overrides = [
        "dataset.dataset_converter_config.include_pinhole_cameras=false",
        "dataset.dataset_converter_config.include_lidars=false",
    ]
    if include_fisheye:
        overrides.append("dataset.dataset_converter_config.include_fisheye_mei_cameras=false")
    return overrides


def _dataset_config(
    extras,
    data_root_key,
    default_splits,
    extra_paths=None,
    include_fisheye=True,
):
    return {
        "extras": extras,
        "data_root_key": data_root_key,
        "extra_paths": extra_paths or {},
        "sensor_overrides": _sensor_overrides(include_fisheye),
        "default_splits": default_splits,
    }


DATASET_CONFIGS = {
    "opendrive": _dataset_config(
        "opendrive",
        "xodr_paths",
        None,
    ),
    "nuplan-mini": _dataset_config(
        "nuplan",
        "nuplan_data_root",
        ["nuplan-mini_train", "nuplan-mini_val", "nuplan-mini_test"],
        extra_paths={"nuplan_maps_root": "maps", "nuplan_sensor_root": "sensor_blobs"},
    ),
    "nuplan": _dataset_config(
        "nuplan",
        "nuplan_data_root",
        ["nuplan_train", "nuplan_val", "nuplan_test"],
        extra_paths={"nuplan_maps_root": "maps", "nuplan_sensor_root": "sensor_blobs"},
    ),
    "wod-motion": _dataset_config(
        "waymo",
        "wod_motion_data_root",
        ["wod-motion_train", "wod-motion_val", "wod-motion_test"],
        include_fisheye=False,
    ),
    "wod-perception": _dataset_config(
        "waymo",
        "wod_perception_data_root",
        ["wod-perception_train", "wod-perception_val", "wod-perception_test"],
        include_fisheye=False,
    ),
    "nuscenes": _dataset_config(
        "nuscenes",
        "nuscenes_data_root",
        ["nuscenes_train", "nuscenes_val", "nuscenes_test"],
    ),
    "nuscenes-mini": _dataset_config(
        "nuscenes",
        "nuscenes_data_root",
        ["nuscenes-mini_train", "nuscenes-mini_val"],
    ),
    "nuscenes-interpolated": _dataset_config(
        "nuscenes",
        "nuscenes_data_root",
        ["nuscenes-interpolated_train", "nuscenes-interpolated_val", "nuscenes-interpolated_test"],
    ),
    "nuscenes-interpolated-mini": _dataset_config(
        "nuscenes",
        "nuscenes_data_root",
        ["nuscenes-interpolated-mini_train", "nuscenes-interpolated-mini_val"],
    ),
    "av2-sensor": _dataset_config(
        "av2",
        "av2_data_root",
        ["av2-sensor_train", "av2-sensor_val", "av2-sensor_test"],
    ),
    "kitti360": _dataset_config(
        "kitti360",
        "kitti360_data_root",
        ["kitti360_train", "kitti360_val", "kitti360_test"],
    ),
    "pandaset": _dataset_config(
        "pandaset",
        "pandaset_data_root",
        ["pandaset_train", "pandaset_val", "pandaset_test"],
    ),
}
