from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from py123d.datatypes.map_objects.map_layer_types import MapLayer


if TYPE_CHECKING:
    from py123d.api.map.map_api import MapAPI


def mps_to_kmh(speed_mps: float | None) -> float:
    if speed_mps is None:
        return -1
    return float(speed_mps) * 3.6


def kmh_to_mph(speed_kmh: float) -> float:
    return speed_kmh / 1.609344


def get_object_xy_points(map_object: object) -> np.ndarray | None:
    if hasattr(map_object, "centerline"):
        return map_object.centerline.array[:, :2]
    if hasattr(map_object, "polyline_3d"):
        return map_object.polyline_3d.array[:, :2]
    if hasattr(map_object, "outline_3d"):
        return map_object.outline_3d.array[:, :2]
    return None


def centered_array(array: np.ndarray, center: np.ndarray) -> np.ndarray:
    centered = array.astype(np.float32, copy=True)
    if centered.shape[1] >= 2:
        centered[:, 0] -= center[0]
        centered[:, 1] -= center[1]
    return centered


def get_lane_position(map_api: MapAPI | None, lane_id: int, center: np.ndarray) -> list[float]:
    if map_api is None:
        return [0.0, 0.0, 0.0]

    lane = map_api.get_map_object(lane_id, MapLayer.LANE)
    if lane is None or not hasattr(lane, "centerline"):
        return [0.0, 0.0, 0.0]

    point = lane.centerline.array[0]
    x = float(point[0]) - float(center[0])
    y = float(point[1]) - float(center[1])
    z = float(point[2]) if len(point) > 2 else 0.0

    return [x, y, z]



def resolve_py123d_data_root(dataset_path: str | None) -> Path:
    """Resolve py123d data root path.

    Args:
        dataset_path: Optional override for the dataset root.

    Returns:
        Path to py123d_data root.
    """
    if dataset_path:
        return Path(dataset_path)

    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "py123d_data"
