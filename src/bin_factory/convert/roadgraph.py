"""Convert road map elements from intermediate format to Puffer format."""

import numpy as np
from py123d.datatypes.map_objects import LaneType, MapLayer, RoadEdgeType, RoadLineType

from bin_factory import logger_utils
from bin_factory.convert import types as puffer_types


logger = logger_utils.get_logger(__name__)

# Datasets with reversed road edge types
REVERSE_ROAD_EDGE_DATASETS = ["av2", "nuplan", "carla"]


def convert_road_map_elements(static_map_elements: dict, dataset_name: str = "") -> list[dict]:
    """Convert static map elements from intermediate format to Puffer road_map_elements."""
    puffer_elements = []

    for element_id, element_data in static_map_elements.items():
        layer = element_data["layer"]
        element_type = element_data["type"]

        if element_data is None or element_type is None:
            raise ValueError(f"Map element {element_id} has unset type")

        # Convert element type to int
        element_type_int = _convert_map_element_type_to_int(layer, element_type)

        if element_type_int == -1:
            continue

        if layer in (MapLayer.LANE, MapLayer.ROAD_LINE, MapLayer.ROAD_EDGE):
            xyz = element_data["polyline"]

            if xyz is None or len(xyz) <= 1:
                continue

            # Ensure polyline has 3D coordinates
            if xyz.shape[1] == 2:
                xyz = np.column_stack([xyz, np.zeros(len(xyz), dtype=np.float64)])

            # Reverse xyz order for road edges in certain datasets
            if dataset_name.split("-")[0] in REVERSE_ROAD_EDGE_DATASETS and layer == MapLayer.ROAD_EDGE:
                xyz = xyz[::-1]

        else:
            xyz = _interpolate_polygon(element_data["polygon"], spacing=3.0)

        puffer_element = {
            "id": element_id,
            "type": element_type_int,
            "xyz": xyz,
        }

        # Add lane-specific attributes if this is a lane
        if layer == MapLayer.LANE:
            puffer_element["speed_limit"] = element_data["speed_limit_mps"]
            puffer_element["entry_lanes"] = element_data["entry_lanes"]
            puffer_element["exit_lanes"] = element_data["exit_lanes"]

        puffer_elements.append(puffer_element)

    return puffer_elements


def _interpolate_polygon(xyz: np.ndarray, spacing: int) -> np.ndarray:
    # Close polygon if not already closed
    if not np.allclose(xyz[0], xyz[-1]):
        xyz = np.vstack([xyz, xyz[0:1]])

    diffs = np.diff(xyz, axis=0)
    seg_lengths = np.linalg.norm(diffs[:, :2], axis=1)
    total_length = seg_lengths.sum()

    if total_length < spacing:
        return xyz

    cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
    num_points = max(int(total_length / spacing), 2)
    target_dists = np.linspace(0, total_length, num_points, endpoint=False)

    interpolated = np.empty((num_points, xyz.shape[1]))
    for i, d in enumerate(target_dists):
        idx = np.searchsorted(cum_lengths[1:], d, side="right")
        idx = min(idx, len(seg_lengths) - 1)
        t = (d - cum_lengths[idx]) / seg_lengths[idx] if seg_lengths[idx] > 0 else 0
        interpolated[i] = xyz[idx] + t * diffs[idx]

    # Close: append first point
    return np.vstack([interpolated, interpolated[0:1]])


def _convert_map_element_type_to_int(layer: MapLayer, element_type) -> int:
    if layer == MapLayer.LANE:
        lane_type_map = {
            LaneType.UNDEFINED: puffer_types.LaneType.UNKNOWN,
            LaneType.FREEWAY: puffer_types.LaneType.FREEWAY,
            LaneType.SURFACE_STREET: puffer_types.LaneType.SURFACE_STREET,
            LaneType.BIKE_LANE: puffer_types.LaneType.BIKE_LANE,
        }
        return lane_type_map.get(element_type, puffer_types.LaneType.UNKNOWN)

    if layer == MapLayer.ROAD_LINE:
        road_line_type_map = {
            RoadLineType.UNKNOWN: puffer_types.RoadLineType.UNKNOWN,
            RoadLineType.DASHED_WHITE: puffer_types.RoadLineType.BROKEN_SINGLE_WHITE,
            RoadLineType.SOLID_WHITE: puffer_types.RoadLineType.SOLID_SINGLE_WHITE,
            RoadLineType.DOUBLE_SOLID_WHITE: puffer_types.RoadLineType.SOLID_DOUBLE_WHITE,
            RoadLineType.DASHED_YELLOW: puffer_types.RoadLineType.BROKEN_SINGLE_YELLOW,
            RoadLineType.DOUBLE_DASH_YELLOW: puffer_types.RoadLineType.BROKEN_DOUBLE_YELLOW,
            RoadLineType.SOLID_YELLOW: puffer_types.RoadLineType.SOLID_SINGLE_YELLOW,
            RoadLineType.DOUBLE_SOLID_YELLOW: puffer_types.RoadLineType.SOLID_DOUBLE_YELLOW,
            RoadLineType.DASH_SOLID_YELLOW: puffer_types.RoadLineType.PASSING_DOUBLE_YELLOW,
            RoadLineType.SOLID_DASH_YELLOW: puffer_types.RoadLineType.PASSING_DOUBLE_YELLOW,
            # Collapsed types
            RoadLineType.DOUBLE_DASH_WHITE: puffer_types.RoadLineType.BROKEN_SINGLE_WHITE,
            RoadLineType.DASH_SOLID_WHITE: puffer_types.RoadLineType.SOLID_SINGLE_WHITE,
            RoadLineType.SOLID_DASH_WHITE: puffer_types.RoadLineType.SOLID_SINGLE_WHITE,
            RoadLineType.SOLID_BLUE: puffer_types.RoadLineType.SOLID_SINGLE_WHITE,
        }
        return road_line_type_map.get(element_type, puffer_types.RoadLineType.UNKNOWN)

    if layer == MapLayer.ROAD_EDGE:
        road_edge_type_map = {
            RoadEdgeType.UNKNOWN: puffer_types.RoadEdgeType.UNKNOWN,
            RoadEdgeType.ROAD_EDGE_BOUNDARY: puffer_types.RoadEdgeType.BOUNDARY,
            RoadEdgeType.ROAD_EDGE_MEDIAN: puffer_types.RoadEdgeType.MEDIAN,
        }
        return road_edge_type_map.get(element_type, puffer_types.RoadEdgeType.UNKNOWN)

    if layer == MapLayer.CROSSWALK:
        return puffer_types.MiscRoadType.CROSSWALK

    return -1
