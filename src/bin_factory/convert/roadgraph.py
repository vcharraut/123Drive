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
            xyz = element_data["polygon"]

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


def _convert_map_element_type_to_int(layer: MapLayer, element_type) -> int:
    if layer == MapLayer.LANE:
        lane_type_map = {
            LaneType.UNDEFINED: puffer_types.LANE_UNKNOWN,
            LaneType.FREEWAY: puffer_types.LANE_FREEWAY,
            LaneType.SURFACE_STREET: puffer_types.LANE_SURFACE_STREET,
            LaneType.BIKE_LANE: puffer_types.LANE_BIKE_LANE,
        }
        return lane_type_map.get(element_type, puffer_types.LANE_UNKNOWN)

    if layer == MapLayer.ROAD_LINE:
        road_line_type_map = {
            RoadLineType.UNKNOWN: puffer_types.ROAD_LINE_UNKNOWN,
            RoadLineType.DASHED_WHITE: puffer_types.ROAD_LINE_BROKEN_SINGLE_WHITE,
            RoadLineType.SOLID_WHITE: puffer_types.ROAD_LINE_SOLID_SINGLE_WHITE,
            RoadLineType.DOUBLE_SOLID_WHITE: puffer_types.ROAD_LINE_SOLID_DOUBLE_WHITE,
            RoadLineType.DASHED_YELLOW: puffer_types.ROAD_LINE_BROKEN_SINGLE_YELLOW,
            RoadLineType.DOUBLE_DASH_YELLOW: puffer_types.ROAD_LINE_BROKEN_DOUBLE_YELLOW,
            RoadLineType.SOLID_YELLOW: puffer_types.ROAD_LINE_SOLID_SINGLE_YELLOW,
            RoadLineType.DOUBLE_SOLID_YELLOW: puffer_types.ROAD_LINE_SOLID_DOUBLE_YELLOW,
            RoadLineType.DASH_SOLID_YELLOW: puffer_types.ROAD_LINE_PASSING_DOUBLE_YELLOW,
            RoadLineType.SOLID_DASH_YELLOW: puffer_types.ROAD_LINE_PASSING_DOUBLE_YELLOW,
            # Collapsed types
            RoadLineType.DOUBLE_DASH_WHITE: puffer_types.ROAD_LINE_BROKEN_SINGLE_WHITE,
            RoadLineType.DASH_SOLID_WHITE: puffer_types.ROAD_LINE_SOLID_SINGLE_WHITE,
            RoadLineType.SOLID_DASH_WHITE: puffer_types.ROAD_LINE_SOLID_SINGLE_WHITE,
            RoadLineType.SOLID_BLUE: puffer_types.ROAD_LINE_SOLID_SINGLE_WHITE,
        }
        return road_line_type_map.get(element_type, puffer_types.ROAD_LINE_UNKNOWN)

    if layer == MapLayer.ROAD_EDGE:
        road_edge_type_map = {
            RoadEdgeType.UNKNOWN: puffer_types.ROAD_EDGE_UNKNOWN,
            RoadEdgeType.ROAD_EDGE_BOUNDARY: puffer_types.ROAD_EDGE_BOUNDARY,
            RoadEdgeType.ROAD_EDGE_MEDIAN: puffer_types.ROAD_EDGE_MEDIAN,
        }
        return road_edge_type_map.get(element_type, puffer_types.ROAD_EDGE_UNKNOWN)

    if layer == MapLayer.CROSSWALK:
        return puffer_types.CROSSWALK

    return -1
