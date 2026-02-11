"""
Convert road map elements from intermediate format to Puffer format.
"""

import numpy as np
from py123d.datatypes.map_objects.map_layer_types import LaneType, RoadEdgeType, RoadLineType, SerialIntEnum

from src import logger_utils, types


logger = logger_utils.get_logger(__name__)

FILTERED_TYPES = []


def convert_road_map_elements(
    static_map_elements: dict,
) -> list[dict]:
    """Convert static map elements from intermediate format to Puffer road_map_elements."""
    puffer_elements = []

    for element_id, element_data in static_map_elements.items():
        element_type = element_data["type"]

        if element_type in FILTERED_TYPES:  # TODO: Add right type mapping when we have more types in the data
            continue

        if not element_type:
            raise ValueError(f"Map element {element_id} has unset type")

        # Convert element type to int
        element_type_int = _convert_map_element_type_to_int(element_type)

        if element_type_int == 0:
            continue

        if element_type_int <= 30:
            xyz = element_data["polyline"]
            # Ensure polyline has 3D coordinates
            if xyz.shape[1] == 2:
                xyz = np.column_stack([xyz, np.zeros(len(xyz))])
        else:
            xyz = element_data["polygon"]

        puffer_element = {
            "id": element_id,
            "type": element_type_int,
            "xyz": xyz,
        }

        # Add lane-specific attributes if this is a lane
        if isinstance(element_type, LaneType):
            # Convert speed limit from km/h to m/s
            speed_limit_kmh = element_data["speed_limit_kmh"]
            puffer_element["speed_limit"] = speed_limit_kmh / 3.6  # m/s

            puffer_element["entry_lanes"] = element_data["entry_lanes"]
            puffer_element["exit_lanes"] = element_data["exit_lanes"]
            puffer_element["neighbors"] = []

        puffer_elements.append(puffer_element)

    return puffer_elements


def _convert_map_element_type_to_int(element_type: SerialIntEnum) -> int:
    # Lane types (1-3)
    lane_type_map = {
        LaneType.UNDEFINED: 0,
        LaneType.FREEWAY: 1,
        LaneType.SURFACE_STREET: 2,
        LaneType.BIKE_LANE: 3,
    }
    if isinstance(element_type, LaneType):
        return lane_type_map.get(element_type, 0)

    # Road line types (11-18)
    road_line_type_map = {
        RoadLineType.UNKNOWN: 0,
        RoadLineType.DASHED_WHITE: 11,
        RoadLineType.SOLID_WHITE: 12,
        RoadLineType.DOUBLE_SOLID_WHITE: 13,
        RoadLineType.DASHED_YELLOW: 14,
        RoadLineType.DOUBLE_DASH_YELLOW: 15,
        RoadLineType.SOLID_YELLOW: 16,
        RoadLineType.DOUBLE_SOLID_YELLOW: 17,
        RoadLineType.DASH_SOLID_YELLOW: 18,
        RoadLineType.SOLID_DASH_YELLOW: 18,
        # Collapsed types
        RoadLineType.DOUBLE_DASH_WHITE: 11,
        RoadLineType.DASH_SOLID_WHITE: 12,
        RoadLineType.SOLID_DASH_WHITE: 12,
        RoadLineType.SOLID_BLUE: 12,
    }
    if isinstance(element_type, RoadLineType):
        return road_line_type_map.get(element_type, 0)

    # Road edge types (21-22)
    road_edge_type_map = {
        RoadEdgeType.UNKNOWN: 0,
        RoadEdgeType.ROAD_EDGE_BOUNDARY: 21,
        RoadEdgeType.ROAD_EDGE_MEDIAN: 22,
    }
    if isinstance(element_type, RoadEdgeType):
        return road_edge_type_map.get(element_type, 0)

    # String map feature types (31+)
    other_type_map = {
        types.CROSSWALK: 31,
        types.SPEED_BUMP: 32,
        types.STOP_SIGN: 33,
        types.DRIVEWAY: 34,
    }
    if element_type in other_type_map:
        return other_type_map[element_type]

    return 0
