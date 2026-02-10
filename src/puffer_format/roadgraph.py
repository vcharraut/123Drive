"""
Convert road map elements from intermediate format to Puffer format.
"""

import numpy as np
from py123d.datatypes.map_objects.map_layer_types import LaneType, RoadEdgeType, RoadLineType, SerialIntEnum

from src import logger_utils, types


logger = logger_utils.get_logger(__name__)

FILTERED_TYPES = []


def calculate_area(p1: dict, p2: dict, p3: dict) -> float:
    """
    Calculate the area of the triangle using the determinant method.

    Args:
        p1: First point dict with 'x' and 'y' keys
        p2: Second point dict with 'x' and 'y' keys
        p3: Third point dict with 'x' and 'y' keys

    Returns:
        Triangle area
    """
    return 0.5 * abs((p1["x"] - p3["x"]) * (p2["y"] - p1["y"]) - (p1["x"] - p2["x"]) * (p3["y"] - p1["y"]))


def simplify_polyline(geometry: list[dict], polyline_reduction_threshold: float, dist_threshold: float) -> list[dict]:
    """
    Simplify the given polyline using a method inspired by Visvalingham-Whyatt, optimized for Python.

    Args:
        geometry: List of point dicts with 'x' and 'y' keys
        polyline_reduction_threshold: Minimum triangle area threshold for point removal
        dist_threshold: Maximum distance between endpoints to consider for simplification

    Returns:
        Simplified polyline (list of point dicts)
    """
    num_points = len(geometry)
    if num_points < 3:
        return geometry  # Not enough points to simplify

    skip = [False] * num_points
    skip_changed = True

    while skip_changed:
        skip_changed = False
        k = 0
        while k < num_points - 1:
            k_1 = k + 1
            while k_1 < num_points - 1 and skip[k_1]:
                k_1 += 1
            if k_1 >= num_points - 1:
                break

            k_2 = k_1 + 1
            while k_2 < num_points and skip[k_2]:
                k_2 += 1
            if k_2 >= num_points:
                break

            point1 = geometry[k]
            point2 = geometry[k_1]
            point3 = geometry[k_2]
            area = calculate_area(point1, point2, point3)
            dist = np.linalg.norm(
                np.array([point1["x"], point1["y"]]) - np.array([point3["x"], point3["y"]]),
            )

            if area < polyline_reduction_threshold and dist < dist_threshold:
                skip[k_1] = True
                skip_changed = True
                k = k_2
            else:
                k = k_1

    return [geometry[i] for i in range(num_points) if not skip[i]]


def convert_road_map_elements(
    static_map_elements: dict,
    polyline_reduction_threshold: float = 0.1,
    dist_threshold: float = 10.0,
) -> list[dict]:
    """
    Convert static map elements from intermediate format to Puffer road_map_elements.

    Args:
        static_map_elements: Dict of static map elements from intermediate scenario
        polyline_reduction_threshold: Minimum triangle area threshold for polyline simplification.
                                       If 0.0 (default), no simplification is applied.
        dist_threshold: Maximum distance between endpoints to consider for simplification

    Returns:
        List of road map element dictionaries in Puffer format
    """
    puffer_elements = []

    for element_id, element_data in static_map_elements.items():
        element_type = element_data["type"]

        if element_type in FILTERED_TYPES: # TODO: Add right type mapping when we have more types in the data
            continue

        if not element_type:
            raise ValueError(f"Map element {element_id} has unset type")

        # Convert element type to int
        element_type_int = _convert_map_element_type_to_int(element_type)

        if element_type_int == 0:
            raise ValueError(f"Unknown map element type: {element_type}")

        if element_type_int <= 30:
            xyz = element_data["polyline"]
            # Ensure polyline has 3D coordinates
            if xyz.shape[1] == 2:
                xyz = np.column_stack([xyz, np.zeros(len(xyz))])

            # Apply polyline reduction if threshold is set
            if polyline_reduction_threshold > 0.0 and len(xyz) >= 3:
                # Convert numpy array to list of dicts for simplification algorithm
                geometry = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in xyz]
                simplified_geometry = simplify_polyline(geometry, polyline_reduction_threshold, dist_threshold)
                # Convert back to numpy array
                xyz = np.array([[p["x"], p["y"], p["z"]] for p in simplified_geometry])
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

            # Convert lane connectivity (entry/exit/neighbors)
            # left_neighbor = element_data["left_neighbor"] # TODO: Add left/right neighbor handling when we have more types in the data
            # right_neighbor = element_data["right_neighbor"]

            # Use int IDs directly
            puffer_element["entry_lanes"] = element_data["entry_lanes"]
            puffer_element["exit_lanes"] = element_data["exit_lanes"]

            # Combine left and right neighbors
            neighbors = []
            # if left_neighbor:
            #     neighbors.extend(left_neighbor)
            # if right_neighbor:
            #     neighbors.extend(right_neighbor)
            puffer_element["neighbors"] = neighbors

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
