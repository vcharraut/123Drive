"""
Convert traffic control elements from intermediate format to Puffer format.
"""

import numpy as np
from py123d.datatypes.detections.traffic_light_detections import TrafficLightStatus
from py123d.datatypes.map_objects.map_layer_types import RoadEdgeType, StopZoneType

from src.bin_factory import logger_utils
from src.bin_factory.convert import types as puffer_types


logger = logger_utils.get_logger(__name__)


def _extract_longest_edge(polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the two 3-D endpoints of the longest edge in a polygon."""
    n = len(polygon)
    best_i, best_sq = 0, -1.0
    for i in range(n):
        d = polygon[(i + 1) % n, :2] - polygon[i, :2]
        sq = float(d @ d)
        if sq > best_sq:
            best_sq, best_i = sq, i
    return polygon[best_i].copy(), polygon[(best_i + 1) % n].copy()


def _ray_segment_intersection_2d(
    origin: np.ndarray,
    direction: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
) -> tuple[float, np.ndarray] | tuple[None, None]:
    """2-D ray-segment intersection via Cramer's rule.

    Returns (t, hit_point) when t > 0 and 0 <= u <= 1, else (None, None).
    """
    dx, dy = direction[0], direction[1]
    ex = seg_b[0] - seg_a[0]
    ey = seg_b[1] - seg_a[1]
    denom = dx * ey - dy * ex
    if abs(denom) < 1e-12:
        return None, None
    fx = seg_a[0] - origin[0]
    fy = seg_a[1] - origin[1]
    t = (fx * ey - fy * ex) / denom
    u = (fx * dy - fy * dx) / denom
    if 0.0 <= u <= 1.0:
        hit = np.array([origin[0] + t * dx, origin[1] + t * dy, 0.0])
        return t, hit
    return None, None


def _find_road_edge_intersection(
    point: np.ndarray,
    direction: np.ndarray,
    road_edge_polylines: list,
) -> np.ndarray:
    """Cast a ray from point along direction and return the nearest road-edge hit.

    The Z-coordinate of the returned point is taken from point.
    If no intersection is found, returns point unchanged.
    """
    max_ext = 20.0  # metres — never extend further than this

    # Pass 1: nearest outward hit within max_ext (endpoint is inside the road).
    best_t, best_hit = None, None
    for polyline in road_edge_polylines:
        pts = np.asarray(polyline)
        for i in range(len(pts) - 1):
            t, hit = _ray_segment_intersection_2d(point, direction, pts[i], pts[i + 1])
            if t is not None and 0 < t <= max_ext and (best_t is None or t < best_t):
                best_t, best_hit = t, hit

    # Pass 2: if nothing outward, snap back (endpoint already past road edge).
    if best_hit is None:
        for polyline in road_edge_polylines:
            pts = np.asarray(polyline)
            for i in range(len(pts) - 1):
                t, hit = _ray_segment_intersection_2d(point, direction, pts[i], pts[i + 1])
                if t is not None and abs(t) <= max_ext and (best_t is None or abs(t) < abs(best_t)):
                    best_t, best_hit = t, hit

    if best_hit is not None:
        result = best_hit.copy()
        result[2] = point[2]
        return result
    return point.copy()


def _extend_to_road_edges(
    p1: np.ndarray,
    p2: np.ndarray,
    road_edge_polylines: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend the stop-line segment [p1, p2] outward until it hits road edges."""
    diff = (p2 - p1)[:2]
    length = float(np.linalg.norm(diff))
    if length < 1e-9:
        return p1.copy(), p2.copy()
    d = diff / length
    new_p1 = _find_road_edge_intersection(p1, -d, road_edge_polylines)
    new_p2 = _find_road_edge_intersection(p2, d, road_edge_polylines)
    return new_p1, new_p2


def _position_to_group_key(position, decimals=1):
    rounded = np.round(np.asarray(position, dtype=np.float64), decimals=decimals)
    return tuple(rounded.tolist())


def convert_traffic_control_elements(traffic_lights: dict, _objects: dict, map: dict) -> list[dict]:
    """
    Convert dynamic map elements to Puffer traffic_control_elements.

    Args:
        traffic_lights: Dict of traffic light elements from intermediate scenario
        _objects: Dict of static map elements from intermediate scenario
        map: Map data for the scenario (used to extract lane information for traffic control elements)

    Returns:
        List of traffic control element dictionaries in Puffer format
    """
    puffer_elements = []

    for element_id, element_data in traffic_lights.items():
        position = element_data["position"]
        states = element_data["states"]
        controlled_lane = element_data["controlled_lane"]

        element_type_int = puffer_types.TRAFFIC_LIGHT

        # Convert states to int array
        # States might be a list or numpy array
        states_list = states.tolist() if isinstance(states, np.ndarray) else states

        states_int = [_convert_traffic_light_state_to_int(s) for s in states_list]
        states_int = np.array(states_int, dtype=np.int64)

        # Normalize controlled_lane to list (PufferDrive expects list)
        if isinstance(controlled_lane, int):
            controlled_lanes = [controlled_lane]
        elif isinstance(controlled_lane, list):
            controlled_lanes = controlled_lane
        else:
            raise TypeError(f"controlled_lane must be int or list[int], got {type(controlled_lane).__name__}")

        puffer_element = {
            "id": int(element_id),
            "type": element_type_int,
            "xyz": position,
            "states": states_int,
            "controlled_lanes": controlled_lanes,
        }

        puffer_elements.append(puffer_element)

    road_edge_polylines = [
        ed["polyline"]
        for ed in map.values()
        if isinstance(ed.get("type"), RoadEdgeType) and "polyline" in ed
    ]

    next_id = max((e["id"] for e in puffer_elements), default=-1) + 1
    for element_id, element_data in map.items():
        if isinstance(element_data["type"], StopZoneType):
            element_type_int = int(element_data["type"])
            lanes = element_data["controlled_lanes"]

            polygon = element_data.get("polygon")
            stop_line = None
            if polygon is not None and len(polygon) >= 2:
                sl_start, sl_end = _extract_longest_edge(np.asarray(polygon))
                sl_start, sl_end = _extend_to_road_edges(sl_start, sl_end, road_edge_polylines)
                stop_line = np.stack([sl_start, sl_end])

            if element_type_int == 1:  # TRAFFIC_LIGHT
                lanes_id_to_position = {lane_id: map[lane_id]["polyline"][0] for lane_id in lanes if lane_id in map}

                # Merge lanes ids with same position into one traffic control element
                position_to_lanes = {}
                for lane_id, position in lanes_id_to_position.items():
                    position_tuple = _position_to_group_key(position)
                    if position_tuple not in position_to_lanes:
                        position_to_lanes[position_tuple] = []
                    position_to_lanes[position_tuple].append(lane_id)

                for _, lanes in position_to_lanes.items():
                    lane_id = lanes[0]
                    lane_data = map[lane_id]
                    position = lane_data["polyline"][0]

                    # Orientation: direction vehicles travel when crossing the stop line
                    # (polyline[0] = stop position, polyline[1] = next point into intersection)
                    orientation = None
                    polyline = lane_data["polyline"]
                    if len(polyline) >= 2:
                        d = np.asarray(polyline[1])[:2] - np.asarray(polyline[0])[:2]
                        norm = float(np.linalg.norm(d))
                        if norm > 1e-9:
                            orientation = d / norm

                    puffer_element = {
                        "id": next_id,
                        "type": element_type_int,
                        "xyz": position,
                        "states": [],
                        "controlled_lanes": lanes,
                    }

                    if stop_line is not None:
                        puffer_element["stop_line"] = stop_line
                    if orientation is not None:
                        puffer_element["orientation"] = orientation

                    puffer_elements.append(puffer_element)
                    next_id += 1

    return puffer_elements


def _convert_traffic_light_state_to_int(state) -> int:
    # None = unobserved
    if state is None:
        return 0

    # int = already converted (WaymonicTLS values from TL processor)
    if isinstance(state, int):
        return state

    # TrafficLightStatus enum from py123d (unprocessed data)
    if isinstance(state, TrafficLightStatus):
        tls_map = {
            TrafficLightStatus.RED: 4,
            TrafficLightStatus.GREEN: 6,
            TrafficLightStatus.YELLOW: 5,
        }
        return tls_map.get(state, 0)

    return 0
