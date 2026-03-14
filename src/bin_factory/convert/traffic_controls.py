"""Convert traffic control elements from intermediate format to Puffer format."""

import numpy as np
from py123d.datatypes.detections import TrafficLightStatus
from py123d.datatypes.map_objects import MapLayer, StopZoneType

from bin_factory import logger_utils
from bin_factory.convert import types as puffer_types


logger = logger_utils.get_logger(__name__)

DEFAULT_LANE_WIDTH = 3.7

PY123D_TO_PUFFER_TL = {
    TrafficLightStatus.GREEN: puffer_types.TLState.GREEN,
    TrafficLightStatus.YELLOW: puffer_types.TLState.YELLOW,
    TrafficLightStatus.RED: puffer_types.TLState.RED,
    TrafficLightStatus.OFF: puffer_types.TLState.OFF,
    TrafficLightStatus.UNKNOWN: puffer_types.TLState.UNKNOWN,
}

STOP_ZONE_TO_PUFFER = {
    StopZoneType.TRAFFIC_LIGHT: puffer_types.TCType.TRAFFIC_LIGHT,
    StopZoneType.STOP_SIGN: puffer_types.TCType.STOP_SIGN,
    StopZoneType.YIELD_SIGN: puffer_types.TCType.YIELD_SIGN,
}


def convert_traffic_control_elements(traffic_lights: dict, map_data: dict, scenario_length: int = 0) -> list[dict]:
    observed_elements, covered_lanes = _convert_observed_traffic_lights(traffic_lights, map_data)
    if scenario_length > 0:
        return observed_elements
    next_id = max((element["id"] for element in observed_elements), default=-1) + 1
    map_elements = _convert_map_traffic_lights(map_data, covered_lanes, next_id, scenario_length)
    return observed_elements + map_elements


def _convert_observed_traffic_lights(traffic_lights: dict, map_data: dict) -> tuple[list[dict], set[int]]:
    puffer_elements = []
    covered_lanes = set()

    lanes_by_id = {eid: edata for eid, edata in map_data.items() if edata.get("layer") == MapLayer.LANE}

    for element_id, element_data in traffic_lights.items():
        controlled_lane_id = element_data["controlled_lane"]
        controlled_lanes = [controlled_lane_id]
        covered_lanes.update(controlled_lanes)

        position = element_data["position"]
        lane = lanes_by_id.get(controlled_lane_id)

        heading = _heading_from_entry_lanes(controlled_lane_id, lanes_by_id)
        stop_line = _stop_line_from_position(position, lane, heading)

        puffer_elements.append(
            {
                "id": int(element_id),
                "type": puffer_types.TCType.TRAFFIC_LIGHT,
                "stop_line": stop_line,
                "heading": heading,
                "states": _traffic_light_states(element_data["states"]),
                "controlled_lanes": controlled_lanes,
            },
        )

    return puffer_elements, covered_lanes


def _convert_map_traffic_lights(
    map_data: dict,
    covered_lanes: set[int],
    next_id: int,
    scenario_length: int,
) -> list[dict]:
    puffer_elements = []

    lanes_by_id = {eid: edata for eid, edata in map_data.items() if edata.get("layer") == MapLayer.LANE}

    for element_data in map_data.values():
        if element_data["layer"] != MapLayer.STOP_ZONE:
            continue

        stop_zone_type = element_data["type"]
        puffer_type = STOP_ZONE_TO_PUFFER.get(stop_zone_type)
        if puffer_type is None:
            continue

        controlled_lanes = [lane_id for lane_id in element_data["controlled_lanes"] if lane_id not in covered_lanes]
        if not controlled_lanes:
            continue

        heading = _heading_from_entry_lanes(controlled_lanes[0], lanes_by_id)
        stop_line = _longest_polygon_edge(element_data["polygon"])

        puffer_elements.append(
            {
                "id": next_id,
                "type": puffer_type,
                "stop_line": stop_line,
                "heading": heading,
                "states": np.array([puffer_types.TLState.UNKNOWN] * scenario_length, dtype=np.int64),
                "controlled_lanes": controlled_lanes,
            },
        )
        next_id += 1

    return puffer_elements


def _stop_line_from_position(position, lane, heading):
    center = np.asarray(position, dtype=np.float64)
    # Compute width based on the boundaries if available, otherwise use default lane width
    width = DEFAULT_LANE_WIDTH
    if lane is not None:
        left_boundary = lane.get("left_boundary")
        right_boundary = lane.get("right_boundary")
        if left_boundary is not None and right_boundary is not None:
            width = float(np.linalg.norm(np.asarray(left_boundary[-1]) - np.asarray(right_boundary[-1])))

    d2 = np.array([np.cos(heading), np.sin(heading)])
    perp = np.array([-d2[1], d2[0], 0.0])
    half = width / 2.0
    return np.array([center - perp * half, center + perp * half], dtype=np.float64)


def _heading_from_entry_lanes(controlled_lane_id, lanes_by_id):
    lane = lanes_by_id.get(controlled_lane_id)
    if lane is None:
        return 0.0

    headings = []
    for entry_id in lane.get("entry_lanes", []):
        entry_lane = lanes_by_id.get(entry_id)
        if entry_lane is None:
            continue
        polyline = entry_lane["polyline"]
        if len(polyline) >= 2:
            d = polyline[-1] - polyline[-2]
            headings.append(np.arctan2(d[1], d[0]))
    if headings:
        return float(np.arctan2(np.mean(np.sin(headings)), np.mean(np.cos(headings))))

    polyline = lane["polyline"]
    if len(polyline) >= 2:
        d = polyline[1] - polyline[0]
        return float(np.arctan2(d[1], d[0]))

    return 0.0


def _longest_polygon_edge(polygon):
    polygon = np.asarray(polygon, dtype=np.float64)
    n = len(polygon)
    best_len, best_a, best_b = -1.0, 0, 1
    for i in range(n):
        j = (i + 1) % n
        edge_len = float(np.linalg.norm(polygon[j] - polygon[i]))
        if edge_len > best_len:
            best_len, best_a, best_b = edge_len, i, j
    return np.array([polygon[best_a], polygon[best_b]], dtype=np.float64)


def _traffic_light_states(states):
    raw = states.tolist() if isinstance(states, np.ndarray) else states
    return np.array(
        [
            int(s) if isinstance(s, (int, np.integer)) else PY123D_TO_PUFFER_TL.get(s, puffer_types.TLState.UNKNOWN)
            for s in raw
        ],
        dtype=np.int64,
    )
