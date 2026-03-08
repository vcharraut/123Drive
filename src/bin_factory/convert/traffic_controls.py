"""
Convert traffic control elements from intermediate format to Puffer format.
"""

import numpy as np
from py123d.datatypes.detections import TrafficLightStatus
from py123d.datatypes.map_objects import LaneType, StopZoneType

from bin_factory import logger_utils
from bin_factory.convert import types as puffer_types


logger = logger_utils.get_logger(__name__)


def _position_to_group_key(position, decimals=1):
    rounded = np.round(np.asarray(position, dtype=np.float64), decimals=decimals)
    return tuple(rounded.tolist())


def _normalize_controlled_lanes(controlled_lane) -> list[int]:
    lanes = [controlled_lane] if isinstance(controlled_lane, int) else controlled_lane
    if not isinstance(lanes, list):
        raise TypeError(f"controlled_lane must be int or list[int], got {type(controlled_lane).__name__}")
    return lanes


def _valid_controlled_lanes(controlled_lane, map_data: dict) -> list[int]:
    lanes = _normalize_controlled_lanes(controlled_lane)
    return [lane_id for lane_id in lanes if lane_id in map_data and isinstance(map_data[lane_id].get("type"), LaneType)]


def _traffic_light_states(states) -> np.ndarray:
    states_list = states.tolist() if isinstance(states, np.ndarray) else states
    return np.array([_convert_traffic_light_state_to_int(state) for state in states_list], dtype=np.int64)


def _group_lanes_by_position(controlled_lanes: list[int], map_data: dict) -> list[list[int]]:
    pos_to_lanes = {}
    for lane_id in controlled_lanes:
        pos_key = _position_to_group_key(map_data[lane_id]["polyline"][0])
        pos_to_lanes.setdefault(pos_key, []).append(lane_id)
    return [sorted(grouped_lanes) for _, grouped_lanes in sorted(pos_to_lanes.items())]


def _convert_observed_traffic_lights(traffic_lights: dict, map_data: dict) -> tuple[list[dict], set[int]]:
    puffer_elements = []
    covered_lanes = set()

    for element_id, element_data in traffic_lights.items():
        controlled_lanes = _valid_controlled_lanes(element_data["controlled_lane"], map_data)
        if not controlled_lanes:
            continue

        covered_lanes.update(controlled_lanes)
        puffer_elements.append(
            {
                "id": int(element_id),
                "type": puffer_types.TRAFFIC_LIGHT,
                "xyz": element_data["position"],
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
    seen_groups = set()
    unique_groups = []

    for element_data in map_data.values():
        if element_data.get("type") != StopZoneType.TRAFFIC_LIGHT:
            continue

        controlled_lanes = [
            lane_id
            for lane_id in _valid_controlled_lanes(element_data.get("controlled_lanes", []), map_data)
            if lane_id not in covered_lanes
        ]
        if not controlled_lanes:
            continue

        for grouped_lanes in _group_lanes_by_position(controlled_lanes, map_data):
            group_key = tuple(grouped_lanes)
            if group_key not in seen_groups:
                seen_groups.add(group_key)
                unique_groups.append(grouped_lanes)

    return [
        {
            "id": element_id,
            "type": puffer_types.TRAFFIC_LIGHT,
            "xyz": np.asarray(map_data[grouped_lanes[0]]["polyline"][0], dtype=np.float64),
            "states": np.zeros((scenario_length,), dtype=np.int64),
            "controlled_lanes": grouped_lanes,
        }
        for element_id, grouped_lanes in enumerate(unique_groups, start=next_id)
    ]


def convert_traffic_control_elements(traffic_lights: dict, map: dict, scenario_length: int = 0) -> list[dict]:
    """Convert dynamic map elements to Puffer traffic_control_elements.

    Args:
        traffic_lights: Dict of traffic light elements from intermediate scenario
        map: Map data for the scenario (used to extract lane information for traffic control elements)

    Returns:
        List of traffic control element dictionaries in Puffer format
    """
    observed_elements, covered_lanes = _convert_observed_traffic_lights(traffic_lights, map)
    next_id = max((element["id"] for element in observed_elements), default=-1) + 1
    map_elements = _convert_map_traffic_lights(map, covered_lanes, next_id, scenario_length)
    return observed_elements + map_elements


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
