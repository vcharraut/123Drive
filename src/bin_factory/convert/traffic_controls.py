"""Convert traffic control elements from intermediate format to Puffer format."""

import numpy as np
from py123d.datatypes.detections import TrafficLightStatus
from py123d.datatypes.map_objects import MapLayer, StopZoneType

from bin_factory import logger_utils
from bin_factory.convert import types as puffer_types


logger = logger_utils.get_logger(__name__)

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
    """Convert dynamic map elements to Puffer traffic_control_elements.

    Args:
        traffic_lights: Dict of traffic light elements from py123d dict
        map_data: Map data for the scenario (used to extract lane information for traffic control elements)
        scenario_length: Number of timesteps in scenario

    Returns:
        List of traffic control element dictionaries in Puffer format
    """
    observed_elements, covered_lanes = _convert_observed_traffic_lights(traffic_lights)
    if scenario_length > 0:
        return observed_elements
    next_id = max((element["id"] for element in observed_elements), default=-1) + 1
    map_elements = _convert_map_traffic_lights(map_data, covered_lanes, next_id, scenario_length)
    return observed_elements + map_elements


def _convert_observed_traffic_lights(traffic_lights: dict) -> tuple[list[dict], set[int]]:
    puffer_elements = []
    covered_lanes = set()

    for element_id, element_data in traffic_lights.items():
        controlled_lanes = [element_data["controlled_lane"]]
        covered_lanes.update(controlled_lanes)
        puffer_elements.append(
            {
                "id": int(element_id),
                "type": puffer_types.TCType.TRAFFIC_LIGHT,
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
        if element_data["layer"] != MapLayer.STOP_ZONE:
            continue

        stop_zone_type = element_data["type"]
        puffer_type = STOP_ZONE_TO_PUFFER.get(stop_zone_type)
        if puffer_type is None:
            continue

        controlled_lanes = [lane_id for lane_id in element_data["controlled_lanes"] if lane_id not in covered_lanes]
        if not controlled_lanes:
            continue

        for grouped_lanes in _group_lanes_by_position(controlled_lanes, map_data):
            group_key = tuple(grouped_lanes)
            if group_key not in seen_groups:
                seen_groups.add(group_key)
                unique_groups.append((puffer_type, grouped_lanes))

    return [
        {
            "id": element_id,
            "type": puffer_type,
            "xyz": np.asarray(map_data[grouped_lanes[0]]["polyline"][0], dtype=np.float64),
            "states": np.zeros((scenario_length,), dtype=np.int64),
            "controlled_lanes": grouped_lanes,
        }
        for element_id, (puffer_type, grouped_lanes) in enumerate(unique_groups, start=next_id)
    ]


def _traffic_light_states(states):
    raw = states.tolist() if isinstance(states, np.ndarray) else states
    return np.array(
        [s if isinstance(s, int) else PY123D_TO_PUFFER_TL.get(s, 0) for s in raw],
        dtype=np.int64,
    )


def _group_lanes_by_position(controlled_lanes, map_data):
    pos_to_lanes = {}
    for lane_id in controlled_lanes:
        pos_key = _position_to_group_key(map_data[lane_id]["polyline"][0])
        pos_to_lanes.setdefault(pos_key, []).append(lane_id)
    return [sorted(grouped_lanes) for _, grouped_lanes in sorted(pos_to_lanes.items())]


def _position_to_group_key(position, decimals=1):
    rounded = np.round(np.asarray(position, dtype=np.float64), decimals=decimals)
    return tuple(rounded.tolist())
