"""
Convert traffic control elements from intermediate format to Puffer format.
"""

import numpy as np
from py123d.datatypes.detections.traffic_light_detections import TrafficLightStatus

from src.bin_factory import logger_utils
from src.bin_factory.convert import types as puffer_types


logger = logger_utils.get_logger(__name__)



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

    next_id = max((e["id"] for e in puffer_elements), default=-1) + 1
    for element_id, element_data in map.items():
        if isinstance(element_data["type"], StopZoneType):
            element_type_int = int(element_data["type"])
            lanes = element_data["controlled_lanes"]

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

                    puffer_element = {
                        "id": next_id,
                        "type": element_type_int,
                        "xyz": position,
                        "states": [],
                        "controlled_lanes": lanes,
                    }

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
