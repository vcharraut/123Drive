"""
Convert traffic control elements from intermediate format to Puffer format.
"""

import numpy as np
from py123d.datatypes.detections.traffic_light_detections import TrafficLightStatus

from src import logger_utils, types


logger = logger_utils.get_logger(__name__)


def convert_traffic_control_elements(dynamic_map_elements: dict, static_map_elements: dict) -> list[dict]:
    """
    Convert dynamic map elements to Puffer traffic_control_elements.

    Args:
        dynamic_map_elements: Dict of dynamic map elements from intermediate scenario
        length: Number of timesteps

    Returns:
        List of traffic control element dictionaries in Puffer format
    """
    puffer_elements = []

    for element_id, element_data in dynamic_map_elements.items():
        element_type = 1  # TODO: Add right type mapping when we have more types in the data
        position = element_data["position"]
        states = element_data["states"]
        controlled_lane = element_data["controlled_lane"]

        # Convert traffic light type to int
        element_type_int = _convert_traffic_control_type_to_int(element_type)

        # Convert states to int array
        # States might be a list or numpy array
        states_list = states.tolist() if isinstance(states, np.ndarray) else states

        states_int = [_convert_traffic_light_state_to_int(s) for s in states_list]
        states_int = np.array(states_int, dtype=np.int32)

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

    for element_id, element_data in static_map_elements.items():
        pass

        # element_type = element_data["type"]

        # # Convert traffic light type to int
        # element_type_int = _convert_traffic_control_type_to_int(element_type)

        # if element_type_int == 0:
        #     continue

        # position = element_data["position"]
        # lanes = element_data["lanes"]

        # # Normalize lanes to list (PufferDrive expects list)
        # if isinstance(lanes, int):
        #     controlled_lanes = [lanes]
        # elif isinstance(lanes, list):
        #     controlled_lanes = lanes
        # else:
        #     raise TypeError(f"lanes must be int or list[int], got {type(lanes).__name__}")

        # puffer_element = {
        #     "id": int(element_id),
        #     "type": element_type_int,
        #     "xyz": position,
        #     "states": [],
        #     "controlled_lanes": controlled_lanes,
        # }

        # puffer_elements.append(puffer_element)

    return puffer_elements


def _convert_traffic_control_type_to_int(element_type: str) -> int:
    """
    Convert traffic light type string to integer.

    Args:
        traffic_light_type: Traffic light type string from types.py

    Returns:
        Integer representation
    """
    # Map traffic light states to types
    type_map = {
        types.TRAFFIC_LIGHT: 1,
        types.STOP_SIGN: 2,
        types.YIELD_SIGN: 3,
        types.TRAFFIC_CONE: 4,
        types.TRAFFIC_BARRIER: 5,
        types.GUARDRAIL: 6,
    }
    return type_map.get(element_type, 0)


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
