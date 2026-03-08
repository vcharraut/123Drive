"""
Convert dynamic agents from intermediate format to Puffer format.
"""

import numpy as np
from py123d.datatypes.detections import DefaultBoxDetectionLabel

from bin_factory import logger_utils
from bin_factory.convert import routes, utils
from bin_factory.convert import types as puffer_types


logger = logger_utils.get_logger(__name__)


def convert_agents(
    dynamic_agents: dict,
    road_map_elements: dict,
    min_route_valid_points: int = 0,
    route_check_timestep: int = 0,
) -> tuple[list[dict], int]:
    """
    Convert dynamic agents from intermediate format to Puffer format.

    Args:
        dynamic_agents: Dict of dynamic agents from intermediate scenario
        road_map_elements: Dict of static map elements (for reference)
        min_route_valid_points: Minimum valid trajectory points required for route computation (0 = no filtering)
        route_check_timestep: Timestep at which agent must be valid for route computation (default: 0)

    Returns:
        Tuple of (list of converted agents, sdc_index)
    """
    puffer_agents = []
    sdc_index = -1

    lane_data = utils.extract_lane_centers(road_map_elements)
    route_cache = routes.build_route_cache(road_map_elements, lane_data)

    for idx, (agent_id, agent_data) in enumerate(dynamic_agents.items()):
        # Get position data (x, y, z)
        position = agent_data["position"]

        if position.shape[1] == 2:
            # Add z=0 if only x,y provided
            position = np.column_stack([position, np.zeros(len(position), dtype=np.float64)])

        # Get heading, velocity, dimensions
        heading = agent_data["heading"]
        velocity = agent_data["velocity"]
        length = agent_data["length"]
        width = agent_data["width"]
        height = agent_data["height"]
        valid = agent_data["valid"]

        if agent_data["type"] == DefaultBoxDetectionLabel.EGO:
            sdc_index = idx

        # Convert agent type to int
        agent_type_int = _convert_agent_type_to_int(agent_data["type"])

        # Routes are computed only for:
        # 1. VEHICLE type (type == 1)
        # 2. Agents valid at route_check_timestep (configurable, default: 0)
        # 3. Agents with sufficient valid trajectory points (configurable, default: 0)
        should_compute_routes = (
            agent_type_int == puffer_types.VEHICLE
            and route_check_timestep < len(valid)
            and valid[route_check_timestep]
            and np.sum(valid) >= min_route_valid_points
        )

        if should_compute_routes:
            _routes = routes.compute_agent_route(
                agent_data=(agent_id, position, heading, valid, length, width),
                route_cache=route_cache,
                route_check_timestep=route_check_timestep,
            )
        else:
            _routes = []

        puffer_agent = {
            "id": idx,
            "type": agent_type_int,
            "states": {
                "xyz": position,
                "heading": heading,
                "velocity": velocity,
                "length": length,
                "width": width,
                "height": height,
                "valid": valid,
            },
            "routes": _routes,
        }

        puffer_agents.append(puffer_agent)

    return puffer_agents, sdc_index


def _convert_agent_type_to_int(agent_type) -> int:
    type_map = {
        DefaultBoxDetectionLabel.EGO: puffer_types.VEHICLE,
        DefaultBoxDetectionLabel.VEHICLE: puffer_types.VEHICLE,
        DefaultBoxDetectionLabel.PERSON: puffer_types.PEDESTRIAN,
        DefaultBoxDetectionLabel.BICYCLE: puffer_types.CYCLIST,
    }
    return type_map.get(agent_type, puffer_types.OTHER)
