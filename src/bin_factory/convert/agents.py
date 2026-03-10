"""Convert dynamic agents from intermediate format to Puffer format."""

import numpy as np
from py123d.datatypes.detections import DefaultBoxDetectionLabel
from py123d.datatypes.map_objects import LaneType

from bin_factory import logger_utils
from bin_factory.convert import routes
from bin_factory.convert import types as puffer_types


logger = logger_utils.get_logger(__name__)


def convert_agents(
    dynamic_agents: dict,
    road_map_elements: dict,
    min_route_valid_points: int = 0,
    route_check_timestep: int = 0,
) -> tuple[list[dict], int]:
    """Convert dynamic agents from intermediate format to Puffer format.

    Args:
        dynamic_agents: Dict of dynamic agents from py123d dict
        road_map_elements: Dict of static map elements (for reference)
        min_route_valid_points: Minimum valid trajectory points required for route computation (0 = no filtering)
        route_check_timestep: Timestep at which agent must be valid for route computation (default: 0)

    Returns:
        Tuple of (list of converted agents, sdc_index)
    """
    puffer_agents = []
    sdc_index = -1

    lane_data = _extract_lane_centers(road_map_elements)
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

        route = routes.compute_agent_route(
            agent_data=(
                agent_id,
                position,
                heading,
                valid,
                length,
                width,
                agent_type_int,
                agent_data["type"] == DefaultBoxDetectionLabel.EGO,
            ),
            route_cache=route_cache,
            route_check_timestep=route_check_timestep,
            min_route_valid_points=min_route_valid_points,
        )

        puffer_agent = {
            "id": agent_id,
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
            "route": route,
        }

        puffer_agents.append(puffer_agent)

    return puffer_agents, sdc_index


def _convert_agent_type_to_int(agent_type) -> int:
    type_map = {
        DefaultBoxDetectionLabel.EGO: puffer_types.AgentType.VEHICLE,
        DefaultBoxDetectionLabel.VEHICLE: puffer_types.AgentType.VEHICLE,
        DefaultBoxDetectionLabel.PERSON: puffer_types.AgentType.PEDESTRIAN,
        DefaultBoxDetectionLabel.BICYCLE: puffer_types.AgentType.CYCLIST,
    }
    return type_map.get(agent_type, puffer_types.AgentType.OTHER)


def _extract_lane_centers(static_map_elements: dict) -> tuple[list, np.ndarray, dict, np.ndarray]:
    """Extract lane center information as numpy arrays for vectorized operations.

    Args:
        static_map_elements: Dict of static map elements

    Returns:
        Tuple of:
        - lane_ids: List of lane IDs (strings)
        - lane_polylines: Array of lane polylines, padded to max length (N_lanes, max_points, 2)
        - lane_metadata: Dict mapping lane_id to connectivity info
        - lane_lengths: Array of actual polyline lengths (N_lanes,) for each lane
    """
    lane_ids = []
    lane_polylines_list = []
    lane_lengths_list = []
    lane_metadata = {}
    max_points = 0

    # First pass: collect lanes and find max polyline length
    for element_id, element_data in static_map_elements.items():
        element_type = element_data["type"]

        # Only process lane centers
        if element_type in (LaneType.SURFACE_STREET, LaneType.FREEWAY):
            polyline = element_data["polyline"]

            if len(polyline) > 0:
                # Convert to 2D if needed
                polyline_2d = polyline[:, :2] if polyline.shape[1] == 3 else polyline

                lane_ids.append(element_id)
                lane_polylines_list.append(polyline_2d)
                lane_lengths_list.append(len(polyline_2d))
                max_points = max(max_points, len(polyline_2d))

                lane_metadata[element_id] = {
                    "entry_lanes": element_data["entry_lanes"],
                    "exit_lanes": element_data["exit_lanes"],
                }

    if not lane_ids:
        return [], np.array([]), {}, np.array([])

    # Second pass: create padded array
    n_lanes = len(lane_ids)
    lane_polylines = np.zeros((n_lanes, max_points, 2), dtype=np.float64)
    lane_lengths = np.array(lane_lengths_list, dtype=np.int64)

    for i, polyline_2d in enumerate(lane_polylines_list):
        lane_polylines[i, : len(polyline_2d), :] = polyline_2d

    return lane_ids, lane_polylines, lane_metadata, lane_lengths
