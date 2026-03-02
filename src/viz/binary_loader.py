"""
Load Puffer binary files and convert to dict format for visualization.

Binary Format Structure (see binary_converter.py for full details):
====================================================================

Header (12 bytes):
    - num_agents (int32)
    - num_road_elements (int32)
    - num_traffic_controls (int32)

Followed by:
    - DynamicAgents (variable size)
    - RoadMapElements (variable size)
    - TrafficControlElements (variable size)
    - Metadata (variable length):
        - scenario_id (char[128])
        - map_id (int32)
        - dataset_name (char[64])
        - length (int32)
        - sdc_index (int32)
        - num_objects_of_interest (int32)
        - objects_of_interest[num_objects_of_interest] (int32[])
        - num_tracks_to_predict (int32)
        - tracks_to_predict[num_tracks_to_predict] (int32[])
"""

import json
import struct
from pathlib import Path

import numpy as np


def load_scenario(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix == ".bin":
        return load_puffer_binary(path)
    with open(path) as f:
        scenario = json.load(f)
    array_keys = ["xyz", "heading", "velocity", "length", "width", "height", "valid"]
    for agent in scenario.get("agents", []):
        states = agent.get("states", {})
        for k in array_keys:
            if k in states and isinstance(states[k], list):
                states[k] = np.array(states[k])
    for elem in scenario.get("road_map_elements", []) + scenario.get("traffic_control_elements", []):
        for k in ("xyz", "dir_xyz", "states"):
            if k in elem and isinstance(elem[k], list):
                elem[k] = np.array(elem[k])
    return scenario


def load_puffer_binary(binary_path: str | Path) -> dict:
    """
    Load Puffer binary file and convert to dict format for visualization.

    Args:
        binary_path: Path to binary file

    Returns:
        Dict with keys:
            - scenario_id: str
            - dynamic_agents: list of agent dicts
            - road_map_elements: list of road dicts
            - traffic_control_elements: list of traffic dicts
            - metadata: dict
    """
    with open(binary_path, "rb") as f:
        # Read header
        num_agents = struct.unpack("i", f.read(4))[0]
        num_roads = struct.unpack("i", f.read(4))[0]
        num_traffic = struct.unpack("i", f.read(4))[0]

        # Read agents
        dynamic_agents = []
        for _ in range(num_agents):
            agent = _read_dynamic_agent(f)
            dynamic_agents.append(agent)

        # Read roads
        road_map_elements = []
        for _ in range(num_roads):
            road = _read_road_map_element(f)
            road_map_elements.append(road)

        # Read traffic controls
        traffic_control_elements = []
        for _ in range(num_traffic):
            traffic = _read_traffic_control_element(f)
            traffic_control_elements.append(traffic)

        # Read metadata
        scenario_id_bytes = f.read(128)
        scenario_id = scenario_id_bytes.rstrip(b"\0").decode("utf-8")

        map_id = struct.unpack("i", f.read(4))[0]

        dataset_name_bytes = f.read(64)
        dataset_name = dataset_name_bytes.rstrip(b"\0").decode("utf-8")

        length = struct.unpack("i", f.read(4))[0]
        sdc_index = struct.unpack("i", f.read(4))[0]

        # Read objects_of_interest
        num_oi = struct.unpack("i", f.read(4))[0]
        objects_of_interest = []
        if num_oi > 0:
            objects_of_interest = list(struct.unpack(f"{num_oi}i", f.read(4 * num_oi)))

        # Read tracks_to_predict
        num_ttp = struct.unpack("i", f.read(4))[0]
        tracks_to_predict = []
        if num_ttp > 0:
            tracks_to_predict = list(struct.unpack(f"{num_ttp}i", f.read(4 * num_ttp)))

    # Build scenario dict
    return {
        "scenario_id": scenario_id,
        "agents": dynamic_agents,
        "road_map_elements": road_map_elements,
        "traffic_control_elements": traffic_control_elements,
        "metadata": {
            "num_agents": num_agents,
            "num_roads": num_roads,
            "num_traffic": num_traffic,
            "map_id": map_id,
            "dataset_name": dataset_name,
            "scenario_length": length,
            "sdc_index": sdc_index,
            "objects_of_interest": objects_of_interest,
            "tracks_to_predict": tracks_to_predict,
        },
    }


def _read_dynamic_agent(f) -> dict:
    """Read a DynamicAgent from binary file."""
    # Read ID and type
    agent_id = struct.unpack("i", f.read(4))[0]
    agent_type = struct.unpack("i", f.read(4))[0]

    # Read trajectory length
    trajectory_length = struct.unpack("i", f.read(4))[0]

    # Read trajectory data - TRANSPOSED format
    traj_x = np.array(struct.unpack(f"{trajectory_length}f", f.read(4 * trajectory_length)))
    traj_y = np.array(struct.unpack(f"{trajectory_length}f", f.read(4 * trajectory_length)))
    traj_z = np.array(struct.unpack(f"{trajectory_length}f", f.read(4 * trajectory_length)))

    # Combine into (N, 3) array
    xyz = np.stack([traj_x, traj_y, traj_z], axis=1)

    # Read heading
    heading = np.array(struct.unpack(f"{trajectory_length}f", f.read(4 * trajectory_length)))

    # Read velocity - TRANSPOSED
    vel_x = np.array(struct.unpack(f"{trajectory_length}f", f.read(4 * trajectory_length)))
    vel_y = np.array(struct.unpack(f"{trajectory_length}f", f.read(4 * trajectory_length)))

    # Combine into (N, 2) array
    velocity = np.stack([vel_x, vel_y], axis=1)

    # Read dimensions
    length = np.array(struct.unpack(f"{trajectory_length}f", f.read(4 * trajectory_length)))
    width = np.array(struct.unpack(f"{trajectory_length}f", f.read(4 * trajectory_length)))
    height = np.array(struct.unpack(f"{trajectory_length}f", f.read(4 * trajectory_length)))

    # Read valid
    valid = np.array(struct.unpack(f"{trajectory_length}i", f.read(4 * trajectory_length)))

    # Read routes (only first route stored, no length prefix)
    num_route_ints = struct.unpack("i", f.read(4))[0]
    routes = []
    if num_route_ints > 0:
        route_data = list(struct.unpack(f"{num_route_ints}i", f.read(4 * num_route_ints)))
        routes.append(route_data)  # Single route, no length prefix

    # Read goal position (ignore for visualization)
    _goal_x = struct.unpack("f", f.read(4))[0]
    _goal_y = struct.unpack("f", f.read(4))[0]
    _goal_z = struct.unpack("f", f.read(4))[0]

    # Read mark_as_expert (ignore for visualization)
    _mark_as_expert = struct.unpack("i", f.read(4))[0]

    return {
        "id": agent_id,
        "type": agent_type,
        "states": {
            "xyz": xyz,
            "heading": heading,
            "velocity": velocity,
            "length": length,
            "width": width,
            "height": height,
            "valid": valid,
        },
        "routes": routes,
    }


def _read_road_map_element(f) -> dict:
    """Read a RoadMapElement from binary file."""
    # Read ID and type
    road_id = struct.unpack("i", f.read(4))[0]
    road_type = struct.unpack("i", f.read(4))[0]

    # Read segment length
    segment_length = struct.unpack("i", f.read(4))[0]

    # Read geometry - TRANSPOSED
    x = np.array(struct.unpack(f"{segment_length}f", f.read(4 * segment_length)))
    y = np.array(struct.unpack(f"{segment_length}f", f.read(4 * segment_length)))
    z = np.array(struct.unpack(f"{segment_length}f", f.read(4 * segment_length)))

    # Combine into (N, 3) array
    xyz = np.stack([x, y, z], axis=1)

    # Lane types (0-9) have entry/exit lanes and speed limit
    entry_lanes = []
    exit_lanes = []
    speed_limit = 0.0

    if 0 <= road_type <= 9:
        # Read entry lanes
        num_entry = struct.unpack("i", f.read(4))[0]
        if num_entry > 0:
            entry_lanes = list(struct.unpack(f"{num_entry}i", f.read(4 * num_entry)))

        # Read exit lanes
        num_exit = struct.unpack("i", f.read(4))[0]
        if num_exit > 0:
            exit_lanes = list(struct.unpack(f"{num_exit}i", f.read(4 * num_exit)))

        # Read speed limit
        speed_limit = struct.unpack("f", f.read(4))[0]

    return {
        "id": road_id,
        "type": road_type,
        "xyz": xyz,
        "entry_lanes": entry_lanes,
        "exit_lanes": exit_lanes,
        "speed_limit": speed_limit,
    }


def _read_traffic_control_element(f) -> dict:
    """Read a TrafficControlElement from binary file."""
    # Read ID and type
    traffic_id = struct.unpack("i", f.read(4))[0]
    traffic_type = struct.unpack("i", f.read(4))[0]

    # Read position
    x = struct.unpack("f", f.read(4))[0]
    y = struct.unpack("f", f.read(4))[0]
    z = struct.unpack("f", f.read(4))[0]

    # Read state length
    state_length = struct.unpack("i", f.read(4))[0]

    # Read states
    states = list(struct.unpack(f"{state_length}i", f.read(4 * state_length))) if state_length > 0 else []

    # Read controlled lanes
    num_controlled_lanes = struct.unpack("i", f.read(4))[0]
    controlled_lanes = []
    if num_controlled_lanes > 0:
        controlled_lanes = list(struct.unpack(f"{num_controlled_lanes}i", f.read(4 * num_controlled_lanes)))

    return {
        "id": traffic_id,
        "type": traffic_type,
        "xyz": np.array([x, y, z]),
        "states": states,
        "controlled_lanes": controlled_lanes,
    }
