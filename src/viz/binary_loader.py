"""Load Puffer binary files and convert to dict format for visualization.

Binary Format Structure (see binary_converter.py for full details):
====================================================================

Header (16 bytes):
    - num_agents (int32)
    - num_road_elements (int32)
    - num_traffic_controls (int32)
    - num_objects (int32)

Followed by:
    - DynamicAgents (variable size)
    - RoadMapElements (variable size)
    - TrafficControlElements (variable size)
    - Objects (variable size)
    - LaneGraphDistances (variable size)
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

import struct
from pathlib import Path

import numpy as np

from bin_factory.convert.types import is_road_lane


def load_puffer_binary(binary_path: str | Path) -> dict:
    """Load Puffer binary file and convert to dict format for visualization.

    Args:
        binary_path: Path to binary file

    Returns:
        Dict with keys:
            - scenario_id: str
            - dynamic_agents: list of agent dicts
            - road_map_elements: list of road dicts
            - traffic_control_elements: list of traffic dicts
            - objects: list of object dicts
            - metadata: dict
    """
    with Path(binary_path).open("rb") as f:
        # Read header
        num_agents = struct.unpack("i", f.read(4))[0]
        num_roads = struct.unpack("i", f.read(4))[0]
        num_traffic = struct.unpack("i", f.read(4))[0]
        num_objects = struct.unpack("i", f.read(4))[0]

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

        # Read objects
        objects = []
        for _ in range(num_objects):
            obj = _read_object(f)
            objects.append(obj)

        # Lane graph distances
        lane_graph_distances = None
        n_graph = struct.unpack("i", f.read(4))[0]
        if n_graph > 0:
            graph_lane_ids = list(struct.unpack(f"{n_graph}i", f.read(4 * n_graph)))
            lane_lengths = np.frombuffer(f.read(4 * n_graph), dtype=np.float32).copy()
            distances = np.frombuffer(f.read(4 * n_graph * n_graph), dtype=np.float32).copy()
            lane_graph_distances = {
                "lane_ids": graph_lane_ids,
                "lane_lengths": lane_lengths,
                "distances": distances.reshape(n_graph, n_graph),
            }

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

    return {
        "scenario_id": scenario_id,
        "agents": dynamic_agents,
        "road_map_elements": road_map_elements,
        "traffic_control_elements": traffic_control_elements,
        "objects": objects,
        "metadata": {
            "num_agents": num_agents,
            "num_roads": num_roads,
            "num_traffic": num_traffic,
            "num_objects": num_objects,
            "map_id": map_id,
            "dataset_name": dataset_name,
            "scenario_length": length,
            "sdc_index": sdc_index,
            "objects_of_interest": objects_of_interest,
            "tracks_to_predict": tracks_to_predict,
            "lane_graph_distances": lane_graph_distances,
        },
    }


def _read_dynamic_agent(f) -> dict:
    """Read a DynamicAgent from binary file."""
    agent_id = struct.unpack("i", f.read(4))[0]
    agent_type = struct.unpack("i", f.read(4))[0]
    trajectory_length = struct.unpack("i", f.read(4))[0]
    states = _read_dynamic_states(f, trajectory_length)

    num_route_ints = struct.unpack("i", f.read(4))[0]
    route = []
    if num_route_ints > 0:
        route = list(struct.unpack(f"{num_route_ints}i", f.read(4 * num_route_ints)))

    f.read(16)  # skip goal_xyz (3 floats) + mark_as_expert (1 int)

    return {
        "id": agent_id,
        "type": agent_type,
        "states": states,
        "route": route,
    }


def _read_object(f) -> dict:
    """Read an Object from binary file."""
    object_id = struct.unpack("i", f.read(4))[0]
    object_type = struct.unpack("i", f.read(4))[0]
    trajectory_length = struct.unpack("i", f.read(4))[0]
    return {
        "id": object_id,
        "type": object_type,
        "states": _read_dynamic_states(f, trajectory_length),
    }


def _read_float_array(f, n):
    return np.frombuffer(f.read(4 * n), dtype=np.float32).copy()


def _read_int_array(f, n):
    return np.frombuffer(f.read(4 * n), dtype=np.int32).copy()


def _read_dynamic_states(f, trajectory_length: int) -> dict:
    n = trajectory_length
    traj_x = _read_float_array(f, n)
    traj_y = _read_float_array(f, n)
    traj_z = _read_float_array(f, n)
    xyz = np.stack([traj_x, traj_y, traj_z], axis=1)

    heading = _read_float_array(f, n)
    vel_x = _read_float_array(f, n)
    vel_y = _read_float_array(f, n)
    velocity = np.stack([vel_x, vel_y], axis=1)

    length = _read_float_array(f, n)
    width = _read_float_array(f, n)
    height = _read_float_array(f, n)
    valid = _read_int_array(f, n)

    return {
        "xyz": xyz,
        "heading": heading,
        "velocity": velocity,
        "length": length,
        "width": width,
        "height": height,
        "valid": valid,
    }


def _read_road_map_element(f) -> dict:
    """Read a RoadMapElement from binary file."""
    # Read ID and type
    road_id = struct.unpack("i", f.read(4))[0]
    road_type = struct.unpack("i", f.read(4))[0]

    # Read segment length
    segment_length = struct.unpack("i", f.read(4))[0]

    # Read geometry - TRANSPOSED
    x = _read_float_array(f, segment_length)
    y = _read_float_array(f, segment_length)
    z = _read_float_array(f, segment_length)

    # Combine into (N, 3) array
    xyz = np.stack([x, y, z], axis=1)

    # Lane types (0-9) have entry/exit lanes and speed limit
    entry_lanes = []
    exit_lanes = []
    speed_limit = 0.0

    if is_road_lane(road_type):
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
    traffic_id = struct.unpack("i", f.read(4))[0]
    traffic_type = struct.unpack("i", f.read(4))[0]

    # Read stop line (2 points x 3 coords)
    x1, y1, z1 = struct.unpack("fff", f.read(12))
    x2, y2, z2 = struct.unpack("fff", f.read(12))
    heading = struct.unpack("f", f.read(4))[0]

    # Read states
    state_length = struct.unpack("i", f.read(4))[0]
    states = list(struct.unpack(f"{state_length}i", f.read(4 * state_length))) if state_length > 0 else []

    # Read controlled lanes
    num_controlled_lanes = struct.unpack("i", f.read(4))[0]
    controlled_lanes = []
    if num_controlled_lanes > 0:
        controlled_lanes = list(struct.unpack(f"{num_controlled_lanes}i", f.read(4 * num_controlled_lanes)))

    return {
        "id": traffic_id,
        "type": traffic_type,
        "stop_line": np.array([[x1, y1, z1], [x2, y2, z2]]),
        "heading": heading,
        "states": states,
        "controlled_lanes": controlled_lanes,
    }
