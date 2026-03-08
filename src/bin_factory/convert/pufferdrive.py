import struct

import numpy as np

from bin_factory import logger_utils
from bin_factory.convert import agents, roadgraph, traffic_controls
from bin_factory.convert import types as puffer_types


logger = logger_utils.get_logger(__name__)


def convert_to_puffer_dict(scenario: dict, min_route_valid_points: int = 0, route_check_timestep: int = 0) -> dict:
    if not isinstance(scenario, dict):
        raise TypeError(f"Expected dict, got {type(scenario).__name__}")

    required_fields = ["id", "agents", "map", "traffic_lights"]
    missing = [f for f in required_fields if f not in scenario]
    if missing:
        raise ValueError(f"scenario missing required fields: {missing}")

    puffer_agents, sdc_index = agents.convert_agents(
        scenario["agents"],
        scenario["map"],
        min_route_valid_points=min_route_valid_points,
        route_check_timestep=route_check_timestep,
    )

    road_map_elements = roadgraph.convert_road_map_elements(scenario["map"], scenario.get("dataset_name", ""))

    traffic_control_elements = traffic_controls.convert_traffic_control_elements(
        scenario["traffic_lights"],
        scenario["map"],
    )

    puffer_metadata = {
        "dataset_name": scenario.get("dataset_name", ""),
        "scenario_length": scenario.get("scenario_length", 0),
        "sdc_index": sdc_index,
        "objects_of_interests": [],
        "tracks_to_predict": [],
    }

    return {
        "scenario_id": scenario["id"],
        "agents": puffer_agents,
        "road_map_elements": road_map_elements,
        "traffic_control_elements": traffic_control_elements,
        "metadata": puffer_metadata,
    }


def _pack_int_list(buffer, items):
    n = len(items)
    buffer.extend(struct.pack("i", n))
    if n > 0:
        buffer.extend(struct.pack(f"{n}i", *[int(x) for x in items]))


def puffer_dict_to_binary(puffer_dict: dict, map_id: int = 0) -> bytes:
    agents = puffer_dict["agents"]
    road_map_elements = puffer_dict["road_map_elements"]
    traffic_control_elements = puffer_dict["traffic_control_elements"]
    metadata = puffer_dict["metadata"]

    buffer = bytearray()
    buffer.extend(struct.pack("iii", len(agents), len(road_map_elements), len(traffic_control_elements)))

    # Agents
    for agent in agents:
        agent_id = int(agent["id"])
        agent_type = int(agent["type"])
        buffer.extend(struct.pack("ii", agent_id, agent_type))

        states = agent["states"]
        xyz = np.asarray(states["xyz"], dtype=np.float32)
        velocity = np.asarray(states["velocity"], dtype=np.float32)
        heading = np.asarray(states["heading"], dtype=np.float32)
        valid = np.asarray(states["valid"], dtype=np.int32)
        width = np.asarray(states["width"], dtype=np.float32)
        length = np.asarray(states["length"], dtype=np.float32)
        height = np.asarray(states["height"], dtype=np.float32)

        trajectory_length = len(xyz)
        buffer.extend(struct.pack("i", trajectory_length))

        # xyz: 3 channels x T (column-major order)
        for i in range(3):
            buffer.extend(xyz[:, i].tobytes())

        buffer.extend(heading.tobytes())

        for i in range(2):
            buffer.extend(velocity[:, i].tobytes())

        buffer.extend(length.tobytes())
        buffer.extend(width.tobytes())
        buffer.extend(height.tobytes())
        buffer.extend(valid.tobytes())

        routes = agent["routes"]
        _pack_int_list(buffer, routes[0] if routes else [])

        goal_x = goal_y = goal_z = 0.0
        if len(valid) > 0:
            valid_indices = np.where(valid > 0)[0]
            if len(valid_indices) > 0:
                last_valid_idx = valid_indices[-1]
                goal_x = float(xyz[last_valid_idx, 0])
                goal_y = float(xyz[last_valid_idx, 1])
                goal_z = float(xyz[last_valid_idx, 2])

        buffer.extend(struct.pack("fff", goal_x, goal_y, goal_z))
        mark_as_expert = 0 if routes else 1
        buffer.extend(struct.pack("i", mark_as_expert))

    # Road Map Elements
    for road in road_map_elements:
        road_id = int(road["id"])
        road_type = int(road["type"])
        buffer.extend(struct.pack("ii", road_id, road_type))

        xyz = np.asarray(road["xyz"], dtype=np.float32)
        segment_length = len(xyz)
        buffer.extend(struct.pack("i", segment_length))

        for i in range(3):
            buffer.extend(xyz[:, i].tobytes())

        if puffer_types.is_road_lane(road_type):
            _pack_int_list(buffer, road["entry_lanes"])
            _pack_int_list(buffer, road["exit_lanes"])

            buffer.extend(struct.pack("f", road["speed_limit"]))

    # Traffic Control Elements
    for element in traffic_control_elements:
        traffic_id = int(element["id"])
        traffic_type = int(element["type"])
        buffer.extend(struct.pack("ii", traffic_id, traffic_type))

        xyz = element["xyz"]
        if isinstance(xyz, list):
            xyz = np.array(xyz)

        x = float(xyz[0]) if len(xyz) > 0 else 0.0
        y = float(xyz[1]) if len(xyz) > 1 else 0.0
        z = float(xyz[2]) if len(xyz) > 2 else 0.0
        buffer.extend(struct.pack("fff", x, y, z))

        _pack_int_list(buffer, element["states"])
        _pack_int_list(buffer, element["controlled_lanes"])

    # Metadata
    scenario_id = puffer_dict["scenario_id"][:128]
    buffer.extend(scenario_id.encode("utf-8").ljust(128, b"\0"))

    buffer.extend(struct.pack("i", int(map_id)))

    dataset_name = metadata["dataset_name"][:64]
    buffer.extend(dataset_name.encode("utf-8").ljust(64, b"\0"))

    buffer.extend(struct.pack("i", int(metadata["scenario_length"])))
    buffer.extend(struct.pack("i", int(metadata["sdc_index"])))

    _pack_int_list(buffer, metadata["objects_of_interests"])
    _pack_int_list(buffer, metadata["tracks_to_predict"])

    return bytes(buffer)
