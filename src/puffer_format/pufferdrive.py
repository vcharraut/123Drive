import struct

import numpy as np

from src import logger_utils
from src.puffer_format import agents, roadgraph, traffic_lights


logger = logger_utils.get_logger(__name__)


def convert_to_puffer_dict(
    scenario: dict,
    polyline_reduction_threshold: float = 0.1,
    dist_threshold: float = 10.0,
    min_route_valid_points: int = 0,
    route_check_timestep: int = 0,
) -> dict:
    if not isinstance(scenario, dict):
        raise TypeError(f"Expected dict, got {type(scenario).__name__}")

    required_fields = ["id", "dynamic_agents", "static_map_elements", "dynamic_map_elements", "metadata"]
    missing = [f for f in required_fields if f not in scenario]
    if missing:
        raise ValueError(f"scenario missing required fields: {missing}")

    scenario_id = scenario["id"]
    road_map_elements = roadgraph.convert_road_map_elements(
        scenario["static_map_elements"],
        polyline_reduction_threshold,
        dist_threshold,
    )

    metadata = scenario["metadata"]
    sdc_agent_id = metadata.get("sdc_index", 0)

    dynamic_agents, sdc_sequential_idx = agents.convert_dynamic_agents(
        scenario["dynamic_agents"],
        scenario["static_map_elements"],
        min_route_valid_points=min_route_valid_points,
        route_check_timestep=route_check_timestep,
        sdc_agent_id=sdc_agent_id,
    )

    traffic_control_elements = traffic_lights.convert_traffic_control_elements(
        scenario["dynamic_map_elements"],
        scenario["static_map_elements"],
    )

    puffer_metadata = {
        "dataset_name": metadata.get("dataset_name", ""),
        "scenario_length": metadata.get("scenario_length", 0),
        "timesteps": metadata.get("timesteps", []),
        "sdc_index": sdc_sequential_idx,
    }

    puffer_metadata["objects_of_interests"] = []
    puffer_metadata["tracks_to_predict"] = []

    return {
        "scenario_id": scenario_id,
        "dynamic_agents": dynamic_agents,
        "road_map_elements": road_map_elements,
        "traffic_control_elements": traffic_control_elements,
        "metadata": puffer_metadata,
    }


def puffer_dict_to_binary(puffer_dict: dict, map_id: int = 0) -> bytes:  # noqa: C901
    dynamic_agents = puffer_dict["dynamic_agents"]
    road_map_elements = puffer_dict["road_map_elements"]
    traffic_control_elements = puffer_dict["traffic_control_elements"]

    buffer = bytearray()
    buffer.extend(struct.pack("iii", len(dynamic_agents), len(road_map_elements), len(traffic_control_elements)))

    for agent in dynamic_agents:
        agent_id = int(agent["id"])
        agent_type = int(agent["type"])
        buffer.extend(struct.pack("ii", agent_id, agent_type))

        states = agent["states"]
        xyz = np.array(states["xyz"])
        velocity = np.array(states["velocity"])
        heading = np.array(states["heading"])
        valid = np.array(states["valid"])
        width = np.array(states["width"])
        length = np.array(states["length"])
        height = np.array(states["height"])

        trajectory_length = len(xyz)
        buffer.extend(struct.pack("i", trajectory_length))

        for i in range(3):
            for j in range(trajectory_length):
                buffer.extend(struct.pack("f", float(xyz[j, i])))

        for j in range(trajectory_length):
            buffer.extend(struct.pack("f", float(heading[j])))

        for i in range(2):
            for j in range(trajectory_length):
                buffer.extend(struct.pack("f", float(velocity[j, i])))

        if isinstance(length, (int, float)):
            length = np.full(trajectory_length, float(length))
        if isinstance(width, (int, float)):
            width = np.full(trajectory_length, float(width))
        if isinstance(height, (int, float)):
            height = np.full(trajectory_length, float(height))

        for j in range(trajectory_length):
            buffer.extend(struct.pack("f", float(length[j])))
        for j in range(trajectory_length):
            buffer.extend(struct.pack("f", float(width[j])))
        for j in range(trajectory_length):
            buffer.extend(struct.pack("f", float(height[j])))

        for j in range(trajectory_length):
            buffer.extend(struct.pack("i", int(valid[j])))

        routes = agent["routes"]
        if routes:
            first_route = routes[0]
            flattened = first_route
            total_route_ints = len(first_route)
        else:
            flattened = []
            total_route_ints = 0

        buffer.extend(struct.pack("i", total_route_ints))
        for route_int in flattened:
            buffer.extend(struct.pack("i", int(route_int)))

        goal_x = goal_y = goal_z = 0.0
        if len(valid) > 0:
            valid_indices = np.where(valid > 0)[0]
            if len(valid_indices) > 0:
                last_valid_idx = valid_indices[-1]
                goal_x = float(xyz[last_valid_idx, 0])
                goal_y = float(xyz[last_valid_idx, 1])
                goal_z = float(xyz[last_valid_idx, 2])

        buffer.extend(struct.pack("fff", goal_x, goal_y, goal_z))
        mark_as_expert = 0 if (routes and len(routes) > 0) else 1
        buffer.extend(struct.pack("i", mark_as_expert))

    for road in road_map_elements:
        road_id = int(road["id"])
        road_type = int(road["type"])
        buffer.extend(struct.pack("ii", road_id, road_type))

        xyz = np.array(road["xyz"])
        segment_length = len(xyz)
        buffer.extend(struct.pack("i", segment_length))

        for i in range(3):
            for j in range(segment_length):
                buffer.extend(struct.pack("f", float(xyz[j, i])))

        if road_type <= 10:
            entry_lanes = road["entry_lanes"]
            exit_lanes = road["exit_lanes"]

            buffer.extend(struct.pack("i", len(entry_lanes)))
            for lane_id in entry_lanes:
                buffer.extend(struct.pack("i", int(lane_id)))

            buffer.extend(struct.pack("i", len(exit_lanes)))
            for lane_id in exit_lanes:
                buffer.extend(struct.pack("i", int(lane_id)))

            buffer.extend(struct.pack("f", road["speed_limit"]))

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

        states = element["states"]
        buffer.extend(struct.pack("i", len(states)))
        for state in states:
            buffer.extend(struct.pack("i", int(state)))

        controlled_lanes = element["controlled_lanes"]
        buffer.extend(struct.pack("i", len(controlled_lanes)))
        for lane in controlled_lanes:
            buffer.extend(struct.pack("i", int(lane)))

    metadata = puffer_dict["metadata"]

    scenario_id = puffer_dict["scenario_id"][:128]
    buffer.extend(scenario_id.encode("utf-8").ljust(128, b"\0"))

    buffer.extend(struct.pack("i", int(map_id)))

    dataset_name = metadata["dataset_name"][:64]
    buffer.extend(dataset_name.encode("utf-8").ljust(64, b"\0"))

    buffer.extend(struct.pack("i", int(metadata["scenario_length"])))
    buffer.extend(struct.pack("i", int(metadata["sdc_index"])))

    objects_of_interest = metadata["objects_of_interests"]
    buffer.extend(struct.pack("i", len(objects_of_interest)))
    for oi in objects_of_interest:
        buffer.extend(struct.pack("i", int(oi)))

    tracks_to_predict = metadata["tracks_to_predict"]
    buffer.extend(struct.pack("i", len(tracks_to_predict)))
    for ttp in tracks_to_predict:
        buffer.extend(struct.pack("i", int(ttp)))

    return bytes(buffer)
