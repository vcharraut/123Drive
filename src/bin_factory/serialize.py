import struct

import numpy as np

from bin_factory import types as puffer_types


METADATA_ID_BYTES = 128
METADATA_DATASET_BYTES = 32


def _pack_int_list(buffer, items) -> None:
    n = len(items)
    buffer.extend(struct.pack("<i", n))
    if n > 0:
        buffer.extend(struct.pack(f"<{n}i", *[int(x) for x in items]))


def _pack_dynamic_states(buffer, states):
    xyz = np.asarray(states["xyz"], dtype=np.float32)
    velocity = np.asarray(states["velocity"], dtype=np.float32)
    heading = np.asarray(states["heading"], dtype=np.float32)
    valid = np.asarray(states["valid"], dtype=np.int32)
    width = np.asarray(states["width"], dtype=np.float32)
    length = np.asarray(states["length"], dtype=np.float32)
    height = np.asarray(states["height"], dtype=np.float32)

    trajectory_length = len(xyz)
    buffer.extend(struct.pack("<i", trajectory_length))
    for i in range(3):
        buffer.extend(xyz[:, i].tobytes())
    buffer.extend(heading.tobytes())
    for i in range(2):
        buffer.extend(velocity[:, i].tobytes())
    buffer.extend(length.tobytes())
    buffer.extend(width.tobytes())
    buffer.extend(height.tobytes())
    buffer.extend(valid.tobytes())
    return xyz, valid


def _pack_fixed_string(buffer, value, size) -> None:
    encoded = str(value).encode("utf-8")[:size]
    buffer.extend(encoded.ljust(size, b"\0"))


def puffer_dict_to_binary(puffer_dict, map_id=0):
    """Serialize a puffer dict to binary format."""
    agents = puffer_dict["agents"]
    road_map_elements = puffer_dict["road_map_elements"]
    traffic_control_elements = puffer_dict["traffic_control_elements"]
    objects = puffer_dict.get("objects", [])
    metadata = puffer_dict["metadata"]

    buffer = bytearray()
    buffer.extend(
        struct.pack("<iiii", len(agents), len(road_map_elements), len(traffic_control_elements), len(objects)),
    )

    for agent in agents:
        buffer.extend(struct.pack("<ii", int(agent["id"]), int(agent["type"])))
        xyz, valid = _pack_dynamic_states(buffer, agent["states"])
        route = agent.get("route", [])
        _pack_int_list(buffer, route)

        goal_x = goal_y = goal_z = 0.0
        if len(valid) > 0:
            valid_indices = np.where(valid > 0)[0]
            if len(valid_indices) > 0:
                last_valid_idx = valid_indices[-1]
                goal_x = float(xyz[last_valid_idx, 0])
                goal_y = float(xyz[last_valid_idx, 1])
                goal_z = float(xyz[last_valid_idx, 2])
        buffer.extend(struct.pack("<fff", goal_x, goal_y, goal_z))
        buffer.extend(struct.pack("<i", 0 if route else 1))

    for road in road_map_elements:
        road_type = int(road["type"])
        buffer.extend(struct.pack("<ii", int(road["id"]), road_type))
        xyz = np.asarray(road["xyz"], dtype=np.float32)
        buffer.extend(struct.pack("<i", len(xyz)))
        for i in range(3):
            buffer.extend(xyz[:, i].tobytes())
        if puffer_types.is_road_lane(road_type):
            _pack_int_list(buffer, road["entry_lanes"])
            _pack_int_list(buffer, road["exit_lanes"])
            buffer.extend(struct.pack("<f", road["speed_limit"]))

    for element in traffic_control_elements:
        buffer.extend(struct.pack("<ii", int(element["id"]), int(element["type"])))
        stop_line = np.asarray(element["stop_line"], dtype=np.float32)
        buffer.extend(struct.pack("<fff", float(stop_line[0, 0]), float(stop_line[0, 1]), float(stop_line[0, 2])))
        buffer.extend(struct.pack("<fff", float(stop_line[1, 0]), float(stop_line[1, 1]), float(stop_line[1, 2])))
        buffer.extend(struct.pack("<f", float(element["heading"])))
        _pack_int_list(buffer, element["states"])
        _pack_int_list(buffer, element["controlled_lanes"])

    for obj in objects:
        buffer.extend(struct.pack("<ii", int(obj["id"]), int(obj["type"])))
        _pack_dynamic_states(buffer, obj["states"])

    lane_graph = puffer_dict.get("lane_graph_distances")
    if lane_graph:
        n = len(lane_graph["lane_ids"])
        buffer.extend(struct.pack("<i", n))
        buffer.extend(struct.pack(f"<{n}i", *lane_graph["lane_ids"]))
        buffer.extend(np.asarray(lane_graph["lane_lengths"], dtype=np.float32).tobytes())
        buffer.extend(np.asarray(lane_graph["distances"], dtype=np.float32).tobytes())
    else:
        buffer.extend(struct.pack("<i", 0))

    _pack_fixed_string(buffer, metadata["id"], METADATA_ID_BYTES)
    buffer.extend(struct.pack("<i", int(map_id)))
    _pack_fixed_string(buffer, metadata["dataset"], METADATA_DATASET_BYTES)
    buffer.extend(struct.pack("<i", int(metadata["scenario_length"])))
    buffer.extend(struct.pack("<f", float(metadata["timestep_seconds"])))
    _pack_int_list(buffer, metadata["objects_of_interest"])
    _pack_int_list(buffer, metadata["tracks_to_predict"])

    return bytes(buffer)


def scenario_to_binary(scenario, map_id=0):
    agents = _flatten_tracks(scenario.agents, include_route=True)
    road_map_elements = _flatten_road_map(scenario.map)
    traffic_control_elements = _flatten_traffic_controls(scenario.traffic_controls)
    objects = _flatten_tracks(scenario.objects)

    puffer_dict = {
        "agents": agents,
        "road_map_elements": road_map_elements,
        "traffic_control_elements": traffic_control_elements,
        "objects": objects,
        "lane_graph_distances": scenario.lane_graph,
        "metadata": {
            "id": scenario.metadata.id,
            "dataset": scenario.metadata.dataset,
            "scenario_length": scenario.metadata.scenario_length,
            "timestep_seconds": scenario.metadata.timestep_seconds,
            "objects_of_interest": [],
            "tracks_to_predict": [],
        },
    }
    return puffer_dict_to_binary(puffer_dict, map_id=map_id)


def _flatten_tracks(tracks, include_route=False):
    return [
        {
            "id": eid,
            "type": track.type,
            "states": {k: getattr(track, k) for k in ("heading", "velocity", "length", "width", "height", "valid")}
            | {"xyz": track.position},
            **({"route": track.route} if include_route else {}),
        }
        for eid, track in tracks.items()
    ]


def _flatten_road_map(static_map):
    elements = []
    for eid, elem in static_map.items():
        road_type = elem["type"]

        if (
            puffer_types.is_road_lane(road_type)
            or puffer_types.is_road_line(road_type)
            or puffer_types.is_road_edge(road_type)
        ):
            xyz = elem.get("polyline")
            if xyz is None or len(xyz) <= 1:
                continue
        else:
            xyz = elem.get("polygon")
            if xyz is None or len(xyz) <= 1:
                continue

        puffer_elem = {"id": eid, "type": road_type, "xyz": xyz}

        if puffer_types.is_road_lane(road_type):
            puffer_elem["speed_limit"] = elem["speed_limit_mps"]
            puffer_elem["entry_lanes"] = elem["entry_lanes"]
            puffer_elem["exit_lanes"] = elem["exit_lanes"]

        elements.append(puffer_elem)

    return elements


def _flatten_traffic_controls(traffic_controls):
    return [
        {
            "id": tc["id"],
            "type": tc["type"],
            "stop_line": tc["stop_line"],
            "heading": tc["heading"],
            "states": np.array(tc["states"]),
            "controlled_lanes": tc["controlled_lanes"],
        }
        for tc in traffic_controls
    ]
