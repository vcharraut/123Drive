"""Load Puffer binary files for visualization."""

import struct
from pathlib import Path

import numpy as np

from bin_factory.puffer_types import is_road_lane
from bin_factory.serialize import METADATA_DATASET_BYTES, METADATA_ID_BYTES


MAX_COUNT = 1_000_000
MAX_GRAPH_LANES = 4096
MAX_TRAJECTORY_LENGTH = 100_000


class BinaryFormatError(ValueError):
    pass


def load_puffer_binary(binary_path: str | Path) -> dict:
    try:
        with Path(binary_path).open("rb") as f:
            return _read_body(f)
    except OSError as exc:
        raise BinaryFormatError(f"Failed to read binary file: {binary_path}") from exc


def _read_body(f):
    num_agents = _read_count(f, "agents")
    num_roads = _read_count(f, "roads")
    num_traffic = _read_count(f, "traffic controls")
    num_objects = _read_count(f, "objects")

    dynamic_agents = [_read_dynamic_agent(f) for _ in range(num_agents)]
    road_map_elements = [_read_road_map_element(f) for _ in range(num_roads)]
    traffic_control_elements = [_read_traffic_control_element(f) for _ in range(num_traffic)]
    objects = [_read_object(f) for _ in range(num_objects)]
    lane_graph_distances = _read_lane_graph(f)

    scenario_id = _read_string(f, METADATA_ID_BYTES)
    map_id = _read_int32(f)
    dataset = _read_string(f, METADATA_DATASET_BYTES)
    scenario_length = _read_int32(f)
    dt = _read_float32(f)
    objects_of_interest = _read_int_list(f, "objects_of_interest")
    tracks_to_predict = _read_int_list(f, "tracks_to_predict")

    return {
        "agents": dynamic_agents,
        "road_map_elements": road_map_elements,
        "traffic_control_elements": traffic_control_elements,
        "objects": objects,
        "metadata": {
            "id": scenario_id,
            "num_agents": num_agents,
            "num_roads": num_roads,
            "num_traffic": num_traffic,
            "num_objects": num_objects,
            "map_id": map_id,
            "dataset": dataset,
            "scenario_length": scenario_length,
            "dt": dt,
            "objects_of_interest": objects_of_interest,
            "tracks_to_predict": tracks_to_predict,
            "lane_graph_distances": lane_graph_distances,
        },
    }


def _read_dynamic_agent(f):
    agent_id = _read_int32(f)
    agent_type = _read_int32(f)
    trajectory_length = _read_count(f, "trajectory", limit=MAX_TRAJECTORY_LENGTH)
    states = _read_dynamic_states(f, trajectory_length)
    route = _read_int_list(f, "route")
    _read_exact(f, 16)
    return {"id": agent_id, "type": agent_type, "states": states, "route": route}


def _read_object(f):
    object_id = _read_int32(f)
    object_type = _read_int32(f)
    trajectory_length = _read_count(f, "trajectory", limit=MAX_TRAJECTORY_LENGTH)
    return {"id": object_id, "type": object_type, "states": _read_dynamic_states(f, trajectory_length)}


def _read_dynamic_states(f, trajectory_length):
    x = _read_float_array(f, trajectory_length)
    y = _read_float_array(f, trajectory_length)
    z = _read_float_array(f, trajectory_length)
    heading = _read_float_array(f, trajectory_length)
    vx = _read_float_array(f, trajectory_length)
    vy = _read_float_array(f, trajectory_length)
    length = _read_float_array(f, trajectory_length)
    width = _read_float_array(f, trajectory_length)
    height = _read_float_array(f, trajectory_length)
    valid = _read_int_array(f, trajectory_length)
    return {
        "xyz": np.stack([x, y, z], axis=1),
        "heading": heading,
        "velocity": np.stack([vx, vy], axis=1),
        "length": length,
        "width": width,
        "height": height,
        "valid": valid,
    }


def _read_road_map_element(f):
    road_id = _read_int32(f)
    road_type = _read_int32(f)
    segment_length = _read_count(f, "road geometry", limit=MAX_COUNT)
    xyz = np.stack(
        [
            _read_float_array(f, segment_length),
            _read_float_array(f, segment_length),
            _read_float_array(f, segment_length),
        ],
        axis=1,
    )
    heading = _read_float_array(f, segment_length)
    entry_lanes = []
    exit_lanes = []
    speed_limit = 0.0
    if is_road_lane(road_type):
        entry_lanes = _read_int_list(f, "entry lanes")
        exit_lanes = _read_int_list(f, "exit lanes")
        speed_limit = _read_float32(f)
    return {
        "id": road_id,
        "type": road_type,
        "xyz": xyz,
        "heading": heading,
        "entry_lanes": entry_lanes,
        "exit_lanes": exit_lanes,
        "speed_limit": speed_limit,
    }


def _read_traffic_control_element(f):
    traffic_id = _read_int32(f)
    traffic_type = _read_int32(f)
    stop_line = np.array(
        [
            [_read_float32(f), _read_float32(f), _read_float32(f)],
            [_read_float32(f), _read_float32(f), _read_float32(f)],
        ],
    )
    heading = _read_float32(f)
    states = _read_int_list(f, "traffic states")
    controlled_lanes = _read_int_list(f, "controlled lanes")
    return {
        "id": traffic_id,
        "type": traffic_type,
        "stop_line": stop_line,
        "heading": heading,
        "states": states,
        "controlled_lanes": controlled_lanes,
    }


def _read_lane_graph(f):
    n_graph = _read_count(f, "lane graph", limit=MAX_GRAPH_LANES)
    if n_graph == 0:
        return None
    lane_ids = _read_int_list(f, "graph lane ids", expected=n_graph)
    lane_lengths = _read_float_array(f, n_graph)
    distances = _read_float_array(f, n_graph * n_graph).reshape(n_graph, n_graph)
    return {"lane_ids": lane_ids, "lane_lengths": lane_lengths, "distances": distances}


def _read_int_list(f, label, expected=None):
    n = _read_count(f, label) if expected is None else expected
    if n == 0:
        return []
    if expected is not None and n != expected:
        raise BinaryFormatError(f"Unexpected {label} count: {n} != {expected}")
    return _read_int_array(f, n).tolist()


def _read_string(f, size):
    try:
        return _read_exact(f, size).split(b"\0", 1)[0].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BinaryFormatError("Invalid UTF-8 string in binary metadata") from exc


def _read_count(f, label, limit=MAX_COUNT):
    value = _read_int32(f)
    if value < 0 or value > limit:
        raise BinaryFormatError(f"Invalid {label} count: {value}")
    return value


def _read_float32(f):
    return struct.unpack("<f", _read_exact(f, 4))[0]


def _read_int32(f):
    return struct.unpack("<i", _read_exact(f, 4))[0]


def _read_float_array(f, n):
    return np.frombuffer(_read_exact(f, 4 * n), dtype=np.float32).copy()


def _read_int_array(f, n):
    return np.frombuffer(_read_exact(f, 4 * n), dtype=np.int32).copy()


def _read_exact(f, size):
    data = f.read(size)
    if len(data) != size:
        raise BinaryFormatError("Unexpected end of file")
    return data
