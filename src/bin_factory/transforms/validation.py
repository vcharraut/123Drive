"""Validation helpers for the PufferDrive dict format.

Validation modes:
0. off
1. schema
2. semantic
"""

import numpy as np


VALIDATION_OFF = "off"
VALIDATION_SCHEMA = "schema"
VALIDATION_SEMANTIC = "semantic"

VALID_AGENT_TYPES = {1, 2, 3, 4}
VALID_OBJECT_TYPES = {1, 2, 3, 4, 5}
LANE_TYPES = set(range(10))
VALID_TL_STATES = {0, 1, 2, 3, 4}

REQUIRED_TOP_LEVEL_KEYS = {
    "scenario_id",
    "agents",
    "objects",
    "road_map_elements",
    "traffic_control_elements",
    "metadata",
}
REQUIRED_METADATA_KEYS = {
    "dataset_name",
    "scenario_length",
    "sdc_index",
    "timestep_seconds",
    "objects_of_interests",
    "tracks_to_predict",
}
REQUIRED_AGENT_KEYS = {"id", "type", "states", "route"}
REQUIRED_OBJECT_KEYS = {"id", "type", "states"}
REQUIRED_AGENT_STATE_KEYS = {"xyz", "heading", "velocity", "length", "width", "height", "valid"}
REQUIRED_ROAD_KEYS = {"id", "type", "xyz"}
REQUIRED_LANE_KEYS = {"entry_lanes", "exit_lanes", "speed_limit"}
REQUIRED_TRAFFIC_CONTROL_KEYS = {"id", "type", "xyz", "states", "controlled_lanes"}


class ValidationError(Exception):
    pass


def validate_puffer_dict(
    puffer_dict: dict,
    validation_mode: str | int = VALIDATION_SCHEMA,
    position_jump_threshold: float = 50.0,
) -> list[str]:
    if validation_mode == VALIDATION_OFF:
        return []

    errors, context = _validate_schema(puffer_dict)
    if errors or validation_mode == VALIDATION_SCHEMA:
        return errors

    return errors + _validate_semantics(context, position_jump_threshold=position_jump_threshold)


def _validate_schema(puffer_dict: dict) -> tuple[list[str], dict]:
    errors = []
    if not isinstance(puffer_dict, dict):
        return [f"Puffer dict must be dict, got {type(puffer_dict).__name__}"], {}

    _validate_top_level_schema(puffer_dict, errors)

    metadata = puffer_dict.get("metadata", {})
    agents = puffer_dict.get("agents", [])
    objects = puffer_dict.get("objects", [])
    roads = puffer_dict.get("road_map_elements", [])
    traffic_controls = puffer_dict.get("traffic_control_elements", [])

    expected_length = _validate_metadata_schema(metadata, errors)
    lane_ids = _validate_roads_schema(roads, errors)
    agent_ids = _validate_agents_schema(agents, expected_length, errors)
    object_ids = _validate_objects_schema(objects, expected_length, errors)
    traffic_control_ids = _validate_traffic_controls_schema(traffic_controls, expected_length, errors)

    context = {
        "puffer_dict": puffer_dict,
        "metadata": metadata,
        "agents": agents,
        "objects": objects,
        "roads": roads,
        "traffic_controls": traffic_controls,
        "expected_length": expected_length,
        "lane_ids": lane_ids,
        "agent_ids": agent_ids,
        "object_ids": object_ids,
        "traffic_control_ids": traffic_control_ids,
    }
    return errors, context


def _validate_top_level_schema(puffer_dict: dict, errors: list[str]):
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in puffer_dict:
            errors.append(f"Missing required top-level key: '{key}'")

    if "scenario_id" in puffer_dict and not isinstance(puffer_dict["scenario_id"], str):
        errors.append(f"'scenario_id' must be string, got {type(puffer_dict['scenario_id']).__name__}")

    for key, expected_type in [
        ("agents", list),
        ("objects", list),
        ("road_map_elements", list),
        ("traffic_control_elements", list),
        ("metadata", dict),
    ]:
        if key in puffer_dict and not isinstance(puffer_dict[key], expected_type):
            errors.append(f"'{key}' must be {expected_type.__name__}, got {type(puffer_dict[key]).__name__}")


def _validate_metadata_schema(metadata: dict, errors: list[str]) -> int:
    if not isinstance(metadata, dict):
        errors.append(f"'metadata' must be dict, got {type(metadata).__name__}")
        return 0

    for key in REQUIRED_METADATA_KEYS:
        if key not in metadata:
            errors.append(f"Missing metadata key: '{key}'")

    scenario_length = metadata.get("scenario_length")
    if "dataset_name" in metadata and not isinstance(metadata["dataset_name"], str):
        errors.append(f"metadata['dataset_name'] must be str, got {type(metadata['dataset_name']).__name__}")
    if "scenario_length" in metadata and not _is_int_like(scenario_length):
        errors.append(f"metadata['scenario_length'] must be int, got {type(scenario_length).__name__}")
        scenario_length = None
    elif _is_int_like(scenario_length) and int(scenario_length) < 0:
        errors.append("metadata['scenario_length'] must be non-negative")
    if "sdc_index" in metadata and not _is_int_like(metadata["sdc_index"]):
        errors.append(f"metadata['sdc_index'] must be int, got {type(metadata['sdc_index']).__name__}")
    if "timestep_seconds" in metadata and not _is_number(metadata["timestep_seconds"]):
        errors.append(
            f"metadata['timestep_seconds'] must be numeric, got {type(metadata['timestep_seconds']).__name__}"
        )
    timestep_seconds = metadata.get("timestep_seconds", 0)
    if (
        _is_int_like(scenario_length)
        and int(scenario_length) > 0
        and _is_number(timestep_seconds)
        and timestep_seconds <= 0
    ):
        errors.append("metadata['timestep_seconds'] must be > 0")

    for key in ["objects_of_interests", "tracks_to_predict"]:
        if key in metadata and not isinstance(metadata[key], list):
            errors.append(f"metadata['{key}'] must be list, got {type(metadata[key]).__name__}")

    return int(scenario_length) if _is_int_like(scenario_length) else 0


def _validate_agents_schema(agents: list, expected_length: int, errors: list[str]) -> set[int]:
    if not isinstance(agents, list):
        errors.append(f"'agents' must be list, got {type(agents).__name__}")
        return set()

    seen_ids = set()
    for index, agent in enumerate(agents):
        if not isinstance(agent, dict):
            errors.append(f"Agent at index {index} must be dict")
            continue

        for key in REQUIRED_AGENT_KEYS:
            if key not in agent:
                errors.append(f"Agent {index} missing key: '{key}'")

        agent_id = _check_id(agent, index, seen_ids, errors, "Agent")

        if "type" in agent:
            agent_type = agent["type"]
            if not _is_int_like(agent_type):
                errors.append(f"Agent {agent_id} type must be int, got {type(agent_type).__name__}")
            elif agent_type not in VALID_AGENT_TYPES:
                errors.append(f"Agent {agent_id} has invalid type {agent_type}")

        if "states" in agent:
            _validate_dynamic_states_schema("Agent", agent_id, agent["states"], expected_length, errors)
        if "route" in agent:
            _validate_route_schema(agent_id, agent["route"], errors)

    return seen_ids


def _validate_objects_schema(objects: list, expected_length: int, errors: list[str]) -> set[int]:
    if not isinstance(objects, list):
        errors.append(f"'objects' must be list, got {type(objects).__name__}")
        return set()

    seen_ids = set()
    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            errors.append(f"Object at index {index} must be dict")
            continue

        for key in REQUIRED_OBJECT_KEYS:
            if key not in obj:
                errors.append(f"Object {index} missing key: '{key}'")

        object_id = _check_id(obj, index, seen_ids, errors, "Object")

        if "type" in obj:
            object_type = obj["type"]
            if not _is_int_like(object_type):
                errors.append(f"Object {object_id} type must be int, got {type(object_type).__name__}")
            elif object_type not in VALID_OBJECT_TYPES:
                errors.append(f"Object {object_id} has invalid type {object_type}")

        if "states" in obj:
            _validate_dynamic_states_schema("Object", object_id, obj["states"], expected_length, errors)

    return seen_ids


def _validate_dynamic_states_schema(prefix: str, item_id, states: dict, expected_length: int, errors: list[str]):
    if not isinstance(states, dict):
        errors.append(f"{prefix} {item_id} states must be dict, got {type(states).__name__}")
        return

    for key in REQUIRED_AGENT_STATE_KEYS:
        if key not in states:
            errors.append(f"{prefix} {item_id} states missing key: '{key}'")

    lengths = {}
    lengths |= _validate_array(states, item_id, "xyz", ndim=2, shape1=3, errors=errors, prefix=prefix)
    lengths |= _validate_array(states, item_id, "velocity", ndim=2, shape1=2, errors=errors, prefix=prefix)
    for key in ["heading", "length", "width", "height", "valid"]:
        lengths |= _validate_array(states, item_id, key, ndim=1, errors=errors, prefix=prefix)

    if len(set(lengths.values())) > 1:
        errors.append(f"{prefix} {item_id} has inconsistent state array lengths: {lengths}")
    if expected_length > 0:
        for key, length in lengths.items():
            if length != expected_length:
                errors.append(f"{prefix} {item_id} state '{key}' has length {length}, expected {expected_length}")


def _validate_route_schema(agent_id, route, errors: list[str]):
    if not isinstance(route, (list, np.ndarray)):
        errors.append(f"Agent {agent_id} route must be list or ndarray, got {type(route).__name__}")
        return
    if isinstance(route, np.ndarray) and route.ndim != 1:
        errors.append(f"Agent {agent_id} route must be 1D, got shape {route.shape}")
        return
    if isinstance(route, list) and not all(_is_int_like(lane_id) for lane_id in route):
        errors.append(f"Agent {agent_id} route must contain only lane ids")


def _validate_roads_schema(roads: list, errors: list[str]) -> set[int]:
    if not isinstance(roads, list):
        errors.append(f"'road_map_elements' must be list, got {type(roads).__name__}")
        return set()

    seen_ids = set()
    lane_ids = set()
    for index, road in enumerate(roads):
        if not isinstance(road, dict):
            errors.append(f"Road element {index} must be dict")
            continue

        for key in REQUIRED_ROAD_KEYS:
            if key not in road:
                errors.append(f"Road {index} missing key: '{key}'")

        road_id = _check_id(road, index, seen_ids, errors, "Road")
        road_type = road.get("type")

        if "type" in road and not _is_int_like(road_type):
            errors.append(f"Road {road_id} type must be int, got {type(road_type).__name__}")

        _validate_array(road, road_id, "xyz", ndim=2, shape1=3, errors=errors, prefix="Road")

        if road_type in LANE_TYPES:
            if _is_int_like(road_id):
                lane_ids.add(int(road_id))
            for key in REQUIRED_LANE_KEYS:
                if key not in road:
                    errors.append(f"Lane {road_id} missing key: '{key}'")
            for key in ["entry_lanes", "exit_lanes"]:
                if key in road and not isinstance(road[key], list):
                    errors.append(f"Lane {road_id} {key} must be list, got {type(road[key]).__name__}")
            if "speed_limit" in road and not _is_number(road["speed_limit"]):
                errors.append(f"Lane {road_id} speed_limit must be numeric, got {type(road['speed_limit']).__name__}")

    return lane_ids


def _validate_traffic_controls_schema(traffic_controls: list, expected_length: int, errors: list[str]) -> set[int]:
    if not isinstance(traffic_controls, list):
        errors.append(f"'traffic_control_elements' must be list, got {type(traffic_controls).__name__}")
        return set()

    seen_ids = set()
    for index, control in enumerate(traffic_controls):
        if not isinstance(control, dict):
            errors.append(f"Traffic control {index} must be dict")
            continue

        for key in REQUIRED_TRAFFIC_CONTROL_KEYS:
            if key not in control:
                errors.append(f"Traffic control {index} missing key: '{key}'")

        control_id = _check_id(control, index, seen_ids, errors, "Traffic control")

        if "type" in control and not _is_int_like(control["type"]):
            errors.append(f"Traffic control {control_id} type must be int, got {type(control['type']).__name__}")

        xyz_lengths = _validate_array(control, control_id, "xyz", ndim=1, errors=errors, prefix="Traffic control")
        if "xyz" in xyz_lengths and xyz_lengths["xyz"] != 3:
            errors.append(
                f"Traffic control {control_id} xyz must be shape (3,), got {np.asarray(control['xyz']).shape}"
            )

        states_lengths = _validate_array(control, control_id, "states", ndim=1, errors=errors, prefix="Traffic control")
        if "states" in states_lengths and expected_length > 0 and states_lengths["states"] != expected_length:
            errors.append(
                f"Traffic control {control_id} states has length {states_lengths['states']}, expected {expected_length}",
            )
        if "controlled_lanes" in control and not isinstance(control["controlled_lanes"], list):
            errors.append(
                f"Traffic control {control_id} controlled_lanes must be list, got "
                f"{type(control['controlled_lanes']).__name__}",
            )

    return seen_ids


def _check_id(item, index, seen_ids, errors, prefix):
    item_id = item.get("id", "?")
    if "id" in item and not _is_int_like(item_id):
        errors.append(f"{prefix} {index} id must be int, got {type(item_id).__name__}")
    elif _is_int_like(item_id) and item_id in seen_ids:
        errors.append(f"Duplicate {prefix.lower()} id: {item_id}")
    elif "id" in item:
        seen_ids.add(int(item_id))
    return item_id


def _validate_array(
    container, item_id, key, ndim: int, errors: list[str], shape1=None, prefix="Agent"
) -> dict[str, int]:
    if key not in container:
        return {}

    value = container[key]
    if not isinstance(value, np.ndarray):
        errors.append(f"{prefix} {item_id} {key} must be ndarray, got {type(value).__name__}")
        return {}
    if value.ndim != ndim:
        errors.append(f"{prefix} {item_id} {key} must be {ndim}D, got shape {value.shape}")
        return {}
    if shape1 is not None and value.shape[1] != shape1:
        errors.append(f"{prefix} {item_id} {key} must be shape (N,{shape1}), got {value.shape}")
        return {}
    return {key: len(value)}


def _validate_semantics(context: dict, position_jump_threshold: float) -> list[str]:
    errors = []
    metadata = context["metadata"]
    agents = context["agents"]
    objects = context["objects"]
    lane_ids = context["lane_ids"]
    roads = context["roads"]
    traffic_controls = context["traffic_controls"]

    _validate_sdc_index(metadata, agents, errors)
    _validate_lane_topology(roads, lane_ids, errors)
    _validate_agent_routes(agents, lane_ids, errors)
    _validate_traffic_control_links(traffic_controls, lane_ids, errors)
    _validate_finite_values(agents, objects, roads, traffic_controls, errors)
    _validate_road_geometry(roads, errors)
    _validate_traffic_control_states(traffic_controls, errors)
    _validate_ego_semantics(metadata, agents, position_jump_threshold, errors)
    return errors


def _validate_sdc_index(metadata: dict, agents: list, errors: list[str]):
    if not isinstance(metadata, dict) or not _is_int_like(metadata.get("sdc_index")):
        return
    sdc_index = int(metadata["sdc_index"])
    if agents and not (0 <= sdc_index < len(agents)):
        errors.append(f"sdc_index {sdc_index} out of range (have {len(agents)} agents)")


def _validate_lane_topology(roads: list, lane_ids: set[int], errors: list[str]):
    lanes_by_id = {
        int(road["id"]): road for road in roads if road.get("type") in LANE_TYPES and _is_int_like(road.get("id"))
    }

    for road in roads:
        if road.get("type") not in LANE_TYPES:
            continue
        lane_id = road.get("id", "?")
        for key in ["entry_lanes", "exit_lanes"]:
            for ref_lane_id in road.get(key, []):
                if not _is_int_like(ref_lane_id):
                    errors.append(f"Lane {lane_id} {key} must contain only lane ids")
                elif int(ref_lane_id) not in lane_ids:
                    errors.append(f"Lane {lane_id} references non-existent lane {ref_lane_id} in {key}")

        if not _is_int_like(lane_id):
            continue

        for fwd_key, rev_key in [("entry_lanes", "exit_lanes"), ("exit_lanes", "entry_lanes")]:
            for ref_id in road.get(fwd_key, []):
                other_lane = lanes_by_id.get(int(ref_id)) if _is_int_like(ref_id) else None
                if other_lane is None:
                    continue
                if other_lane.get("type") not in [1, 2]:  # Only validate lane-lane links
                    continue
                if int(lane_id) not in other_lane.get(rev_key, []):
                    errors.append(
                        f"Lane {lane_id} lists {ref_id} as {fwd_key[:-1]}, but lane {ref_id} "
                        f"does not list {lane_id} in {rev_key}"
                    )


def _validate_agent_routes(agents: list, lane_ids: set[int], errors: list[str]):
    for agent in agents:
        agent_id = agent.get("id", "?")
        route = agent.get("route", [])
        for lane_id in route:
            if not _is_int_like(lane_id):
                errors.append(f"Agent {agent_id} route must contain only lane ids")
            elif int(lane_id) not in lane_ids:
                errors.append(f"Agent {agent_id} route references non-existent lane {lane_id}")


def _validate_traffic_control_links(traffic_controls: list, lane_ids: set[int], errors: list[str]):
    for control in traffic_controls:
        control_id = control.get("id", "?")
        for lane_id in control.get("controlled_lanes", []):
            if not _is_int_like(lane_id):
                errors.append(f"Traffic control {control_id} controlled_lanes must contain only lane ids")
            elif int(lane_id) not in lane_ids:
                errors.append(f"Traffic control {control_id} references non-existent lane {lane_id}")


def _validate_finite_values(agents: list, objects: list, roads: list, traffic_controls: list, errors: list[str]):
    for agent in agents:
        agent_id = agent.get("id", "?")
        states = agent.get("states", {})
        for key in REQUIRED_AGENT_STATE_KEYS:
            if key in states and not _is_finite_array(states[key]):
                errors.append(f"Agent {agent_id} states.{key} contains NaN or Inf")

    for obj in objects:
        object_id = obj.get("id", "?")
        states = obj.get("states", {})
        for key in REQUIRED_AGENT_STATE_KEYS:
            if key in states and not _is_finite_array(states[key]):
                errors.append(f"Object {object_id} states.{key} contains NaN or Inf")

    for road in roads:
        road_id = road.get("id", "?")
        if "xyz" in road and not _is_finite_array(road["xyz"]):
            errors.append(f"Road {road_id} xyz contains NaN or Inf")
        if "speed_limit" in road and not _is_finite_scalar(road["speed_limit"]):
            errors.append(f"Lane {road_id} speed_limit contains NaN or Inf")

    for control in traffic_controls:
        control_id = control.get("id", "?")
        for key in ["xyz", "states"]:
            if key in control and not _is_finite_array(control[key]):
                errors.append(f"Traffic control {control_id} {key} contains NaN or Inf")


def _validate_road_geometry(roads: list, errors: list[str]):
    for road in roads:
        road_id = road.get("id", "?")
        xyz = road.get("xyz")
        if isinstance(xyz, np.ndarray) and len(xyz) < 2:
            errors.append(f"Road {road_id} xyz must have at least 2 points")


def _validate_traffic_control_states(traffic_controls: list, errors: list[str]):
    for control in traffic_controls:
        control_id = control.get("id", "?")
        states = control.get("states")
        if isinstance(states, np.ndarray) and not np.all(np.isin(states, list(VALID_TL_STATES))):
            errors.append(f"Traffic control {control_id} has invalid light states")


def _validate_ego_semantics(metadata: dict, agents: list, position_jump_threshold: float, errors: list[str]):
    if not _is_int_like(metadata.get("sdc_index")):
        return

    sdc_index = int(metadata["sdc_index"])
    if not (0 <= sdc_index < len(agents)):
        return

    ego = agents[sdc_index]
    ego_id = ego.get("id", sdc_index)
    states = ego.get("states", {})
    if not {"xyz", "valid", "length", "width", "height", "velocity"}.issubset(states):
        return

    xyz = states["xyz"]
    valid = states["valid"].astype(bool)
    velocity = states["velocity"]
    length = states["length"]
    width = states["width"]
    height = states["height"]

    if np.any(valid):
        valid_indices = np.flatnonzero(valid)
        first_valid = int(valid_indices[0])
        last_valid = int(valid_indices[-1])
        if not np.all(valid[first_valid : last_valid + 1]):
            errors.append(f"Ego agent {ego_id} has gaps in valid timesteps")

    for i in range(len(xyz) - 1):
        if valid[i] and valid[i + 1]:
            dist = float(np.linalg.norm(xyz[i + 1, :2] - xyz[i, :2]))
            if dist > max(position_jump_threshold, 1.0):
                errors.append(f"Ego agent {ego_id} teleports at timestep {i}: moved {dist:.2f}m")

    for key, values in [("length", length), ("width", width), ("height", height)]:
        if np.any(values[valid] <= 0):
            errors.append(f"Ego agent {ego_id} has non-positive {key}")

    speeds = np.linalg.norm(velocity[valid, :2], axis=1) if np.any(valid) else np.array([])
    if np.any(speeds > 60.0):
        errors.append(f"Ego agent {ego_id} speed exceeds 60.0 m/s")


def _is_int_like(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _is_finite_array(value) -> bool:
    arr = np.asarray(value)
    return np.issubdtype(arr.dtype, np.number) and bool(np.all(np.isfinite(arr)))


def _is_finite_scalar(value) -> bool:
    return _is_number(value) and np.isfinite(value)
