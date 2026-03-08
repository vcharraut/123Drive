"""
Validation helpers for the PufferDrive dict format.

Validation levels:
0. Disabled
1. Mandatory schema/integrity checks only
2. Mandatory errors + physical checks as warnings
3. Mandatory errors + hard physical errors + soft physical warnings
4. Mandatory errors + all physical checks as errors
"""

import numpy as np


AGENT_TYPE_MAP = {1: "VEHICLE", 2: "PEDESTRIAN", 3: "CYCLIST", 4: "OTHER"}
VALID_AGENT_TYPES = set(AGENT_TYPE_MAP.keys())

LANE_TYPES = set(range(1, 10))
VALID_TL_STATES = set(range(9))

_REQUIRED_TOP_LEVEL_KEYS = {
    "scenario_id",
    "agents",
    "road_map_elements",
    "traffic_control_elements",
    "metadata",
}
_REQUIRED_METADATA_KEYS = {
    "dataset_name",
    "scenario_length",
    "sdc_index",
    "timestep_seconds",
    "objects_of_interests",
    "tracks_to_predict",
}
_REQUIRED_AGENT_KEYS = {"id", "type", "states", "routes"}
_REQUIRED_AGENT_STATE_KEYS = {"xyz", "heading", "velocity", "length", "width", "height", "valid"}
_REQUIRED_ROAD_KEYS = {"id", "type", "xyz"}
_LANE_EXTRA_KEYS = {"entry_lanes", "exit_lanes", "speed_limit"}
_REQUIRED_TRAFFIC_CONTROL_KEYS = {"id", "type", "xyz", "states", "controlled_lanes"}

_SOFT = "soft"
_HARD = "hard"


def validate_puffer_dict(
    puffer_dict: dict,
    validation_level: int = 1,
    position_jump_threshold: float = 50.0,
    velocity_tolerance: float = 2.0,
    heading_tolerance_deg: float = 30.0,
) -> tuple[list[str], list[str]]:
    validation_level = min(max(int(validation_level), 0), 4)
    if validation_level == 0:
        return [], []

    mandatory_errors, mandatory_warnings = _collect_mandatory_issues(puffer_dict)
    if validation_level == 1 or mandatory_errors:
        return mandatory_errors, mandatory_warnings

    physical_issues = _collect_physics_issues(
        puffer_dict,
        position_jump_threshold=position_jump_threshold,
        velocity_tolerance=velocity_tolerance,
        heading_tolerance_deg=heading_tolerance_deg,
    )
    physics_errors, physics_warnings = _classify_physics_issues(physical_issues, validation_level)

    errors = mandatory_errors + physics_errors
    warnings = mandatory_warnings + physics_warnings
    return errors, warnings


def _collect_mandatory_issues(puffer_dict: dict) -> tuple[list[str], list[str]]:
    errors = []
    warnings = []

    if not isinstance(puffer_dict, dict):
        return [f"Puffer dict must be dict, got {type(puffer_dict).__name__}"], []

    _validate_top_level(puffer_dict, errors)

    metadata = puffer_dict.get("metadata", {})
    agents = puffer_dict.get("agents", [])
    roads = puffer_dict.get("road_map_elements", [])
    controls = puffer_dict.get("traffic_control_elements", [])
    expected_length = metadata.get("scenario_length", 0) if isinstance(metadata, dict) else 0

    _validate_metadata(metadata, errors)
    _validate_dynamic_agents(agents, expected_length, errors, warnings)
    lane_ids = _validate_road_map_elements(roads, errors)
    _validate_traffic_control_elements(controls, expected_length, lane_ids, errors)
    _validate_scenario_coherence(metadata, agents, errors, warnings)

    return errors, warnings


def _collect_physics_issues(
    puffer_dict: dict,
    position_jump_threshold: float,
    velocity_tolerance: float,
    heading_tolerance_deg: float,
) -> list[tuple[str, str]]:
    metadata = puffer_dict.get("metadata", {})
    dt = metadata.get("timestep_seconds", 0.1)
    agents = puffer_dict.get("agents", [])
    roads = puffer_dict.get("road_map_elements", [])
    controls = puffer_dict.get("traffic_control_elements", [])

    return [
        *_collect_agent_physics_issues(
            agents,
            dt,
            position_jump_threshold=position_jump_threshold,
            velocity_tolerance=velocity_tolerance,
            heading_tolerance_deg=heading_tolerance_deg,
        ),
        *_collect_road_physics_issues(roads),
        *_collect_traffic_control_physics_issues(controls, roads),
    ]


def _classify_physics_issues(
    issues: list[tuple[str, str]],
    validation_level: int,
) -> tuple[list[str], list[str]]:
    if validation_level <= 2:
        return [], [message for _, message in issues]
    if validation_level == 3:
        return (
            [message for severity, message in issues if severity == _HARD],
            [message for severity, message in issues if severity == _SOFT],
        )
    return [message for _, message in issues], []


def _validate_top_level(puffer_dict: dict, errors: list[str]):
    for key in _REQUIRED_TOP_LEVEL_KEYS:
        if key not in puffer_dict:
            errors.append(f"Missing required top-level key: '{key}'")

    if "scenario_id" in puffer_dict and not isinstance(puffer_dict["scenario_id"], str):
        errors.append(f"'scenario_id' must be string, got {type(puffer_dict['scenario_id']).__name__}")

    for key, expected_type in [
        ("agents", list),
        ("road_map_elements", list),
        ("traffic_control_elements", list),
        ("metadata", dict),
    ]:
        if key in puffer_dict and not isinstance(puffer_dict[key], expected_type):
            errors.append(f"'{key}' must be {expected_type.__name__}, got {type(puffer_dict[key]).__name__}")


def _validate_metadata(metadata: dict, errors: list[str]):
    if not isinstance(metadata, dict):
        errors.append(f"'metadata' must be dict, got {type(metadata).__name__}")
        return

    for key in _REQUIRED_METADATA_KEYS:
        if key not in metadata:
            errors.append(f"Missing metadata key: '{key}'")

    if "dataset_name" in metadata and not isinstance(metadata["dataset_name"], str):
        errors.append(f"metadata['dataset_name'] must be str, got {type(metadata['dataset_name']).__name__}")

    scenario_length = metadata.get("scenario_length")

    if "scenario_length" in metadata:
        if not _is_int_like(scenario_length):
            errors.append(f"metadata['scenario_length'] must be int, got {type(metadata['scenario_length']).__name__}")
        elif scenario_length < 0:
            errors.append("metadata['scenario_length'] must be non-negative")

    if "sdc_index" in metadata and not _is_int_like(metadata["sdc_index"]):
        errors.append(f"metadata['sdc_index'] must be int, got {type(metadata['sdc_index']).__name__}")

    if "timestep_seconds" in metadata:
        if not _is_number(metadata["timestep_seconds"]):
            errors.append(
                f"metadata['timestep_seconds'] must be numeric, got {type(metadata['timestep_seconds']).__name__}",
            )
        elif scenario_length and metadata["timestep_seconds"] <= 0:
            errors.append("metadata['timestep_seconds'] must be > 0")

    for key in ["objects_of_interests", "tracks_to_predict"]:
        if key in metadata and not isinstance(metadata[key], list):
            errors.append(f"metadata['{key}'] must be list, got {type(metadata[key]).__name__}")


def _validate_dynamic_agents(
    agents: list,
    expected_length: int,
    errors: list[str],
    warnings: list[str],
):
    if not isinstance(agents, list):
        errors.append(f"'agents' must be list, got {type(agents).__name__}")
        return

    seen_ids = set()
    for i, agent in enumerate(agents):
        if not isinstance(agent, dict):
            errors.append(f"Agent at index {i} must be dict")
            continue

        for key in _REQUIRED_AGENT_KEYS:
            if key not in agent:
                errors.append(f"Agent {i} missing key: '{key}'")

        agent_id = agent.get("id", "?")
        if "id" in agent and not _is_int_like(agent_id):
            errors.append(f"Agent {i} id must be int, got {type(agent_id).__name__}")
        elif "id" in agent and agent_id in seen_ids:
            errors.append(f"Duplicate agent id: {agent_id}")
        elif "id" in agent:
            seen_ids.add(agent_id)

        if _is_int_like(agent_id) and agent_id != i:
            warnings.append(f"Agent {i} has id={agent_id}, expected sequential id={i}")

        if "type" in agent:
            agent_type = agent["type"]
            if not _is_int_like(agent_type):
                errors.append(f"Agent {agent_id} type must be int, got {type(agent_type).__name__}")
            elif agent_type not in VALID_AGENT_TYPES:
                errors.append(f"Agent {agent_id} has invalid type {agent_type}, must be in {VALID_AGENT_TYPES}")

        if "states" in agent:
            _validate_agent_states(agent_id, agent["states"], expected_length, errors)

        if "routes" in agent:
            _validate_routes(agent_id, agent["routes"], errors)


def _validate_agent_states(agent_id, states: dict, expected_length: int, errors: list[str]):
    if not isinstance(states, dict):
        errors.append(f"Agent {agent_id} states must be dict, got {type(states).__name__}")
        return

    for key in _REQUIRED_AGENT_STATE_KEYS:
        if key not in states:
            errors.append(f"Agent {agent_id} states missing key: '{key}'")

    lengths = {}

    if "xyz" in states:
        xyz = states["xyz"]
        if not isinstance(xyz, np.ndarray):
            errors.append(f"Agent {agent_id} states.xyz must be ndarray, got {type(xyz).__name__}")
        elif xyz.ndim != 2 or xyz.shape[1] != 3:
            errors.append(f"Agent {agent_id} states.xyz must be shape (T,3), got {xyz.shape}")
        else:
            lengths["xyz"] = xyz.shape[0]

    if "velocity" in states:
        velocity = states["velocity"]
        if not isinstance(velocity, np.ndarray):
            errors.append(f"Agent {agent_id} states.velocity must be ndarray, got {type(velocity).__name__}")
        elif velocity.ndim != 2 or velocity.shape[1] != 2:
            errors.append(f"Agent {agent_id} states.velocity must be shape (T,2), got {velocity.shape}")
        else:
            lengths["velocity"] = velocity.shape[0]

    for key in ["heading", "length", "width", "height", "valid"]:
        if key not in states:
            continue
        arr = states[key]
        if not isinstance(arr, np.ndarray):
            errors.append(f"Agent {agent_id} states.{key} must be ndarray, got {type(arr).__name__}")
            continue
        if arr.ndim != 1:
            errors.append(f"Agent {agent_id} states.{key} must be 1D, got shape {arr.shape}")
            continue
        lengths[key] = len(arr)

    if lengths:
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            errors.append(f"Agent {agent_id} has inconsistent state array lengths: {lengths}")
        if expected_length > 0:
            wrong_lengths = {key: length for key, length in lengths.items() if length != expected_length}
            for key, length in wrong_lengths.items():
                errors.append(f"Agent {agent_id} state '{key}' has length {length}, expected {expected_length}")


def _validate_routes(agent_id, routes, errors: list[str]):
    if not isinstance(routes, list):
        errors.append(f"Agent {agent_id} routes must be list, got {type(routes).__name__}")
        return

    for route_idx, route in enumerate(routes):
        if isinstance(route, np.ndarray):
            if route.ndim != 1:
                errors.append(f"Agent {agent_id} route {route_idx} must be 1D, got shape {route.shape}")
            continue
        if not isinstance(route, list):
            errors.append(f"Agent {agent_id} route {route_idx} must be list or ndarray")
            continue
        if not all(_is_int_like(lane_id) for lane_id in route):
            errors.append(f"Agent {agent_id} route {route_idx} must contain only lane ids")


def _validate_road_map_elements(roads: list, errors: list[str]) -> set[int]:
    if not isinstance(roads, list):
        errors.append(f"'road_map_elements' must be list, got {type(roads).__name__}")
        return set()

    lane_ids = set()
    seen_ids = set()
    lanes = []

    for i, road in enumerate(roads):
        if not isinstance(road, dict):
            errors.append(f"Road element {i} must be dict")
            continue

        for key in _REQUIRED_ROAD_KEYS:
            if key not in road:
                errors.append(f"Road {i} missing key: '{key}'")

        road_id = road.get("id", "?")
        road_type = road.get("type", 0)

        if "id" in road and not _is_int_like(road_id):
            errors.append(f"Road {i} id must be int, got {type(road_id).__name__}")
        elif "id" in road and road_id in seen_ids:
            errors.append(f"Duplicate road id: {road_id}")
        elif "id" in road:
            seen_ids.add(road_id)

        if "type" in road and not _is_int_like(road_type):
            errors.append(f"Road {road_id} type must be int, got {type(road_type).__name__}")

        if "xyz" in road:
            xyz = road["xyz"]
            if not isinstance(xyz, np.ndarray):
                errors.append(f"Road {road_id} xyz must be ndarray, got {type(xyz).__name__}")
            elif xyz.ndim != 2 or xyz.shape[1] != 3:
                errors.append(f"Road {road_id} xyz must be shape (N,3), got {xyz.shape}")
            elif len(xyz) == 0:
                errors.append(f"Road {road_id} xyz must not be empty")

        if road_type in LANE_TYPES:
            lane_ids.add(road_id)
            lanes.append(road)
            for key in _LANE_EXTRA_KEYS:
                if key not in road:
                    errors.append(f"Lane {road_id} missing key: '{key}'")

            for key in ["entry_lanes", "exit_lanes"]:
                if key in road and not isinstance(road[key], list):
                    errors.append(f"Lane {road_id} {key} must be list, got {type(road[key]).__name__}")

            if "speed_limit" in road and not _is_number(road["speed_limit"]):
                errors.append(f"Lane {road_id} speed_limit must be numeric, got {type(road['speed_limit']).__name__}")

    for lane in lanes:
        lane_id = lane.get("id", "?")
        for key in ["entry_lanes", "exit_lanes"]:
            for ref_lane_id in lane.get(key, []):
                if not _is_int_like(ref_lane_id):
                    errors.append(f"Lane {lane_id} {key} must contain only lane ids")
                elif ref_lane_id not in lane_ids:
                    errors.append(f"Lane {lane_id} references non-existent lane {ref_lane_id} in {key}")

    return lane_ids


def _validate_traffic_control_elements(
    elements: list,
    expected_length: int,
    lane_ids: set[int],
    errors: list[str],
):
    if not isinstance(elements, list):
        errors.append(f"'traffic_control_elements' must be list, got {type(elements).__name__}")
        return

    seen_ids = set()
    for i, elem in enumerate(elements):
        if not isinstance(elem, dict):
            errors.append(f"Traffic control {i} must be dict")
            continue

        for key in _REQUIRED_TRAFFIC_CONTROL_KEYS:
            if key not in elem:
                errors.append(f"Traffic control {i} missing key: '{key}'")

        elem_id = elem.get("id", "?")
        if "id" in elem and not _is_int_like(elem_id):
            errors.append(f"Traffic control {i} id must be int, got {type(elem_id).__name__}")
        elif "id" in elem and elem_id in seen_ids:
            errors.append(f"Duplicate traffic control id: {elem_id}")
        elif "id" in elem:
            seen_ids.add(elem_id)

        if "type" in elem and not _is_int_like(elem["type"]):
            errors.append(f"Traffic control {elem_id} type must be int, got {type(elem['type']).__name__}")

        if "xyz" in elem:
            xyz = np.asarray(elem["xyz"])
            if xyz.ndim != 1 or xyz.shape[0] != 3:
                errors.append(f"Traffic control {elem_id} xyz must be shape (3,), got {xyz.shape}")

        if "states" in elem:
            states = np.asarray(elem["states"])
            if states.ndim != 1:
                errors.append(f"Traffic control {elem_id} states must be 1D, got {states.shape}")
            elif expected_length > 0 and len(states) != expected_length:
                errors.append(f"Traffic control {elem_id} has {len(states)} states, expected {expected_length}")
            elif not np.all(np.isin(states, list(VALID_TL_STATES))):
                errors.append(f"Traffic control {elem_id} has invalid light states")

        if "controlled_lanes" in elem:
            lanes = elem["controlled_lanes"]
            if not isinstance(lanes, list):
                errors.append(f"Traffic control {elem_id} controlled_lanes must be list, got {type(lanes).__name__}")
            else:
                for lane_id in lanes:
                    if not _is_int_like(lane_id):
                        errors.append(f"Traffic control {elem_id} controlled_lanes must contain only lane ids")
                    elif lane_id not in lane_ids:
                        errors.append(f"Traffic control {elem_id} references non-existent lane {lane_id}")


def _validate_scenario_coherence(
    metadata: dict,
    agents: list,
    errors: list[str],
    warnings: list[str],
):
    if not isinstance(metadata, dict) or not isinstance(agents, list) or "sdc_index" not in metadata:
        return

    sdc_index = metadata["sdc_index"]
    if agents and not (0 <= sdc_index < len(agents)):
        errors.append(f"sdc_index {sdc_index} out of range (have {len(agents)} agents)")
    if not agents and sdc_index != -1:
        warnings.append(f"Scenario has no agents but sdc_index={sdc_index}")


def _collect_agent_physics_issues(
    agents: list,
    dt: float,
    position_jump_threshold: float,
    velocity_tolerance: float,
    heading_tolerance_deg: float,
) -> list[tuple[str, str]]:
    issues = []

    for agent in agents:
        agent_id = agent.get("id", "?")
        agent_type = agent.get("type", 4)
        states = agent.get("states", {})

        issues.extend(
            _collect_trajectory_coherence_issues(
                agent_id,
                states,
                dt,
                position_jump_threshold=position_jump_threshold,
                velocity_tolerance=velocity_tolerance,
                heading_tolerance_deg=heading_tolerance_deg,
            ),
        )
        issues.extend(_collect_physical_constraint_issues(agent_id, agent_type, states, dt))

    return issues


def _collect_trajectory_coherence_issues(
    agent_id,
    states: dict,
    dt: float,
    position_jump_threshold: float,
    velocity_tolerance: float,
    heading_tolerance_deg: float,
) -> list[tuple[str, str]]:
    if not {"xyz", "valid", "velocity", "heading"}.issubset(states):
        return []

    xyz = np.asarray(states["xyz"])
    valid = np.asarray(states["valid"])
    velocity = np.asarray(states["velocity"])
    heading = np.asarray(states["heading"])
    valid_pairs = [(i, i + 1) for i in range(len(xyz) - 1) if valid[i] and valid[i + 1]]

    teleport_issues = [
        (_HARD, f"Agent {agent_id} teleports at timestep {i}: moved {dist:.2f}m")
        for i, j in valid_pairs
        for dist in [np.linalg.norm(xyz[j, :2] - xyz[i, :2])]
        if dist > max(position_jump_threshold, 1.0)
    ]

    velocity_issues = []
    if dt > 0:
        velocity_issues = [
            (_SOFT, f"Agent {agent_id} timestep {i}: velocity mismatch (error={vel_error:.2f} m/s)")
            for i, j in valid_pairs
            for computed_vel in [(xyz[j, :2] - xyz[i, :2]) / dt]
            for vel_error in [np.linalg.norm(computed_vel - velocity[i, :2])]
            if vel_error > velocity_tolerance
        ]

    heading_issues = [
        (_SOFT, f"Agent {agent_id} timestep {i}: heading-velocity misalignment")
        for i in range(len(heading))
        if valid[i]
        for speed in [np.linalg.norm(velocity[i, :2])]
        if speed > 0.5
        for vel_angle in [np.arctan2(velocity[i, 1], velocity[i, 0])]
        for angle_diff in [_wrap_angle(vel_angle - heading[i])]
        if np.degrees(np.abs(angle_diff)) > heading_tolerance_deg
    ]

    return teleport_issues + velocity_issues + heading_issues


def _collect_physical_constraint_issues(agent_id, agent_type, states: dict, dt: float) -> list[tuple[str, str]]:
    type_name = AGENT_TYPE_MAP.get(agent_type, "OTHER")
    issues = []

    issues.extend(
        [
            (_HARD, f"Agent {agent_id} has non-positive {dim_key}")
            for dim_key in ["length", "width", "height"]
            if dim_key in states
            for dim in [np.asarray(states[dim_key])]
            if np.any(dim <= 0)
        ],
    )

    if all(key in states for key in ["length", "width", "height"]):
        length = float(np.mean(states["length"]))
        width = float(np.mean(states["width"]))
        height = float(np.mean(states["height"]))
        issues.extend(_collect_dimension_range_issues(agent_id, agent_type, length, width, height))

    if not {"velocity", "valid"}.issubset(states):
        return issues

    velocity = np.asarray(states["velocity"])
    valid = np.asarray(states["valid"])
    max_speed = {1: 60.0, 2: 8.0, 3: 30.0}.get(agent_type, 60.0)
    max_accel = {1: 10.0, 2: 3.0, 3: 5.0}.get(agent_type, 10.0)

    issues.extend(
        [
            (_HARD, f"Agent {agent_id} ({type_name}) timestep {i}: speed={speed:.1f} m/s exceeds {max_speed}")
            for i in range(len(velocity))
            if valid[i]
            for speed in [np.linalg.norm(velocity[i, :2])]
            if speed > max_speed
        ],
    )

    if dt <= 0:
        return issues

    issues.extend(
        [
            (_SOFT, f"Agent {agent_id} timestep {i}: acceleration={accel:.2f} m/s^2 exceeds {max_accel}")
            for i in range(len(velocity) - 1)
            if valid[i] and valid[i + 1]
            for dv in [velocity[i + 1, :2] - velocity[i, :2]]
            for accel in [np.linalg.norm(dv) / dt]
            if accel > max_accel
        ],
    )

    return issues


_DIMENSION_RANGES = {
    # agent_type -> {dim: (min, max)}
    1: {"length": (2.0, 20.0), "width": (1.5, 5.0), "height": (1.0, 4.0)},
    2: {"length": (0.3, 0.6), "width": (0.3, 0.6), "height": (1.0, 3.0)},
    3: {"length": (1.5, 2.0), "width": (0.5, 0.8)},
}


def _collect_dimension_range_issues(
    agent_id,
    agent_type,
    length: float,
    width: float,
    height: float,
) -> list[tuple[str, str]]:
    ranges = _DIMENSION_RANGES.get(agent_type)
    if not ranges:
        return []

    type_name = AGENT_TYPE_MAP.get(agent_type, "OTHER")
    dims = {"length": length, "width": width, "height": height}
    return [
        (_SOFT, f"{type_name} {agent_id} has unusual {dim}={val:.2f}m (expected {lo}-{hi}m)")
        for dim, (lo, hi) in ranges.items()
        if not (lo <= (val := dims[dim]) <= hi)
    ]


def _collect_road_physics_issues(roads: list) -> list[tuple[str, str]]:
    issues = []

    for road in roads:
        road_id = road.get("id", "?")
        xyz = np.asarray(road.get("xyz", []))
        if len(xyz) < 2:
            continue

        dists = [np.linalg.norm(xyz[i + 1] - xyz[i]) for i in range(len(xyz) - 1)]
        total_length = sum(dists)
        if total_length < 1.0:
            issues.append((_SOFT, f"Road {road_id} has very short polyline (length={total_length:.2f}m)"))

        if len(xyz) <= 2:
            continue

        issues.extend(
            [
                (_SOFT, f"Road {road_id} has sharp corner at point {i} (angle={angle_deg:.1f} deg)")
                for i in range(1, len(xyz) - 1)
                for v1 in [xyz[i] - xyz[i - 1]]
                for v2 in [xyz[i + 1] - xyz[i]]
                for v1_norm in [np.linalg.norm(v1)]
                for v2_norm in [np.linalg.norm(v2)]
                if v1_norm > 1e-6 and v2_norm > 1e-6
                for dot in [np.clip(np.dot(v1[:2] / v1_norm, v2[:2] / v2_norm), -1.0, 1.0)]
                for angle_deg in [np.degrees(np.arccos(dot))]
                if angle_deg > 170
            ],
        )

    return issues


def _collect_traffic_control_physics_issues(elements: list, roads: list) -> list[tuple[str, str]]:
    lanes_by_id = {road["id"]: road for road in roads if road.get("type", 0) in LANE_TYPES}
    issues = []

    for elem in elements:
        elem_id = elem.get("id", "?")
        tl_pos = np.asarray(elem.get("xyz", [0.0, 0.0, 0.0]))[:2]
        states = np.asarray(elem.get("states", []))

        issues.extend(
            [
                (_SOFT, f"Traffic control {elem_id} is {dist:.1f}m away from lane {lane_id} (expected < 10m)")
                for lane_id in elem.get("controlled_lanes", [])
                if lane_id in lanes_by_id
                for lane_xyz in [np.asarray(lanes_by_id[lane_id]["xyz"])]
                for dist in [np.linalg.norm(lane_xyz[0, :2] - tl_pos)]
                if dist > 10.0
            ],
        )

        if len(states) <= 1:
            continue

        issues.extend(
            [
                (_SOFT, f"Traffic control {elem_id} timestep {i}: invalid GREEN->RED transition")
                for i in range(len(states) - 1)
                if states[i] in (3, 6) and states[i + 1] in (1, 4)
            ],
        )

        state_changes = sum(1 for i in range(len(states) - 1) if states[i] != states[i + 1])
        if state_changes > len(states) * 0.5:
            issues.append(
                (
                    _SOFT,
                    f"Traffic control {elem_id} has rapid flickering: "
                    f"{state_changes} state changes in {len(states)} timesteps",
                ),
            )

    return issues


def _is_int_like(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi
