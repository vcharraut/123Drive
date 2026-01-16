"""
Validation functions for PufferDrive dict format.

Two validation levels:
1. Soft validation: Structure checks (keys, types, shapes)
2. Strict validation: Physics checks (trajectory coherence, constraints)
"""

import numpy as np


# Type mappings
AGENT_TYPE_MAP = {0: "PEDESTRIAN", 1: "VEHICLE", 2: "CYCLIST", 3: "OTHER"}
VALID_AGENT_TYPES = set(AGENT_TYPE_MAP.keys())

# Road types: 1-10 = lanes, 11-18 = road lines, 21-23 = road edges, 31+ = other
LANE_TYPES = set(range(1, 11))
ROAD_LINE_TYPES = set(range(11, 19))
ROAD_EDGE_TYPES = set(range(21, 24))

# Traffic light state encodings
TL_STATE_UNKNOWN = 0
TL_STATE_RED = 1
TL_STATE_YELLOW = 2
TL_STATE_GREEN = 3
VALID_TL_STATES = {TL_STATE_UNKNOWN, TL_STATE_RED, TL_STATE_YELLOW, TL_STATE_GREEN}

# Expected keys
_REQUIRED_TOP_LEVEL_KEYS = {"scenario_id", "dynamic_agents", "road_map_elements", "traffic_control_elements", "metadata"}
_REQUIRED_METADATA_KEYS = {"dataset_name", "scenario_length", "timesteps", "sdc_index"}
_REQUIRED_AGENT_KEYS = {"id", "type", "states", "routes"}
_REQUIRED_AGENT_STATE_KEYS = {"xyz", "heading", "velocity", "length", "width", "height", "valid"}
_REQUIRED_ROAD_KEYS = {"id", "type", "xyz"}
_LANE_EXTRA_KEYS = {"entry_lanes", "exit_lanes", "speed_limit"}
_REQUIRED_TRAFFIC_CONTROL_KEYS = {"id", "type", "xyz", "states", "controlled_lanes"}


def soft_validate(puffer_dict: dict) -> tuple[bool, list[str], list[str]]:
    """
    Soft validation for puffer_dict structure.

    Returns:
        (is_valid, errors, warnings)
    """
    errors = []
    warnings = []

    _validate_top_level(puffer_dict, errors, warnings)
    _validate_metadata(puffer_dict.get("metadata", {}), errors, warnings)
    _validate_dynamic_agents(puffer_dict.get("dynamic_agents", []), errors, warnings)
    _validate_road_map_elements(puffer_dict.get("road_map_elements", []), errors, warnings)
    _validate_traffic_control_elements(puffer_dict.get("traffic_control_elements", []), errors, warnings)

    return len(errors) == 0, errors, warnings


def _validate_top_level(puffer_dict: dict, errors: list, warnings: list):
    for key in _REQUIRED_TOP_LEVEL_KEYS:
        if key not in puffer_dict:
            errors.append(f"Missing required top-level key: '{key}'")

    if "scenario_id" in puffer_dict and not isinstance(puffer_dict["scenario_id"], str):
        errors.append(f"'scenario_id' must be string, got {type(puffer_dict['scenario_id']).__name__}")


def _validate_metadata(metadata: dict, errors: list, warnings: list):
    for key in _REQUIRED_METADATA_KEYS:
        if key not in metadata:
            errors.append(f"Missing metadata key: '{key}'")

    if "sdc_index" in metadata and not isinstance(metadata["sdc_index"], int):
        errors.append(f"metadata['sdc_index'] must be int, got {type(metadata['sdc_index']).__name__}")

    if "scenario_length" in metadata and not isinstance(metadata["scenario_length"], int):
        errors.append(f"metadata['scenario_length'] must be int, got {type(metadata['scenario_length']).__name__}")


def _validate_dynamic_agents(agents: list, errors: list, warnings: list):
    if not isinstance(agents, list):
        errors.append(f"'dynamic_agents' must be list, got {type(agents).__name__}")
        return

    for i, agent in enumerate(agents):
        if not isinstance(agent, dict):
            errors.append(f"Agent at index {i} must be dict")
            continue

        for key in _REQUIRED_AGENT_KEYS:
            if key not in agent:
                errors.append(f"Agent {i} missing key: '{key}'")

        if "id" in agent and agent["id"] != i:
            warnings.append(f"Agent {i} has id={agent['id']}, expected sequential id={i}")

        if "type" in agent:
            agent_type = agent["type"]
            if not isinstance(agent_type, int):
                errors.append(f"Agent {i} type must be int, got {type(agent_type).__name__}")
            elif agent_type not in VALID_AGENT_TYPES:
                errors.append(f"Agent {i} has invalid type {agent_type}, must be in {VALID_AGENT_TYPES}")

        if "states" in agent:
            _validate_agent_states(i, agent["states"], errors, warnings)

        if "routes" in agent:
            routes = agent["routes"]
            if not isinstance(routes, list):
                errors.append(f"Agent {i} routes must be list, got {type(routes).__name__}")


def _validate_agent_states(agent_idx: int, states: dict, errors: list, warnings: list):
    if not isinstance(states, dict):
        errors.append(f"Agent {agent_idx} states must be dict, got {type(states).__name__}")
        return

    for key in _REQUIRED_AGENT_STATE_KEYS:
        if key not in states:
            errors.append(f"Agent {agent_idx} states missing key: '{key}'")

    if "xyz" in states:
        xyz = states["xyz"]
        if not isinstance(xyz, np.ndarray):
            errors.append(f"Agent {agent_idx} states.xyz must be ndarray, got {type(xyz).__name__}")
        elif xyz.ndim != 2 or xyz.shape[1] != 3:
            errors.append(f"Agent {agent_idx} states.xyz must be shape (T,3), got {xyz.shape}")

    for key in ["heading", "length", "width", "height"]:
        if key in states:
            arr = states[key]
            if isinstance(arr, np.ndarray) and arr.ndim != 1:
                errors.append(f"Agent {agent_idx} states.{key} must be 1D, got shape {arr.shape}")

    if "velocity" in states:
        vel = states["velocity"]
        if isinstance(vel, np.ndarray) and (vel.ndim != 2 or vel.shape[1] != 2):
            errors.append(f"Agent {agent_idx} states.velocity must be shape (T,2), got {vel.shape}")

    if "valid" in states:
        valid = states["valid"]
        if not isinstance(valid, np.ndarray):
            errors.append(f"Agent {agent_idx} states.valid must be ndarray, got {type(valid).__name__}")


def _validate_road_map_elements(roads: list, errors: list, warnings: list):
    if not isinstance(roads, list):
        errors.append(f"'road_map_elements' must be list, got {type(roads).__name__}")
        return

    for i, road in enumerate(roads):
        if not isinstance(road, dict):
            errors.append(f"Road element {i} must be dict")
            continue

        for key in _REQUIRED_ROAD_KEYS:
            if key not in road:
                errors.append(f"Road {i} missing key: '{key}'")

        if "type" in road:
            road_type = road["type"]
            if not isinstance(road_type, int):
                errors.append(f"Road {i} type must be int, got {type(road_type).__name__}")

        if "xyz" in road:
            xyz = road["xyz"]
            if not isinstance(xyz, np.ndarray):
                errors.append(f"Road {i} xyz must be ndarray, got {type(xyz).__name__}")
            elif xyz.ndim != 2 or xyz.shape[1] != 3:
                errors.append(f"Road {i} xyz must be shape (N,3), got {xyz.shape}")

        # Lane-specific checks
        if "type" in road and road["type"] in LANE_TYPES:
            for key in _LANE_EXTRA_KEYS:
                if key not in road:
                    errors.append(f"Lane {i} missing key: '{key}'")


def _validate_traffic_control_elements(elements: list, errors: list, warnings: list):
    if not isinstance(elements, list):
        errors.append(f"'traffic_control_elements' must be list, got {type(elements).__name__}")
        return

    for i, elem in enumerate(elements):
        if not isinstance(elem, dict):
            errors.append(f"Traffic control {i} must be dict")
            continue

        for key in _REQUIRED_TRAFFIC_CONTROL_KEYS:
            if key not in elem:
                errors.append(f"Traffic control {i} missing key: '{key}'")

        if "states" in elem:
            states = elem["states"]
            if not isinstance(states, (list, np.ndarray)):
                errors.append(f"Traffic control {i} states must be list/array, got {type(states).__name__}")


# Strict validation (physics)

def strict_validate(
    puffer_dict: dict,
    validation_level: int = 2,
    position_jump_threshold: float = 50.0,
    velocity_tolerance: float = 2.0,
    heading_tolerance_deg: float = 30.0,
) -> tuple[bool, list[str], list[str]]:
    """
    Strict validation with physics checks.

    Returns:
        (is_valid, errors, warnings)
    """
    errors = []
    warnings = []

    metadata = puffer_dict.get("metadata", {})
    timesteps = metadata.get("timesteps", [])
    dt = _compute_timestep(timesteps)
    expected_length = metadata.get("scenario_length", 0)
    agents = puffer_dict.get("dynamic_agents", [])

    # Skip agent validation for map_only scenarios
    if agents:
        _validate_strict_agents(
            agents,
            dt,
            expected_length,
            position_jump_threshold,
            velocity_tolerance,
            heading_tolerance_deg,
            validation_level,
            errors,
            warnings,
        )

    _validate_strict_roads(
        puffer_dict.get("road_map_elements", []),
        validation_level,
        errors,
        warnings,
    )

    _validate_strict_traffic_control(
        puffer_dict.get("traffic_control_elements", []),
        puffer_dict.get("road_map_elements", []),
        expected_length,
        validation_level,
        errors,
        warnings,
    )

    _validate_scenario_coherence(puffer_dict, validation_level, errors, warnings)

    return len(errors) == 0, errors, warnings


def _compute_timestep(timesteps) -> float:
    if isinstance(timesteps, np.ndarray) and len(timesteps) >= 2:
        return float(np.mean(np.diff(timesteps)))
    if isinstance(timesteps, list) and len(timesteps) >= 2:
        dts = [timesteps[i+1] - timesteps[i] for i in range(len(timesteps)-1)]
        return sum(dts) / len(dts)
    return 0.1


def _validate_strict_agents(
    agents: list,
    dt: float,
    expected_length: int,
    position_jump_threshold: float,
    velocity_tolerance: float,
    heading_tolerance_deg: float,
    validation_level: int,
    errors: list,
    warnings: list,
):
    for agent in agents:
        agent_id = agent.get("id", "?")
        agent_type = agent.get("type", 3)
        states = agent.get("states", {})

        _validate_trajectory_coherence(
            agent_id, agent_type, states, dt,
            position_jump_threshold, velocity_tolerance, heading_tolerance_deg,
            errors, warnings,
        )
        _validate_physical_constraints(agent_id, agent_type, states, dt, validation_level, errors, warnings)
        _validate_temporal_consistency(agent_id, states, expected_length, errors, warnings)


def _validate_trajectory_coherence(
    agent_id, agent_type, states: dict, dt: float,
    position_jump_threshold: float, velocity_tolerance: float, heading_tolerance_deg: float,
    errors: list, warnings: list,
):
    if "xyz" not in states or "valid" not in states:
        return

    xyz = np.array(states["xyz"])
    valid = np.array(states["valid"])
    velocity = np.array(states.get("velocity")) if "velocity" in states else None
    heading = np.array(states.get("heading")) if "heading" in states else None

    # Check for teleportation
    for i in range(len(xyz) - 1):
        if valid[i] and valid[i + 1]:
            dist = np.linalg.norm(xyz[i + 1, :2] - xyz[i, :2])
            if dist > position_jump_threshold and dist > 1.0:
                errors.append(f"Agent {agent_id} teleports at timestep {i}: moved {dist:.2f}m")

    # Velocity-position consistency
    if velocity is not None and dt > 0:
        for i in range(len(xyz) - 1):
            if valid[i] and valid[i + 1] and i < len(velocity):
                computed_vel = (xyz[i + 1, :2] - xyz[i, :2]) / dt
                stated_vel = velocity[i, :2]
                vel_error = np.linalg.norm(computed_vel - stated_vel)
                if vel_error > velocity_tolerance:
                    warnings.append(f"Agent {agent_id} timestep {i}: velocity mismatch (error={vel_error:.2f} m/s)")

    # Heading-velocity consistency
    if heading is not None and velocity is not None:
        for i in range(len(heading)):
            if valid[i] and i < len(velocity):
                speed = np.linalg.norm(velocity[i, :2])
                if speed > 0.5:
                    vel_angle = np.arctan2(velocity[i, 1], velocity[i, 0])
                    angle_diff = np.abs(vel_angle - heading[i])
                    angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                    if np.degrees(angle_diff) > heading_tolerance_deg:
                        warnings.append(f"Agent {agent_id} timestep {i}: heading-velocity misalignment")


def _validate_physical_constraints(
    agent_id, agent_type, states: dict, dt: float, validation_level: int,
    errors: list, warnings: list,
):
    type_name = AGENT_TYPE_MAP.get(agent_type, "OTHER")

    # Dimension validation
    for dim_key, (min_val, max_val) in [("length", (0.1, 30.0)), ("width", (0.1, 10.0)), ("height", (0.1, 10.0))]:
        if dim_key in states:
            dim = states[dim_key]
            if isinstance(dim, np.ndarray):
                if np.any(dim < 0):
                    errors.append(f"Agent {agent_id} has non-positive {dim_key}")
                mean_dim = np.mean(dim)
                if not (min_val <= mean_dim <= max_val):
                    warnings.append(f"Agent {agent_id} has unusual {dim_key}={mean_dim:.2f}m")
                # Dimension variance check
                if len(dim) > 1 and np.std(dim) > 0.01:
                    warnings.append(f"Agent {agent_id} has varying {dim_key} over time (should be constant)")

    # Type-specific dimension ranges
    if all(k in states for k in ["length", "width", "height"]):
        length = np.mean(states["length"])
        width = np.mean(states["width"])
        height = np.mean(states["height"])

        if agent_type == 1:  # VEHICLE
            if not (2.0 <= length <= 20.0):
                warnings.append(f"Vehicle {agent_id} has unusual length={length:.2f}m (expected 2-20m)")
            if not (1.5 <= width <= 3.0):
                warnings.append(f"Vehicle {agent_id} has unusual width={width:.2f}m (expected 1.5-3m)")
            if not (1.0 <= height <= 4.0):
                warnings.append(f"Vehicle {agent_id} has unusual height={height:.2f}m (expected 1-4m)")
        elif agent_type == 0:  # PEDESTRIAN
            if not (0.3 <= length <= 0.6):
                warnings.append(f"Pedestrian {agent_id} has unusual length={length:.2f}m (expected 0.3-0.6m)")
            if not (0.3 <= width <= 0.6):
                warnings.append(f"Pedestrian {agent_id} has unusual width={width:.2f}m (expected 0.3-0.6m)")
            if not (1.4 <= height <= 2.0):
                warnings.append(f"Pedestrian {agent_id} has unusual height={height:.2f}m (expected 1.4-2m)")
        elif agent_type == 2:  # CYCLIST
            if not (1.5 <= length <= 2.0):
                warnings.append(f"Cyclist {agent_id} has unusual length={length:.2f}m (expected 1.5-2m)")
            if not (0.5 <= width <= 0.8):
                warnings.append(f"Cyclist {agent_id} has unusual width={width:.2f}m (expected 0.5-0.8m)")

    # Velocity limits
    if "velocity" in states and "valid" in states:
        velocity = np.array(states["velocity"])
        valid = np.array(states["valid"])
        max_speed = {0: 5.0, 1: 60.0, 2: 15.0, 3: 60.0}.get(agent_type, 60.0)

        for i in range(len(velocity)):
            if valid[i]:
                speed = np.linalg.norm(velocity[i, :2])
                if speed > max_speed:
                    warnings.append(f"Agent {agent_id} ({type_name}) timestep {i}: speed={speed:.1f} m/s exceeds {max_speed}")

        # Acceleration limits (level >= 2)
        if validation_level >= 2 and dt > 0:
            max_accel = {0: 3.0, 2: 5.0}.get(agent_type, 10.0)
            for i in range(len(velocity) - 1):
                if valid[i] and valid[i + 1]:
                    dv = velocity[i + 1, :2] - velocity[i, :2]
                    accel = np.linalg.norm(dv) / dt
                    if accel > max_accel:
                        warnings.append(f"Agent {agent_id} timestep {i}: acceleration={accel:.2f} m/s² exceeds {max_accel}")


def _validate_temporal_consistency(agent_id, states: dict, expected_length: int, errors: list, warnings: list):
    lengths = {k: len(v) for k, v in states.items() if isinstance(v, np.ndarray)}
    if lengths:
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            errors.append(f"Agent {agent_id} has inconsistent state array lengths: {lengths}")
        for key, length in lengths.items():
            if length != expected_length and expected_length > 0:
                errors.append(f"Agent {agent_id} state '{key}' has length {length}, expected {expected_length}")

    # Check for appearing with NaN or at origin
    if "valid" in states and "xyz" in states:
        valid = np.array(states["valid"])
        xyz = np.array(states["xyz"])
        for i in range(1, len(valid)):
            if not valid[i - 1] and valid[i]:
                if np.any(np.isnan(xyz[i])):
                    warnings.append(f"Agent {agent_id} appears at timestep {i} with NaN position")
                elif np.linalg.norm(xyz[i, :2]) < 1e-3:
                    warnings.append(f"Agent {agent_id} appears at timestep {i} at origin (suspicious)")


def _validate_strict_roads(roads: list, validation_level: int, errors: list, warnings: list):
    lane_ids = {r["id"] for r in roads if r.get("type", 0) in LANE_TYPES}
    lanes_by_id = {r["id"]: r for r in roads if r.get("type", 0) in LANE_TYPES}

    for road in roads:
        road_id = road.get("id", "?")
        road_type = road.get("type", 0)

        if "xyz" in road:
            xyz = np.array(road["xyz"])
            if len(xyz) >= 2:
                # Check for duplicate points
                dists = []
                for i in range(len(xyz) - 1):
                    d = np.linalg.norm(xyz[i + 1] - xyz[i])
                    dists.append(d)
                    # if d < 1e-6:
                    #     errors.append(f"Road {road_id} has duplicate points at index {i}")

                # Total length check
                total_length = sum(dists)
                if total_length < 0.5:
                    warnings.append(f"Road {road_id} has very short polyline (length={total_length:.2f}m)")

                # Sharp corner detection (level >= 3)
                if validation_level >= 3 and len(xyz) > 2:
                    for i in range(1, len(xyz) - 1):
                        v1 = xyz[i] - xyz[i - 1]
                        v2 = xyz[i + 1] - xyz[i]
                        v1_norm, v2_norm = np.linalg.norm(v1), np.linalg.norm(v2)
                        if v1_norm > 1e-6 and v2_norm > 1e-6:
                            dot = np.clip(np.dot(v1[:2] / v1_norm, v2[:2] / v2_norm), -1.0, 1.0)
                            angle_deg = np.degrees(np.arccos(dot))
                            if angle_deg > 170:
                                warnings.append(f"Road {road_id} has sharp corner at point {i} (angle={angle_deg:.1f}°)")

        # Lane connectivity
        if road_type in LANE_TYPES:
            for entry_id in road.get("entry_lanes", []):
                if entry_id not in lane_ids:
                    errors.append(f"Lane {road_id} references non-existent entry lane {entry_id}")
            for exit_id in road.get("exit_lanes", []):
                if exit_id not in lane_ids:
                    errors.append(f"Lane {road_id} references non-existent exit lane {exit_id}")

            # Bidirectional connectivity check (level >= 3)
            if validation_level >= 3:
                for exit_id in road.get("exit_lanes", []):
                    if exit_id in lanes_by_id:
                        exit_entries = lanes_by_id[exit_id].get("entry_lanes", [])
                        if road_id not in exit_entries:
                            warnings.append(f"Lane {road_id} exits to {exit_id}, but {exit_id} doesn't list {road_id} as entry")

            # # Speed limit validation (m/s)
            # if "speed_limit" in road:
            #     speed_mps = road["speed_limit"]
            #     if speed_mps < 0:
            #         errors.append(f"Lane {road_id} has negative speed limit - {speed_mps} m/s")
            #     elif speed_mps > 50:  # ~180 km/h
            #         warnings.append(f"Lane {road_id} has unusually high speed limit: {speed_mps:.1f} m/s")


def _validate_strict_traffic_control(
    elements: list, roads: list, expected_length: int, validation_level: int,
    errors: list, warnings: list,
):
    lanes_by_id = {r["id"]: r for r in roads if r.get("type", 0) in LANE_TYPES}
    lane_ids = set(lanes_by_id.keys())

    for elem in elements:
        elem_id = elem.get("id", "?")

        # Controlled lanes validation
        for lane_id in elem.get("controlled_lanes", []):
            if lane_id not in lane_ids:
                errors.append(f"Traffic control {elem_id} references non-existent lane {lane_id}")

        # Position proximity to controlled lane
        if "xyz" in elem and "controlled_lanes" in elem:
            tl_pos = np.array(elem["xyz"])[:2]
            for lane_id in elem["controlled_lanes"]:
                if lane_id in lanes_by_id and "xyz" in lanes_by_id[lane_id]:
                    lane_xyz = np.array(lanes_by_id[lane_id]["xyz"])[:, :2]
                    min_dist = np.min(np.linalg.norm(lane_xyz - tl_pos, axis=1))
                    if min_dist > 50.0:
                        warnings.append(f"Traffic control {elem_id} is {min_dist:.1f}m away from lane {lane_id} (expected < 50m)")

        # Height validation
        if "xyz" in elem:
            z_height = elem["xyz"][2] if len(elem["xyz"]) > 2 else 0
            if not (0.0 <= z_height <= 10.0):
                warnings.append(f"Traffic control {elem_id} has unusual height z={z_height:.2f}m (expected 0-10m)")

        # States validation
        if "states" in elem:
            states = elem["states"]
            if len(states) != expected_length and expected_length > 0:
                errors.append(f"Traffic control {elem_id} has {len(states)} states, expected {expected_length}")

            # State transition validation
            if validation_level >= 2 and len(states) > 1:
                for i in range(len(states) - 1):
                    if states[i] == TL_STATE_GREEN and states[i + 1] == TL_STATE_RED:
                        warnings.append(f"Traffic control {elem_id} timestep {i}: invalid GREEN->RED transition")

                # Rapid flickering detection
                state_changes = sum(1 for i in range(len(states) - 1) if states[i] != states[i + 1])
                if state_changes > len(states) * 0.5:
                    warnings.append(f"Traffic control {elem_id} has rapid flickering: {state_changes} state changes in {len(states)} timesteps")


def _validate_scenario_coherence(puffer_dict: dict, validation_level: int, errors: list, warnings: list):
    metadata = puffer_dict.get("metadata", {})
    agents = puffer_dict.get("dynamic_agents", [])
    roads = puffer_dict.get("road_map_elements", [])

    # Skip sdc_index check for map_only scenarios (no agents)
    if len(agents) > 0:
        sdc_index = metadata.get("sdc_index", 0)
        if sdc_index >= len(agents):
            errors.append(f"sdc_index {sdc_index} out of range (have {len(agents)} agents)")

    if "timesteps" in metadata:
        timesteps = metadata["timesteps"]
        if isinstance(timesteps, np.ndarray) and len(timesteps) > 1:
            if not np.all(np.diff(timesteps) > 0):
                errors.append("Timestamps are not monotonically increasing")
            if len(np.unique(timesteps)) < len(timesteps):
                errors.append("Duplicate timestamps found")

    if validation_level >= 3:
        if len(agents) > 500:
            warnings.append(f"Scenario has {len(agents)} agents (unusually high)")
        elif len(agents) == 0:
            warnings.append("Scenario has no dynamic agents")

        if len(roads) > 1000:
            warnings.append(f"Scenario has {len(roads)} map elements (unusually high, expected 10-500)")
