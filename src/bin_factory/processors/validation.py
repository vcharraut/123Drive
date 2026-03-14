"""Validation for PufferDrive dicts.

All keys and types are guaranteed by the pipeline (build_puffer_dict).
Validation only checks value correctness: shapes, ranges, consistency, semantics.

Modes: 0=off, 1=schema, 2=semantic.
"""

import numpy as np


VALIDATION_OFF = "off"
VALIDATION_SCHEMA = "schema"
VALIDATION_SEMANTIC = "semantic"

VALID_AGENT_TYPES = {1, 2, 3, 4}
VALID_OBJECT_TYPES = {1, 2, 3, 4, 5}
LANE_TYPES = set(range(10))
VALID_TL_STATES = {0, 1, 2, 3, 4}

# key -> (ndim, shape1_or_None)
STATE_ARRAY_SPECS = {
    "xyz": (2, 3),
    "velocity": (2, 2),
    **dict.fromkeys(("heading", "length", "width", "height", "valid"), (1, None)),
}


class ValidationError(Exception):
    pass


def validate_puffer_dict(puffer_dict, validation_mode=VALIDATION_SCHEMA, position_jump_threshold=50.0):
    if validation_mode == VALIDATION_OFF:
        return []

    errors = []
    metadata = puffer_dict["metadata"]
    agents = puffer_dict["agents"]
    objects = puffer_dict["objects"]
    roads = puffer_dict["road_map_elements"]
    tcs = puffer_dict["traffic_control_elements"]

    expected_length = int(metadata["scenario_length"])

    # Schema: value ranges, shapes, consistency
    if expected_length < 0:
        errors.append("metadata['scenario_length'] must be non-negative")
    timestep = metadata["timestep_seconds"]
    if expected_length > 0 and (not _is_number(timestep) or timestep <= 0):
        errors.append("metadata['timestep_seconds'] must be > 0")

    lane_ids = _validate_roads(roads, errors)
    _validate_entities(agents, "Agent", VALID_AGENT_TYPES, expected_length, errors, check_route=True)
    _validate_entities(objects, "Object", VALID_OBJECT_TYPES, expected_length, errors)
    _validate_tcs(tcs, expected_length, errors)

    if errors or validation_mode == VALIDATION_SCHEMA:
        return errors

    # Semantic: cross-references, geometry, physics
    _validate_lane_topology(roads, lane_ids, errors)
    _validate_lane_refs(agents, "Agent", "route", lane_ids, errors)
    _validate_lane_refs(tcs, "Traffic control", "controlled_lanes", lane_ids, errors)
    _validate_finite_values(agents, objects, roads, tcs, errors)
    _validate_ego_semantics(metadata, agents, position_jump_threshold, errors)
    return errors


# ── Schema checks (value correctness) ──────────────────────────────────────────────────


def _validate_entities(items, prefix, valid_types, expected_length, errors, check_route=False):
    seen_ids = set()
    for item in items:
        eid = item["id"]
        if eid in seen_ids:
            errors.append(f"Duplicate {prefix.lower()} id: {eid}")
        seen_ids.add(eid)

        if item["type"] not in valid_types:
            errors.append(f"{prefix} {eid} has invalid type {item['type']}")

        _validate_states(prefix, eid, item["states"], expected_length, errors)

        if check_route and not all(isinstance(lid, int | np.integer) for lid in item["route"]):
            errors.append(f"{prefix} {eid} route must contain only lane ids")


def _validate_states(prefix, eid, states, expected_length, errors):
    lengths = {}
    for key, (ndim, shape1) in STATE_ARRAY_SPECS.items():
        arr = states[key]
        if arr.ndim != ndim:
            errors.append(f"{prefix} {eid} {key} must be {ndim}D, got shape {arr.shape}")
            continue
        if shape1 is not None and arr.shape[1] != shape1:
            errors.append(f"{prefix} {eid} {key} must be shape (N,{shape1}), got {arr.shape}")
            continue
        lengths[key] = len(arr)

    if len(set(lengths.values())) > 1:
        errors.append(f"{prefix} {eid} has inconsistent state array lengths: {lengths}")
    if expected_length > 0:
        for key, length in lengths.items():
            if length != expected_length:
                errors.append(f"{prefix} {eid} state '{key}' has length {length}, expected {expected_length}")


def _validate_roads(roads, errors):
    seen_ids, lane_ids = set(), set()
    for road in roads:
        rid = road["id"]
        if rid in seen_ids:
            errors.append(f"Duplicate road id: {rid}")
        seen_ids.add(rid)

        xyz = road["xyz"]
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            errors.append(f"Road {rid} xyz must be shape (N,3), got {xyz.shape}")
        elif len(xyz) < 2:
            errors.append(f"Road {rid} xyz must have at least 2 points")

        if road["type"] in LANE_TYPES:
            lane_ids.add(rid)

    return lane_ids


def _validate_tcs(tcs, expected_length, errors):
    seen_ids = set()
    for tc in tcs:
        tid = tc["id"]
        if tid in seen_ids:
            errors.append(f"Duplicate traffic control id: {tid}")
        seen_ids.add(tid)

        sl = tc["stop_line"]
        if not isinstance(sl, np.ndarray) or sl.shape != (2, 3):
            errors.append(f"Traffic control {tid} stop_line must be shape (2,3), got {np.asarray(sl).shape}")

        states = tc["states"]
        if isinstance(states, np.ndarray) and expected_length > 0 and len(states) != expected_length:
            errors.append(f"Traffic control {tid} states has length {len(states)}, expected {expected_length}")


# ── Semantic checks ────────────────────────────────────────────────────────────────────


def _validate_lane_topology(roads, lane_ids, errors):
    lanes_by_id = {r["id"]: r for r in roads if r["type"] in LANE_TYPES}
    for lane in lanes_by_id.values():
        lid = lane["id"]
        for key in ("entry_lanes", "exit_lanes"):
            for ref in lane.get(key, []):
                if ref not in lane_ids:
                    errors.append(f"Lane {lid} references non-existent lane {ref} in {key}")

        for fwd, rev in [("entry_lanes", "exit_lanes"), ("exit_lanes", "entry_lanes")]:
            for ref_id in lane.get(fwd, []):
                other = lanes_by_id.get(ref_id)
                if other is None or other["type"] not in (1, 2):
                    continue
                if lid not in other.get(rev, []):
                    errors.append(
                        f"Lane {lid} lists {ref_id} as {fwd[:-1]}, but lane {ref_id} does not list {lid} in {rev}"
                    )


def _validate_lane_refs(items, prefix, key, lane_ids, errors):
    for item in items:
        for lid in item.get(key, []):
            if lid not in lane_ids:
                errors.append(f"{prefix} {item['id']} {key} references non-existent lane {lid}")


def _validate_finite_values(agents, objects, roads, tcs, errors):
    for prefix, items in [("Agent", agents), ("Object", objects)]:
        for item in items:
            for key in STATE_ARRAY_SPECS:
                if not _is_finite(item["states"][key]):
                    errors.append(f"{prefix} {item['id']} states.{key} contains NaN or Inf")

    for road in roads:
        if not _is_finite(road["xyz"]):
            errors.append(f"Road {road['id']} xyz contains NaN or Inf")
        if "speed_limit" in road and not _is_finite_scalar(road["speed_limit"]):
            errors.append(f"Lane {road['id']} speed_limit contains NaN or Inf")

    for tc in tcs:
        if not _is_finite(tc["stop_line"]):
            errors.append(f"Traffic control {tc['id']} stop_line contains NaN or Inf")
        if isinstance(tc["states"], np.ndarray):
            if not _is_finite(tc["states"]):
                errors.append(f"Traffic control {tc['id']} states contains NaN or Inf")
            if not np.all(np.isin(tc["states"], list(VALID_TL_STATES))):
                errors.append(f"Traffic control {tc['id']} has invalid light states")
        if not _is_finite_scalar(tc["heading"]):
            errors.append(f"Traffic control {tc['id']} heading contains NaN or Inf")


def _validate_ego_semantics(metadata, agents, position_jump_threshold, errors):
    ego = agents[0]
    eid = ego["id"]
    states = ego["states"]
    xyz, valid = states["xyz"], states["valid"].astype(bool)
    velocity = states["velocity"]

    if np.any(valid):
        vi = np.flatnonzero(valid)
        if not np.all(valid[vi[0] : vi[-1] + 1]):
            errors.append(f"Ego agent {eid} has gaps in valid timesteps")

    for i in range(len(xyz) - 1):
        if valid[i] and valid[i + 1]:
            dist = float(np.linalg.norm(xyz[i + 1, :2] - xyz[i, :2]))
            if dist > max(position_jump_threshold, 1.0):
                errors.append(f"Ego agent {eid} teleports at timestep {i}: moved {dist:.2f}m")

    for key in ("length", "width", "height"):
        if np.any(states[key][valid] <= 0):
            errors.append(f"Ego agent {eid} has non-positive {key}")

    speeds = np.linalg.norm(velocity[valid, :2], axis=1) if np.any(valid) else np.array([])
    if np.any(speeds > 60.0):
        errors.append(f"Ego agent {eid} speed exceeds 60.0 m/s")


# ── Helpers ────────────────────────────────────────────────────────────────────────────


def _is_number(value):
    return isinstance(value, int | float | np.integer | np.floating) and not isinstance(value, bool)


def _is_finite(arr):
    a = np.asarray(arr)
    return np.issubdtype(a.dtype, np.number) and bool(np.all(np.isfinite(a)))


def _is_finite_scalar(value):
    return _is_number(value) and np.isfinite(value)
