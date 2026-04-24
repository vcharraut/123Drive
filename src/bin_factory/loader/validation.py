"""Input validation for Scenario — runs after extraction, before processing.

Level 1 (schema): shapes, dtypes, required fields.
Level 2 (semantic): NaN/Inf, cross-refs, physics.
"""

import numpy as np

from bin_factory import puffer_types


class ValidationError(Exception):
    pass


_DYNAMIC_STATE_SPECS = {
    "position": (2, 3),
    "velocity": (2, 2),
    "heading": (1, None),
    "valid": (1, None),
    "length": (1, None),
    "width": (1, None),
    "height": (1, None),
}


def validate_scenario(scenario, extras=None, level=1):
    """Returns list of error strings. Empty = valid.

    extras: {"traffic_lights": ..., "stop_zones": ...} from extraction.
    """
    if level <= 0:
        return []

    extras = extras or {}
    traffic_lights = extras.get("traffic_lights", {})
    stop_zones = extras.get("stop_zones", [])

    errors = []
    meta = scenario.metadata
    length = meta.scenario_length

    # ── Schema ──
    if length < 0:
        errors.append("metadata.scenario_length must be non-negative")
    if length > 0 and meta.dt <= 0:
        errors.append("metadata.dt must be > 0 when scenario_length > 0")

    _validate_dynamic_states(scenario.agents, "Agent", length, errors)
    _validate_dynamic_states(scenario.objects, "Object", length, errors)
    _validate_map_elements(scenario.map, errors)
    _validate_stop_zones(stop_zones, errors)
    _validate_traffic_lights(traffic_lights, length, scenario.map, errors)

    if errors or level < 2:
        return errors

    # ── Semantic ──
    _validate_no_nan_inf(scenario, errors)
    lane_ids = {eid for eid, e in scenario.map.items() if puffer_types.is_road_lane(e["type"])}
    _validate_lane_topology(scenario.map, lane_ids, errors)
    _validate_tl_lane_refs(traffic_lights, lane_ids, errors)
    if scenario.agents:
        _validate_ego(scenario.agents, length, errors)
        _validate_agent_sizes(scenario.agents, "Agent", errors)
    if scenario.objects:
        _validate_agent_sizes(scenario.objects, "Object", errors)

    return errors


def _validate_dynamic_states(items, prefix, length, errors) -> None:
    for eid, ds in items.items():
        for field, (ndim, dim1) in _DYNAMIC_STATE_SPECS.items():
            arr = getattr(ds, field)
            if not isinstance(arr, np.ndarray):
                errors.append(f"{prefix} {eid} {field} is not ndarray")
                continue
            if arr.ndim != ndim:
                errors.append(f"{prefix} {eid} {field} must be {ndim}D, got shape {arr.shape}")
                continue
            if dim1 is not None and arr.shape[-1] != dim1:
                errors.append(f"{prefix} {eid} {field} last dim must be {dim1}, got {arr.shape}")
            if length > 0 and arr.shape[0] != length:
                errors.append(f"{prefix} {eid} {field} length {arr.shape[0]} != scenario_length {length}")


def _validate_geometry(elem, key, label, min_points, errors):
    geom = elem.get(key)
    if geom is None:
        errors.append(f"{label} missing {key}")
    elif not isinstance(geom, np.ndarray) or geom.ndim != 2 or geom.shape[1] < 2:
        errors.append(f"{label} {key} invalid shape {getattr(geom, 'shape', None)}")
    elif len(geom) < min_points:
        errors.append(f"{label} {key} needs >= {min_points} points, got {len(geom)}")


def _validate_map_elements(map_data, errors) -> None:
    for eid, elem in map_data.items():
        t = elem["type"]
        if puffer_types.is_road_lane(t) or puffer_types.is_road_line(t) or puffer_types.is_road_edge(t):
            _validate_geometry(elem, "polyline", f"Map {eid}", 2, errors)
            if puffer_types.is_road_lane(t):
                for key in ("entry_lanes", "exit_lanes"):
                    if not isinstance(elem.get(key), list):
                        errors.append(f"Map {eid} missing or invalid {key}")

        elif puffer_types.is_crosswalk(t):
            _validate_geometry(elem, "polygon", f"Map {eid}", 3, errors)


def _validate_stop_zones(stop_zones, errors) -> None:
    for i, sz in enumerate(stop_zones):
        _validate_geometry(sz, "polygon", f"StopZone {i}", 3, errors)
        if not isinstance(sz.get("controlled_lanes"), list):
            errors.append(f"StopZone {i} missing or invalid controlled_lanes")


def _validate_traffic_lights(tl_data, length, map_data, errors) -> None:
    for eid, tl in tl_data.items():
        if not isinstance(tl.position, np.ndarray) or tl.position.shape != (3,):
            errors.append(f"TL {eid} position must be shape (3,), got {getattr(tl.position, 'shape', None)}")
        if length > 0 and len(tl.states) != length:
            errors.append(f"TL {eid} states length {len(tl.states)} != scenario_length {length}")
        if tl.controlled_lane not in map_data:
            errors.append(f"TL {eid} controlled_lane {tl.controlled_lane} not in map")


# ── Semantic checks ──


def _validate_no_nan_inf(scenario, errors) -> None:
    for prefix, items in (("Agent", scenario.agents), ("Object", scenario.objects)):
        for eid, ds in items.items():
            for field in _DYNAMIC_STATE_SPECS:
                arr = getattr(ds, field)
                if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
                    errors.append(f"{prefix} {eid} {field} contains NaN or Inf")

    for eid, elem in scenario.map.items():
        for key in ("polyline", "polygon"):
            if (arr := elem.get(key)) is not None and isinstance(arr, np.ndarray) and not np.all(np.isfinite(arr)):
                errors.append(f"Map {eid} {key} contains NaN or Inf")


def _validate_ego(agents, length, errors) -> None:
    if 0 not in agents:
        errors.append("Ego agent (id=0) missing")
        return
    ego = agents[0]
    valid = ego.valid.astype(bool)
    if not np.any(valid):
        errors.append("Ego agent has no valid frames")
        return

    if length > 0 and len(valid) != length:
        errors.append(f"Ego agent valid length {len(valid)} != scenario_length {length}")
        return

    vi = np.flatnonzero(valid)
    if not np.all(valid[vi[0] : vi[-1] + 1]):
        errors.append("Ego agent has gaps in valid frames")

    xyz = ego.position
    dists = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
    both_valid = valid[:-1] & valid[1:]
    for i in np.flatnonzero((dists > 5.0) & both_valid):
        errors.append(f"Ego teleports at timestep {i}: moved {dists[i]:.2f}m")


def _validate_lane_topology(map_data, lane_ids, errors) -> None:
    for eid, elem in map_data.items():
        if not puffer_types.is_road_lane(elem["type"]):
            continue
        for key in ("entry_lanes", "exit_lanes"):
            for ref in elem.get(key, []):
                if ref not in lane_ids:
                    errors.append(f"Lane {eid} {key} references non-existent lane {ref}")


def _validate_tl_lane_refs(tl_data, lane_ids, errors) -> None:
    for eid, tl in tl_data.items():
        if tl.controlled_lane not in lane_ids:
            errors.append(f"TL {eid} controlled_lane {tl.controlled_lane} references non-lane element")


def _validate_agent_sizes(items, prefix, errors) -> None:
    for eid, ds in items.items():
        valid = ds.valid.astype(bool)
        if not np.any(valid):
            continue
        for key in ("length", "width", "height"):
            vals = getattr(ds, key)[valid]
            if np.any(vals <= 0):
                errors.append(f"{prefix} {eid} has non-positive {key}")
