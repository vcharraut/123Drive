"""Input validation for ArrowScenario — runs after extraction, before processing.

Level 1 (schema): shapes, dtypes, required fields.
Level 2 (semantic): NaN/Inf, cross-refs, physics.
"""

import numpy as np
from py123d.datatypes.map_objects import MapLayer


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


def validate_scenario(scenario, level=1):
    """Returns list of error strings. Empty = valid."""
    if level <= 0:
        return []

    errors = []
    meta = scenario.metadata
    length = meta.scenario_length

    # ── Schema ──
    if length < 0:
        errors.append("metadata.scenario_length must be non-negative")
    if length > 0 and meta.timestep_seconds <= 0:
        errors.append("metadata.timestep_seconds must be > 0 when scenario_length > 0")

    _validate_dynamic_states(scenario.agents, "Agent", length, errors)
    _validate_dynamic_states(scenario.objects, "Object", length, errors)
    _validate_map_elements(scenario.map, errors)
    _validate_traffic_lights(scenario.traffic_lights, length, scenario.map, errors)

    if errors or level < 2:
        return errors

    # ── Semantic ──
    _validate_no_nan_inf(scenario, errors)
    _validate_ego(scenario.agents, length, errors)
    _validate_lane_topology(scenario.map, errors)
    _validate_tl_lane_refs(scenario.traffic_lights, scenario.map, errors)
    _validate_agent_sizes(scenario.agents, "Agent", errors)
    _validate_agent_sizes(scenario.objects, "Object", errors)

    return errors


def _validate_dynamic_states(items, prefix, length, errors):
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


def _validate_map_elements(map_data, errors):
    for eid, elem in map_data.items():
        layer = elem.get("layer")
        if layer in (MapLayer.LANE, MapLayer.ROAD_LINE, MapLayer.ROAD_EDGE):
            poly = elem.get("polyline")
            if poly is None:
                errors.append(f"Map {eid} missing polyline")
            elif not isinstance(poly, np.ndarray) or poly.ndim != 2 or poly.shape[1] < 2:
                errors.append(f"Map {eid} polyline invalid shape {getattr(poly, 'shape', None)}")
            elif len(poly) < 2:
                errors.append(f"Map {eid} polyline needs >= 2 points, got {len(poly)}")

            if layer == MapLayer.LANE:
                for key in ("entry_lanes", "exit_lanes"):
                    if not isinstance(elem.get(key), list):
                        errors.append(f"Map {eid} missing or invalid {key}")

        elif layer in (MapLayer.CROSSWALK, MapLayer.STOP_ZONE):
            poly = elem.get("polygon")
            if poly is None:
                errors.append(f"Map {eid} missing polygon")
            elif not isinstance(poly, np.ndarray) or poly.ndim != 2 or poly.shape[1] < 2:
                errors.append(f"Map {eid} polygon invalid shape {getattr(poly, 'shape', None)}")
            elif len(poly) < 3:
                errors.append(f"Map {eid} polygon needs >= 3 points, got {len(poly)}")


def _validate_traffic_lights(tl_data, length, map_data, errors):
    for eid, tl in tl_data.items():
        if not isinstance(tl.position, np.ndarray) or tl.position.shape != (3,):
            errors.append(f"TL {eid} position must be shape (3,), got {getattr(tl.position, 'shape', None)}")
        if length > 0 and len(tl.states) != length:
            errors.append(f"TL {eid} states length {len(tl.states)} != scenario_length {length}")
        if tl.controlled_lane not in map_data:
            errors.append(f"TL {eid} controlled_lane {tl.controlled_lane} not in map")


# ── Semantic checks ──


def _validate_no_nan_inf(scenario, errors):
    for prefix, items in [("Agent", scenario.agents), ("Object", scenario.objects)]:
        for eid, ds in items.items():
            for field in _DYNAMIC_STATE_SPECS:
                arr = getattr(ds, field)
                if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
                    errors.append(f"{prefix} {eid} {field} contains NaN or Inf")

    for eid, elem in scenario.map.items():
        for key in ("polyline", "polygon"):
            arr = elem.get(key)
            if arr is not None and isinstance(arr, np.ndarray) and not np.all(np.isfinite(arr)):
                errors.append(f"Map {eid} {key} contains NaN or Inf")


def _validate_ego(agents, length, errors):
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
    for i in range(len(xyz) - 1):
        if valid[i] and valid[i + 1]:
            dist = float(np.linalg.norm(xyz[i + 1, :2] - xyz[i, :2]))
            if dist > 5.0:
                errors.append(f"Ego teleports at timestep {i}: moved {dist:.2f}m")


def _validate_lane_topology(map_data, errors):
    lane_ids = {eid for eid, e in map_data.items() if e.get("layer") == MapLayer.LANE}
    for eid, elem in map_data.items():
        if elem.get("layer") != MapLayer.LANE:
            continue
        for key in ("entry_lanes", "exit_lanes"):
            for ref in elem.get(key, []):
                if ref not in lane_ids:
                    errors.append(f"Lane {eid} {key} references non-existent lane {ref}")


def _validate_tl_lane_refs(tl_data, map_data, errors):
    lane_ids = {eid for eid, e in map_data.items() if e.get("layer") == MapLayer.LANE}
    for eid, tl in tl_data.items():
        if tl.controlled_lane not in lane_ids:
            errors.append(f"TL {eid} controlled_lane {tl.controlled_lane} references non-lane element")


def _validate_agent_sizes(items, prefix, errors):
    for eid, ds in items.items():
        valid = ds.valid.astype(bool)
        if not np.any(valid):
            continue
        for key in ("length", "width", "height"):
            vals = getattr(ds, key)[valid]
            if np.any(vals <= 0):
                errors.append(f"{prefix} {eid} has non-positive {key}")

        speeds = np.linalg.norm(ds.velocity[valid, :2], axis=1)
        if np.any(speeds > 60.0):
            errors.append(f"{prefix} {eid} speed exceeds 60 m/s")
