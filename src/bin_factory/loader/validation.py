"""Input validation for Scenario — runs after extraction, before processing.

Level 1 (schema): shapes, dtypes, required fields.
Level 2 (semantic): NaN/Inf, cross-refs, physics.
"""

import numpy as np
from shapely.geometry import LineString, MultiPoint, Point
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from bin_factory import puffer_types, schema


_Z_BRIDGE_MIN = 0.5
_Z_BRIDGE_MAX = 4.0


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


def validate_scenario(
    scenario: schema.PufferScenario,
    extras: schema.ExtractionExtras | None = None,
    level: int = 1,
) -> list[str]:
    """Returns list of error strings. Empty = valid.

    extras: ExtractionExtras (traffic_lights, stop_zones) from extraction.
    """
    if level <= 0:
        return []

    extras = extras or schema.ExtractionExtras()
    traffic_lights = extras.traffic_lights
    stop_zones = extras.stop_zones

    errors: list[str] = []
    meta = scenario.metadata
    length = meta.scenario_length

    # ── Schema ──
    if length < 0:
        errors.append("metadata.scenario_length must be non-negative")
    if not np.isfinite(meta.dt):
        errors.append("metadata.dt must be finite")
    elif length > 0 and meta.dt <= 0:
        errors.append("metadata.dt must be > 0 when scenario_length > 0")

    _validate_dynamic_states(scenario.agents, "Agent", length, errors)
    _validate_dynamic_states(scenario.objects, "Object", length, errors)
    _validate_map_elements(scenario.map, errors)
    _validate_stop_zones(stop_zones, errors)
    _validate_traffic_lights(traffic_lights, length, scenario.map, errors)
    _validate_traffic_controls(scenario.traffic_controls, length, errors)
    _validate_lane_graph(scenario.lane_graph, errors)

    if errors or level < 2:
        return errors

    # ── Semantic ──
    _validate_no_nan_inf(scenario, errors)
    lane_ids = {eid for eid, e in scenario.map.items() if e.is_lane}
    _validate_lane_topology(scenario.map, lane_ids, errors)
    _validate_tl_lane_refs(traffic_lights, lane_ids, errors)
    _validate_lane_refs(scenario, stop_zones, lane_ids, errors)
    if scenario.agents:
        _validate_ego(scenario.agents, length, meta.dt, errors)
        _validate_agent_sizes(scenario.agents, "Agent", errors)
    if scenario.objects:
        _validate_agent_sizes(scenario.objects, "Object", errors)
    _validate_road_edge_z_overlap(scenario.map, errors)

    return errors


def _validate_dynamic_states(items: dict[int, schema.Track], prefix: str, length: int, errors: list[str]) -> None:
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
            if arr.shape[0] != length:
                errors.append(f"{prefix} {eid} {field} length {arr.shape[0]} != scenario_length {length}")


def _validate_geometry(elem: object, key: str, label: str, min_points: int, errors: list[str]) -> None:
    geom = getattr(elem, key, None)
    if geom is None:
        errors.append(f"{label} missing {key}")
    elif not isinstance(geom, np.ndarray) or geom.ndim != 2 or geom.shape[1] != 3:
        errors.append(f"{label} {key} invalid shape {getattr(geom, 'shape', None)}")
    elif len(geom) < min_points:
        errors.append(f"{label} {key} needs >= {min_points} points, got {len(geom)}")


def _validate_map_elements(map_data: dict[int, schema.MapElement], errors: list[str]) -> None:
    has_lane_lengths = any(elem.is_lane and elem.cum_length is not None for elem in map_data.values())
    for eid, elem in map_data.items():
        if elem.uses_polyline:
            _validate_geometry(elem, "polyline", f"Map {eid}", 2, errors)
            if elem.is_lane:
                for key in ("entry_lanes", "exit_lanes"):
                    if not isinstance(getattr(elem, key, None), list):
                        errors.append(f"Map {eid} missing or invalid {key}")
                if has_lane_lengths:
                    expected = len(elem.polyline) if isinstance(elem.polyline, np.ndarray) else 0
                    if not isinstance(elem.cum_length, np.ndarray) or elem.cum_length.shape != (expected,):
                        errors.append(f"Map {eid} cum_length must be shape ({expected},)")
        else:
            _validate_geometry(elem, "polygon", f"Map {eid}", 3, errors)

        if elem.is_lane:
            for key in ("left_boundary", "right_boundary"):
                if getattr(elem, key) is not None:
                    _validate_geometry(elem, key, f"Map {eid}", 0, errors)


def _validate_stop_zones(stop_zones: list[schema.StopZone], errors: list[str]) -> None:
    for i, sz in enumerate(stop_zones):
        _validate_geometry(sz, "polygon", f"StopZone {i}", 3, errors)
        if not isinstance(getattr(sz, "controlled_lanes", None), list):
            errors.append(f"StopZone {i} missing or invalid controlled_lanes")


def _validate_traffic_lights(
    tl_data: dict[int, schema.TrafficLightTrack],
    length: int,
    map_data: dict[int, schema.MapElement],
    errors: list[str],
) -> None:
    for eid, tl in tl_data.items():
        if not isinstance(tl.position, np.ndarray) or tl.position.shape != (3,):
            errors.append(f"TL {eid} position must be shape (3,), got {getattr(tl.position, 'shape', None)}")
        if not isinstance(tl.states, list):
            errors.append(f"TL {eid} states must be a list")
        elif any(not isinstance(state, (int, np.integer)) for state in tl.states):
            errors.append(f"TL {eid} states must contain integers")
        elif len(tl.states) != length:
            errors.append(f"TL {eid} states length {len(tl.states)} != scenario_length {length}")
        if tl.controlled_lane not in map_data:
            errors.append(f"TL {eid} controlled_lane {tl.controlled_lane} not in map")


def _validate_traffic_controls(traffic_controls: list[schema.TrafficControl], length: int, errors: list[str]) -> None:
    for index, control in enumerate(traffic_controls):
        if control.type not in puffer_types.TC_TYPE_NAMES:
            errors.append(f"TrafficControl {index} type is invalid")
        if control.stop_line.shape != (2, 3):
            errors.append(f"TrafficControl {index} stop_line must be shape (2, 3), got {control.stop_line.shape}")
        if not control.controlled_lanes:
            errors.append(f"TrafficControl {index} controlled_lanes must be non-empty")
        if any(not isinstance(state, (int, np.integer)) for state in control.states):
            errors.append(f"TrafficControl {index} states must contain integers")
        elif control.type == puffer_types.TCType.TRAFFIC_LIGHT and len(control.states) != length:
            errors.append(f"TrafficControl {index} states length {len(control.states)} != scenario_length {length}")
    if len({control.id for control in traffic_controls}) != len(traffic_controls):
        errors.append("TrafficControl ids must be unique")


def _validate_lane_graph(lane_graph: dict | None, errors: list[str]) -> None:
    if lane_graph is None:
        return
    if not isinstance(lane_graph, dict):
        errors.append("lane_graph must be a dict")
        return
    lane_ids = lane_graph.get("lane_ids")
    distances = lane_graph.get("distances")
    if not isinstance(lane_ids, list):
        errors.append("lane_graph.lane_ids must be a list")
        return
    if not isinstance(distances, np.ndarray) or distances.shape != (len(lane_ids), len(lane_ids)):
        errors.append(
            f"lane_graph.distances must be shape ({len(lane_ids)}, {len(lane_ids)}), "
            f"got {getattr(distances, 'shape', None)}"
        )


# ── Semantic checks ──


def _validate_no_nan_inf(scenario: schema.PufferScenario, errors: list[str]) -> None:
    for prefix, items in (("Agent", scenario.agents), ("Object", scenario.objects)):
        for eid, ds in items.items():
            for field in _DYNAMIC_STATE_SPECS:
                arr = getattr(ds, field)
                if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
                    errors.append(f"{prefix} {eid} {field} contains NaN or Inf")

    for eid, elem in scenario.map.items():
        for key in ("polyline", "polygon", "left_boundary", "right_boundary", "cum_length"):
            if (
                (arr := getattr(elem, key, None)) is not None
                and isinstance(arr, np.ndarray)
                and not np.all(np.isfinite(arr))
            ):
                errors.append(f"Map {eid} {key} contains NaN or Inf")

    for index, control in enumerate(scenario.traffic_controls):
        if not np.all(np.isfinite(control.stop_line)):
            errors.append(f"TrafficControl {index} stop_line contains NaN or Inf")
        if not np.isfinite(control.heading):
            errors.append(f"TrafficControl {index} heading must be finite")

    if scenario.lane_graph is not None and np.any(np.isnan(scenario.lane_graph["distances"])):
        errors.append("lane_graph.distances contains NaN")


def _validate_ego(agents: dict[int, schema.Track], length: int, dt: float, errors: list[str]) -> None:
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
    for i in np.flatnonzero((dists > 50.0 * dt) & both_valid):
        errors.append(f"Ego teleports at timestep {i}: moved {dists[i]:.2f}m")


def _validate_lane_topology(map_data: dict[int, schema.MapElement], lane_ids: set[int], errors: list[str]) -> None:
    for eid, elem in map_data.items():
        if not elem.is_lane:
            continue
        for key in ("entry_lanes", "exit_lanes", "left_neighbor", "right_neighbor"):
            for ref in getattr(elem, key):
                if ref not in lane_ids:
                    errors.append(f"Lane {eid} {key} references non-existent lane {ref}")


def _validate_tl_lane_refs(tl_data: dict[int, schema.TrafficLightTrack], lane_ids: set[int], errors: list[str]) -> None:
    for eid, tl in tl_data.items():
        if tl.controlled_lane not in lane_ids:
            errors.append(f"TL {eid} controlled_lane {tl.controlled_lane} references non-lane element")
        if not np.all(np.isfinite(tl.position)):
            errors.append(f"TL {eid} position contains NaN or Inf")
        if any(int(state) not in puffer_types.TL_STATE_NAMES for state in tl.states):
            errors.append(f"TL {eid} contains invalid state")


def _validate_lane_refs(
    scenario: schema.PufferScenario, stop_zones: list[schema.StopZone], lane_ids: set[int], errors: list[str]
) -> None:
    for index, zone in enumerate(stop_zones):
        if not np.all(np.isfinite(zone.polygon)):
            errors.append(f"StopZone {index} polygon contains NaN or Inf")
        for lane_id in zone.controlled_lanes:
            if lane_id not in lane_ids:
                errors.append(f"StopZone {index} controlled_lanes references non-lane element {lane_id}")

    for index, control in enumerate(scenario.traffic_controls):
        for lane_id in control.controlled_lanes:
            if lane_id not in lane_ids:
                errors.append(f"TrafficControl {index} controlled_lanes references non-lane element {lane_id}")
        if any(int(state) not in puffer_types.TL_STATE_NAMES for state in control.states):
            errors.append(f"TrafficControl {index} contains invalid state")

    for items in (scenario.agents, scenario.objects):
        for track_id, track in items.items():
            if not 0 <= track.route_gt_len <= len(track.route):
                errors.append(f"Track {track_id} route_gt_len is outside its route")
            for lane_id in track.route:
                if lane_id not in lane_ids:
                    errors.append(f"Track {track_id} route references non-lane element {lane_id}")

    if scenario.lane_graph is None:
        return
    graph_lane_ids = scenario.lane_graph["lane_ids"]
    if len(graph_lane_ids) != len(set(graph_lane_ids)):
        errors.append("lane_graph.lane_ids contains duplicates")
    for lane_id in graph_lane_ids:
        if lane_id not in lane_ids:
            errors.append(f"lane_graph.lane_ids references non-lane element {lane_id}")


def _validate_agent_sizes(items: dict[int, schema.Track], prefix: str, errors: list[str]) -> None:
    for eid, ds in items.items():
        valid = ds.valid.astype(bool)
        if not np.any(valid):
            continue
        for key in ("length", "width", "height"):
            vals = getattr(ds, key)[valid]
            if np.any(vals <= 0):
                errors.append(f"{prefix} {eid} has non-positive {key}")


def _z_at_xy(polyline: np.ndarray, point: Point) -> float:
    seg = np.linalg.norm(np.diff(polyline[:, :2], axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = LineString(polyline[:, :2]).project(point)
    i = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1))
    t = 0.0 if seg[i] == 0 else (s - cum[i]) / seg[i]
    return polyline[i, 2] + t * (polyline[i + 1, 2] - polyline[i, 2])


def _intersection_points(geom: BaseGeometry) -> list[Point]:
    if geom.is_empty:
        return []
    if isinstance(geom, Point):
        return [geom]
    if isinstance(geom, MultiPoint):
        return list(geom.geoms)
    if hasattr(geom, "geoms"):
        return [p for g in geom.geoms for p in _intersection_points(g)]
    if isinstance(geom, LineString):
        coords = list(geom.coords)
        return [Point(coords[0]), Point(coords[-1])] if coords else []
    return []


def _validate_road_edge_z_overlap(map_data: dict[int, schema.MapElement], errors: list[str]) -> None:
    edges = [
        (eid, np.asarray(elem.polyline))
        for eid, elem in map_data.items()
        if elem.is_edge
        and isinstance(elem.polyline, np.ndarray)
        and elem.polyline.ndim == 2
        and elem.polyline.shape[0] >= 2
        and elem.polyline.shape[1] >= 3
    ]
    if len(edges) < 2:
        return

    lines = [LineString(p[:, :2]) for _, p in edges]
    tree = STRtree(lines)

    for i, (eid_a, poly_a) in enumerate(edges):
        for j in tree.query(lines[i]):
            j = int(j)
            if j <= i:
                continue
            inter = lines[i].intersection(lines[j])
            for p in _intersection_points(inter):
                dz = abs(_z_at_xy(poly_a, p) - _z_at_xy(edges[j][1], p))
                if _Z_BRIDGE_MIN < dz < _Z_BRIDGE_MAX:
                    errors.append(
                        f"Map edges {eid_a} and {edges[j][0]} cross in XY with |dz|={dz:.2f}m (sub-4m bridge)"
                    )
                    break
