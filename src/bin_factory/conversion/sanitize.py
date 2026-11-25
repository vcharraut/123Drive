import collections
import dataclasses
from typing import cast

import numpy as np
import shapely

from bin_factory import puffer_types, schema
from bin_factory.schema import remap_element_refs


MAX_LANE_GAP = 3.0  # Maximum endpoint gap in metres for connected entry and exit lanes
LINE_DUPLICATE_DISTANCE = 0.1  # Hausdorff distance in metres that marks a road line as a duplicate
HOOK_MAX_LENGTH = 0.5  # Maximum end-segment length in metres that counts as a sharp hook
HOOK_MIN_TURN = np.radians(30.0)
NEIGHBOR_MIN_OVERLAP = 10.0  # Metres a neighbour must run alongside a lane, capped at half the shorter lane
NEIGHBOR_GAP = (1.5, 6.0)  # Lateral distance range in metres of a neighbour centerline
NEIGHBOR_MAX_ANGLE = np.radians(25.0)


def clean_map(polylines: dict[int, schema.RoadPolyline]) -> None:
    """Correct py123d map quirks in freshly extracted polylines, in place.

    In order: trim lane end hooks, drop disconnected lane links and infer undefined lane types, keep only
    neighbours that run alongside, spread speed limits along lane links, and drop duplicate road lines.
    """
    lanes = {element_id: element for element_id, element in polylines.items() if element.is_lane}
    for lane in lanes.values():
        lane.polyline = _trim_end_hooks(cast("np.ndarray", lane.polyline))  # lanes always carry a centerline
    _fix_lane_topology(lanes)
    _drop_partial_neighbors(lanes)
    fill_missing_speed_limits(lanes)
    drop_duplicate_lines(polylines)


def prune_invalid_map_elements(scenario: schema.PufferScenario, extras: schema.ExtractionExtras | None = None) -> None:
    valid_ids = {
        element_id
        for element_id, element in scenario.polylines.items()
        if element.polyline is not None and len(element.polyline) >= 2
    }
    if len(valid_ids) != len(scenario.polylines):
        identity_map = {element_id: element_id for element_id in valid_ids}
        scenario.polylines = {
            element_id: remap_element_refs(element, identity_map)
            for element_id, element in scenario.polylines.items()
            if element_id in valid_ids
        }
    scenario.surfaces = {
        element_id: element for element_id, element in scenario.surfaces.items() if len(element.polygon) >= 3
    }

    if extras is None:
        return

    lane_ids = {element_id for element_id, element in scenario.polylines.items() if element.is_lane}
    extras.traffic_lights = {
        element_id: light for element_id, light in extras.traffic_lights.items() if light.controlled_lane in lane_ids
    }
    extras.stop_zones = [
        dataclasses.replace(zone, controlled_lanes=[lid for lid in zone.controlled_lanes if lid in lane_ids])
        for zone in extras.stop_zones
        if any(lid in lane_ids for lid in zone.controlled_lanes)
    ]


def fill_missing_speed_limits(lanes: dict[int, schema.RoadPolyline]) -> None:
    """Give each lane with an unknown speed limit the limit of its nearest connected lane via BFS."""
    # PY123D-REPORT[opendrive]: py123d only copies road-type speeds one hop into junction lanes and matches the
    # unit "mps" where OpenDRIVE spells "m/s", leaving ~50% of CARLA Town01-10 lanes without a limit.
    # PY123D-REPORT[wod-motion]: WOMD lanes with speed_limit_mph == 0 come through as unknown.
    queue = collections.deque(lane_id for lane_id, lane in lanes.items() if lane.speed_limit_mps > 0)
    while queue:
        lane = lanes[queue.popleft()]
        for ref in [*lane.entry_lanes, *lane.exit_lanes]:
            if lanes[ref].speed_limit_mps <= 0:
                lanes[ref].speed_limit_mps = lane.speed_limit_mps
                queue.append(ref)


def drop_duplicate_lines(elements: dict[int, schema.RoadPolyline]) -> None:
    """Remove road lines that repeat an earlier line of the same type, in either direction or resampled."""
    # PY123D-REPORT[nuplan,opendrive]: ~40% of road lines come twice, once per adjacent lane, often reversed and
    # sometimes sampled differently. CARLA types the reversed copy of a DASH_SOLID line as SOLID_DASH, so reversed
    # lines also match across that pair.
    ids = [eid for eid, element in elements.items() if element.is_line]
    if len(ids) < 2:
        return
    polylines = [cast("np.ndarray", elements[eid].polyline) for eid in ids]  # road lines always carry one
    types = np.array([elements[eid].type for eid in ids], dtype=int)
    mirrored = types.copy()
    for dash_solid, solid_dash in (
        (puffer_types.RoadLineType.DASH_SOLID_YELLOW, puffer_types.RoadLineType.SOLID_DASH_YELLOW),
        (puffer_types.RoadLineType.DASH_SOLID_WHITE, puffer_types.RoadLineType.SOLID_DASH_WHITE),
    ):
        mirrored[types == dash_solid], mirrored[types == solid_dash] = solid_dash, dash_solid
    ends = np.array([polyline[[0, -1], :2] for polyline in polylines]).reshape(-1, 2, 2)
    pts = shapely.points(ends.min(axis=1))
    i, j = shapely.STRtree(pts).query(pts, predicate="dwithin", distance=LINE_DUPLICATE_DISTANCE * np.sqrt(2.0))
    same_type = types[i] == types[j]
    mask = (i < j) & (same_type | (types[i] == mirrored[j]))
    i, j, same_type = i[mask], j[mask], same_type[mask]
    same = same_type & (np.linalg.norm(ends[i] - ends[j], axis=-1).max(-1) < LINE_DUPLICATE_DISTANCE)
    flipped = np.linalg.norm(ends[i] - ends[j, ::-1], axis=-1).max(-1) < LINE_DUPLICATE_DISTANCE
    cand = same | flipped
    i, j = i[cand], j[cand]
    if len(i) == 0:
        return
    geoms_i = [shapely.LineString(polylines[idx][:, :2]) for idx in i]
    geoms_j = [shapely.LineString(polylines[idx][:, :2]) for idx in j]
    dup = shapely.hausdorff_distance(geoms_i, geoms_j) < LINE_DUPLICATE_DISTANCE
    for k in np.unique(j[dup]):
        del elements[ids[k]]


def _trim_end_hooks(line: np.ndarray) -> np.ndarray:
    """Remove points near a centerline's ends when a segment turns more than ``HOOK_MIN_TURN``.

    Check the first and last ``HOOK_MAX_LENGTH`` against the chord to the first point at least 1 m away.
    Keep both endpoints.
    """
    # PY123D-REPORT[opendrive]: ~2% of Town12/13 junction lane centerlines start or end with a 0.25 m zig-zag
    # that turns the lane's end heading and spikes its curvature.
    if len(line) < 3:
        return line
    hook_sq = HOOK_MAX_LENGTH**2
    cos_min = np.cos(HOOK_MIN_TURN)
    for _ in range(2):
        if len(line) >= 3 and ((line[1, :2] - line[0, :2]) ** 2).sum() < hook_sq:
            dist_sq = ((line[:, :2] - line[0, :2]) ** 2).sum(axis=1)
            if (dist_sq >= 1.0).any() and (dist_sq >= hook_sq).any():
                chord = line[np.argmax(dist_sq >= 1.0), :2] - line[0, :2]
                end = int(np.argmax(dist_sq >= hook_sq))  # first point past the end stretch
                steps = np.diff(line[: end + 1, :2], axis=0)
                cos_turn = steps @ chord / np.maximum(np.linalg.norm(steps, axis=1) * np.linalg.norm(chord), 1e-12)
                if (cos_turn < cos_min).any():
                    line = np.delete(line, np.arange(1, end), axis=0)
        line = line[::-1]
    return line


def _fix_lane_topology(lanes: dict[int, schema.RoadPolyline]) -> None:
    """Infer undefined lane types from connected lanes and drop missing or disconnected references.

    Keep entry and exit directions from py123d, which already orients nuPlan connections.
    The previous endpoint-distance reversal created cycles on broken WOMD links.
    """
    # PY123D-REPORT[nuplan]: most nuPlan lane connectors come out of py123d as LaneType.UNDEFINED.
    # PY123D-REPORT[wod-motion]: some WOMD entry/exit refs point at lanes up to ~100 m away.
    undefined_lane_ids = [lane_id for lane_id, lane in lanes.items() if lane.type == puffer_types.LaneType.UNDEFINED]
    gap_sq = MAX_LANE_GAP**2
    exits_by_lane = {lane_id: set(lane.exit_lanes) for lane_id, lane in lanes.items()}

    def _gap_sq(src_id: int, dst_id: int) -> float:
        src_end = cast("np.ndarray", lanes[src_id].polyline)[-1, :2]
        return float(((src_end - cast("np.ndarray", lanes[dst_id].polyline)[0, :2]) ** 2).sum())

    def _connected(src_id: int, dst_id: int) -> bool:
        # A dropped lane joins its successor sideways; a neighbour reaching that successor end to start marks the merge
        neighbor_ids = [*lanes[src_id].left_neighbor, *lanes[src_id].right_neighbor]
        return _gap_sq(src_id, dst_id) <= gap_sq or (
            _gap_sq(src_id, dst_id) <= NEIGHBOR_GAP[1] ** 2
            and any(
                nid in lanes and dst_id in exits_by_lane[nid] and _gap_sq(nid, dst_id) <= gap_sq for nid in neighbor_ids
            )
        )

    for lane_id, element in lanes.items():
        element.entry_lanes = [lid for lid in element.entry_lanes if lid in lanes and _connected(lid, lane_id)]
        element.exit_lanes = [lid for lid in element.exit_lanes if lid in lanes and _connected(lane_id, lid)]
        element.left_neighbor = [lid for lid in element.left_neighbor if lid in lanes]
        element.right_neighbor = [lid for lid in element.right_neighbor if lid in lanes]

    for lane_id in undefined_lane_ids:
        lane = lanes[lane_id]
        connected_types = {lanes[nid].type for nid in [*lane.entry_lanes, *lane.exit_lanes]}
        if len(connected_types) == 1:
            lane.type = connected_types.pop()


def _drop_partial_neighbors(lanes: dict[int, schema.RoadPolyline]) -> None:
    """Keep a left or right neighbour only if it runs alongside the lane over a lane change's length."""
    # PY123D-REPORT[wod-motion]: WOMD neighbours carry the index range where two lanes are adjacent. py123d keeps one
    # neighbour per side and drops the range, so lanes side by side for a few metres come out as whole-lane neighbours.
    for lane in lanes.values():
        line = cast("np.ndarray", lane.polyline)  # lanes always carry a centerline
        lane.left_neighbor = [
            ref for ref in lane.left_neighbor if _alongside(line, cast("np.ndarray", lanes[ref].polyline), 1.0)
        ]
        lane.right_neighbor = [
            ref for ref in lane.right_neighbor if _alongside(line, cast("np.ndarray", lanes[ref].polyline), -1.0)
        ]


def _alongside(line: np.ndarray, other: np.ndarray, side: float) -> bool:
    """Return True when ``other`` runs parallel to ``line`` on ``side`` (+1 left, -1 right) long enough."""
    line, other = line[:, :2], other[:, :2]
    steps = np.diff(line, axis=0)
    step_norms = np.linalg.norm(steps, axis=1)
    cum = np.r_[0.0, np.cumsum(step_norms)]
    other_steps = np.diff(other, axis=0)
    other_norms = np.linalg.norm(other_steps, axis=1)
    other_length = float(other_norms.sum())
    if cum[-1] <= 0 or other_length <= 0:
        return False
    samples = np.arange(0.0, cum[-1], 1.0)
    points = np.column_stack([np.interp(samples, cum, line[:, 0]), np.interp(samples, cum, line[:, 1])])
    seg_idx = np.clip(np.searchsorted(cum, samples, side="right") - 1, 0, len(steps) - 1)
    direction = steps[seg_idx] / np.maximum(step_norms[seg_idx, None], 1e-9)

    sq = np.maximum((other_steps**2).sum(1), 1e-9)
    dx = points[:, None, 0] - other[:-1, 0]
    dy = points[:, None, 1] - other[:-1, 1]
    t = np.clip((dx * other_steps[:, 0] + dy * other_steps[:, 1]) / sq, 0.0, 1.0)
    rx = t * other_steps[:, 0] - dx
    ry = t * other_steps[:, 1] - dy
    nearest = (rx * rx + ry * ry).argmin(1)
    p_idx = np.arange(len(points))
    rel_x = rx[p_idx, nearest]
    rel_y = ry[p_idx, nearest]
    lateral = side * (direction[:, 0] * rel_y - direction[:, 1] * rel_x)
    other_dir = other_steps[nearest] / np.sqrt(sq[nearest])[:, None]
    parallel = (direction * other_dir).sum(1) > np.cos(NEIGHBOR_MAX_ANGLE)
    adjacent = (lateral > NEIGHBOR_GAP[0]) & (lateral < NEIGHBOR_GAP[1]) & parallel
    return adjacent.sum() >= min(NEIGHBOR_MIN_OVERLAP, 0.5 * min(cum[-1], other_length))
