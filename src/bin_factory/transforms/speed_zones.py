"""Speed zones: connected lane sets that share one speed limit, the unit for speed-limit randomization.

A zone follows real-world signage: parallel lanes of one lane group, both driving directions of the
same road, and consecutive road pieces up to the next junction or posted-limit change. Junction
connector lanes carry no zone (``speed_zone_idx == -1``); they inherit from their entry lanes downstream.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import shapely
from scipy import spatial as scipy_spatial

from bin_factory.log_context import log


if TYPE_CHECKING:
    from bin_factory import schema


DEFAULT_MIN_ZONE_EXTENT_M = 100.0
DEFAULT_OPPOSITE_GAP_M = 8.0
# Two opposite-direction lanes count as one road when this share of the inner boundary lies within the gap
OPPOSITE_OVERLAP_MIN_FRACTION = 0.5
OPPOSITE_HEADING_MIN_RAD = np.deg2rad(150.0)
SPEED_LIMIT_MATCH_TOLERANCE_MPS = 0.05
MAX_JUNCTION_HOPS = 4
# A short zone only merges across a junction along a straight continuation, never around a corner
MAX_MERGE_TURN_RAD = np.deg2rad(30.0)


def compute_speed_zones(
    scenario: schema.PufferScenario,
    min_zone_extent_m: float = DEFAULT_MIN_ZONE_EXTENT_M,
    opposite_gap_m: float = DEFAULT_OPPOSITE_GAP_M,
) -> None:
    """Annotate each lane with ``speed_zone_idx`` (compact 0..N-1, -1 for junction lanes).

    Must run AFTER process_polylines (geometry is final) and BEFORE reindex (uses source ids).
    """
    lanes = {eid: e for eid, e in scenario.map.items() if e.is_lane and e.polyline is not None}
    for lane in lanes.values():
        lane.speed_zone_idx = -1
    if not lanes:
        return
    if all(lane.lane_group_id is None for lane in lanes.values()):
        log.warning("speed zones skipped: map carries no lane groups")
        return

    parent = {eid: eid for eid, lane in lanes.items() if not lane.in_junction}
    _union_lane_groups(lanes, parent)
    _union_consecutive_lanes(lanes, parent)
    _union_opposite_directions(lanes, parent, opposite_gap_m)
    _merge_short_zones(lanes, parent, min_zone_extent_m)

    zones = _zones_by_root(parent)
    ordered_roots = sorted(zones, key=lambda root: min(zones[root]))
    for zone_idx, root in enumerate(ordered_roots):
        for eid in zones[root]:
            lanes[eid].speed_zone_idx = zone_idx
    junction_count = len(lanes) - len(parent)
    log.info("speed zones: %d zones over %d lanes (%d junction lanes)", len(zones), len(lanes), junction_count)


# ── Union-find ───────────────────────────────────────────────────────────────


def _find(parent: dict[int, int], eid: int) -> int:
    root = eid
    while parent[root] != root:
        root = parent[root]
    while parent[eid] != root:
        parent[eid], eid = root, parent[eid]
    return root


def _union(parent: dict[int, int], a: int, b: int) -> None:
    root_a, root_b = _find(parent, a), _find(parent, b)
    if root_a != root_b:
        parent[max(root_a, root_b)] = min(root_a, root_b)


def _zones_by_root(parent: dict[int, int]) -> dict[int, list[int]]:
    zones: dict[int, list[int]] = defaultdict(list)
    for eid in parent:
        zones[_find(parent, eid)].append(eid)
    return dict(zones)


def _same_limit(a: schema.MapElement, b: schema.MapElement) -> bool:
    return abs(a.speed_limit_mps - b.speed_limit_mps) <= SPEED_LIMIT_MATCH_TOLERANCE_MPS


# ── Union rules ──────────────────────────────────────────────────────────────


def _union_lane_groups(lanes: dict[int, schema.MapElement], parent: dict[int, int]) -> None:
    by_group: dict[int, list[int]] = defaultdict(list)
    for eid in parent:
        group_id = lanes[eid].lane_group_id
        if group_id is not None:
            by_group[group_id].append(eid)
    for members in by_group.values():
        for eid in members[1:]:
            _union(parent, members[0], eid)


def _union_consecutive_lanes(lanes: dict[int, schema.MapElement], parent: dict[int, int]) -> None:
    for eid in parent:
        for exit_id in lanes[eid].exit_lanes:
            if exit_id in parent and _same_limit(lanes[eid], lanes[exit_id]):
                _union(parent, eid, exit_id)


def _union_opposite_directions(lanes: dict[int, schema.MapElement], parent: dict[int, int], gap_m: float) -> None:
    if gap_m <= 0.0:
        return
    inner_ids = [
        eid
        for eid in parent
        if not lanes[eid].left_neighbor and lanes[eid].left_boundary is not None and len(lanes[eid].left_boundary) >= 2
    ]
    if len(inner_ids) < 2:
        return
    boundaries = {eid: np.asarray(lanes[eid].left_boundary, dtype=np.float64)[:, :2] for eid in inner_ids}
    lines = [shapely.LineString(boundaries[eid]) for eid in inner_ids]
    tree = shapely.STRtree(lines)
    pairs = tree.query(lines, predicate="dwithin", distance=gap_m)
    for i, j in pairs.T:
        if i >= j:
            continue
        eid_a, eid_b = inner_ids[i], inner_ids[j]
        if _find(parent, eid_a) == _find(parent, eid_b) or not _same_limit(lanes[eid_a], lanes[eid_b]):
            continue
        if _is_opposite_neighbor(boundaries[eid_a], boundaries[eid_b], gap_m) or _is_opposite_neighbor(
            boundaries[eid_b], boundaries[eid_a], gap_m
        ):
            _union(parent, eid_a, eid_b)


def _is_opposite_neighbor(boundary_a: np.ndarray, boundary_b: np.ndarray, gap_m: float) -> bool:
    """True when most of boundary_a runs within gap_m of boundary_b, heading the opposite way."""
    headings_a = _point_headings(boundary_a)
    headings_b = _point_headings(boundary_b)
    distances, nearest_idx = scipy_spatial.cKDTree(boundary_b).query(boundary_a)
    heading_diff = np.abs(np.angle(np.exp(1j * (headings_a - headings_b[nearest_idx]))))
    close_and_opposite = (distances <= gap_m) & (heading_diff >= OPPOSITE_HEADING_MIN_RAD)
    return bool(np.mean(close_and_opposite) >= OPPOSITE_OVERLAP_MIN_FRACTION)


def _point_headings(polyline: np.ndarray) -> np.ndarray:
    segment_headings = np.arctan2(np.diff(polyline[:, 1]), np.diff(polyline[:, 0]))
    return np.append(segment_headings, segment_headings[-1])


# ── Short-zone merging through junctions ─────────────────────────────────────


def _merge_short_zones(lanes: dict[int, schema.MapElement], parent: dict[int, int], min_extent_m: float) -> None:
    if min_extent_m <= 0.0:
        return
    max_passes = len(parent)
    for _ in range(max_passes):
        zones = _zones_by_root(parent)
        extents = {root: _zone_extent_m(lanes, members) for root, members in zones.items()}
        short_roots = sorted((root for root in zones if extents[root] < min_extent_m), key=lambda r: (extents[r], r))
        merged = False
        for root in short_roots:
            candidates = _zones_across_junctions(lanes, parent, zones[root])
            candidates.discard(root)
            if not candidates:
                continue
            target = min(candidates, key=lambda r: (extents[r], r))
            _union(parent, root, target)
            merged = True
            break
        if not merged:
            return


def _zone_extent_m(lanes: dict[int, schema.MapElement], members: list[int]) -> float:
    points = np.vstack([np.asarray(lanes[eid].polyline, dtype=np.float64)[:, :2] for eid in members])
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def _zones_across_junctions(
    lanes: dict[int, schema.MapElement], parent: dict[int, int], members: list[int]
) -> set[int]:
    """Roots of same-limit zones reached straight through junction lanes from ``members`` (both directions)."""
    limit_lane = lanes[members[0]]
    roots: set[int] = set()
    for eid in members:
        for key in ("exit_lanes", "entry_lanes"):
            frontier = [nid for nid in getattr(lanes[eid], key) if nid in lanes and nid not in parent]
            for _ in range(MAX_JUNCTION_HOPS):
                next_frontier = []
                for junction_id in frontier:
                    for nid in getattr(lanes[junction_id], key):
                        if nid not in lanes:
                            continue
                        if nid not in parent:
                            next_frontier.append(nid)
                        elif _same_limit(lanes[nid], limit_lane) and _is_straight_continuation(
                            lanes[eid], lanes[nid], key
                        ):
                            roots.add(_find(parent, nid))
                frontier = next_frontier
                if not frontier:
                    break
    return roots


def _is_straight_continuation(lane: schema.MapElement, other: schema.MapElement, key: str) -> bool:
    if key == "exit_lanes":
        heading_out, heading_in = _end_heading(lane.polyline), _start_heading(other.polyline)
    else:
        heading_out, heading_in = _end_heading(other.polyline), _start_heading(lane.polyline)
    return abs(np.angle(np.exp(1j * (heading_in - heading_out)))) <= MAX_MERGE_TURN_RAD


def _start_heading(polyline: np.ndarray) -> float:
    return float(np.arctan2(polyline[1, 1] - polyline[0, 1], polyline[1, 0] - polyline[0, 0]))


def _end_heading(polyline: np.ndarray) -> float:
    return float(np.arctan2(polyline[-1, 1] - polyline[-2, 1], polyline[-1, 0] - polyline[-2, 0]))
