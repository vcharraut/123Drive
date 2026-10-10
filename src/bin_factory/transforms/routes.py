"""Compute per-agent lane routes from observed trajectories.

The route pipeline is graph-constrained:
1. Build a per-scenario lane cache once.
2. Gate non-ego agents on validity and on-road status at the check timestep.
3. Build per-point lane candidates from GT geometry.
4. Solve a best lane sequence with dynamic programming.
5. Extend beyond GT to the map dead-end.
"""

import itertools
from collections import deque
from typing import Any

import numpy as np
import shapely
from shapely import geometry as shapely_geom

from bin_factory import puffer_types, schema
from bin_factory.log_context import log
from bin_factory.transforms.geometry import arc_length, polyline_length


LANE_WIDTH_THRESHOLD = 7.0  # meters — reject point-to-lane matches farther than this
ALIGNMENT_THRESHOLD = 0.3  # cos(heading) — minimum dot product for "same direction"
ROUTE_CANDIDATE_BBOX_MARGIN = 12.0  # meters — bbox expansion when filtering nearby lanes
MAX_CANDIDATES_PER_POINT = 7
MAX_TRANSITION_HOPS = 3
BACKWARD_PROGRESS_TOLERANCE = 2.0  # meters — allow small projection noise on same lane
OFFROAD_DISTANCE_THRESHOLD = 5.0  # meters — max lane distance at the check timestep
MIN_ROUTE_REMAINING = 15.0  # meters of route ahead at the check timestep; above the simulator's largest goal radius
ELEVATION_THRESHOLD = 2.0  # meters — reject matches to vertically separated roads
SMOOTH_WINDOW = 7  # frames — odd median-filter window; kills per-frame perception spikes
PARKED_MOTION_THRESHOLD = 1.5
PARKED_MOTION_WINDOW_SECONDS = 1.5
PARKED_DIRECTIONAL_CONSISTENCY = 0.8
SKIPPED_POINT_COST = 50.0
HOP_COST = 1.0
LANE_CHANGE_COST = 6.0

_RouteCache = dict[str, Any]
# (point_idx, lane_id, distance, s)
_Candidate = tuple[int, int, float, float]


def process_agent_routes(scenario: schema.PufferScenario, route_check_timestep: int = 0) -> None:
    """Compute a lane route per agent and assign it back onto ``scenario``.

    Routes are graph-constrained lane sequences. For each vehicle agent, the ground-truth
    trajectory is projected onto lane centerlines and a best-matching lane sequence is
    solved via DP, then extended past the last GT lane toward a dead-end.

    Arguments:
        scenario: PufferScenario whose ``agents`` and ``map`` are read; routes are written
            back onto each Track (``route``, ``route_gt_len``, ``control_state``).
        route_check_timestep: Timestep at which the agent must be valid (and on-road) for
            non-ego routes. Ego (vehicle id 0) bypasses this gate and the parked filter;
            failure on ego raises.
    """
    scenario_length = scenario.metadata.scenario_length
    if scenario_length > 0 and route_check_timestep >= scenario_length:
        raise ValueError(
            f"route_check_timestep={route_check_timestep} is out of range for scenario length {scenario_length}",
        )

    route_cache = build_route_cache(scenario.map)
    dt = scenario.metadata.dt
    for agent_id, agent_data in scenario.agents.items():
        is_vehicle = agent_data.type == puffer_types.AgentType.VEHICLE
        is_ego = is_vehicle and agent_id == 0
        is_static = not is_ego and _is_static(agent_data, dt)

        # Parked cars are NON_CONTROLLABLE_STATIC regardless of route, so skip the route DP for them.
        route, route_gt_len = (
            ([], 0)
            if not is_vehicle or is_static
            else compute_agent_route(
                agent_id=agent_id,
                positions=agent_data.position,
                headings=agent_data.heading,
                valid=agent_data.valid,
                lengths=agent_data.length,
                widths=agent_data.width,
                is_ego=is_ego,
                route_cache=route_cache,
                route_check_timestep=route_check_timestep,
            )
        )
        t = route_check_timestep
        # The simulator retires an agent that spawns at its route end, so such a vehicle is replayed, not controlled
        has_road_ahead = bool(route) and (
            _route_remaining(agent_data.position[t], route, route_cache) >= MIN_ROUTE_REMAINING
        )
        if is_ego and (
            not has_road_ahead
            or _is_offroad(
                agent_data.position[t], agent_data.heading[t], agent_data.length[t], agent_data.width[t], route_cache
            )
        ):
            raise ValueError(
                f"Ego vehicle (agent 0) has no route, under {MIN_ROUTE_REMAINING:g} m of route ahead or an off-road "
                f"start in scenario {scenario.metadata.id}"
            )
        agent_data.route = route
        agent_data.route_gt_len = route_gt_len
        agent_data.control_state = int(
            puffer_types.ControlState.NON_CONTROLLABLE_STATIC
            if is_static
            else puffer_types.ControlState.CONTROLLABLE
            if has_road_ahead
            else puffer_types.ControlState.NON_CONTROLLABLE_MOVING
        )
    _replay_log_conflicts(scenario.agents, route_check_timestep)


def _replay_log_conflicts(agents: dict[int, schema.Track], start: int) -> None:
    """Hand to log replay (NON_CONTROLLABLE_MOVING) the controllable vehicles whose logged box overlaps another agent
    at the start frame, or a log-driven (replayed or frozen) agent at any later frame.

    Controlling them would start the policy in, or drive it along its route into, a collision the log already holds
    (duplicate tracks, perception noise). Ego stays controllable. Only the control state changes: trajectories,
    validity and routes are kept, so log replay and WOSAC still see the whole scene.
    """
    tracks = list(agents.values())
    corners = _box_corners(tracks) if tracks else np.zeros((0, 0, 4, 2))
    hits = []  # (agent, other, frame) index triples of intersecting boxes
    for t in range(start, corners.shape[1]):
        live = np.flatnonzero([track.valid[t] for track in tracks])
        if len(live) < 2:
            continue
        boxes = shapely.polygons(corners[live, t])
        i, j = shapely.STRtree(boxes).query(boxes, predicate="intersects")
        hits.append(np.stack([live[i[i != j]], live[j[i != j]], np.full((i != j).sum(), t)], 1))
    if not hits:
        return
    agent, other, frame = np.concatenate(hits).T
    was_controllable = np.array([track.control_state == puffer_types.ControlState.CONTROLLABLE for track in tracks])
    is_ego = np.array([aid == 0 for aid in agents])
    if (is_ego[agent] & was_controllable[agent] & (frame == start)).any():
        raise ValueError("Ego vehicle (agent 0) overlaps another agent at the start frame")
    controlled = was_controllable.copy()
    # A vehicle handed to replay is log-driven too, so controlled vehicles it overlaps later are handed over in turn
    while (hit := controlled[agent] & ~is_ego[agent] & ((frame == start) | ~controlled[other])).any():
        controlled[agent[hit]] = False
    for k in np.flatnonzero(was_controllable & ~controlled):
        tracks[k].control_state = int(puffer_types.ControlState.NON_CONTROLLABLE_MOVING)


def _route_remaining(position: np.ndarray, route: list[int], route_cache: _RouteCache) -> float:
    """Route length (m) ahead of ``position``: the rest of the nearest route lane plus every later lane."""
    idx = np.array([route_cache["lane_id_to_idx"][lane_id] for lane_id in route])
    distances, closest, closest_t = _points_to_polylines_distance(
        position[:2].reshape(1, 2), route_cache["lane_polylines"][idx], route_cache["lane_lengths"][idx]
    )
    nearest = int(np.argmin(distances[0]))
    segment = closest[0, nearest]
    lengths = route_cache["lane_cum_lengths"][idx, route_cache["lane_lengths"][idx] - 1]
    progress = (
        route_cache["lane_cum_lengths"][idx[nearest], segment]
        + closest_t[0, nearest] * route_cache["lane_segment_lengths"][idx[nearest], segment]
    )
    return float(lengths[nearest] - progress + lengths[nearest + 1 :].sum())


def _box_corners(tracks: list[schema.Track]) -> np.ndarray:
    """Oriented box corners (N, T, 4, 2) of every track at every frame."""
    position = np.stack([track.position[:, :2] for track in tracks])
    heading = np.stack([track.heading for track in tracks])
    half = np.stack([np.stack([track.length, track.width], -1) / 2 for track in tracks])
    local = half[:, :, None] * np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])
    cos, sin = np.cos(heading)[..., None], np.sin(heading)[..., None]
    rotated = np.stack([cos * local[..., 0] - sin * local[..., 1], sin * local[..., 0] + cos * local[..., 1]], -1)
    return rotated + position[:, :, None]


def build_route_cache(static_map_elements: dict[int, schema.MapElement]) -> _RouteCache:
    """Precompute lane geometry and connectivity shared by all agents."""
    lanes = {
        element_id: element
        for element_id, element in static_map_elements.items()
        if element.type in (puffer_types.LaneType.SURFACE_STREET, puffer_types.LaneType.FREEWAY)
        and element.polyline is not None
        and len(element.polyline) > 0
    }
    lane_ids = list(lanes)
    n_lanes = len(lane_ids)
    lane_lengths = np.array([len(lanes[lane_id].polyline) for lane_id in lane_ids], dtype=np.int64)
    max_points = int(lane_lengths.max(initial=0))

    lane_polylines_xyz = np.zeros((n_lanes, max_points, 3), dtype=np.float64)
    lane_bbox_mins = np.zeros((n_lanes, 2), dtype=np.float64)
    lane_bbox_maxs = np.zeros((n_lanes, 2), dtype=np.float64)
    lane_head_dirs = np.zeros((n_lanes, 2), dtype=np.float64)
    lane_tail_dirs = np.zeros((n_lanes, 2), dtype=np.float64)
    lane_segment_lengths = np.zeros((n_lanes, max(0, max_points - 1)), dtype=np.float64)
    lane_cum_lengths = np.zeros((n_lanes, max_points), dtype=np.float64)

    for idx, lane_id in enumerate(lane_ids):
        n_points = lane_lengths[idx]
        lane_polylines_xyz[idx, :n_points] = lanes[lane_id].polyline[:, :3]
        polyline = lane_polylines_xyz[idx, :n_points, :2]
        lane_bbox_mins[idx] = polyline.min(axis=0)
        lane_bbox_maxs[idx] = polyline.max(axis=0)
        lane_head_dirs[idx] = _get_lane_endpoint_direction(polyline, from_start=True)
        lane_tail_dirs[idx] = _get_lane_endpoint_direction(polyline, from_start=False)
        seg_lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
        lane_segment_lengths[idx, : len(seg_lengths)] = seg_lengths
        lane_cum_lengths[idx, :n_points] = np.concatenate(([0.0], np.cumsum(seg_lengths)))

    lane_id_to_idx = {lane_id: idx for idx, lane_id in enumerate(lane_ids)}
    edge_polylines = [
        element.polyline[:, :3]
        for element in static_map_elements.values()
        if element.is_edge and element.polyline is not None and len(element.polyline) >= 2
    ]

    return {
        "lane_id_array": np.asarray(lane_ids, dtype=np.int64),
        "lane_polylines": lane_polylines_xyz[:, :, :2],
        "lane_polylines_xyz": lane_polylines_xyz,
        "lane_lengths": lane_lengths,
        "lane_id_to_idx": lane_id_to_idx,
        "lane_graph": {
            lane_id: tuple(exit_id for exit_id in lanes[lane_id].exit_lanes if exit_id in lane_id_to_idx)
            for lane_id in lane_ids
        },
        "lane_bbox_mins": lane_bbox_mins,
        "lane_bbox_maxs": lane_bbox_maxs,
        "lane_head_dirs": lane_head_dirs,
        "lane_tail_dirs": lane_tail_dirs,
        "lane_segment_lengths": lane_segment_lengths,
        "lane_cum_lengths": lane_cum_lengths,
        "path_cache": {},
        "road_edges": tuple((p, p[:, :2].min(axis=0), p[:, :2].max(axis=0)) for p in edge_polylines),
    }


def compute_agent_route(
    agent_id: int,
    positions: np.ndarray,
    headings: np.ndarray,
    valid: np.ndarray,
    lengths: np.ndarray,
    widths: np.ndarray,
    is_ego: bool,
    route_cache: _RouteCache,
    route_check_timestep: int = 0,
) -> tuple[list[int], int]:
    """Return ``(route, route_gt_len)`` for one agent, or ``([], 0)``.

    Ego bypasses the check-timestep gate; a non-ego agent must be valid and on-road there.
    """
    valid = np.asarray(valid, dtype=bool)
    t = route_check_timestep
    if (
        not valid.any()
        or len(route_cache["lane_id_array"]) == 0
        or (not is_ego and (not valid[t] or _is_offroad(positions[t], headings[t], lengths[t], widths[t], route_cache)))
    ):
        log.debug("agent=%d: skipping route computation (insufficient valid data or offroad start)", agent_id)
        return [], 0

    observations = _build_point_observations(positions[valid], headings[valid], route_cache)
    if not observations:
        log.debug("agent=%d: no GT-supported lane candidates found", agent_id)
        return [], 0

    candidate_path = _select_candidate_path(observations, int(valid.sum()), route_cache)
    gt_route = [candidate_path[0][1]]
    for prev_candidate, next_candidate in itertools.pairwise(candidate_path):
        gt_route.extend(_find_shortest_lane_path(prev_candidate[1], next_candidate[1], route_cache)[1:])

    return _extend_route_to_dead_end(gt_route.copy(), route_cache), len(gt_route)


def _build_point_observations(
    positions: np.ndarray,
    headings: np.ndarray,
    route_cache: _RouteCache,
) -> list[list[_Candidate]]:
    """Return, per valid point with any match, its top lane candidates ranked by distance then alignment."""
    trajectory = positions[:, :2]
    traj_min = trajectory.min(axis=0) - ROUTE_CANDIDATE_BBOX_MARGIN
    traj_max = trajectory.max(axis=0) + ROUTE_CANDIDATE_BBOX_MARGIN
    lane_indices = np.where(
        (route_cache["lane_bbox_maxs"][:, 0] >= traj_min[0])
        & (route_cache["lane_bbox_mins"][:, 0] <= traj_max[0])
        & (route_cache["lane_bbox_maxs"][:, 1] >= traj_min[1])
        & (route_cache["lane_bbox_mins"][:, 1] <= traj_max[1]),
    )[0]
    polylines = route_cache["lane_polylines"][lane_indices]
    lane_ids = route_cache["lane_id_array"][lane_indices]

    distances, closest, closest_t = _points_to_polylines_distance(
        trajectory, polylines, route_cache["lane_lengths"][lane_indices]
    )
    lane_axis = np.arange(len(lane_indices))[np.newaxis, :]
    seg_dirs = polylines[lane_axis, np.minimum(closest + 1, polylines.shape[1] - 1)] - polylines[lane_axis, closest]
    lane_dirs = seg_dirs / (np.linalg.norm(seg_dirs, axis=2, keepdims=True) + 1e-6)
    agent_dirs = np.stack([np.cos(headings), np.sin(headings)], axis=1)
    alignments = np.sum(lane_dirs * agent_dirs[:, np.newaxis, :], axis=2)
    valid_mask = (
        (distances <= LANE_WIDTH_THRESHOLD)
        & (alignments > ALIGNMENT_THRESHOLD)
        & _elevation_ok(route_cache["lane_polylines_xyz"], lane_indices, closest, closest_t, positions[:, 2:3])
    )
    projected_s = (
        route_cache["lane_cum_lengths"][lane_indices][lane_axis, closest]
        + closest_t * route_cache["lane_segment_lengths"][lane_indices][lane_axis, closest]
    )

    observations = []
    for point_idx in range(len(trajectory)):
        ranked = sorted(
            np.where(valid_mask[point_idx])[0],
            key=lambda idx: (distances[point_idx, idx], -alignments[point_idx, idx], int(lane_ids[idx])),
        )[:MAX_CANDIDATES_PER_POINT]
        if ranked:
            observations.append(
                [
                    (point_idx, int(lane_ids[idx]), float(distances[point_idx, idx]), float(projected_s[point_idx, idx]))
                    for idx in ranked
                ],
            )
    return observations


def _select_candidate_path(
    observations: list[list[_Candidate]],
    total_points: int,
    route_cache: _RouteCache,
) -> list[_Candidate]:
    # Full-history backward scan: O(T^2 * C^2) over observations x candidates. Kept exact
    # because SKIPPED_POINT_COST dominates long gaps; windowing would need real-data parity checks.
    costs: list[list[float]] = []
    backrefs: list[list[tuple[int, int] | None]] = []

    for obs_idx, candidates in enumerate(observations):
        obs_costs: list[float] = []
        obs_backrefs: list[tuple[int, int] | None] = []

        for point_idx, lane_id, distance, s in candidates:
            best_cost = point_idx * SKIPPED_POINT_COST + distance
            best_backref: tuple[int, int] | None = None

            for prev_idx in range(obs_idx):
                skipped_cost = (point_idx - observations[prev_idx][0][0] - 1) * SKIPPED_POINT_COST

                for prev_cand_idx, (_, prev_lane_id, _, prev_s) in enumerate(observations[prev_idx]):
                    if prev_lane_id == lane_id:
                        if s + BACKWARD_PROGRESS_TOLERANCE < prev_s:
                            continue
                        transition = skipped_cost
                    else:
                        path = _find_shortest_lane_path(prev_lane_id, lane_id, route_cache)
                        if not path:
                            continue
                        transition = skipped_cost + (len(path) - 1) * HOP_COST + LANE_CHANGE_COST

                    total_cost = costs[prev_idx][prev_cand_idx] + transition + distance
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_backref = (prev_idx, prev_cand_idx)

            obs_costs.append(best_cost)
            obs_backrefs.append(best_backref)

        costs.append(obs_costs)
        backrefs.append(obs_backrefs)

    obs_idx, cand_idx = min(
        ((obs_idx, cand_idx) for obs_idx, candidates in enumerate(observations) for cand_idx in range(len(candidates))),
        key=lambda state: costs[state[0]][state[1]]
        + (total_points - observations[state[0]][0][0] - 1) * SKIPPED_POINT_COST,
    )
    path = [observations[obs_idx][cand_idx]]
    while (backref := backrefs[obs_idx][cand_idx]) is not None:
        obs_idx, cand_idx = backref
        path.append(observations[obs_idx][cand_idx])
    return path[::-1]


def _find_shortest_lane_path(start_lane_id: int, end_lane_id: int, route_cache: _RouteCache) -> tuple[int, ...]:
    if start_lane_id == end_lane_id:
        return (start_lane_id,)

    cache_key = (start_lane_id, end_lane_id)
    if cache_key in route_cache["path_cache"]:
        return route_cache["path_cache"][cache_key]

    lane_graph = route_cache["lane_graph"]
    queue: deque[tuple[int, tuple[int, ...]]] = deque([(start_lane_id, (start_lane_id,))])
    seen = {start_lane_id}

    while queue:
        lane_id, path = queue.popleft()
        if len(path) - 1 >= MAX_TRANSITION_HOPS:
            continue

        for exit_lane_id in lane_graph.get(lane_id, ()):
            next_path = (*path, exit_lane_id)
            if exit_lane_id == end_lane_id:
                route_cache["path_cache"][cache_key] = next_path
                return next_path
            if exit_lane_id in seen:
                continue
            seen.add(exit_lane_id)
            queue.append((exit_lane_id, next_path))

    route_cache["path_cache"][cache_key] = ()
    return ()


def _extend_route_to_dead_end(route: list[int], route_cache: _RouteCache) -> list[int]:
    """Greedily follow the straightest unvisited exit until none remain."""
    lane_id_to_idx = route_cache["lane_id_to_idx"]
    visited = set(route)

    while exit_lane_ids := [lane_id for lane_id in route_cache["lane_graph"][route[-1]] if lane_id not in visited]:
        current_dir = route_cache["lane_tail_dirs"][lane_id_to_idx[route[-1]]]
        next_lane_id = max(
            exit_lane_ids,
            key=lambda lane_id: (
                float(np.dot(current_dir, route_cache["lane_head_dirs"][lane_id_to_idx[lane_id]])),
                -int(lane_id),
            ),
        )
        route.append(next_lane_id)
        visited.add(next_lane_id)

    return route


def _get_lane_endpoint_direction(polyline: np.ndarray, from_start: bool) -> np.ndarray:
    if len(polyline) < 2:
        return np.zeros(2, dtype=np.float64)

    segment_iter = itertools.pairwise(polyline)
    segments = segment_iter if from_start else reversed(tuple(segment_iter))

    for start, end in segments:
        direction = end - start
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            return direction / norm

    return np.zeros(2, dtype=np.float64)


def _is_offroad(position: np.ndarray, heading: float, length: float, width: float, route_cache: _RouteCache) -> bool:
    """True if no elevation-matched lane is within reach, or the agent box touches a same-level road edge."""
    xy, z = position[:2], position[2]
    min_distances, closest_indices, closest_t = _points_to_polylines_distance(
        xy.reshape(1, 2),
        route_cache["lane_polylines"],
        route_cache["lane_lengths"],
    )
    elevation_ok = _elevation_ok(
        route_cache["lane_polylines_xyz"],
        np.arange(len(route_cache["lane_id_array"])),
        closest_indices,
        closest_t,
        z,
    )
    if not np.any(elevation_ok) or np.min(min_distances[elevation_ok]) > OFFROAD_DISTANCE_THRESHOLD:
        return True

    cos_h, sin_h = np.cos(heading), np.sin(heading)
    half_len, half_w = length / 2, width / 2
    local_corners = np.array([[half_len, -half_w], [half_len, half_w], [-half_len, half_w], [-half_len, -half_w]])
    rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners = local_corners @ rotation.T + xy

    agent_poly = shapely_geom.Polygon(corners)
    bbox_min = corners.min(axis=0) - max(half_len, half_w)
    bbox_max = corners.max(axis=0) + max(half_len, half_w)

    for polyline, edge_min, edge_max in route_cache["road_edges"]:
        if edge_max[0] < bbox_min[0] or edge_min[0] > bbox_max[0]:
            continue
        if edge_max[1] < bbox_min[1] or edge_min[1] > bbox_max[1]:
            continue
        edge_line = shapely_geom.LineString(polyline[:, :2])
        if not agent_poly.intersects(edge_line):
            continue
        station = edge_line.project(shapely_geom.Point(xy))
        edge_z = np.interp(station, arc_length(polyline[:, :2]), polyline[:, 2])
        if abs(z - edge_z) <= ELEVATION_THRESHOLD:
            return True

    return False


def _elevation_ok(
    lane_polylines_xyz: np.ndarray,
    lane_indices: np.ndarray,
    closest_indices: np.ndarray,
    closest_t: np.ndarray,
    z: np.ndarray | float,
) -> np.ndarray:
    rows = lane_indices[np.newaxis, :]
    start_z = lane_polylines_xyz[rows, closest_indices, 2]
    end_z = lane_polylines_xyz[rows, closest_indices + 1, 2]
    return np.abs(z - (start_z + closest_t * (end_z - start_z))) <= ELEVATION_THRESHOLD


def _is_static(agent_data: schema.Track, dt: float) -> bool:
    valid = np.asarray(agent_data.valid, dtype=bool)
    trajectory = agent_data.position[:, :2][valid]
    # Static agents are frozen in self-play. A track shorter than the motion window can't show 1.5 m of travel even
    # at speed (a car passing by for 1 s would freeze mid-lane), so judge it by its velocity at the same 1 m/s rate.
    if len(trajectory) * dt < PARKED_MOTION_WINDOW_SECONDS:
        speed = np.linalg.norm(agent_data.velocity[valid], axis=1)
        return len(speed) == 0 or float(np.median(speed)) <= PARKED_MOTION_THRESHOLD / PARKED_MOTION_WINDOW_SECONDS
    smoothed = _median_smooth(trajectory, SMOOTH_WINDOW)
    if np.linalg.norm(np.ptp(smoothed, axis=0)) <= PARKED_MOTION_THRESHOLD:
        return True
    if _peak_motion(smoothed, dt) > PARKED_MOTION_THRESHOLD:
        return False

    path_length = polyline_length(smoothed)
    if path_length == 0:
        return True
    net_displacement = float(np.linalg.norm(smoothed[-1] - smoothed[0]))
    return net_displacement / path_length < PARKED_DIRECTIONAL_CONSISTENCY


def _median_smooth(trajectory: np.ndarray, window: int) -> np.ndarray:
    n = len(trajectory)
    w = min(window, n if n % 2 else n - 1)
    if w < 3:
        return trajectory
    half = w // 2
    idx = np.clip(np.arange(n)[:, None] + np.arange(-half, half + 1)[None, :], 0, n - 1)
    return np.median(trajectory[idx], axis=1)


def _peak_motion(smoothed: np.ndarray, dt: float) -> float:
    window = max(1, round(PARKED_MOTION_WINDOW_SECONDS / dt))
    if len(smoothed) <= window:
        return float(np.linalg.norm(smoothed[-1] - smoothed[0]))
    return float(np.max(np.linalg.norm(smoothed[window:] - smoothed[:-window], axis=1)))


def _points_to_polylines_distance(
    points: np.ndarray,
    polylines: np.ndarray,
    polyline_lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (min_distances, closest_indices, closest_t), each shaped (n_points, n_lanes)."""
    n_points = len(points)
    n_lanes = len(polylines)
    max_segments = polylines.shape[1] - 1 if n_lanes > 0 else 0

    if n_points == 0 or n_lanes == 0 or max_segments <= 0:
        return (
            np.zeros((n_points, n_lanes), dtype=np.float64),
            np.zeros((n_points, n_lanes), dtype=np.int64),
            np.zeros((n_points, n_lanes), dtype=np.float64),
        )

    seg_starts = polylines[:, :-1, :]
    seg_ends = polylines[:, 1:, :]

    polyline_lengths = np.asarray(polyline_lengths, dtype=np.int64)
    seg_counts = np.clip(polyline_lengths - 1, 0, max_segments)
    valid_segs = np.arange(max_segments)[np.newaxis, :] < seg_counts[:, np.newaxis]

    seg_vecs = seg_ends - seg_starts
    seg_lens_sq = np.einsum("ijk,ijk->ij", seg_vecs, seg_vecs)
    valid_segs = valid_segs & (seg_lens_sq > 1e-10)
    seg_lens_sq_safe = seg_lens_sq + 1e-10

    points_bc = points.reshape(n_points, 1, 1, 2)
    point_to_start = points_bc - seg_starts.reshape(1, n_lanes, max_segments, 2)
    t = np.einsum("ijkl,jkl->ijk", point_to_start, seg_vecs) / seg_lens_sq_safe.reshape(1, n_lanes, max_segments)
    t = np.clip(t, 0.0, 1.0)
    closest_on_seg = seg_starts.reshape(1, n_lanes, max_segments, 2) + t[..., np.newaxis] * seg_vecs.reshape(
        1,
        n_lanes,
        max_segments,
        2,
    )
    diff = points_bc - closest_on_seg
    distances_sq = np.einsum("ijkl,ijkl->ijk", diff, diff)
    distances_sq = np.where(valid_segs.reshape(1, n_lanes, max_segments), distances_sq, np.inf)

    closest_indices = np.argmin(distances_sq, axis=2).astype(np.int64)
    min_distances = np.sqrt(np.min(distances_sq, axis=2))

    lane_axis = np.arange(n_lanes)[np.newaxis, :]
    closest_t = t[np.arange(n_points)[:, np.newaxis], lane_axis, closest_indices]
    return min_distances, closest_indices, closest_t


