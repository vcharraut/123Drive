"""Compute per-agent lane routes from observed trajectories.

Cache lane geometry and connectivity once per scenario. Match valid, on-road vehicles to
lanes with dynamic programming, then extend their routes beyond the observed trajectory.
"""

import itertools
from collections import deque
from typing import Any, cast

import numpy as np
import shapely
from shapely import geometry as shapely_geom

from bin_factory import puffer_types, schema
from bin_factory.conversion.geometry import arc_length, polyline_length
from bin_factory.log_context import log


LANE_WIDTH_THRESHOLD = 7.0  # meters, maximum point-to-lane distance
ALIGNMENT_THRESHOLD = 0.3  # minimum dot product of agent and lane heading vectors
ROUTE_CANDIDATE_BBOX_MARGIN = 12.0  # meters, bounding-box expansion for nearby lanes
MAX_CANDIDATES_PER_POINT = 7
MAX_TRANSITION_HOPS = 3
BACKWARD_PROGRESS_TOLERANCE = 2.0  # meters, allowed backward projection on one lane
OFFROAD_DISTANCE_THRESHOLD = 5.0  # meters, maximum lane distance at an eligible start timestep
MIN_ROUTE_REMAINING = 15.0  # meters of route ahead at an eligible start timestep, above largest goal radius
MAX_INIT_TIMESTEP = 10  # latest timestep checked for route eligibility, covering t=0 through WOD's t=10 init step
ELEVATION_THRESHOLD = 2.0  # meters, maximum elevation difference for lane matches
SMOOTH_WINDOW = 7  # frames, odd median-filter window for perception spikes
PARKED_MOTION_THRESHOLD = 1.5
PARKED_MOTION_WINDOW_SECONDS = 1.5
PARKED_DIRECTIONAL_CONSISTENCY = 0.8
MAX_JUMP_SPEED = 50.0  # m/s between consecutive valid frames; faster log jumps shift goals
SKIPPED_POINT_COST = 50.0
HOP_COST = 1.0
LANE_CHANGE_COST = 6.0

_RouteCache = dict[str, Any]
_Candidate = tuple[int, int, float, float]  # (point_idx, lane_id, distance, s)


def process_agent_routes(scenario: schema.PufferScenario) -> None:
    """Set each agent's ``route``. A nonempty route marks the agent as eligible for control.

    Project vehicle trajectories onto lane centerlines, choose a connected sequence with
    dynamic programming, then extend it to a dead end. Non-ego vehicles must be valid,
    on-road, and have at least ``MIN_ROUTE_REMAINING`` ahead at their first eligible
    timestep in ``0..MAX_INIT_TIMESTEP``, with no log jump above ``MAX_JUMP_SPEED``. Ego bypasses
    the parked and jump filters and must be valid,
    on-road, and have enough route ahead at timestep 0.
    """
    scenario_length = scenario.metadata.scenario_length
    max_check = min(MAX_INIT_TIMESTEP + 1, scenario_length)
    route_cache = build_route_cache(scenario.map)
    dt = scenario.metadata.dt
    start_frames = np.zeros(len(scenario.agents), dtype=np.int64)

    for agent_idx, (agent_id, agent_data) in enumerate(scenario.agents.items()):
        is_vehicle = agent_data.type == puffer_types.AgentType.VEHICLE
        is_ego = is_vehicle and agent_id == 0
        if not is_vehicle or (not is_ego and (_is_static(agent_data, dt) or _has_jump(agent_data, dt))):
            agent_data.route = []
            continue

        if is_ego:
            route = _compute_track_route(agent_id, agent_data, route_cache)
            has_road_ahead = (
                bool(route)
                and _is_onroad_at(agent_data, 0, route_cache)
                and _route_remaining(agent_data.position[0], route, route_cache) >= MIN_ROUTE_REMAINING
            )
            if not has_road_ahead:
                raise ValueError(
                    f"Ego vehicle (agent 0) has no route, under {MIN_ROUTE_REMAINING:g} m of route ahead or an "
                    f"off-road start in scenario {scenario.metadata.id}"
                )
            agent_data.route = route
            continue

        onroad_t = next((t for t in range(max_check) if _is_onroad_at(agent_data, t, route_cache)), None)
        if onroad_t is None:
            agent_data.route = []
            continue

        route = _compute_track_route(agent_id, agent_data, route_cache)
        init_t = (
            next(
                (
                    t
                    for t in range(onroad_t, max_check)
                    if (t == onroad_t or _is_onroad_at(agent_data, t, route_cache))
                    and _route_remaining(agent_data.position[t], route, route_cache) >= MIN_ROUTE_REMAINING
                ),
                None,
            )
            if route
            else None
        )
        if init_t is not None:
            agent_data.route = route
            start_frames[agent_idx] = init_t
        else:
            agent_data.route = []

    _clear_conflicting_routes(scenario.agents, start_frames)


def _is_onroad_at(agent: schema.Track, t: int, route_cache: _RouteCache) -> bool:
    return bool(agent.valid[t]) and not _is_offroad(
        agent.position[t], agent.heading[t], agent.length[t], agent.width[t], route_cache
    )


def _compute_track_route(agent_id: int, agent: schema.Track, route_cache: _RouteCache) -> list[int]:
    return compute_agent_route(
        agent_id=agent_id,
        positions=agent.position,
        headings=agent.heading,
        valid=agent.valid,
        lengths=agent.length,
        widths=agent.width,
        is_ego=True,
        route_cache=route_cache,
    )


def _clear_conflicting_routes(agents: dict[int, schema.Track], start_frames: np.ndarray) -> None:
    """Clear non-ego routes when logged boxes overlap another agent at the agent's start frame
    or an agent without a route later.

    This avoids collisions already present in the log. Clear routes but keep trajectories
    and validity for log replay and WOSAC. An ego overlap at frame 0 raises an error.
    """
    tracks = list(agents.values())
    was_controllable = np.array([bool(track.route) for track in tracks], dtype=bool)
    ctrl_idx = np.flatnonzero(was_controllable)
    if len(ctrl_idx) == 0 or not tracks or tracks[0].position.shape[0] == 0:
        return

    is_ego = np.array([aid == 0 for aid in agents], dtype=bool)
    pos = np.stack([track.position[:, :2] for track in tracks])
    valid = np.stack([track.valid > 0 for track in tracks])
    lengths = np.stack([track.length for track in tracks])
    widths = np.stack([track.width for track in tracks])
    radii = 0.5 * np.hypot(lengths, widths)

    frame_idx = np.arange(pos.shape[1])[None, None, :]
    in_window = np.where(is_ego[ctrl_idx, None, None], frame_idx == 0, frame_idx >= start_frames[ctrl_idx, None, None])
    both_valid = valid[ctrl_idx, None, :] & valid[None, :, :] & in_window
    both_valid[np.arange(len(ctrl_idx)), ctrl_idx, :] = False

    diff = pos[ctrl_idx, None, :, :] - pos[None, :, :, :]
    dist_sq = diff[..., 0] ** 2 + diff[..., 1] ** 2
    r_sum = radii[ctrl_idx, None, :] + radii[None, :, :] + 1e-6
    c_idx, other_idx, frame = np.nonzero(both_valid & (dist_sq <= r_sum * r_sum))
    if len(c_idx) == 0:
        return

    agent_idx = ctrl_idx[c_idx]
    headings = np.stack([track.heading for track in tracks])
    corners_a = _pair_box_corners(
        pos[agent_idx, frame],
        headings[agent_idx, frame],
        lengths[agent_idx, frame],
        widths[agent_idx, frame],
    )
    corners_b = _pair_box_corners(
        pos[other_idx, frame],
        headings[other_idx, frame],
        lengths[other_idx, frame],
        widths[other_idx, frame],
    )
    keep = shapely.intersects(shapely.polygons(corners_a), shapely.polygons(corners_b))
    if not np.any(keep):
        return

    agent, other, frame = agent_idx[keep], other_idx[keep], frame[keep]
    if (is_ego[agent] & was_controllable[agent] & (frame == 0)).any():
        raise ValueError("Ego vehicle (agent 0) overlaps another agent at the start frame")
    controlled = was_controllable.copy()
    # A vehicle losing its route becomes background, so clear routes of vehicles it overlaps later too.
    while (hit := controlled[agent] & ~is_ego[agent] & ((frame == start_frames[agent]) | ~controlled[other])).any():
        controlled[agent[hit]] = False
    for k in np.flatnonzero(was_controllable & ~controlled):
        tracks[k].route = []


def _route_remaining(position: np.ndarray, route: list[int], route_cache: _RouteCache) -> float:
    """Return metres ahead of ``position``, adding the rest of the nearest route lane and all later lanes."""
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


def _pair_box_corners(position: np.ndarray, heading: np.ndarray, length: np.ndarray, width: np.ndarray) -> np.ndarray:
    """Return oriented box corners with shape (K, 4, 2) for K agent-frame pairs."""
    half = np.stack([length, width], -1) / 2
    local = half[:, None] * np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])
    cos, sin = np.cos(heading)[:, None], np.sin(heading)[:, None]
    rotated = np.stack([cos * local[..., 0] - sin * local[..., 1], sin * local[..., 0] + cos * local[..., 1]], -1)
    return rotated + position[:, None]


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
    lane_polylines = [cast("np.ndarray", lanes[lane_id].polyline) for lane_id in lane_ids]  # filtered above
    lane_lengths = np.array([len(polyline) for polyline in lane_polylines], dtype=np.int64)
    max_points = int(lane_lengths.max(initial=0))

    lane_polylines_xyz = np.zeros((n_lanes, max_points, 3), dtype=np.float64)
    lane_bbox_mins = np.zeros((n_lanes, 2), dtype=np.float64)
    lane_bbox_maxs = np.zeros((n_lanes, 2), dtype=np.float64)
    lane_head_dirs = np.zeros((n_lanes, 2), dtype=np.float64)
    lane_tail_dirs = np.zeros((n_lanes, 2), dtype=np.float64)
    lane_segment_lengths = np.zeros((n_lanes, max(0, max_points - 1)), dtype=np.float64)
    lane_cum_lengths = np.zeros((n_lanes, max_points), dtype=np.float64)

    for idx, lane_polyline in enumerate(lane_polylines):
        n_points = lane_lengths[idx]
        lane_polylines_xyz[idx, :n_points] = lane_polyline[:, :3]
        polyline = lane_polylines_xyz[idx, :n_points, :2]
        lane_bbox_mins[idx] = polyline.min(axis=0)
        lane_bbox_maxs[idx] = polyline.max(axis=0)
        lane_head_dirs[idx] = _get_lane_endpoint_direction(polyline, from_start=True)
        lane_tail_dirs[idx] = _get_lane_endpoint_direction(polyline, from_start=False)
        seg_lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
        lane_segment_lengths[idx, : len(seg_lengths)] = seg_lengths
        lane_cum_lengths[idx, :n_points] = np.concatenate(([0.0], np.cumsum(seg_lengths)))

    lane_id_to_idx = {lane_id: idx for idx, lane_id in enumerate(lane_ids)}
    lane_graph = {
        lane_id: tuple(exit_id for exit_id in lanes[lane_id].exit_lanes if exit_id in lane_id_to_idx)
        for lane_id in lane_ids
    }
    lane_rev_graph: dict[int, list[int]] = {lane_id: [] for lane_id in lane_ids}
    for lane_id, exits in lane_graph.items():
        for exit_id in exits:
            lane_rev_graph[exit_id].append(lane_id)

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
        "lane_graph": lane_graph,
        "lane_rev_graph": lane_rev_graph,
        "lane_bbox_mins": lane_bbox_mins,
        "lane_bbox_maxs": lane_bbox_maxs,
        "lane_head_dirs": lane_head_dirs,
        "lane_tail_dirs": lane_tail_dirs,
        "lane_segment_lengths": lane_segment_lengths,
        "lane_cum_lengths": lane_cum_lengths,
        "path_cache": {},
        "in_cost_cache": {},
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
) -> list[int]:
    """Return the lane route for one agent, or ``[]``.

    Ego bypasses the on-road gate. Non-ego agents must be valid and on-road at some
    timestep in ``0..MAX_INIT_TIMESTEP``.
    """
    valid = np.asarray(valid, dtype=bool)
    if (
        not valid.any()
        or len(route_cache["lane_id_array"]) == 0
        or (
            not is_ego
            and not any(
                valid[t] and not _is_offroad(positions[t], headings[t], lengths[t], widths[t], route_cache)
                for t in range(min(MAX_INIT_TIMESTEP + 1, len(valid)))
            )
        )
    ):
        log.debug("agent=%d: skipping route computation (insufficient valid data or offroad start)", agent_id)
        return []

    observations = _build_point_observations(positions[valid], headings[valid], route_cache)
    if not observations:
        log.debug("agent=%d: no GT-supported lane candidates found", agent_id)
        return []

    candidate_path = _select_candidate_path(observations, int(valid.sum()), route_cache)
    gt_route = [candidate_path[0][1]]
    for prev_candidate, next_candidate in itertools.pairwise(candidate_path):
        gt_route.extend(_find_shortest_lane_path(prev_candidate[1], next_candidate[1], route_cache)[1:])

    return _extend_route_to_dead_end(gt_route, route_cache)


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
                    (
                        point_idx,
                        int(lane_ids[idx]),
                        float(distances[point_idx, idx]),
                        float(projected_s[point_idx, idx]),
                    )
                    for idx in ranked
                ],
            )
    return observations


def _select_candidate_path(
    observations: list[list[_Candidate]],
    total_points: int,
    route_cache: _RouteCache,
) -> list[_Candidate]:
    # Scan recent history first. Nonnegative costs and prefix-min bounds let us stop
    # once no earlier observation can beat the best cost, without a fixed history limit.
    costs: list[list[float]] = []
    backrefs: list[list[tuple[int, int] | None]] = []
    adj_costs: list[float] = []
    prefix_min_adj: list[float] = []
    lane_last_obs: dict[int, int] = {}

    for obs_idx, candidates in enumerate(observations):
        obs_costs: list[float] = []
        obs_backrefs: list[tuple[int, int] | None] = []

        for point_idx, lane_id, distance, s in candidates:
            best_cost = point_idx * SKIPPED_POINT_COST + distance
            best_backref: tuple[int, int] | None = None
            in_costs = _get_incoming_transition_costs(lane_id, route_cache)
            start_prev = max((lane_last_obs.get(pid, -1) for pid in in_costs), default=-1)
            base_skip = (point_idx - 1) * SKIPPED_POINT_COST + distance

            for prev_idx in range(start_prev, -1, -1):
                if prefix_min_adj[prev_idx] + base_skip - 1e-9 > best_cost:
                    break
                if adj_costs[prev_idx] + base_skip - 1e-9 > best_cost:
                    continue
                skipped_cost = (point_idx - observations[prev_idx][0][0] - 1) * SKIPPED_POINT_COST

                for prev_cand_idx, (_, prev_lane_id, _, prev_s) in enumerate(observations[prev_idx]):
                    hop_cost = in_costs.get(prev_lane_id)
                    if hop_cost is None:
                        continue
                    if prev_lane_id == lane_id and s + BACKWARD_PROGRESS_TOLERANCE < prev_s:
                        continue

                    total_cost = costs[prev_idx][prev_cand_idx] + skipped_cost + hop_cost + distance
                    # Preserve the forward scan's earliest predecessor on ties.
                    if total_cost < best_cost or (
                        total_cost == best_cost
                        and best_backref is not None
                        and (prev_idx, prev_cand_idx) < best_backref
                    ):
                        best_cost = total_cost
                        best_backref = (prev_idx, prev_cand_idx)

            obs_costs.append(best_cost)
            obs_backrefs.append(best_backref)

        costs.append(obs_costs)
        backrefs.append(obs_backrefs)
        for _, lane_id, _, _ in candidates:
            lane_last_obs[lane_id] = obs_idx
        adj = min(obs_costs) - candidates[0][0] * SKIPPED_POINT_COST
        adj_costs.append(adj)
        prefix_min_adj.append(adj if not prefix_min_adj or adj < prefix_min_adj[-1] else prefix_min_adj[-1])

    obs_idx, cand_idx = min(
        ((obs_idx, cand_idx) for obs_idx, candidates in enumerate(observations) for cand_idx in range(len(candidates))),
        key=lambda state: (
            costs[state[0]][state[1]] + (total_points - observations[state[0]][0][0] - 1) * SKIPPED_POINT_COST
        ),
    )
    path = [observations[obs_idx][cand_idx]]
    while (backref := backrefs[obs_idx][cand_idx]) is not None:
        obs_idx, cand_idx = backref
        path.append(observations[obs_idx][cand_idx])
    return path[::-1]


def _get_incoming_transition_costs(end_lane_id: int, route_cache: _RouteCache) -> dict[int, float]:
    cached = route_cache["in_cost_cache"].get(end_lane_id)
    if cached is not None:
        return cached
    rev_graph = route_cache["lane_rev_graph"]
    costs: dict[int, float] = {end_lane_id: 0.0}
    frontier = [end_lane_id]
    for hops in range(1, MAX_TRANSITION_HOPS + 1):
        penalty = hops * HOP_COST + LANE_CHANGE_COST
        next_frontier = []
        for curr in frontier:
            for prev_id in rev_graph.get(curr, ()):
                if prev_id not in costs:
                    costs[prev_id] = penalty
                    next_frontier.append(prev_id)
        if not next_frontier:
            break
        frontier = next_frontier
    route_cache["in_cost_cache"][end_lane_id] = costs
    return costs


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
    """Return True when no elevation-matched lane is within reach or the agent box touches a same-level road edge."""
    xy, z = position[:2], position[2]
    near_lanes = np.flatnonzero(
        (route_cache["lane_bbox_maxs"][:, 0] >= xy[0] - OFFROAD_DISTANCE_THRESHOLD)
        & (route_cache["lane_bbox_mins"][:, 0] <= xy[0] + OFFROAD_DISTANCE_THRESHOLD)
        & (route_cache["lane_bbox_maxs"][:, 1] >= xy[1] - OFFROAD_DISTANCE_THRESHOLD)
        & (route_cache["lane_bbox_mins"][:, 1] <= xy[1] + OFFROAD_DISTANCE_THRESHOLD)
    )
    if len(near_lanes) == 0:
        return True
    min_distances, closest_indices, closest_t = _points_to_polylines_distance(
        xy.reshape(1, 2),
        route_cache["lane_polylines"][near_lanes],
        route_cache["lane_lengths"][near_lanes],
    )
    elevation_ok = _elevation_ok(
        route_cache["lane_polylines_xyz"],
        near_lanes,
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
    # Static agents are frozen in self-play. Tracks shorter than the motion window cannot show 1.5 m of travel even
    # at speed, so judge them by median speed at the same 1 m/s rate.
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


def _has_jump(agent_data: schema.Track, dt: float) -> bool:
    valid = np.asarray(agent_data.valid, dtype=bool)
    step = np.linalg.norm(np.diff(agent_data.position[:, :2], axis=0), axis=1)[valid[1:] & valid[:-1]]
    return bool((step > MAX_JUMP_SPEED * dt).any())


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
    """Return min_distances, closest_indices, and closest_t, each shaped (n_points, n_lanes)."""
    n_points = len(points)
    n_lanes = len(polylines)
    polyline_lengths = np.asarray(polyline_lengths, dtype=np.int64)
    max_pts = min(polylines.shape[1], int(polyline_lengths.max(initial=0))) if n_lanes > 0 else 0
    max_segments = max_pts - 1

    if n_points == 0 or n_lanes == 0 or max_segments <= 0:
        return (
            np.zeros((n_points, n_lanes), dtype=np.float64),
            np.zeros((n_points, n_lanes), dtype=np.int64),
            np.zeros((n_points, n_lanes), dtype=np.float64),
        )

    polylines = polylines[:, :max_pts, :]
    seg_starts = polylines[:, :-1, :]
    seg_ends = polylines[:, 1:, :]

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
