"""Compute per-agent lane routes from observed trajectories.

The route pipeline is graph-constrained:
1. Build a per-scenario lane cache once.
2. Keep current agent eligibility and offroad filtering.
3. Build per-point lane candidates from GT geometry.
4. Solve a best lane sequence with dynamic programming.
5. Extend beyond GT to the map dead-end.
"""

import itertools
import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from shapely import geometry as shapely_geom

from bin_factory import puffer_types
from bin_factory.log_context import log


LANE_WIDTH_THRESHOLD = 7.0  # meters — reject point-to-lane matches farther than this
ALIGNMENT_THRESHOLD = 0.3  # cos(heading) — minimum dot product for "same direction"
ROUTE_CANDIDATE_BBOX_MARGIN = 12.0  # meters — bbox expansion when filtering nearby lanes
MAX_CANDIDATES_PER_POINT = 7
MAX_TRANSITION_HOPS = 3
BACKWARD_PROGRESS_TOLERANCE = 2.0  # meters — allow small projection noise on same lane
OFFROAD_DISTANCE_THRESHOLD = 5.0  # meters — max lane distance for moving agents
STATIONARY_OFFROAD_DISTANCE_THRESHOLD = 0.3  # meters — max lane distance for stationary agents
MOVEMENT_THRESHOLD = 1.0  # meters — spatial extent below this = stationary (noise-robust)
SKIPPED_POINT_COST = 50.0
HOP_COST = 1.0
LANE_CHANGE_COST = 6.0


@dataclass(frozen=True)
class PointLaneCandidate:
    point_idx: int
    lane_id: int
    distance: float
    s: float


def process_agent_routes(scenario, min_route_valid_points=0.0, route_check_timestep=0) -> None:
    """Compute a lane route per agent and assign it back onto ``scenario``.

    Routes are graph-constrained lane sequences. For each vehicle agent, the ground-truth
    trajectory is projected onto lane centerlines and a best-matching lane sequence is
    solved via DP, then extended past the last GT lane toward a dead-end.

    Arguments:
        scenario: PufferScenario whose ``agents`` and ``map`` are read; routes are written
            back onto each Track (``route``, ``route_gt_len``, ``control_state``).
        min_route_valid_points: Minimum valid trajectory percentage from
            ``route_check_timestep`` onwards required to attempt route computation for a
            non-ego agent. Below this threshold the agent gets an empty route.
        route_check_timestep: Timestep at which the agent must be valid (and on-road) for
            non-ego routes. Ego (vehicle id 0) bypasses this gate; failure on ego raises.
    """
    scenario_length = scenario.metadata.scenario_length
    if scenario_length > 0 and route_check_timestep >= scenario_length:
        raise ValueError(
            f"route_check_timestep={route_check_timestep} is out of range for scenario length {scenario_length}",
        )
    route_check_horizon = scenario_length - route_check_timestep
    min_route_valid_count = math.ceil(route_check_horizon * min_route_valid_points / 100)

    lane_data = _extract_lane_centers(scenario.map)
    route_cache = build_route_cache(scenario.map, lane_data)

    for agent_id, agent_data in scenario.agents.items():
        is_ego = agent_data.type == puffer_types.AgentType.VEHICLE and agent_id == 0
        is_vehicle = agent_data.type == puffer_types.AgentType.VEHICLE

        if not is_vehicle:
            agent_data.route = []
            agent_data.route_gt_len = 0
            agent_data.control_state = _compute_control_state(agent_data)
            continue

        route, route_gt_len = compute_agent_route(
            agent_id=agent_id,
            positions=agent_data.position,
            headings=agent_data.heading,
            valid=agent_data.valid,
            lengths=agent_data.length,
            widths=agent_data.width,
            is_ego=is_ego,
            route_cache=route_cache,
            route_check_timestep=route_check_timestep,
            min_route_valid_count=min_route_valid_count,
        )
        if is_ego and not route:
            raise ValueError(f"Route computation failed for ego vehicle (agent 0) in scenario {scenario.metadata.id}")
        agent_data.route = route
        agent_data.route_gt_len = route_gt_len
        agent_data.control_state = _compute_control_state(agent_data)


def _extract_lane_centers(static_map_elements):
    lane_ids = []
    lane_polylines_list = []
    lane_lengths_list = []
    lane_metadata = {}
    max_points = 0

    for element_id, element_data in static_map_elements.items():
        element_type = element_data.type
        if element_type not in (puffer_types.LaneType.SURFACE_STREET, puffer_types.LaneType.FREEWAY):
            continue

        polyline = element_data.polyline
        if len(polyline) == 0:
            continue

        polyline_2d = polyline[:, :2] if polyline.shape[1] == 3 else polyline
        lane_ids.append(element_id)
        lane_polylines_list.append(polyline_2d)
        lane_lengths_list.append(len(polyline_2d))
        max_points = max(max_points, len(polyline_2d))
        lane_metadata[element_id] = {
            "entry_lanes": element_data.entry_lanes,
            "exit_lanes": element_data.exit_lanes,
        }

    if not lane_ids:
        return [], np.array([]), {}, np.array([])

    n_lanes = len(lane_ids)
    lane_polylines = np.zeros((n_lanes, max_points, 2), dtype=np.float64)
    lane_lengths = np.array(lane_lengths_list, dtype=np.int64)

    for idx, polyline_2d in enumerate(lane_polylines_list):
        lane_polylines[idx, : len(polyline_2d), :] = polyline_2d

    return lane_ids, lane_polylines, lane_metadata, lane_lengths


def build_route_cache(static_map_elements, lane_data):
    """Precompute lane geometry and connectivity shared by all agents."""
    lane_ids, lane_polylines, lane_metadata, lane_lengths = lane_data
    lane_ids = list(lane_ids)
    lane_id_array = np.asarray(lane_ids, dtype=np.int64)
    lane_lengths = np.asarray(lane_lengths, dtype=np.int64)
    lane_id_to_idx = {lane_id: idx for idx, lane_id in enumerate(lane_ids)}
    trimmed_polylines = tuple(lane_polylines[idx, : lane_lengths[idx], :].copy() for idx in range(len(lane_ids)))

    if trimmed_polylines:
        lane_bbox_mins = np.stack([polyline.min(axis=0) for polyline in trimmed_polylines])
        lane_bbox_maxs = np.stack([polyline.max(axis=0) for polyline in trimmed_polylines])
        lane_head_dirs = np.stack(
            [_get_lane_endpoint_direction(polyline, from_start=True) for polyline in trimmed_polylines],
        )
        lane_tail_dirs = np.stack(
            [_get_lane_endpoint_direction(polyline, from_start=False) for polyline in trimmed_polylines],
        )
        lane_segment_lengths, lane_cum_lengths = _compute_lane_length_tables(trimmed_polylines, lane_lengths.max())
    else:
        lane_bbox_mins = np.zeros((0, 2), dtype=np.float64)
        lane_bbox_maxs = np.zeros((0, 2), dtype=np.float64)
        lane_head_dirs = np.zeros((0, 2), dtype=np.float64)
        lane_tail_dirs = np.zeros((0, 2), dtype=np.float64)
        lane_segment_lengths = np.zeros((0, 0), dtype=np.float64)
        lane_cum_lengths = np.zeros((0, 0), dtype=np.float64)

    lane_graph = {
        lane_id: tuple(
            exit_id for exit_id in lane_metadata.get(lane_id, {}).get("exit_lanes", []) if exit_id in lane_id_to_idx
        )
        for lane_id in lane_ids
    }

    road_edges = tuple(
        edge for element in static_map_elements.values() if (edge := _extract_road_edge(element)) is not None
    )

    return {
        "lane_id_array": lane_id_array,
        "lane_polylines": lane_polylines,
        "lane_lengths": lane_lengths,
        "lane_id_to_idx": lane_id_to_idx,
        "trimmed_polylines": trimmed_polylines,
        "lane_graph": lane_graph,
        "lane_bbox_mins": lane_bbox_mins,
        "lane_bbox_maxs": lane_bbox_maxs,
        "lane_head_dirs": lane_head_dirs,
        "lane_tail_dirs": lane_tail_dirs,
        "lane_segment_lengths": lane_segment_lengths,
        "lane_cum_lengths": lane_cum_lengths,
        "path_cache": {},
        "road_edges": road_edges,
    }


def compute_agent_route(
    agent_id,
    positions,
    headings,
    valid,
    lengths,
    widths,
    is_ego,
    route_cache,
    route_check_timestep=0,
    min_route_valid_count=0,
):
    """Return the best lane sequence for one agent, or an empty list."""
    positions_2d = positions[:, :2] if positions.shape[1] == 3 else positions
    valid = np.asarray(valid, dtype=bool)
    trajectory = positions_2d[valid]
    heading_valid = headings[valid]

    agent_context = {
        "positions_2d": positions_2d,
        "headings": headings,
        "valid": valid,
        "lengths": lengths,
        "widths": widths,
        "trajectory": trajectory,
        "heading_valid": heading_valid,
    }

    if not _can_compute_route(
        agent_context,
        route_cache,
        is_ego,
        route_check_timestep,
        min_route_valid_count,
    ):
        log.debug("agent=%d: skipping route computation (insufficient valid data or offroad start)", agent_id)
        return [], 0

    observations = _build_point_observations(agent_context, route_cache)
    if not observations:
        log.debug("agent=%d: no GT-supported lane candidates found", agent_id)
        return [], 0

    candidate_path = _select_candidate_path(observations, len(trajectory), route_cache)
    if not candidate_path:
        log.debug("agent=%d: no topologically valid lane path found", agent_id)
        return [], 0

    gt_route = _expand_candidate_path(candidate_path, route_cache)
    if not gt_route:
        log.debug("agent=%d: failed to reconstruct GT-supported route", agent_id)
        return [], 0

    route_gt_len = len(gt_route)
    route = _extend_route_to_dead_end(gt_route.copy(), route_cache)
    return route, route_gt_len


def _can_compute_route(agent_context, route_cache, is_ego, route_check_timestep=0, min_route_valid_count=0) -> bool:
    """Return True if a non-ego agent qualifies for route computation at ``route_check_timestep``.

    Ego always qualifies. A non-ego agent qualifies only if it has a trajectory, the map has
    routable lanes, it is valid and on-road at the check timestep, and it has enough valid
    samples from that timestep onwards.
    """
    if is_ego:
        return True

    if len(agent_context["trajectory"]) == 0 or len(route_cache["lane_id_array"]) == 0:
        return False

    if route_check_timestep >= len(agent_context["valid"]):
        return False

    if not agent_context["valid"][route_check_timestep]:
        return False

    if np.sum(agent_context["valid"][route_check_timestep:]) < min_route_valid_count:
        return False

    return not _is_offroad_at_timestep(agent_context, route_cache, route_check_timestep)


def _build_point_observations(agent_context, route_cache):
    trajectory = agent_context["trajectory"]
    if len(trajectory) == 0:
        return []

    candidate_lane_indices = _select_nearby_lane_indices(trajectory, route_cache)
    if len(candidate_lane_indices) == 0:
        return []

    candidate_polylines = route_cache["lane_polylines"][candidate_lane_indices]
    candidate_lengths = route_cache["lane_lengths"][candidate_lane_indices]
    candidate_lane_ids = route_cache["lane_id_array"][candidate_lane_indices]

    min_distances_all, closest_indices_all, closest_t_all = _points_to_polylines_distance(
        trajectory,
        candidate_polylines,
        candidate_lengths,
    )
    lane_directions_all = _get_lane_directions_at_indices_batch(candidate_polylines, closest_indices_all)
    agent_dirs = np.stack([np.cos(agent_context["heading_valid"]), np.sin(agent_context["heading_valid"])], axis=1)
    alignments_all = np.sum(lane_directions_all * agent_dirs[:, np.newaxis, :], axis=2)
    valid_mask = (min_distances_all <= LANE_WIDTH_THRESHOLD) & (alignments_all > ALIGNMENT_THRESHOLD)
    projected_s_all = _project_arc_lengths(closest_indices_all, closest_t_all, candidate_lane_indices, route_cache)

    observations = [
        {
            "point_idx": point_idx,
            "candidates": _select_point_candidates(
                point_idx,
                point_distances,
                point_alignments,
                point_s,
                point_mask,
                candidate_lane_ids,
            ),
        }
        for point_idx, (point_distances, point_alignments, point_s, point_mask) in enumerate(
            zip(min_distances_all, alignments_all, projected_s_all, valid_mask, strict=True),
        )
    ]
    return [observation for observation in observations if observation["candidates"]]


def _select_nearby_lane_indices(trajectory, route_cache):
    if len(route_cache["lane_bbox_mins"]) == 0:
        return np.array([], dtype=np.int64)

    traj_min = trajectory.min(axis=0) - ROUTE_CANDIDATE_BBOX_MARGIN
    traj_max = trajectory.max(axis=0) + ROUTE_CANDIDATE_BBOX_MARGIN
    overlaps = (
        (route_cache["lane_bbox_maxs"][:, 0] >= traj_min[0])
        & (route_cache["lane_bbox_mins"][:, 0] <= traj_max[0])
        & (route_cache["lane_bbox_maxs"][:, 1] >= traj_min[1])
        & (route_cache["lane_bbox_mins"][:, 1] <= traj_max[1])
    )
    return np.where(overlaps)[0]


def _select_point_candidates(point_idx, point_distances, point_alignments, point_s, point_mask, candidate_lane_ids):
    valid_indices = np.where(point_mask)[0]
    ranked = sorted(
        valid_indices,
        key=lambda idx: (
            point_distances[idx],
            -point_alignments[idx],
            int(candidate_lane_ids[idx]),
        ),
    )[:MAX_CANDIDATES_PER_POINT]
    return [
        PointLaneCandidate(
            point_idx=point_idx,
            lane_id=int(candidate_lane_ids[idx]),
            distance=float(point_distances[idx]),
            s=float(point_s[idx]),
        )
        for idx in ranked
    ]


def _project_arc_lengths(closest_indices_all, closest_t_all, candidate_lane_indices, route_cache):
    if len(candidate_lane_indices) == 0:
        return np.zeros((len(closest_indices_all), 0), dtype=np.float64)

    candidate_axis = np.arange(len(candidate_lane_indices))[np.newaxis, :]
    cum_lengths = route_cache["lane_cum_lengths"][candidate_lane_indices]
    seg_lengths = route_cache["lane_segment_lengths"][candidate_lane_indices]
    base_lengths = cum_lengths[candidate_axis, closest_indices_all]
    segment_lengths = seg_lengths[candidate_axis, closest_indices_all]
    return base_lengths + closest_t_all * segment_lengths


def _select_candidate_path(observations, total_points, route_cache):
    if not observations:
        return []

    # Full-history backward scan: O(T^2 * C^2) over observations x candidates. Kept exact
    # because SKIPPED_POINT_COST dominates long gaps; windowing would need real-data parity checks.
    costs = []
    backrefs = []

    for obs_idx, observation in enumerate(observations):
        obs_costs = [None] * len(observation["candidates"])
        obs_backrefs = [None] * len(observation["candidates"])

        for cand_idx, candidate in enumerate(observation["candidates"]):
            start_cost = _make_path_cost(candidate.point_idx, 0, 0, candidate.distance)
            best_cost = start_cost
            best_backref = None

            for prev_idx in range(obs_idx):
                skipped_points = candidate.point_idx - observations[prev_idx]["point_idx"] - 1
                if skipped_points < 0:
                    continue

                for prev_cand_idx, prev_candidate in enumerate(observations[prev_idx]["candidates"]):
                    prev_cost = costs[prev_idx][prev_cand_idx]
                    if prev_cost is None:
                        continue

                    transition = _transition_cost(prev_candidate, candidate, skipped_points, route_cache)
                    if transition is None:
                        continue

                    total_cost = _add_costs(prev_cost, transition, _make_path_cost(0, 0, 0, candidate.distance))
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_backref = (prev_idx, prev_cand_idx)

            obs_costs[cand_idx] = best_cost
            obs_backrefs[cand_idx] = best_backref

        costs.append(obs_costs)
        backrefs.append(obs_backrefs)

    best_final = None
    best_state = None
    for obs_idx, observation in enumerate(observations):
        trailing_unmatched = total_points - observation["point_idx"] - 1
        for cand_idx, cost in enumerate(costs[obs_idx]):
            final_cost = _add_costs(cost, _make_path_cost(trailing_unmatched, 0, 0, 0))
            if best_final is None or final_cost < best_final:
                best_final = final_cost
                best_state = (obs_idx, cand_idx)

    return _backtrack_candidate_path(best_state, observations, backrefs)


def _transition_cost(prev_candidate, next_candidate, skipped_points, route_cache):
    if prev_candidate.lane_id == next_candidate.lane_id:
        is_forward = next_candidate.s + BACKWARD_PROGRESS_TOLERANCE >= prev_candidate.s
        return _make_path_cost(skipped_points, 0, 0, 0) if is_forward else None

    path = _find_shortest_lane_path(prev_candidate.lane_id, next_candidate.lane_id, route_cache)
    if not path:
        return None

    hop_count = len(path) - 1
    return _make_path_cost(skipped_points, hop_count, 1, 0)


def _add_costs(*costs):
    return tuple(sum(parts) for parts in zip(*costs, strict=True))


def _make_path_cost(skipped_points, hop_count, lane_changes, total_distance):
    return (
        skipped_points * SKIPPED_POINT_COST + hop_count * HOP_COST + lane_changes * LANE_CHANGE_COST + total_distance,
        skipped_points,
        hop_count,
        lane_changes,
        total_distance,
    )


def _backtrack_candidate_path(best_state, observations, backrefs):
    if best_state is None:
        return []

    obs_idx, cand_idx = best_state
    path = []

    while obs_idx is not None:
        path.append(observations[obs_idx]["candidates"][cand_idx])
        backref = backrefs[obs_idx][cand_idx]
        if backref is None:
            break
        obs_idx, cand_idx = backref

    return list(reversed(path))


def _expand_candidate_path(candidate_path, route_cache):
    if not candidate_path:
        return []

    route = [candidate_path[0].lane_id]
    for prev_candidate, next_candidate in itertools.pairwise(candidate_path):
        path = _find_shortest_lane_path(prev_candidate.lane_id, next_candidate.lane_id, route_cache)
        if not path:
            return _collapse_lane_ids(route)
        route.extend(path[1:])

    return _collapse_lane_ids(route)


def _collapse_lane_ids(route):
    return [lane_id for idx, lane_id in enumerate(route) if idx == 0 or lane_id != route[idx - 1]]


def _find_shortest_lane_path(start_lane_id, end_lane_id, route_cache, max_hops=MAX_TRANSITION_HOPS):
    cache_key = (start_lane_id, end_lane_id)
    if cache_key in route_cache["path_cache"]:
        return route_cache["path_cache"][cache_key]

    if start_lane_id == end_lane_id:
        path = (start_lane_id,)
        route_cache["path_cache"][cache_key] = path
        return path

    lane_graph = route_cache["lane_graph"]
    queue = deque([(start_lane_id, (start_lane_id,))])
    seen = {start_lane_id}

    while queue:
        lane_id, path = queue.popleft()
        if len(path) - 1 >= max_hops:
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


def _extend_route_to_dead_end(route, route_cache):
    if not route:
        return []

    lane_graph = route_cache["lane_graph"]
    visited = set(route)
    max_length = max(1, len(route_cache["lane_id_array"]))

    while len(route) < max_length:
        current_lane_id = route[-1]
        exit_lane_ids = [lane_id for lane_id in lane_graph.get(current_lane_id, ()) if lane_id not in visited]
        if not exit_lane_ids:
            break

        next_lane_id = _select_straightest_exit(current_lane_id, exit_lane_ids, route_cache)
        route.append(next_lane_id)
        visited.add(next_lane_id)

    return route


def _select_straightest_exit(current_lane_id, exit_lane_ids, route_cache):
    lane_id_to_idx = route_cache["lane_id_to_idx"]
    current_dir = route_cache["lane_tail_dirs"][lane_id_to_idx[current_lane_id]]
    return max(
        exit_lane_ids,
        key=lambda lane_id: (
            float(np.dot(current_dir, route_cache["lane_head_dirs"][lane_id_to_idx[lane_id]])),
            -int(lane_id),
        ),
    )


def _compute_lane_length_tables(trimmed_polylines, max_points):
    segment_lengths = np.zeros((len(trimmed_polylines), max(0, max_points - 1)), dtype=np.float64)
    cum_lengths = np.zeros((len(trimmed_polylines), max_points), dtype=np.float64)

    for lane_idx, polyline in enumerate(trimmed_polylines):
        if len(polyline) < 2:
            continue
        seg_lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
        segment_lengths[lane_idx, : len(seg_lengths)] = seg_lengths
        cum_lengths[lane_idx, : len(polyline)] = np.concatenate(([0.0], np.cumsum(seg_lengths)))

    return segment_lengths, cum_lengths


def _get_lane_endpoint_direction(polyline, from_start):
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


def _is_offroad_at_timestep(agent_context, route_cache, route_check_timestep=0) -> bool:
    position = agent_context["positions_2d"][route_check_timestep]
    heading = agent_context["headings"][route_check_timestep]
    length = agent_context["lengths"][route_check_timestep]
    width = agent_context["widths"][route_check_timestep]
    trajectory = agent_context["trajectory"]

    extent = _compute_trajectory_extent(trajectory)
    distance_threshold = (
        OFFROAD_DISTANCE_THRESHOLD if extent > MOVEMENT_THRESHOLD else STATIONARY_OFFROAD_DISTANCE_THRESHOLD
    )

    min_distances, _, _ = _points_to_polylines_distance(
        position.reshape(1, 2),
        route_cache["lane_polylines"],
        route_cache["lane_lengths"],
    )
    if np.min(min_distances) > distance_threshold:
        return True

    cos_h, sin_h = np.cos(heading), np.sin(heading)
    half_len, half_w = length / 2, width / 2
    local_corners = np.array([[half_len, -half_w], [half_len, half_w], [-half_len, half_w], [-half_len, -half_w]])
    rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners = local_corners @ rotation.T + position

    agent_poly = shapely_geom.Polygon(corners)
    bbox_min = corners.min(axis=0) - max(half_len, half_w)
    bbox_max = corners.max(axis=0) + max(half_len, half_w)

    for polyline_2d, edge_min, edge_max in route_cache["road_edges"]:
        if edge_max[0] < bbox_min[0] or edge_min[0] > bbox_max[0]:
            continue
        if edge_max[1] < bbox_min[1] or edge_min[1] > bbox_max[1]:
            continue
        if agent_poly.intersects(shapely_geom.LineString(polyline_2d)):
            return True

    return False


def _compute_control_state(agent_data):
    if agent_data.route:
        return int(puffer_types.ControlState.CONTROLLABLE)
    if _compute_trajectory_extent(_valid_trajectory(agent_data.position, agent_data.valid)) > MOVEMENT_THRESHOLD:
        return int(puffer_types.ControlState.NON_CONTROLLABLE_MOVING)
    return int(puffer_types.ControlState.NON_CONTROLLABLE_STATIC)


def _valid_trajectory(positions, valid):
    positions_2d = positions[:, :2] if positions.shape[1] == 3 else positions
    return positions_2d[np.asarray(valid, dtype=bool)]


def _compute_trajectory_extent(trajectory):
    if len(trajectory) < 2:
        return 0.0
    center = np.median(trajectory, axis=0)
    return float(np.linalg.norm(trajectory - center, axis=1).max())


def _points_to_polylines_distance(points, polylines, polyline_lengths):
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


def _get_lane_directions_at_indices_batch(polylines, indices):
    n_lanes = indices.shape[1]
    max_points = polylines.shape[1]
    lane_idx = np.arange(n_lanes)[np.newaxis, :]
    seg_starts = polylines[lane_idx, indices, :]
    seg_ends = polylines[lane_idx, np.minimum(indices + 1, max_points - 1), :]
    directions = seg_ends - seg_starts
    norms = np.linalg.norm(directions, axis=2, keepdims=True)
    return directions / (norms + 1e-6)


def _extract_road_edge(element):
    if not element.is_edge:
        return None

    polyline = element.polyline
    if polyline is None or len(polyline) < 2:
        return None

    polyline_2d = polyline[:, :2]
    return polyline_2d, polyline_2d.min(axis=0), polyline_2d.max(axis=0)
