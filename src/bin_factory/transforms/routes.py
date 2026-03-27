"""Compute per-agent lane routes from observed trajectories.

The pipeline is intentionally split into a few stages:
1. Build a per-scenario lane cache once.
2. Normalize a single agent into valid trajectory samples.
3. Find plausible root lanes near the early trajectory.
4. Expand those roots with a small beam search over lane exits.
5. Score each candidate route with geometric coverage + heading alignment.

The score favors routes whose concatenated centerline stays close to the
trajectory while preserving the original directional consistency check.
"""

import logging
from dataclasses import dataclass

import numpy as np
import shapely
from shapely import geometry as shapely_geom

from bin_factory import puffer_types


logger = logging.getLogger(__name__)


LANE_WIDTH_THRESHOLD = 6.0  # meters — reject point-to-lane matches farther than this
ALIGNMENT_THRESHOLD = 0.3  # cos(heading) — minimum dot product for "same direction"
MAX_PATH_LENGTH = 10  # max lanes in a route sequence
MAX_ROOT_CANDIDATES = 3  # top-k starting lanes to seed beam search
ROOT_LANE_MIN_SCORE = 0.3  # minimum aggregate score to consider a root lane
BEAM_WIDTH = 3  # parallel candidates kept at each beam expansion step
ROOT_CANDIDATE_BBOX_MARGIN = 12.0  # meters — bbox expansion when filtering candidate lanes
ALIGNMENT_WEIGHT = 0.7  # weight of heading alignment in combined lane score
DISTANCE_WEIGHT = 0.3  # weight of proximity in combined lane score
OFFROAD_DISTANCE_THRESHOLD = 5.0  # meters — max lane distance for moving agents
STATIONARY_OFFROAD_DISTANCE_THRESHOLD = 1.0  # meters — max lane distance for stationary agents
MOVEMENT_THRESHOLD = 0.5  # meters — total displacement below this = stationary


@dataclass(frozen=True)
class AgentRouteInput:
    agent_id: int
    positions: np.ndarray
    headings: np.ndarray
    valid: np.ndarray
    lengths: np.ndarray
    widths: np.ndarray
    is_ego: bool


@dataclass(frozen=True)
class BeamState:
    route: tuple[int, ...]
    visited: frozenset[int]
    score: float


def process_agent_routes(scenario, min_route_valid_points=0, route_check_timestep=0) -> None:
    scenario_length = scenario.metadata.scenario_length
    if scenario_length > 0 and route_check_timestep >= scenario_length:
        return

    lane_data = _extract_lane_centers(scenario.map)
    route_cache = build_route_cache(scenario.map, lane_data)

    for agent_id, agent_data in scenario.agents.items():
        is_ego = agent_data.type == puffer_types.AgentType.VEHICLE and agent_id == 0
        is_vehicle = agent_data.type == puffer_types.AgentType.VEHICLE

        if not is_vehicle:
            agent_data.route = []
            continue

        route = compute_agent_route(
            agent_data=AgentRouteInput(
                agent_id=agent_id,
                positions=agent_data.position,
                headings=agent_data.heading,
                valid=agent_data.valid,
                lengths=agent_data.length,
                widths=agent_data.width,
                is_ego=is_ego,
            ),
            route_cache=route_cache,
            route_check_timestep=route_check_timestep,
            min_route_valid_points=min_route_valid_points,
        )
        if is_ego and not route:
            raise ValueError(f"Route computation failed for ego vehicle (agent 0) in scenario {scenario.metadata.id}")
        agent_data.route = route


def _extract_lane_centers(static_map_elements):
    lane_ids = []
    lane_polylines_list = []
    lane_lengths_list = []
    lane_metadata = {}
    max_points = 0

    for element_id, element_data in static_map_elements.items():
        element_type = element_data["type"]

        if element_type in (puffer_types.LaneType.SURFACE_STREET, puffer_types.LaneType.FREEWAY):
            polyline = element_data["polyline"]

            if len(polyline) > 0:
                polyline_2d = polyline[:, :2] if polyline.shape[1] == 3 else polyline

                lane_ids.append(element_id)
                lane_polylines_list.append(polyline_2d)
                lane_lengths_list.append(len(polyline_2d))
                max_points = max(max_points, len(polyline_2d))

                lane_metadata[element_id] = {
                    "entry_lanes": element_data["entry_lanes"],
                    "exit_lanes": element_data["exit_lanes"],
                }

    if not lane_ids:
        return [], np.array([]), {}, np.array([])

    n_lanes = len(lane_ids)
    lane_polylines = np.zeros((n_lanes, max_points, 2), dtype=np.float64)
    lane_lengths = np.array(lane_lengths_list, dtype=np.int64)

    for i, polyline_2d in enumerate(lane_polylines_list):
        lane_polylines[i, : len(polyline_2d), :] = polyline_2d

    return lane_ids, lane_polylines, lane_metadata, lane_lengths


def build_route_cache(static_map_elements, lane_data):
    """Precompute lane geometry and connectivity shared by all agents."""
    lane_ids, lane_polylines, lane_metadata, lane_lengths = lane_data
    lane_ids = list(lane_ids)
    lane_lengths = np.asarray(lane_lengths, dtype=np.int64)
    lane_id_to_idx = {lane_id: idx for idx, lane_id in enumerate(lane_ids)}
    trimmed_polylines = tuple(lane_polylines[idx, : lane_lengths[idx], :].copy() for idx in range(len(lane_ids)))

    if trimmed_polylines:
        lane_bbox_mins = np.stack([polyline.min(axis=0) for polyline in trimmed_polylines])
        lane_bbox_maxs = np.stack([polyline.max(axis=0) for polyline in trimmed_polylines])
    else:
        lane_bbox_mins = np.zeros((0, 2), dtype=np.float64)
        lane_bbox_maxs = np.zeros((0, 2), dtype=np.float64)

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
        "lane_ids": lane_ids,
        "lane_polylines": lane_polylines,
        "lane_lengths": lane_lengths,
        "lane_id_to_idx": lane_id_to_idx,
        "trimmed_polylines": trimmed_polylines,
        "lane_graph": lane_graph,
        "lane_bbox_mins": lane_bbox_mins,
        "lane_bbox_maxs": lane_bbox_maxs,
        "road_edges": road_edges,
    }


def compute_agent_route(
    agent_data,
    route_cache,
    route_check_timestep=0,
    min_route_valid_points=0,
):
    """Return the best lane sequence for one agent, or an empty list."""
    positions_2d = agent_data.positions[:, :2] if agent_data.positions.shape[1] == 3 else agent_data.positions
    headings = agent_data.headings
    valid = np.asarray(agent_data.valid, dtype=bool)
    trajectory = positions_2d[valid]
    heading_valid = headings[valid]

    agent_dirs = np.stack([np.cos(heading_valid), np.sin(heading_valid)], axis=1)
    sample_indices = np.linspace(0, len(trajectory) - 1, min(10, len(trajectory)), dtype=int)

    agent_context = {
        "positions_2d": positions_2d,
        "headings": headings,
        "valid": valid,
        "lengths": agent_data.lengths,
        "widths": agent_data.widths,
        "trajectory": trajectory,
        "heading_valid": heading_valid,
        "sample_positions": trajectory[sample_indices],
        "sample_agent_dirs": agent_dirs[sample_indices],
    }

    agent_str = f"Agent {agent_data.agent_id}" if agent_data.agent_id is not None else "Agent"

    if not _can_compute_route(
        agent_context,
        route_cache,
        agent_data.is_ego,
        route_check_timestep,
        min_route_valid_points,
    ):
        logger.debug(f"{agent_str}: Skipping route computation due to insufficient valid data or offroad start")
        return []

    root_candidates = _select_root_lane_candidates(agent_context, route_cache)
    if not root_candidates:
        logger.debug(f"{agent_str}: No current lane found")
        return []

    best_route, best_score = _search_route_beam(root_candidates, route_cache, agent_context)
    if not best_route or best_score <= 0:
        logger.debug(f"{agent_str}: No valid route found from candidates")
        return []

    return best_route


def _can_compute_route(agent_context, route_cache, is_ego, route_check_timestep=0, min_route_valid_points=0) -> bool:
    if is_ego:
        return True

    if len(agent_context["trajectory"]) == 0 or len(route_cache["lane_ids"]) == 0:
        return False

    if not agent_context["valid"][route_check_timestep]:
        return False

    if np.sum(agent_context["valid"][route_check_timestep:]) < min_route_valid_points:
        return False

    return not _is_offroad_at_timestep(agent_context, route_cache, route_check_timestep)


def _search_route_beam(root_candidates, route_cache, agent_context, max_length=MAX_PATH_LENGTH, beam_width=BEAM_WIDTH):
    lane_id_to_idx = route_cache["lane_id_to_idx"]
    trimmed_polylines = route_cache["trimmed_polylines"]

    initial_sequences = [(lane_id,) for lane_id, _ in root_candidates]
    beam = _select_top_beam(
        _score_lane_sequences(initial_sequences, lane_id_to_idx, trimmed_polylines, agent_context),
        beam_width,
    )
    if not beam:
        return [], 0.0

    best_beam_state = max(beam, key=_beam_rank)

    for _ in range(max_length - 1):
        candidate_sequences = _expand_beam(beam, route_cache["lane_graph"])
        if not candidate_sequences:
            break

        beam = _select_top_beam(
            _score_lane_sequences(candidate_sequences, lane_id_to_idx, trimmed_polylines, agent_context),
            beam_width,
        )
        if not beam:
            break

        candidate_best = max(beam, key=_beam_rank)
        if _beam_rank(candidate_best) > _beam_rank(best_beam_state):
            best_beam_state = candidate_best

    return list(best_beam_state.route), best_beam_state.score


def _beam_rank(state):
    return (state.score, len(state.route))


def _merge_route_centerlines(route, lane_id_to_idx, trimmed_polylines):
    parts = []
    for lane_id in route:
        lane_idx = lane_id_to_idx.get(lane_id)
        if lane_idx is None:
            continue
        polyline = trimmed_polylines[lane_idx]
        if len(polyline) == 0:
            continue
        if parts and np.allclose(parts[-1][-1], polyline[0]):
            parts.append(polyline[1:])
        else:
            parts.append(polyline)

    if not parts:
        return np.zeros((0, 2), dtype=np.float64)
    return np.concatenate(parts) if len(parts) > 1 else parts[0].copy()


def _score_lane_sequences(lane_sequences, lane_id_to_idx, trimmed_polylines, agent_context):
    if not lane_sequences:
        return []

    polylines = [_merge_route_centerlines(sequence, lane_id_to_idx, trimmed_polylines) for sequence in lane_sequences]
    scores = _score_route_polylines_batch(polylines, agent_context)
    return [
        BeamState(route=sequence, visited=frozenset(sequence), score=float(score))
        for sequence, score in zip(lane_sequences, scores, strict=False)
        if score > 0
    ]


def _select_top_beam(states, beam_width):
    selected = []
    seen = set()
    for state in sorted(states, key=_beam_rank, reverse=True):
        if state.route in seen:
            continue
        selected.append(state)
        seen.add(state.route)
        if len(selected) == beam_width:
            break
    return selected


def _expand_beam(beam, lane_graph):
    candidates = []
    for state in beam:
        exit_lanes = lane_graph.get(state.route[-1], ())
        candidates.extend((*state.route, exit_id) for exit_id in exit_lanes if exit_id not in state.visited)
    return candidates


def _select_root_lane_candidates(
    agent_context,
    route_cache,
    max_candidates=MAX_ROOT_CANDIDATES,
    min_score=ROOT_LANE_MIN_SCORE,
):
    trajectory = agent_context["trajectory"]
    heading = agent_context["heading_valid"]
    lane_ids = route_cache["lane_ids"]
    lane_polylines = route_cache["lane_polylines"]
    lane_lengths = route_cache["lane_lengths"]

    traj_len = len(trajectory)
    if traj_len == 0:
        return []
    window = min(traj_len, 8)
    sample_count = min(window, 5)
    sample_indices = np.unique(np.linspace(0, window - 1, sample_count, dtype=int))
    if len(sample_indices) == 0:
        return []

    sample_points = trajectory[sample_indices]
    sample_headings = heading[sample_indices]

    if len(sample_points) > 1:
        diffs = np.linalg.norm(np.diff(sample_points, axis=0), axis=1)
        keep_mask = np.concatenate(([True], diffs >= 1e-3))
        sample_points = sample_points[keep_mask]
        sample_headings = sample_headings[keep_mask]
    if len(sample_points) == 0:
        return []

    lane_bbox_mins = route_cache["lane_bbox_mins"]
    if len(lane_bbox_mins) == 0:
        return []
    clipped = np.clip(
        sample_points[:, np.newaxis, :],
        lane_bbox_mins[np.newaxis, :, :],
        route_cache["lane_bbox_maxs"][np.newaxis, :, :],
    )
    diff = sample_points[:, np.newaxis, :] - clipped
    bbox_distances = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    candidate_lane_indices = np.where(np.any(bbox_distances <= ROOT_CANDIDATE_BBOX_MARGIN, axis=0))[0]
    if len(candidate_lane_indices) == 0:
        return []

    candidate_polylines = lane_polylines[candidate_lane_indices]
    candidate_lengths = lane_lengths[candidate_lane_indices]

    min_distances_all, closest_indices_all = _points_to_polylines_distance(
        sample_points,
        candidate_polylines,
        polyline_lengths=candidate_lengths,
    )
    lane_directions_all = _get_lane_directions_at_indices_batch(candidate_polylines, closest_indices_all)
    agent_dirs = np.stack([np.cos(sample_headings), np.sin(sample_headings)], axis=1)
    alignments_all = np.sum(lane_directions_all * agent_dirs[:, np.newaxis, :], axis=2)

    valid_mask = (min_distances_all < LANE_WIDTH_THRESHOLD) & (alignments_all > ALIGNMENT_THRESHOLD)
    distance_scores_all = 1.0 / (1.0 + min_distances_all)
    scores_all = ALIGNMENT_WEIGHT * alignments_all + DISTANCE_WEIGHT * distance_scores_all
    lane_total_scores = np.sum(np.where(valid_mask, scores_all, 0.0), axis=0)

    valid_lane_indices = np.where(lane_total_scores >= min_score)[0]
    if len(valid_lane_indices) == 0:
        return []

    ranked = valid_lane_indices[np.argsort(lane_total_scores[valid_lane_indices])[::-1][:max_candidates]]
    return [(lane_ids[candidate_lane_indices[idx]], lane_total_scores[idx]) for idx in ranked]


def _score_route_polylines_batch(route_polylines, agent_context):
    scores = np.zeros(len(route_polylines), dtype=np.float64)
    if not route_polylines:
        return scores

    lengths = np.array([len(polyline) for polyline in route_polylines], dtype=np.int64)
    valid_mask = lengths > 1
    if not np.any(valid_mask):
        return scores

    valid_indices = np.where(valid_mask)[0]
    valid_polylines = [route_polylines[idx] for idx in valid_indices]
    valid_lengths = lengths[valid_indices]

    max_points = max(len(p) for p in valid_polylines)
    padded_polylines = np.zeros((len(valid_polylines), max_points, 2), dtype=np.float64)
    for idx, polyline in enumerate(valid_polylines):
        padded_polylines[idx, : len(polyline), :] = polyline

    sample_positions = agent_context["sample_positions"]
    sample_agent_dirs = agent_context["sample_agent_dirs"]
    if len(padded_polylines) == 0 or len(sample_positions) == 0:
        alignment_mask = np.zeros(len(padded_polylines), dtype=bool)
    else:
        _, closest_indices = _points_to_polylines_distance(
            sample_positions,
            padded_polylines,
            polyline_lengths=valid_lengths,
        )
        route_directions = _get_lane_directions_at_indices_batch(padded_polylines, closest_indices)
        alignments = np.sum(route_directions * sample_agent_dirs[:, np.newaxis, :], axis=2)
        alignment_ratio = np.mean(alignments > ALIGNMENT_THRESHOLD, axis=0)
        alignment_mask = alignment_ratio >= 0.7

    if not np.any(alignment_mask):
        return scores

    aligned_indices = valid_indices[alignment_mask]
    aligned_polylines = padded_polylines[alignment_mask]
    aligned_lengths = valid_lengths[alignment_mask]
    min_distances = _points_to_polylines_distance(
        agent_context["trajectory"],
        aligned_polylines,
        polyline_lengths=aligned_lengths,
        return_indices=False,
    )
    min_distances = np.asarray(min_distances, dtype=np.float64)

    coverage_mask = min_distances < LANE_WIDTH_THRESHOLD
    covered_counts = coverage_mask.sum(axis=0)
    positive_coverage = covered_counts > 0
    if not np.any(positive_coverage):
        return scores

    avg_distances = np.full(len(aligned_indices), np.inf, dtype=np.float64)
    covered_distances = np.where(coverage_mask, min_distances, 0.0).sum(axis=0)
    avg_distances[positive_coverage] = covered_distances[positive_coverage] / covered_counts[positive_coverage]

    if not np.all(np.isfinite(avg_distances[positive_coverage])):
        raise ValueError("Invalid route distance. This may indicate degenerate route polylines.")

    coverage_ratios = covered_counts[positive_coverage] / len(agent_context["trajectory"])
    scores[aligned_indices[positive_coverage]] = coverage_ratios / (1.0 + avg_distances[positive_coverage])
    return scores


def _is_offroad_at_timestep(agent_context, route_cache, route_check_timestep=0) -> bool:
    position = agent_context["positions_2d"][route_check_timestep]
    heading = agent_context["headings"][route_check_timestep]
    length = agent_context["lengths"][route_check_timestep]
    width = agent_context["widths"][route_check_timestep]
    trajectory = agent_context["trajectory"]

    displacement = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) if len(trajectory) >= 2 else 0.0
    distance_threshold = (
        OFFROAD_DISTANCE_THRESHOLD if displacement > MOVEMENT_THRESHOLD else STATIONARY_OFFROAD_DISTANCE_THRESHOLD
    )

    min_distances = _points_to_polylines_distance(
        position.reshape(1, 2),
        route_cache["lane_polylines"],
        polyline_lengths=route_cache["lane_lengths"],
        return_indices=False,
    )
    if np.min(min_distances) > distance_threshold:
        return True

    cos_h, sin_h = np.cos(heading), np.sin(heading)
    half_len, half_w = length / 2, width / 2
    local_corners = np.array([[half_len, -half_w], [half_len, half_w], [-half_len, half_w], [-half_len, -half_w]])
    rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners = local_corners @ rotation.T + position

    agent_poly = shapely_geom.Polygon(corners)
    shapely.prepare(agent_poly)

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


def _points_to_polylines_distance(points, polylines, polyline_lengths=None, return_indices=True):
    n_points = len(points)
    n_lanes = len(polylines)
    max_segments = polylines.shape[1] - 1 if n_lanes > 0 else 0

    if n_points == 0 or n_lanes == 0 or max_segments <= 0:
        empty_distances = np.zeros((n_points, n_lanes), dtype=np.float64)
        empty_indices = np.zeros((n_points, n_lanes), dtype=np.int64)
        return (empty_distances, empty_indices) if return_indices else empty_distances

    seg_starts = polylines[:, :-1, :]
    seg_ends = polylines[:, 1:, :]

    if polyline_lengths is not None:
        polyline_lengths = np.asarray(polyline_lengths, dtype=np.int64)
        if len(polyline_lengths) != n_lanes:
            raise ValueError("polyline_lengths must match number of polylines")
        seg_counts = np.clip(polyline_lengths - 1, 0, max_segments)
        valid_segs = np.arange(max_segments)[np.newaxis, :] < seg_counts[:, np.newaxis]
    else:
        valid_segs = np.any(seg_starts != 0, axis=2) & np.any(seg_ends != 0, axis=2)

    seg_vecs = seg_ends - seg_starts
    seg_lens_sq = np.einsum("ijk,ijk->ij", seg_vecs, seg_vecs)
    valid_segs = valid_segs & (seg_lens_sq > 1e-10)
    seg_lens_sq_safe = seg_lens_sq + 1e-10

    points_bc = points.reshape(n_points, 1, 1, 2)
    point_to_start = points_bc - seg_starts.reshape(1, n_lanes, max_segments, 2)
    t = np.einsum("ijkl,jkl->ijk", point_to_start, seg_vecs) / seg_lens_sq_safe.reshape(1, n_lanes, max_segments)
    t = np.clip(t, 0.0, 1.0, out=t)
    closest_on_seg = seg_starts.reshape(1, n_lanes, max_segments, 2) + t[..., np.newaxis] * seg_vecs.reshape(
        1,
        n_lanes,
        max_segments,
        2,
    )
    diff = points_bc - closest_on_seg
    distances_sq = np.einsum("ijkl,ijkl->ijk", diff, diff)
    distances_sq = np.where(valid_segs.reshape(1, n_lanes, max_segments), distances_sq, np.inf)

    min_distances_sq = np.min(distances_sq, axis=2)
    min_distances = np.sqrt(min_distances_sq)

    if not return_indices:
        return min_distances

    closest_indices = np.argmin(distances_sq, axis=2).astype(np.int64)
    return min_distances, closest_indices


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
    if not puffer_types.is_road_edge(element["type"]):
        return None

    polyline = element.get("polyline")
    if polyline is None or len(polyline) < 2:
        return None

    polyline_2d = polyline[:, :2]
    return polyline_2d, polyline_2d.min(axis=0), polyline_2d.max(axis=0)
