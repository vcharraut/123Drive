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

import numpy as np
from py123d.datatypes.map_objects import RoadEdgeType

from bin_factory import logger_utils


logger = logger_utils.get_logger(__name__)


LANE_WIDTH_THRESHOLD = 6.0
ALIGNMENT_THRESHOLD = 0.3
MAX_PATH_LENGTH = 10
MAX_ROOT_CANDIDATES = 3
ROOT_LANE_MIN_SCORE = 0.3
BEAM_WIDTH = 3
ROOT_CANDIDATE_BBOX_MARGIN = 12.0
ALIGNMENT_WEIGHT = 0.7
DISTANCE_WEIGHT = 0.3
OFFROAD_DISTANCE_THRESHOLD = 5.0
STATIONARY_OFFROAD_DISTANCE_THRESHOLD = 1.0
MOVEMENT_THRESHOLD = 0.5
ROAD_EDGE_TYPES = {
    RoadEdgeType.ROAD_EDGE_BOUNDARY,
    RoadEdgeType.ROAD_EDGE_MEDIAN,
}


def build_route_cache(static_map_elements: dict, lane_data: tuple) -> dict:
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


def compute_agent_route(agent_data: tuple, route_cache: dict, route_check_timestep: int = 0) -> list[list[int]]:
    """Return the best lane sequence for one agent, or an empty list."""
    agent_context = _build_agent_route_context(agent_data)
    agent_id = agent_data[0]
    agent_str = f"Agent {agent_id}" if agent_id is not None else "Agent"

    if agent_context is None or len(route_cache["lane_ids"]) == 0:
        return []

    if _is_offroad_at_timestep(agent_context, route_cache, route_check_timestep):
        logger.debug(f"{agent_str}: Off-road at timestep {route_check_timestep}")
        return []

    root_candidates = _select_root_lane_candidates(agent_context, route_cache)
    if not root_candidates:
        logger.debug(f"{agent_str}: No current lane found")
        return []

    best_route, best_score = _search_route_beam(root_candidates, route_cache, agent_context)
    if not best_route or best_score <= 0:
        logger.debug(f"{agent_str}: No valid route found from candidates")
        return []

    return [best_route]


def _build_agent_route_context(agent_data: tuple) -> dict | None:
    """Convert raw per-timestep states into route-specific derived arrays."""
    agent_id, positions, headings, valid, lengths, widths = agent_data
    positions_2d = positions[:, :2] if positions.shape[1] == 3 else positions
    valid = np.asarray(valid, dtype=bool)
    trajectory = positions_2d[valid]
    heading_valid = headings[valid]

    if len(trajectory) == 0:
        return None

    agent_dirs = np.stack([np.cos(heading_valid), np.sin(heading_valid)], axis=1)
    sample_indices = np.linspace(0, len(trajectory) - 1, min(10, len(trajectory)), dtype=int)

    return {
        "id": agent_id,
        "positions_2d": positions_2d,
        "headings": headings,
        "valid": valid,
        "lengths": lengths,
        "widths": widths,
        "trajectory": trajectory,
        "heading_valid": heading_valid,
        "sample_positions": trajectory[sample_indices],
        "sample_agent_dirs": agent_dirs[sample_indices],
    }


def _search_route_beam(
    root_candidates: list[tuple[int | str, float]],
    route_cache: dict,
    agent_context: dict,
    max_length: int = MAX_PATH_LENGTH,
    beam_width: int = BEAM_WIDTH,
) -> tuple[list, float]:
    """Search over consecutive exit lanes while keeping only top candidates.

    Each beam state stores a route prefix and its current geometric score.
    We keep the best-scoring prefixes, expand them through exit lanes, and use
    score then route length as a tie-breaker so flat-score continuations are not
    dropped too early.
    """
    initial_states = _score_route_candidates([[lane_id] for lane_id, _ in root_candidates], route_cache, agent_context)
    beam = _select_top_states(initial_states, beam_width)
    if not beam:
        return [], 0.0

    best_beam_state = max(beam, key=_beam_state_rank)

    for _ in range(max_length - 1):
        candidates = []
        for beam_state in beam:
            last_lane = beam_state["route"][-1]
            exit_lanes = route_cache["lane_graph"].get(last_lane, ())
            candidates.extend(
                beam_state["route"] + [exit_id] for exit_id in exit_lanes if exit_id not in beam_state["visited"]
            )

        if not candidates:
            break

        expanded_states = _score_route_candidates(candidates, route_cache, agent_context)
        beam = _select_top_states(expanded_states, beam_width)
        if not beam:
            break

        candidate_best = max(beam, key=_beam_state_rank)
        if _beam_state_rank(candidate_best) > _beam_state_rank(best_beam_state):
            best_beam_state = candidate_best

    return best_beam_state["route"], best_beam_state["score"]


def _score_route_candidates(lane_sequences: list[list], route_cache: dict, agent_context: dict) -> list[dict]:
    """Materialize and score a batch of candidate lane sequences."""
    if not lane_sequences:
        return []

    polylines = [_merge_lane_centerlines(sequence, route_cache) for sequence in lane_sequences]
    scores = _score_route_polylines_batch(polylines, agent_context)

    return [
        {
            "route": sequence,
            "visited": set(sequence),
            "score": float(score),
        }
        for sequence, score in zip(lane_sequences, scores)
        if score > 0
    ]


def _merge_lane_centerlines(route: list, route_cache: dict) -> np.ndarray:
    """Concatenate lane centerlines into one polyline for route scoring."""
    lane_id_to_idx = route_cache["lane_id_to_idx"]
    trimmed_polylines = route_cache["trimmed_polylines"]
    route_polyline = np.zeros((0, 2), dtype=np.float64)

    for lane_id in route:
        lane_idx = lane_id_to_idx.get(lane_id)
        if lane_idx is None:
            continue
        route_polyline = _append_polyline(route_polyline, trimmed_polylines[lane_idx])

    return route_polyline


def _select_root_lane_candidates(
    agent_context: dict,
    route_cache: dict,
    max_candidates: int = MAX_ROOT_CANDIDATES,
    min_score: float = ROOT_LANE_MIN_SCORE,
) -> list[tuple[int | str, float]]:
    """Score likely starting lanes from the first part of the trajectory.

    We first prefilter lanes by bounding-box distance, then apply the exact
    point-to-polyline distance and local heading checks on a few early samples.
    """
    trajectory = agent_context["trajectory"]
    heading = agent_context["heading_valid"]
    lane_ids = route_cache["lane_ids"]
    lane_polylines = route_cache["lane_polylines"]
    lane_lengths = route_cache["lane_lengths"]

    sample_indices = _get_root_sample_indices(len(trajectory))
    if len(sample_indices) == 0:
        return []

    sample_points = trajectory[sample_indices]
    sample_headings = heading[sample_indices]
    sample_points, sample_headings = _deduplicate_sample_points(sample_points, sample_headings)
    if len(sample_points) == 0:
        return []

    candidate_lane_indices = _filter_root_candidate_lanes(sample_points, route_cache)
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


def _get_root_sample_indices(traj_len: int) -> np.ndarray:
    if traj_len == 0:
        return np.array([], dtype=np.int64)
    window = min(traj_len, 8)
    sample_count = min(window, 5)
    return np.unique(np.linspace(0, window - 1, sample_count, dtype=int))


def _deduplicate_sample_points(points: np.ndarray, headings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= 1:
        return points, headings

    keep_mask = np.ones(len(points), dtype=bool)
    unique_points = []
    for idx, point in enumerate(points):
        if any(np.linalg.norm(point - seen) < 1e-3 for seen in unique_points):
            keep_mask[idx] = False
            continue
        unique_points.append(point)

    return points[keep_mask], headings[keep_mask]


def _filter_root_candidate_lanes(sample_points: np.ndarray, route_cache: dict) -> np.ndarray:
    lane_bbox_mins = route_cache["lane_bbox_mins"]
    if len(lane_bbox_mins) == 0:
        return np.array([], dtype=np.int64)

    bbox_distances = _points_to_bboxes_distance(sample_points, lane_bbox_mins, route_cache["lane_bbox_maxs"])
    return np.where(np.any(bbox_distances <= ROOT_CANDIDATE_BBOX_MARGIN, axis=0))[0]


def _points_to_bboxes_distance(points: np.ndarray, bbox_mins: np.ndarray, bbox_maxs: np.ndarray) -> np.ndarray:
    clipped = np.clip(points[:, np.newaxis, :], bbox_mins[np.newaxis, :, :], bbox_maxs[np.newaxis, :, :])
    diff = points[:, np.newaxis, :] - clipped
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def _select_top_states(states: list[dict], beam_width: int) -> list[dict]:
    selected = []
    seen = set()

    for beam_state in sorted(states, key=_beam_state_rank, reverse=True):
        route_key = tuple(beam_state["route"])
        if route_key in seen:
            continue
        selected.append(beam_state)
        seen.add(route_key)
        if len(selected) == beam_width:
            break

    return selected


def _beam_state_rank(beam_state: dict) -> tuple[float, int]:
    return beam_state["score"], len(beam_state["route"])


def _score_route_polylines_batch(route_polylines: list[np.ndarray], agent_context: dict) -> np.ndarray:
    """Score many candidate route polylines against one agent trajectory."""
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
    padded_polylines = _pad_polylines(valid_polylines)

    # Reject routes that mostly point against the agent's motion before doing
    # the more expensive full-trajectory coverage computation.
    alignment_mask = _check_route_heading_alignment_batch(
        padded_polylines,
        valid_lengths,
        agent_context["sample_positions"],
        agent_context["sample_agent_dirs"],
    )
    if not np.any(alignment_mask):
        return scores

    aligned_indices = valid_indices[alignment_mask]
    aligned_polylines = padded_polylines[alignment_mask]
    aligned_lengths = valid_lengths[alignment_mask]
    min_distances = _points_to_polylines_min_distance(
        agent_context["trajectory"],
        aligned_polylines,
        polyline_lengths=aligned_lengths,
    )

    # Final score matches the original intent: coverage times inverse average
    # distance over covered trajectory points only.
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


def _check_route_heading_alignment_batch(
    route_polylines: np.ndarray,
    route_lengths: np.ndarray,
    sample_positions: np.ndarray,
    sample_agent_dirs: np.ndarray,
    min_alignment_ratio: float = 0.7,
) -> np.ndarray:
    """Binary heading filter for a batch of candidate route polylines."""
    if len(route_polylines) == 0 or len(sample_positions) == 0:
        return np.zeros(len(route_polylines), dtype=bool)

    _, closest_indices = _points_to_polylines_distance(
        sample_positions,
        route_polylines,
        polyline_lengths=route_lengths,
    )
    route_directions = _get_lane_directions_at_indices_batch(route_polylines, closest_indices)
    alignments = np.sum(route_directions * sample_agent_dirs[:, np.newaxis, :], axis=2)
    alignment_ratio = np.mean(alignments > ALIGNMENT_THRESHOLD, axis=0)
    return alignment_ratio >= min_alignment_ratio


def _pad_polylines(polylines: list[np.ndarray]) -> np.ndarray:
    max_points = max(len(polyline) for polyline in polylines)
    padded = np.zeros((len(polylines), max_points, 2), dtype=np.float64)

    for idx, polyline in enumerate(polylines):
        padded[idx, : len(polyline), :] = polyline

    return padded


def _append_polyline(base: np.ndarray, addition: np.ndarray) -> np.ndarray:
    if len(addition) == 0:
        return base.copy()
    if len(base) == 0:
        return addition.copy()
    if np.allclose(base[-1], addition[0]):
        return np.vstack((base, addition[1:]))
    return np.vstack((base, addition))


def _extract_road_edge(element: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if element["type"] not in ROAD_EDGE_TYPES:
        return None

    polyline = element["polyline"]
    if polyline is None or len(polyline) < 2:
        return None

    polyline_2d = polyline[:, :2]
    return polyline_2d, polyline_2d.min(axis=0), polyline_2d.max(axis=0)


def _is_offroad_at_timestep(agent_context: dict, route_cache: dict, route_check_timestep: int = 0) -> bool:
    """Skip route generation for vehicles that start clearly off drivable road.

    The check mirrors the previous behavior: far from any lane centerline or
    bounding box intersecting a road edge. Movement is estimated from total path
    length so looped trajectories are not treated as stationary.
    """
    positions_2d = agent_context["positions_2d"]
    if route_check_timestep >= len(positions_2d):
        return True

    position = positions_2d[route_check_timestep]
    heading = agent_context["headings"][route_check_timestep]
    length = agent_context["lengths"][route_check_timestep]
    width = agent_context["widths"][route_check_timestep]
    trajectory = agent_context["trajectory"]

    displacement = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) if len(trajectory) >= 2 else 0.0
    distance_threshold = (
        OFFROAD_DISTANCE_THRESHOLD if displacement > MOVEMENT_THRESHOLD else STATIONARY_OFFROAD_DISTANCE_THRESHOLD
    )

    if len(route_cache["lane_polylines"]) > 0:
        min_distances = _points_to_polylines_min_distance(
            position.reshape(1, 2),
            route_cache["lane_polylines"],
            polyline_lengths=route_cache["lane_lengths"],
        )
        if np.min(min_distances) > distance_threshold:
            return True

    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    half_len = length / 2
    half_width = width / 2
    local_corners = np.array(
        [
            [half_len, -half_width],
            [half_len, half_width],
            [-half_len, half_width],
            [-half_len, -half_width],
        ],
    )
    rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners = local_corners @ rotation_matrix.T + position

    bbox_min = corners.min(axis=0) - max(half_len, half_width)
    bbox_max = corners.max(axis=0) + max(half_len, half_width)

    for polyline_2d, edge_min, edge_max in route_cache["road_edges"]:
        if edge_max[0] < bbox_min[0] or edge_min[0] > bbox_max[0]:
            continue
        if edge_max[1] < bbox_min[1] or edge_min[1] > bbox_max[1]:
            continue

        for idx in range(len(polyline_2d) - 1):
            seg_start = polyline_2d[idx]
            seg_end = polyline_2d[idx + 1]
            seg_min = np.minimum(seg_start, seg_end)
            seg_max = np.maximum(seg_start, seg_end)

            if seg_max[0] < bbox_min[0] or seg_min[0] > bbox_max[0]:
                continue
            if seg_max[1] < bbox_min[1] or seg_min[1] > bbox_max[1]:
                continue
            if _segment_intersects_polygon(seg_start, seg_end, corners):
                return True

    return False


def _points_to_polylines_distance(
    points: np.ndarray,
    polylines: np.ndarray,
    polyline_lengths: np.ndarray | None = None,
    return_indices: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Vectorized point-to-polyline distance with optional closest segment ids."""
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


def _points_to_polylines_min_distance(
    points: np.ndarray,
    polylines: np.ndarray,
    polyline_lengths: np.ndarray | None = None,
) -> np.ndarray:
    distances = _points_to_polylines_distance(
        points,
        polylines,
        polyline_lengths=polyline_lengths,
        return_indices=False,
    )
    if isinstance(distances, tuple):
        return distances[0]
    return distances


def _get_lane_directions_at_indices_batch(polylines: np.ndarray, indices: np.ndarray) -> np.ndarray:
    n_lanes = indices.shape[1]
    max_points = polylines.shape[1]
    lane_idx = np.arange(n_lanes)[np.newaxis, :]
    seg_starts = polylines[lane_idx, indices, :]
    seg_ends = polylines[lane_idx, np.minimum(indices + 1, max_points - 1), :]
    directions = seg_ends - seg_starts
    norms = np.linalg.norm(directions, axis=2, keepdims=True)
    return directions / (norms + 1e-6)


def _segment_intersects_polygon(seg_start: np.ndarray, seg_end: np.ndarray, polygon: np.ndarray) -> bool:
    for idx in range(len(polygon)):
        poly_start = polygon[idx]
        poly_end = polygon[(idx + 1) % len(polygon)]
        if _segments_intersect(seg_start, seg_end, poly_start, poly_end):
            return True
    return False


def _segments_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    d1 = p2 - p1
    d2 = p4 - p3

    def cross_2d(v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]

    denom = cross_2d(d1, d2)
    if abs(denom) < 1e-10:
        if abs(cross_2d(p3 - p1, d1)) > 1e-10 or abs(cross_2d(p4 - p1, d1)) > 1e-10:
            return False

        def ranges_overlap(a1, a2, b1, b2) -> bool:
            return max(min(a1, a2), min(b1, b2)) <= min(max(a1, a2), max(b1, b2)) + 1e-10

        return ranges_overlap(p1[0], p2[0], p3[0], p4[0]) and ranges_overlap(p1[1], p2[1], p3[1], p4[1])

    t = cross_2d(p3 - p1, d2) / denom
    u = cross_2d(p3 - p1, d1) / denom
    return 0 <= t <= 1 and 0 <= u <= 1
