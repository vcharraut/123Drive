"""Graph-based agent route computation using precomputed lane distance matrix.

Replaces the heavy beam-search route finder (routes.py) with a simple:
  1. Find start lane (geometric match over early trajectory window)
  2. Find goal lane (geometric match over late trajectory window)
  3. Reconstruct shortest path via greedy Dijkstra walk
"""

import numpy as np

from bin_factory.convert.graph import GRAPH_LANE_TYPES


HEADING_ALIGNMENT_MIN = 0.3
LANE_DISTANCE_MAX = 6.0
WINDOW_SIZE = 8
SAMPLE_COUNT = 5


def compute_routes_from_graph(puffer_agents, road_map_elements, lane_graph):
    if lane_graph is None:
        return

    lanes = [r for r in road_map_elements if r["type"] in GRAPH_LANE_TYPES]
    if not lanes:
        return

    lane_cache = _build_lane_cache(lanes)
    id_to_idx = {lid: i for i, lid in enumerate(lane_graph["lane_ids"])}
    n = len(lane_graph["lane_ids"])
    dist_matrix = lane_graph["distances"]
    lane_lengths = lane_graph["lane_lengths"]

    for agent in puffer_agents:
        if agent.get("route"):
            continue
        if agent["type"] != 1:  # vehicles only
            continue

        states = agent["states"]
        valid = np.asarray(states["valid"], dtype=bool)
        if valid.sum() < 2:
            continue

        xyz = states["xyz"]
        pos_2d = xyz[:, :2] if xyz.shape[1] == 3 else xyz
        headings = states["heading"]

        start = _find_lane_window(pos_2d, headings, valid, lane_cache, from_end=False)
        goal = _find_lane_window(pos_2d, headings, valid, lane_cache, from_end=True)
        if start is None or goal is None:
            continue
        if start == goal:
            agent["route"] = [start]
            continue

        path = _reconstruct_path(start, goal, id_to_idx, dist_matrix, lane_lengths, lane_cache, n)
        if path:
            agent["route"] = path


def _build_lane_cache(lanes):
    ids = []
    polylines = []
    directions = []
    id_to_exits = {}

    for lane in lanes:
        xyz = np.asarray(lane["xyz"], dtype=np.float64)
        if len(xyz) < 2:
            continue
        xy = xyz[:, :2]
        ids.append(lane["id"])
        polylines.append(xy)

        # Precompute segment directions (normalized)
        diffs = np.diff(xy, axis=0)
        norms = np.linalg.norm(diffs, axis=1, keepdims=True)
        dirs = diffs / np.maximum(norms, 1e-8)
        directions.append(dirs)
        id_to_exits[lane["id"]] = [eid for eid in lane.get("exit_lanes", []) if eid is not None]

    return {
        "ids": ids,
        "polylines": polylines,
        "directions": directions,
        "id_to_exits": id_to_exits,
    }


def _find_lane_window(pos_2d, headings, valid, lane_cache, from_end=False):
    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        return None

    window_indices = valid_indices[-WINDOW_SIZE:] if from_end else valid_indices[:WINDOW_SIZE]

    # Subsample
    sample_idx = np.unique(np.linspace(0, len(window_indices) - 1, min(SAMPLE_COUNT, len(window_indices)), dtype=int))
    sampled = window_indices[sample_idx]

    points = pos_2d[sampled]
    agent_dirs = np.column_stack([np.cos(headings[sampled]), np.sin(headings[sampled])])

    best_id = None
    best_score = -np.inf

    for i, (lid, polyline, dirs) in enumerate(
        zip(lane_cache["ids"], lane_cache["polylines"], lane_cache["directions"])
    ):
        score = _score_lane_match(points, agent_dirs, polyline, dirs)
        if score > best_score:
            best_score = score
            best_id = lid

    return best_id if best_score > 0 else None


def _score_lane_match(points, agent_dirs, polyline, seg_dirs):
    # Point-to-polyline distances
    seg_starts = polyline[:-1]
    seg_ends = polyline[1:]
    seg_vecs = seg_ends - seg_starts
    seg_lens_sq = np.sum(seg_vecs ** 2, axis=1)
    seg_lens_sq_safe = np.maximum(seg_lens_sq, 1e-10)

    total_score = 0.0
    n_valid = 0

    for k in range(len(points)):
        pt = points[k]
        # Project onto each segment
        t = np.sum((pt - seg_starts) * seg_vecs, axis=1) / seg_lens_sq_safe
        t = np.clip(t, 0, 1)
        closest = seg_starts + t[:, None] * seg_vecs
        dists = np.linalg.norm(pt - closest, axis=1)

        best_seg = np.argmin(dists)
        min_dist = dists[best_seg]
        if min_dist > LANE_DISTANCE_MAX:
            continue

        # Heading alignment
        alignment = np.dot(agent_dirs[k], seg_dirs[best_seg])
        if alignment < HEADING_ALIGNMENT_MIN:
            continue

        # Score: alignment weighted by inverse distance
        total_score += alignment / (1.0 + min_dist)
        n_valid += 1

    if n_valid < max(1, len(points) // 2):
        return -1.0
    return total_score


def _reconstruct_path(start, goal, id_to_idx, dist_matrix, lane_lengths, lane_cache, n):
    src_idx = id_to_idx.get(start)
    dst_idx = id_to_idx.get(goal)
    if src_idx is None or dst_idx is None:
        return None

    total_dist = dist_matrix[src_idx, dst_idx]
    if not np.isfinite(total_dist):
        return None

    exits_map = lane_cache["id_to_exits"]
    ids = list(id_to_idx.keys())

    path = [start]
    cur_id, cur_idx = start, src_idx
    for _ in range(n):
        if cur_id == goal:
            return path
        exit_ids = [eid for eid in exits_map.get(cur_id, []) if eid in id_to_idx]
        found = False
        for eid in exit_ids:
            eidx = id_to_idx[eid]
            expected = lane_lengths[cur_idx] + dist_matrix[eidx, dst_idx]
            if abs(expected - dist_matrix[cur_idx, dst_idx]) < 0.01:
                path.append(eid)
                cur_id, cur_idx = eid, eidx
                found = True
                break
        if not found:
            return None

    return path if cur_id == goal else None
