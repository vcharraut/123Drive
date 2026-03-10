import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


GRAPH_LANE_TYPES = {1, 2}  # FREEWAY, SURFACE_STREET


def compute_lane_length(xyz):
    return float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))


def build_lane_distance_matrix(road_map_elements):
    lanes = [r for r in road_map_elements if r["type"] in GRAPH_LANE_TYPES]
    if not lanes:
        return None

    lane_ids = [r["id"] for r in lanes]
    id_to_idx = {lid: i for i, lid in enumerate(lane_ids)}
    n = len(lanes)

    lane_lengths = np.array([compute_lane_length(np.asarray(r["xyz"])) for r in lanes], dtype=np.float32)

    rows, cols, weights = [], [], []
    for lane in lanes:
        src = id_to_idx[lane["id"]]
        for exit_id in lane.get("exit_lanes", []):
            if exit_id in id_to_idx:
                rows.append(src)
                cols.append(id_to_idx[exit_id])
                weights.append(lane_lengths[src])

    graph = csr_matrix((weights, (rows, cols)), shape=(n, n)) if rows else csr_matrix((n, n))
    dist_matrix = dijkstra(graph, directed=True).astype(np.float32)

    return {"lane_ids": lane_ids, "distances": dist_matrix, "lane_lengths": lane_lengths}
