import numpy as np
from scipy import sparse as scipy_sparse
from scipy.sparse import csgraph as scipy_csgraph

from bin_factory import types as puffer_types


GRAPH_LANE_TYPES = {puffer_types.LaneType.FREEWAY, puffer_types.LaneType.SURFACE_STREET}


def build_lane_distance_matrix(map_elements):
    lanes = [(eid, e) for eid, e in map_elements.items() if e["type"] in GRAPH_LANE_TYPES]
    if not lanes:
        return None

    lane_ids = [eid for eid, _ in lanes]
    id_to_idx = {lid: i for i, lid in enumerate(lane_ids)}
    n = len(lanes)

    lane_lengths = np.array([_compute_lane_length(np.asarray(e["polyline"])) for _, e in lanes], dtype=np.float32)

    rows, cols, weights = [], [], []
    for eid, element in lanes:
        src = id_to_idx[eid]
        for exit_id in element.get("exit_lanes", []):
            if exit_id in id_to_idx:
                rows.append(src)
                cols.append(id_to_idx[exit_id])
                weights.append(lane_lengths[src])

    graph = scipy_sparse.csr_matrix((weights, (rows, cols)), shape=(n, n)) if rows else scipy_sparse.csr_matrix((n, n))
    dist_matrix = scipy_csgraph.dijkstra(graph, directed=True).astype(np.float32)

    return {"lane_ids": lane_ids, "distances": dist_matrix, "lane_lengths": lane_lengths}


def _compute_lane_length(polyline):
    return float(np.sum(np.linalg.norm(np.diff(polyline, axis=0), axis=1)))
