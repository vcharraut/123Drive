import numpy as np
from scipy import sparse as scipy_sparse
from scipy.sparse import csgraph as scipy_csgraph

from bin_factory import puffer_types
from bin_factory.transforms.geometry import arc_length


GRAPH_LANE_TYPES = {puffer_types.LaneType.FREEWAY, puffer_types.LaneType.SURFACE_STREET}


def build_lane_distance_matrix(map_elements):
    """Build the all-pairs shortest-path distance matrix over drivable lanes.

    Only SURFACE_STREET and FREEWAY lanes participate. Directed edges follow each lane's
    ``exit_lanes`` and are weighted by the source lane's polyline length, so distances
    measure travel along the lane network (row = source, col = destination).

    Arguments:
        map_elements: Scenario map dict ``{id: element}`` as on ``PufferScenario.map``.

    Returns:
        ``None`` if the map has no drivable lanes, else ``{"lane_ids": [int, ...],
        "distances": float64 NxN array}``. Unreachable pairs are ``+inf``.
    """
    lanes = [(eid, e) for eid, e in map_elements.items() if e.type in GRAPH_LANE_TYPES]
    if not lanes:
        return None

    lane_ids = [eid for eid, _ in lanes]
    id_to_idx = {lid: i for i, lid in enumerate(lane_ids)}
    n = len(lanes)

    lane_lengths = [e.length for _, e in lanes]

    rows, cols, weights = [], [], []
    for eid, element in lanes:
        src = id_to_idx[eid]
        for exit_id in element.exit_lanes:
            if exit_id in id_to_idx:
                rows.append(src)
                cols.append(id_to_idx[exit_id])
                weights.append(lane_lengths[src])

    graph = scipy_sparse.csr_matrix((weights, (rows, cols)), shape=(n, n)) if rows else scipy_sparse.csr_matrix((n, n))
    dist_matrix = scipy_csgraph.dijkstra(graph, directed=True).astype(np.float64)

    return {"lane_ids": lane_ids, "distances": dist_matrix}


def compute_lane_lengths(scenario):
    """Annotate each lane element with `length` (scalar) and `cum_length` (per-point arc-length).

    Must run AFTER geometry.process_polylines so values match the serialized polyline.
    """
    for elem in scenario.map.values():
        if not elem.is_lane:
            continue
        cum = arc_length(elem.polyline)
        elem.cum_length = cum
        elem.length = float(cum[-1]) if len(cum) else 0.0
