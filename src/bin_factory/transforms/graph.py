import numpy as np
from scipy import sparse as scipy_sparse
from scipy.sparse import csgraph as scipy_csgraph

from bin_factory import puffer_types, schema
from bin_factory.transforms.geometry import arc_length


GRAPH_LANE_TYPES = {puffer_types.LaneType.FREEWAY, puffer_types.LaneType.SURFACE_STREET}
LANE_CHANGE_COST = 25.0  # metres, about the distance a lane change takes; keeps the right lane ahead of a split


def build_lane_distance_matrix(map_elements: dict[int, schema.MapElement]) -> dict | None:
    """Build the all-pairs shortest-path distance matrix over drivable lanes.

    Only SURFACE_STREET and FREEWAY lanes participate. Exit edges cost the source lane's
    length. Lane changes to either neighbour cost ``LANE_CHANGE_COST`` in both directions.

    Return ``{"lane_ids": [...], "distances": float64 NxN array}``, or ``None`` without
    drivable lanes. Rows are sources, columns are destinations, and unreachable pairs are ``+inf``.
    """
    lanes = [(eid, e) for eid, e in map_elements.items() if e.type in GRAPH_LANE_TYPES]
    if not lanes:
        return None

    lane_ids = [eid for eid, _ in lanes]
    id_to_idx = {lid: i for i, lid in enumerate(lane_ids)}
    n = len(lanes)

    # Keyed by (src, dst) so a pair listed twice is not summed by csr_matrix; the cheaper edge wins.
    edges: dict[tuple[int, int], float] = {}
    for eid, element in lanes:
        for exit_id in element.exit_lanes:
            if exit_id in id_to_idx:
                edges[(id_to_idx[eid], id_to_idx[exit_id])] = element.length
    for eid, element in lanes:
        for neighbor_id in [*element.left_neighbor, *element.right_neighbor]:
            if neighbor_id in id_to_idx:
                for pair in ((id_to_idx[eid], id_to_idx[neighbor_id]), (id_to_idx[neighbor_id], id_to_idx[eid])):
                    edges[pair] = min(edges.get(pair, np.inf), LANE_CHANGE_COST)

    rows, cols = zip(*edges, strict=False) if edges else ((), ())
    graph = scipy_sparse.csr_matrix((list(edges.values()), (rows, cols)), shape=(n, n))
    dist_matrix = scipy_csgraph.dijkstra(graph, directed=True).astype(np.float64)

    return {"lane_ids": lane_ids, "distances": dist_matrix}


def compute_lane_lengths(scenario: schema.PufferScenario) -> None:
    """Set each lane's total length and cumulative arc length per point.

    Run after ``geometry.process_polylines`` so lengths match the serialized geometry.
    """
    for elem in scenario.map.values():
        if not elem.is_lane or elem.polyline is None:
            continue
        cum = arc_length(elem.polyline)
        elem.cum_length = cum
        elem.length = float(cum[-1]) if len(cum) else 0.0
