import numpy as np

from bin_factory.transforms.routes import _extract_lane_centers, build_route_cache, compute_agent_route


def test_route_matching_handles_empty_lane_set():
    cache = build_route_cache({}, _extract_lane_centers({}))

    assert compute_agent_route(
        agent_id=1,
        positions=np.array([[0.0, 0.0, 0.0]]),
        headings=np.zeros(1),
        valid=np.ones(1),
        lengths=np.ones(1),
        widths=np.ones(1),
        is_ego=False,
        route_cache=cache,
    ) == ([], 0)
