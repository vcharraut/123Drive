import numpy as np

from bin_factory import puffer_types, schema
from bin_factory.transforms.routes import (
    _extract_lane_centers,
    _is_static,
    build_route_cache,
    compute_agent_route,
)


def _lane(z):
    return schema.MapElement(
        type=int(puffer_types.LaneType.SURFACE_STREET),
        polyline=np.array([[0.0, 0.0, z], [10.0, 0.0, z]]),
    )


def _track(xy):
    positions = np.column_stack([xy, np.zeros(len(xy))])
    length = len(xy)
    return schema.Track(
        type=int(puffer_types.AgentType.VEHICLE),
        position=positions,
        heading=np.zeros(length),
        velocity=np.zeros((length, 2)),
        valid=np.ones(length, dtype=np.int32),
        length=np.ones(length),
        width=np.ones(length),
        height=np.ones(length),
    )


def test_route_matching_rejects_stacked_lane_at_wrong_elevation():
    road_map = {1: _lane(0.0), 2: _lane(5.0)}
    cache = build_route_cache(road_map, _extract_lane_centers(road_map))
    positions = np.array([[1.0, 0.0, 5.0], [9.0, 0.0, 5.0]])

    route, route_gt_len = compute_agent_route(
        agent_id=0,
        positions=positions,
        headings=np.zeros(2),
        valid=np.ones(2),
        lengths=np.ones(2),
        widths=np.ones(2),
        is_ego=True,
        route_cache=cache,
    )

    assert route == [2]
    assert route_gt_len == 1


def test_offroad_gate_ignores_wrong_elevation_lane():
    road_map = {
        1: _lane(0.0),
        2: schema.MapElement(
            type=int(puffer_types.LaneType.SURFACE_STREET),
            polyline=np.array([[0.0, 6.0, 5.0], [10.0, 6.0, 5.0]]),
        ),
    }
    cache = build_route_cache(road_map, _extract_lane_centers(road_map))

    route = compute_agent_route(
        agent_id=1,
        positions=np.array([[1.0, 0.0, 5.0], [9.0, 0.0, 5.0]]),
        headings=np.zeros(2),
        valid=np.ones(2),
        lengths=np.ones(2),
        widths=np.ones(2),
        is_ego=False,
        route_cache=cache,
    )

    assert route == ([], 0)


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


def test_full_track_span_detects_slow_and_looping_motion():
    slow = _track(np.column_stack([np.linspace(0, 2, 21), np.zeros(21)]))
    angle = np.linspace(0, 2 * np.pi, 41)
    loop = _track(np.column_stack([2 * np.cos(angle), 2 * np.sin(angle)]))
    parked = _track(np.column_stack([np.linspace(0, 0.2, 21), np.zeros(21)]))

    assert not _is_static(slow, 0.1)
    assert not _is_static(loop, 0.1)
    assert _is_static(parked, 0.1)


def test_non_directional_detector_drift_is_static():
    base = np.array(
        [
            [-0.5, 0.0],
            [-0.25, 0.25],
            [0.0, 0.5],
            [0.25, 0.25],
            [0.5, 0.0],
            [0.25, -0.25],
            [0.0, -0.5],
            [-0.25, -0.25],
        ]
    )
    xy = np.tile(base, (8, 1))
    xy[:, 0] += np.repeat(np.arange(8) * 0.3, len(base))

    assert _is_static(_track(xy), 0.1)
