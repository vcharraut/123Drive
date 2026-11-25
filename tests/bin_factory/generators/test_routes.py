import numpy as np
import pytest

from bin_factory import puffer_types, schema
from bin_factory.generators.routes import _is_static, process_agent_routes


def _lane(y=0.0, z=0.0, length=60.0):
    xs = np.linspace(0.0, length, 7)
    return schema.RoadPolyline(
        layer=int(puffer_types.MapLayer.LANE),
        type=int(puffer_types.LaneType.SURFACE_STREET),
        polyline=np.column_stack([xs, np.full(len(xs), y), np.full(len(xs), z)]),
    )


def _path_lane(xy, exits=()):
    xy = np.asarray(xy, dtype=float)
    return schema.RoadPolyline(
        layer=int(puffer_types.MapLayer.LANE),
        type=int(puffer_types.LaneType.SURFACE_STREET),
        polyline=np.column_stack([xy, np.zeros(len(xy))]),
        exit_lanes=list(exits),
    )


def _quarter_turn(start, radius, side):
    """Quarter circle leaving ``start`` along +x and turning left (side=1) or right (side=-1)."""
    angle = np.linspace(0.0, np.pi / 2, 10)
    return np.column_stack([start[0] + radius * np.sin(angle), start[1] + side * radius * (1 - np.cos(angle))])


def _track(xy, z=0.0, kind=puffer_types.AgentType.VEHICLE):
    length = len(xy)
    return schema.Track(
        type=int(kind),
        position=np.column_stack([xy, np.full(length, z)]),
        heading=np.zeros(length),
        velocity=np.zeros((length, 2)),
        valid=np.ones(length, dtype=np.int32),
        length=np.ones(length),
        width=np.ones(length),
        height=np.ones(length),
    )


def _drive(x0, x1, y=0.0, z=0.0, frames=41):
    return _track(np.column_stack([np.linspace(x0, x1, frames), np.full(frames, y)]), z)


def _routes(agents, lanes):
    frames = len(next(iter(agents.values())).valid)
    scenario = schema.PufferScenario(
        agents=agents,
        objects={},
        polylines=lanes,
        metadata=schema.ScenarioMetadata(id="test", dataset="test", scenario_length=frames, dt=0.1),
    )
    process_agent_routes(scenario)
    return scenario.agents


def test_route_follows_the_lane_at_the_vehicle_elevation():
    agents = _routes({0: _drive(1.0, 9.0, z=5.0)}, {1: _lane(z=0.0), 2: _lane(z=5.0)})

    assert agents[0].route == [2]


def test_vehicle_over_a_lane_at_another_elevation_is_offroad():
    lanes = {1: _lane(z=0.0), 2: _lane(y=6.0, z=5.0)}

    assert _routes({1: _drive(1.0, 21.0, z=5.0)}, lanes)[1].route == []
    assert _routes({1: _drive(1.0, 21.0, z=0.0)}, lanes)[1].route == [1]


def test_vehicle_without_lanes_gets_no_route():
    assert _routes({1: _drive(1.0, 21.0)}, {})[1].route == []


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
        [[-0.5, 0.0], [-0.25, 0.25], [0.0, 0.5], [0.25, 0.25], [0.5, 0.0], [0.25, -0.25], [0.0, -0.5], [-0.25, -0.25]]
    )
    xy = np.tile(base, (8, 1))
    xy[:, 0] += np.repeat(np.arange(8) * 0.3, len(base))

    assert _is_static(_track(xy), 0.1)


def test_ego_with_too_little_route_ahead_is_rejected():
    with pytest.raises(ValueError, match="Ego vehicle"):
        _routes({0: _drive(1.0, 3.0, frames=21)}, {1: _lane(length=10.0)})


def test_lane_change_splits_the_route_at_the_logged_switch_point():
    x = np.linspace(5.0, 45.0, 41)
    y = np.clip((x - 20.0) / 10.0, 0.0, 1.0) * 3.5  # moves from lane 1 to lane 2 between x=20 and x=30

    track = _routes({1: _track(np.column_stack([x, y]))}, {1: _lane(), 2: _lane(y=3.5)})[1]

    assert track.route == [1, 2]
    assert track.route_entry_s[0] == 0.0
    assert track.route_exit_s[0] == pytest.approx(track.route_entry_s[1])
    assert 20.0 <= track.route_exit_s[0] <= 30.0
    assert track.route_exit_s[1] == pytest.approx(60.0)


def test_extension_goes_straight_when_a_turn_starts_more_aligned():
    right_turn = np.vstack([[60.0, 0.0], _quarter_turn((61.0, 0.0), 10.0, side=-1)])  # first segment exactly along +x
    lanes = {
        1: _path_lane([[0.0, 0.0], [60.0, 0.0]], exits=(2, 3)),
        2: _path_lane([[60.0, 0.0], [100.0, 0.7]]),  # straight exit, first segment 1 degree off
        3: _path_lane(right_turn),
    }

    assert _routes({1: _drive(5.0, 45.0)}, lanes)[1].route == [1, 2]


def test_log_ending_on_overlapping_connectors_keeps_the_straight_one_unless_turning():
    lanes = {
        1: _path_lane([[0.0, 0.0], [60.0, 0.0]], exits=(2, 3)),
        2: _path_lane([[60.0, 0.0], [100.0, 0.0]]),
        3: _path_lane(_quarter_turn((60.0, 0.0), 20.0, side=1)),  # left turn, 0.1-0.25 m closer to the last points
    }
    xy = np.column_stack([np.linspace(30.0, 64.0, 41), np.full(41, 0.3)])
    straight = _track(xy)
    turning = _track(xy)
    turning.heading[-5:] = np.radians(15.0)

    assert _routes({1: straight}, lanes)[1].route == [1, 2]
    assert _routes({1: turning}, lanes)[1].route == [1, 3]


def test_route_missing_the_last_logged_position_is_cleared_or_rejects_the_ego():
    xy = np.vstack(
        [np.column_stack([np.linspace(5.0, 35.0, 31), np.zeros(31)]), [[35.0 + k, 1.2 * k] for k in range(1, 11)]]
    )  # leaves the lane and ends 12 m from it

    assert _routes({1: _track(xy)}, {1: _lane()})[1].route == []
    with pytest.raises(ValueError, match="Ego vehicle"):
        _routes({0: _track(xy)}, {1: _lane()})


def test_overlap_with_a_background_agent_clears_routes_in_cascade():
    pedestrian = _track(np.full((41, 2), [14.0, 0.0]), kind=puffer_types.AgentType.PEDESTRIAN)
    pedestrian.valid[11:] = 0  # gone before the second vehicle reaches x=14
    agents = {
        1: _drive(10.0, 30.0),  # passes the pedestrian around frame 8
        2: _drive(0.0, 40.0),  # catches the first vehicle at x=20 in frame 20
        3: _drive(0.0, 40.0, y=10.0),
        4: pedestrian,
    }

    agents = _routes(agents, {1: _lane(length=100.0), 2: _lane(y=10.0, length=100.0)})

    assert agents[1].route == []
    assert agents[2].route == []
    assert agents[3].route == [2]


def test_ego_overlapping_another_vehicle_at_start_is_rejected():
    with pytest.raises(ValueError, match="overlaps another agent"):
        _routes({0: _drive(5.0, 25.0), 1: _drive(5.5, 30.0)}, {1: _lane()})
