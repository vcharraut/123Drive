"""Tests for traffic-light inference after arXiv 2506.07150 (Yan et al.).

The confidence functions, Table 2 and Table 3 are checked against the paper. A hand-built four-way intersection
checks the whole method: vehicles cross on one axis while the other queues, one light is logged wrong, one approach
has no logged light, and the logs are incomplete.
"""

import numpy as np
import pytest

from bin_factory import puffer_types, schema
from bin_factory.generators import traffic_light_inference as tli


GREEN, YELLOW, RED, UNKNOWN = (int(puffer_types.TLState[name]) for name in ("GREEN", "YELLOW", "RED", "UNKNOWN"))
GO, STOP = tli.GO, tli.STOP
FRAMES = 120
HEADINGS = (np.pi / 2, 0.0, -np.pi / 2, np.pi)  # approach k drives north, east, south, west
TURNS = {0: 0, 1: 3, 2: 1}  # connector kind (straight, left, right) -> exit heading index offset


def _unit(theta):
    return np.array([np.cos(theta), np.sin(theta)])


def _right(theta):
    return np.array([np.sin(theta), -np.cos(theta)])


def _lane(points, entry=(), exits=()):
    return schema.RoadPolyline(
        layer=int(puffer_types.MapLayer.LANE),
        type=int(puffer_types.LaneType.SURFACE_STREET),
        polyline=np.column_stack([points, np.zeros(len(points))]),
        entry_lanes=list(entry),
        exit_lanes=list(exits),
    )


def _four_way():
    """Approach lanes 1-4 end 10 m before the centre; connector 100 + 10 k + kind (0 S, 1 L, 2 R) joins exit 11 + m."""
    lanes = {}
    for k, theta in enumerate(HEADINGS):
        u, r = _unit(theta), 1.75 * _right(theta)
        lanes[1 + k] = _lane(np.linspace(-70 * u + r, -10 * u + r, 61), exits=[100 + 10 * k + kind for kind in TURNS])
        lanes[11 + k] = _lane(np.linspace(10 * u + r, 70 * u + r, 61), entry=[])
    for k, theta in enumerate(HEADINGS):
        start = -10 * _unit(theta) + 1.75 * _right(theta)
        for kind, offset in TURNS.items():
            m = (k + offset) % 4
            end = 10 * _unit(HEADINGS[m]) + 1.75 * _right(HEADINGS[m])
            matrix = np.column_stack([_unit(theta), _unit(HEADINGS[m])])
            if abs(np.linalg.det(matrix)) < 1e-9:
                control = (start + end) / 2
            else:
                a, _ = np.linalg.solve(matrix, end - start)
                control = start + a * _unit(theta)
            t = np.linspace(0, 1, 21)[:, None]
            curve = (1 - t) ** 2 * start + 2 * t * (1 - t) * control + t**2 * end
            lanes[100 + 10 * k + kind] = _lane(curve, entry=[1 + k], exits=[11 + m])
            lanes[11 + m].entry_lanes.append(100 + 10 * k + kind)
    return lanes


def _vehicle(xy, heading):
    speed = np.linalg.norm(np.gradient(xy, 0.1, axis=0), axis=1)
    return schema.Track(
        type=int(puffer_types.AgentType.VEHICLE),
        position=np.column_stack([xy, np.zeros(FRAMES)]),
        heading=np.full(FRAMES, heading),
        velocity=np.column_stack([speed * np.cos(heading), speed * np.sin(heading)]),
        valid=np.ones(FRAMES, dtype=np.int32),
        length=np.full(FRAMES, 4.5),
        width=np.full(FRAMES, 1.8),
        height=np.full(FRAMES, 1.5),
    )


def _along(k, distance):
    """Positions ``distance`` metres along approach k's path, measured from its stop line."""
    theta = HEADINGS[k]
    return (-10 + np.asarray(distance))[:, None] * _unit(theta) + 1.75 * _right(theta)


def _light(frames):
    states = [puffer_types.TLState.UNKNOWN] * FRAMES
    for start, end, state in frames:
        states[start:end] = [state] * (end - start)
    return states


@pytest.fixture(scope="module")
def inferred():
    """Approaches 0 and 2 cross at 10 m/s at frames 20 and 30; approaches 1 and 3 queue 4 m before the line and leave
    at frame 80, crossing at frame 100. Approach 0 is logged green until frame 40, approach 1 red until frame 60,
    approach 3 wrongly green until frame 40, and approach 2 not at all."""
    t = np.arange(FRAMES)
    leave = np.where(t < 80, 0.0, 0.01 * (t - 80) ** 2)  # 2 m/s^2 from frame 80
    agents = {
        0: _vehicle(_along(0, -20 + t), HEADINGS[0]),
        1: _vehicle(_along(2, -30 + t), HEADINGS[2]),
        2: _vehicle(_along(1, -4 + leave), HEADINGS[1]),
        3: _vehicle(_along(3, -4 + leave), HEADINGS[3]),
    }
    lanes = _four_way()
    scenario = schema.PufferScenario(
        agents=agents,
        objects={},
        polylines=lanes,
        metadata=schema.ScenarioMetadata(id="four-way", dataset="test", scenario_length=FRAMES, dt=0.1),
    )
    logged = {
        100: _light([(0, 40, puffer_types.TLState.GREEN)]),
        110: _light([(0, 60, puffer_types.TLState.RED)]),
        130: _light([(0, 40, puffer_types.TLState.GREEN)]),
    }
    extras = schema.ExtractionExtras(
        traffic_lights={lane: schema.TrafficLightTrack(np.zeros(3), states, lane) for lane, states in logged.items()}
    )
    tli.infer_traffic_lights(scenario, extras)
    return {track.controlled_lane: np.array([int(s) for s in track.states]) for track in extras.traffic_lights.values()}


def test_crossing_vehicles_see_green_and_queues_red(inferred):
    assert inferred[100][20] == GREEN  # approach 0 crosses at frame 20
    assert inferred[120][30] == GREEN  # approach 2, never logged, crosses at frame 30
    assert inferred[110][100] == GREEN  # the queues leave and cross at frame 100
    assert inferred[130][100] == GREEN
    assert (inferred[130][:40] == RED).all()  # the logged green on approach 3 contradicts its queue


def test_every_signalized_connector_gets_a_complete_conflict_free_light(inferred):
    assert set(inferred) == {100 + 10 * k + kind for k in range(4) for kind in TURNS}
    go = {lane: np.isin(states, (GREEN, YELLOW)) for lane, states in inferred.items()}
    for north_south in (100, 101, 120, 121):
        for east_west in (110, 111, 130, 131):
            assert not (go[north_south] & go[east_west]).any()
    for lane, states in inferred.items():
        free = lane % 10 == 2
        assert (states != RED).all() if free else (states != UNKNOWN).all()


def test_green_ends_with_two_seconds_of_yellow(inferred):
    states = inferred[100]
    switch = int(np.flatnonzero(states == RED)[0])
    assert (states[switch - 20 : switch] == YELLOW).all()
    assert states[switch - 21] == GREEN


def test_confidence_functions_follow_appendix_a():
    d = np.array([10.0, -2.0, -2.0, 22.5, 30.5, -9.0])
    a = np.array([-1.0, -1.0, 1.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(tli._f(d, a), [1, 0, 1, 0.25, 0, 0])
    # g0 is 15 m standing still, 6 m at 6 m/s and 25 m at 14 m/s (Fig. 14b)
    d = np.array([15.0, 22.5, 31.0, 6.0, 9.0, 25.0, 37.5, -9.0])
    v = np.array([0.0, 0.0, 0.0, 6.0, 6.0, 14.0, 14.0, 0.0])
    np.testing.assert_allclose(tli._g(d, v), [1, 0.25, 0, 1, 0.25, 1, 0.25, 0])


def test_imputation_follows_table_2():
    logged = np.array([0, 0, GO, GO, GO, GO])
    estimated = np.array([0, STOP, 0, GO, STOP, STOP])
    confidence = np.array([0.0, 0.7, 0.0, 2.0, 3.0, 0.5])
    imputed, weight = tli._impute(logged, estimated, confidence)
    np.testing.assert_array_equal(imputed, [0, STOP, GO, GO, STOP, GO])
    np.testing.assert_allclose(weight, [0, 0.7, tli.W_SMALL, tli.W_BIG, 3.0, 0])


def test_four_way_configurations_are_table_3():
    approaches = [0, 0, 1, 1, 2, 2, 3, 3]  # north, east, south, west; a left and a straight-right movement each
    turns = [frozenset({"L"}), frozenset({"S", "R"})] * 4
    movements = [tli._Movement(a, t, [], [], {}) for a, t in zip(approaches, turns, strict=True)]
    configurations = tli._feasible(movements, [(0, 2), (1, 3)], {})
    # Table 3 rows over m1..m8 = north L, south S, east L, west S, south L, north S, west L, east S
    index = {"m1": 0, "m6": 1, "m3": 2, "m8": 3, "m5": 4, "m2": 5, "m7": 6, "m4": 7}
    table_3 = [
        "m1 m5", "m1 m6", "m2 m5", "m2 m6", "m3 m7", "m3 m8", "m4 m7", "m4 m8", "m1 m2 m5 m6", "m3 m4 m7 m8"
    ]  # fmt: skip
    expected = {frozenset(index[m] for m in row.split()) for row in table_3}
    assert {frozenset(np.flatnonzero(row == GO).tolist()) for row in configurations} == expected
    assert len(configurations) == 10


def test_logged_yellow_is_kept_until_the_green_ends():
    green_then_red = np.r_[[GREEN] * 30, [YELLOW] * 20, [RED] * 50]  # after Eq. 14 with t_yellow 2 s
    shown = np.r_[[-1] * 20, [1] * 30, [0] * 50]  # logged green, then a 3 s yellow
    flashing = np.full(100, GREEN)
    blip = np.r_[[0] * 10, [1] * 5, [-1] * 5, [0] * 80]  # a logged yellow that a logged green follows
    kept = tli._keep_logged_yellow(np.array([green_then_red, flashing, flashing]), np.array([shown, [1] * 100, blip]))
    np.testing.assert_array_equal(kept[0], np.r_[[GREEN] * 20, [YELLOW] * 30, [RED] * 50])
    np.testing.assert_array_equal(kept[1], [YELLOW] * 100)
    np.testing.assert_array_equal(kept[2], flashing)


def test_short_states_are_smoothed_without_undoing_evidenced_switches():
    evidenced = np.r_[[STOP] * 50, [GO] * 100]  # green from frame 50, with strong evidence
    unobserved = np.r_[[GO] * 50, [STOP] * 25, [GO] * 10, [STOP] * 65]  # follows configuration changes
    imputed = np.array([evidenced, np.zeros(150, dtype=int)])
    weight = np.array([np.full(150, tli.W_BIG), np.zeros(150)])
    smoothed = tli._smooth(np.array([evidenced, unobserved]), imputed, weight, 30)
    np.testing.assert_array_equal(smoothed[0], evidenced)
    np.testing.assert_array_equal(smoothed[1], np.r_[[GO] * 50, [STOP] * 100])

    island = np.array([np.r_[[STOP] * 40, [GO] * 20, [STOP] * 40]])  # 2 s of green between two reds
    np.testing.assert_array_equal(tli._smooth(island, island, np.ones((1, 100)), 30), [[STOP] * 100])


def test_two_crossing_approaches_are_never_green_together():
    lanes = {lane_id: lane for lane_id, lane in _four_way().items() if lane_id not in (3, 4) and lane_id // 10 < 12}
    green = _light([(0, FRAMES, puffer_types.TLState.GREEN)])
    extras = schema.ExtractionExtras(
        traffic_lights={lane: schema.TrafficLightTrack(np.zeros(3), list(green), lane) for lane in (100, 110)}
    )
    north = _vehicle(_along(0, -20 + np.arange(FRAMES)), HEADINGS[0])  # crosses at 10 m/s at frame 20
    scenario = schema.PufferScenario(
        agents={0: north},
        objects={},
        polylines=lanes,
        metadata=schema.ScenarioMetadata(id="crossing", dataset="test", scenario_length=FRAMES, dt=0.1),
    )
    tli.infer_traffic_lights(scenario, extras)
    states = {
        track.controlled_lane: np.array([int(s) for s in track.states]) for track in extras.traffic_lights.values()
    }
    assert states[100][20] == GREEN
    assert not (np.isin(states[100], (GREEN, YELLOW)) & np.isin(states[110], (GREEN, YELLOW))).any()


def test_connector_joined_by_two_lanes_belongs_to_the_approach_it_continues():
    lanes = _four_way()
    start = lanes[100].polyline[0, :2]
    ramp = np.radians(40.0)  # joins the northbound straight connector from 50 degrees off its heading
    lanes[50] = _lane(np.linspace(start - 40 * _unit(ramp), start, 41), exits=[100])
    lanes[100].entry_lanes = [50, 1]
    connectors = sorted(lane_id for lane_id in lanes if lane_id >= 100)
    movements, _ = tli._movements(connectors, set(connectors), lanes, lanes, [], False)
    assert len({movement.approach for movement in movements}) == 4
    assert any({100, 101} <= set(movement.connectors) for movement in movements)


def _curve(start, headings, length):
    """Points along steps of equal length in the directions ``headings`` (degrees), ``length`` metres in all."""
    headings = np.radians(headings)
    steps = length / len(headings) * np.column_stack([np.cos(headings), np.sin(headings)])
    return np.vstack([start, start + np.cumsum(steps, axis=0)])


def test_turn_inside_the_junction_stands_when_the_road_bends_back_after_it():
    slip = _curve(np.zeros(2), np.linspace(0, -35, 16), 15.0)  # turns 35 degrees to the right
    road = _curve(slip[-1], np.linspace(-35, -29, 16), 20.0)  # bends 6 degrees back within 20 m
    lanes = {
        1: _lane(np.linspace([-30.0, 0.0], [0.0, 0.0], 31), exits=[2]),
        2: _lane(slip, entry=[1], exits=[3]),
        3: _lane(road, entry=[2]),
    }
    assert tli._turn_class(np.vstack([slip, road[1:]]), False) == "S"
    movements, _ = tli._movements([2], {2}, lanes, lanes, [], False)
    assert movements[0].turns == {"R"}
    assert movements[0].free == [2]


def test_logged_yellow_never_follows_a_red_directly():
    states = np.array([np.r_[[RED] * 20, [GREEN] * 30, [RED] * 50]])
    shown = np.array([np.r_[[0] * 20, [1] * 30, [0] * 50]])  # yellow logged from the first green frame on
    kept = tli._keep_logged_yellow(states, shown)
    np.testing.assert_array_equal(kept[0], np.r_[[RED] * 20, [GREEN], [YELLOW] * 29, [RED] * 50])


def test_free_turn_is_green_between_greens_and_unknown_outside():
    cycle = np.r_[[RED] * 10, [GREEN] * 10, [YELLOW] * 5, [RED] * 10, [GREEN] * 10, [YELLOW] * 5, [RED] * 10]
    lanes = {1: _lane(np.zeros((2, 2))), 2: _lane(np.zeros((2, 2)))}
    extras = schema.ExtractionExtras()
    tli._write([tli._Movement(0, frozenset("SR"), [1, 2], [2], {})], np.array([cycle]), lanes, {}, extras)
    np.testing.assert_array_equal([int(s) for s in extras.traffic_lights[1].states], cycle)
    np.testing.assert_array_equal(
        [int(s) for s in extras.traffic_lights[2].states],
        np.r_[[UNKNOWN] * 10, [GREEN] * 35, [YELLOW] * 5, [UNKNOWN] * 10],
    )
