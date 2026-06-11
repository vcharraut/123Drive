"""Tests for the traffic-light interpolation module (Yan et al. 2025 port).

Covers the pure geometry/graph helpers, the kinematic weight functions, the
state-derivation / feasible-state / scoring logic, vehicle-to-lane assignment,
sequence smoothing / yellow insertion, signalized-intersection detection, and the
two regression cases for the bugs fixed against the upstream reference:
  * the trajectory-metric window is `[curr-dt, curr+dt)` (exclusive end), and
  * acceleration is the 0.5 s difference anchored at `tt-5` (walking older).
It also locks in the two intentional deviations from upstream:
  * `_g`'s plateau runs `[-12 m, D2]` (consistent with `_f`), and
  * `_neighbor_type` returns "real" for non-"low" parallel neighbours.
"""

import numpy as np
import pytest

from bin_factory import puffer_types, schema
from bin_factory.transforms.traffic_light_interpolation import (
    _TLS,
    _angle_of_headings,
    _angle_of_two_vectors,
    _ApproachingLane,
    _as_xyz_array,
    _assign_vehicle_states_to_lanes,
    _classify_direction,
    _copy_state,
    _Direction,
    _distance,
    _from_puffer_tls,
    _group_lanes_into_ways,
    _group_vectors_by_angles,
    _InJunctionLane,
    _neighbor_type,
    _real_neighbor_type,
    _TLSGenerator,
    _to_puffer_tls,
    _TrafficLightInterpolator,
    _two_lines_parallel,
    _UnionFind,
    _vector_heading,
    _VehicleState,
    interpolate_traffic_lights,
)


# ── Shared geometry: lane shapes that classify to a known turn direction ──────────────────

S_SHAPE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
L_SHAPE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
R_SHAPE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, -1.0, 0.0], [2.0, -2.0, 0.0]])

S = (_Direction.S,)
L = (_Direction.L,)
R = (_Direction.R,)
SR = (_Direction.S, _Direction.R)


def _inj(shape, *, record_tls=None, record_vehs=None, horizon=40, id=0):
    return _InJunctionLane(
        id=id,
        shape=shape,
        record_tls=[_TLS.ABSENT] * horizon if record_tls is None else record_tls,
        record_vehs=[{} for _ in range(horizon)] if record_vehs is None else record_vehs,
    )


def _approach(injs, *, shape=None, record_vehs=None, horizon=40, id=0):
    return _ApproachingLane(
        id=id,
        shape=S_SHAPE if shape is None else shape,
        record_vehs=[{} for _ in range(horizon)] if record_vehs is None else record_vehs,
        injunction_lanes=injs,
    )


def _straight_lane_shape(n=11):
    return np.column_stack([np.arange(float(n)), np.zeros(n), np.zeros(n)])


# ── _distance ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([0, 0, 0], [3, 4, 0], 5.0),
        ([0, 0, 0], [0, 0, 0], 0.0),
        ([1, 2, 2], [0, 0, 0], 3.0),
        ([0, 0], [3, 4], 5.0),  # 2-D inputs
    ],
)
def test_distance(a, b, expected):
    assert _distance(np.array(a, float), np.array(b, float)) == pytest.approx(expected)


# ── _two_lines_parallel ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("v2", "parallel"),
    [
        ([[0, 1], [1, 1]], True),  # same +x direction, offset
        ([[0, 0], [10, 1]], True),  # ~5.7 deg < 15
        ([[0, 0], [0, 1]], False),  # perpendicular
        ([[0, 0], [-1, 0]], False),  # opposite direction (180 deg)
        ([[0, 0], [0, 0]], False),  # zero-length -> not parallel
    ],
)
def test_two_lines_parallel(v2, parallel):
    line1 = np.array([[0.0, 0.0], [1.0, 0.0]])  # +x
    assert _two_lines_parallel(line1, np.array(v2, float)) is parallel


# ── _angle_of_two_vectors / _vector_heading / _angle_of_headings ────────────────────────────


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1, 0], [1, 0], 0.0),
        ([1, 0], [-1, 0], np.pi),
        ([1, 0], [0, 1], np.pi / 2),
        ([0, 0], [1, 0], np.pi),  # zero vector -> pi
    ],
)
def test_angle_of_two_vectors(left, right, expected):
    assert _angle_of_two_vectors(np.array(left, float), np.array(right, float)) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("vec", "heading"),
    [([1, 0], 0.0), ([0, 1], np.pi / 2), ([-1, 0], np.pi), ([0, -1], -np.pi / 2)],
)
def test_vector_heading(vec, heading):
    assert _vector_heading(np.array(vec, float)) == pytest.approx(heading)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (0.0, 0.0, 0.0),
        (0.0, np.pi, np.pi),
        (0.0, 3 * np.pi / 2, np.pi / 2),  # wraps the short way
        (0.1, -0.1, 0.2),
    ],
)
def test_angle_of_headings(a, b, expected):
    assert _angle_of_headings(a, b) == pytest.approx(expected)


# ── _classify_direction ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("shape", "direction"),
    [
        (S_SHAPE, _Direction.S),
        (L_SHAPE, _Direction.L),
        (R_SHAPE, _Direction.R),
        (np.array([[0, 0], [1, 0.05], [2, 0.0]]), _Direction.S),  # tiny wiggle stays straight
    ],
)
def test_classify_direction(shape, direction):
    assert _classify_direction(np.asarray(shape, float)[:, :2]) == direction


# ── _neighbor_type (locks the "trust py123d" decision: parallel non-low -> "real") ──────────


@pytest.mark.parametrize(
    ("p2", "expected"),
    [
        ([[0, 3.5, 0], [10, 3.5, 0]], "real"),  # parallel, both mid -> real (was "other")
        ([[0, 0.5, 0], [10, 0.8, 0]], "bifurcated-parallel"),  # parallel, start low
        ([[0, 2, 0], [10, 0.5, 0]], "merged-parallel"),  # parallel, end low
        ([[0, 0, 0], [0, 10, 0]], "bifurcated"),  # not parallel, start low
        ([[0, -10, 0], [10, 0, 0]], "merged"),  # not parallel, end low
        ([[20, 20, 0], [20, 30, 0]], "other"),  # not parallel, both far
    ],
)
def test_neighbor_type(p2, expected):
    p1 = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    assert _neighbor_type(p1, np.array(p2, float)) == expected


# ── _real_neighbor_type ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("p1", "p2", "expected"),
    [
        ([[0, 0, 0], [10, 0, 0]], [[0, 3, 0], [10, 3, 0]], "complete"),
        ([[0, 0, 0], [20, 0, 0]], [[0, 1, 0], [8, 1, 0]], "side-start"),
        ([[0, 0, 0], [20, 0, 0]], [[12, 1, 0], [20, 1, 0]], "side-end"),
        ([[0, 0, 0], [10, 0, 0]], [[50, 50, 0], [60, 50, 0]], "other"),
    ],
)
def test_real_neighbor_type(p1, p2, expected):
    assert _real_neighbor_type(np.array(p1, float), np.array(p2, float)) == expected


# ── _as_xyz_array ───────────────────────────────────────────────────────────────────────────


def test_as_xyz_array_pads_2d():
    np.testing.assert_array_equal(_as_xyz_array([[1, 2], [3, 4]]), [[1, 2, 0], [3, 4, 0]])


def test_as_xyz_array_truncates_to_three_cols():
    np.testing.assert_array_equal(_as_xyz_array([[1, 2, 3, 9], [4, 5, 6, 9]]), [[1, 2, 3], [4, 5, 6]])


@pytest.mark.parametrize("bad", [[], [1, 2, 3]])  # empty, or 1-D
def test_as_xyz_array_degenerate(bad):
    assert _as_xyz_array(bad).shape == (0, 3)


# ── TLS <-> puffer conversions ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("puffer", "simple"),
    [
        (puffer_types.TLState.GREEN, _TLS.GREEN),
        (puffer_types.TLState.YELLOW, _TLS.YELLOW),
        (puffer_types.TLState.RED, _TLS.RED),
        (puffer_types.TLState.UNKNOWN, _TLS.UNKNOWN),
        (puffer_types.TLState.OFF, _TLS.UNKNOWN),  # unmapped -> unknown
    ],
)
def test_from_puffer_tls(puffer, simple):
    assert _from_puffer_tls(puffer) == simple


@pytest.mark.parametrize("simple", [_TLS.GREEN, _TLS.YELLOW, _TLS.RED, _TLS.UNKNOWN])
def test_tls_roundtrip(simple):
    assert _from_puffer_tls(_to_puffer_tls(simple)) == simple


def test_to_puffer_tls_absent_is_unknown():
    assert _to_puffer_tls(_TLS.ABSENT) == puffer_types.TLState.UNKNOWN


# ── _copy_state ─────────────────────────────────────────────────────────────────────────────


def test_copy_state_is_independent():
    state = [{S: _TLS.GREEN}, {L: _TLS.RED}]
    clone = _copy_state(state)
    clone[0][S] = _TLS.RED
    assert state[0][S] == _TLS.GREEN  # original untouched


# ── _UnionFind ──────────────────────────────────────────────────────────────────────────────


def test_unionfind_groups():
    uf = _UnionFind([0, 1, 2, 3])
    uf.union(0, 1)
    uf.union(2, 3)
    assert {frozenset(g) for g in uf.groups()} == {frozenset({0, 1}), frozenset({2, 3})}


def test_unionfind_transitive_merge():
    uf = _UnionFind(range(4))
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(2, 3)
    assert {frozenset(g) for g in uf.groups()} == {frozenset({0, 1, 2, 3})}


def test_unionfind_singletons():
    uf = _UnionFind([5, 6, 7])
    assert {frozenset(g) for g in uf.groups()} == {frozenset({5}), frozenset({6}), frozenset({7})}


# ── _group_vectors_by_angles ────────────────────────────────────────────────────────────────


def test_group_vectors_by_angles_cardinals_split():
    vectors = [np.array(v, float) for v in ([1, 0], [0, 1], [-1, 0], [0, -1])]
    groups = {frozenset(g) for g in _group_vectors_by_angles(vectors)}
    assert groups == {frozenset({0}), frozenset({1}), frozenset({2}), frozenset({3})}


def test_group_vectors_by_angles_merges_near_parallel():
    vectors = [np.array([1.0, 0.0]), np.array([np.cos(0.1), np.sin(0.1)]), np.array([0.0, 1.0])]
    groups = {frozenset(g) for g in _group_vectors_by_angles(vectors)}
    assert groups == {frozenset({0, 1}), frozenset({2})}


# ── _group_lanes_into_ways ──────────────────────────────────────────────────────────────────


def _heading_lane(start, end):
    return _ApproachingLane(id=0, shape=np.array([start, end], float), record_vehs=[], injunction_lanes=[])


def test_group_lanes_into_ways_4way_sorted_by_heading():
    lanes = [
        _heading_lane([-2, 0, 0], [0, 0, 0]),  # east, heading 0
        _heading_lane([0, -2, 0], [0, 0, 0]),  # north, heading +pi/2
        _heading_lane([2, 0, 0], [0, 0, 0]),  # west, heading pi
        _heading_lane([0, 2, 0], [0, 0, 0]),  # south, heading -pi/2
    ]
    ways = _group_lanes_into_ways(lanes)
    assert len(ways) == 4
    headings = [_vector_heading(w[0].shape[-1, :2] - w[0].shape[0, :2]) for w in ways]
    assert headings == sorted(headings)  # ascending heading order


def test_group_lanes_into_ways_3way_orders_opposites_first():
    lanes = [
        _heading_lane([-2, 0, 0], [0, 0, 0]),  # east
        _heading_lane([0, -2, 0], [0, 0, 0]),  # north
        _heading_lane([2, 0, 0], [0, 0, 0]),  # west
    ]
    ways = _group_lanes_into_ways(lanes)
    assert len(ways) == 3
    v0 = ways[0][0].shape[-1, :2] - ways[0][0].shape[0, :2]
    v1 = ways[1][0].shape[-1, :2] - ways[1][0].shape[0, :2]
    assert _angle_of_two_vectors(v0, v1) == pytest.approx(np.pi)  # way0 & way1 are opposite


# ── _TLSGenerator._f (acceleration relevance weight) ────────────────────────────────────────


@pytest.mark.parametrize(
    ("index", "acc", "expected"),
    [
        (0, 1.0, 1.0),  # at stop line
        (30, 1.0, 1.0),  # d=15, edge of plateau
        (31, 1.0, (15.5 - 30) ** 2 / 225),  # decaying
        (60, 1.0, 0.0),  # d=30, far before
        (-20, 1.0, 0.0),  # d=-10 < -8, far past
        (-16, 1.0, 1.0),  # d=-8 exactly -> still relevant
        (-17, 1.0, 0.0),  # d=-8.5 < -8
        (-2, -1.0, 0.0),  # just past line & decelerating -> irrelevant
        (-2, 1.0, 1.0),  # just past line & accelerating -> relevant
    ],
)
def test_relevance_weight_f(index, acc, expected):
    assert _TLSGenerator._f(index, acc) == pytest.approx(expected)


# ── _TLSGenerator._g (speed relevance weight) — locks corrected [-12, D2] plateau ───────────


@pytest.mark.parametrize(
    ("index", "speed", "expected"),
    [
        # speed 0 -> D2 = 15
        (0, 0.0, 1.0),
        (30, 0.0, 1.0),
        (31, 0.0, (15.5 - 30) ** 2 / 225),
        (60, 0.0, 0.0),
        (-24, 0.0, 1.0),  # d=-12 exactly
        (-25, 0.0, 0.0),  # d=-12.5 < -12
        # speed 6 -> D2 = 6: these are the cases where upstream's typo'd `g` diverges
        (0, 6.0, 1.0),  # upstream would give 4.0
        (8, 6.0, 1.0),  # d=4 inside plateau; upstream ~1.78
        (12, 6.0, 1.0),  # d=6 == D2
        (16, 6.0, (8 - 12) ** 2 / 36),  # d=8 decaying
        (24, 6.0, 0.0),  # d=12 == 2*D2
        (-10, 6.0, 1.0),  # just past stop line
        # speed 20 -> D2 = 23
        (0, 20.0, 1.0),
        (46, 20.0, 1.0),
        (60, 20.0, (30 - 46) ** 2 / 23**2),
        (92, 20.0, 0.0),
        # speed 30 -> D2 capped at 30
        (60, 30.0, 1.0),
        (90, 30.0, 0.25),
        (120, 30.0, 0.0),
    ],
)
def test_relevance_weight_g(index, speed, expected):
    assert _TLSGenerator._g(index, speed) == pytest.approx(expected)


def test_g_plateau_is_flat_unlike_upstream():
    # Whole plateau [-12, D2] is exactly 1.0 (upstream peaks >1 near the stop line for v~6).
    gen_g = _TLSGenerator._g
    assert all(gen_g(idx, 6.0) == 1.0 for idx in range(-20, 13))  # d in [-10, 6]


# ── container / feasible-state generation ───────────────────────────────────────────────────


def test_gen_state_container_unions_directions_per_lane():
    # one lane carries L+S+R -> a single combined phase (L,S,R)
    approach = _approach([_inj(L_SHAPE, id=0), _inj(S_SHAPE, id=1), _inj(R_SHAPE, id=2)])
    container = _TLSGenerator(40)._gen_state_container([[approach]])
    assert set(container[0]) == {(_Direction.L, _Direction.S, _Direction.R)}


def test_gen_state_container_splits_phases_across_lanes():
    # lane A: L only; lane B: S+R  ->  phases (L,) and (S,R)
    lane_a = _approach([_inj(L_SHAPE, id=0)], id=0)
    lane_b = _approach([_inj(S_SHAPE, id=1), _inj(R_SHAPE, id=2)], id=1)
    container = _TLSGenerator(40)._gen_state_container([[lane_a, lane_b]])
    assert set(container[0]) == {L, SR}


def _single_green(template, way):
    return [dict.fromkeys(template[i], _TLS.GREEN if i == way else _TLS.RED) for i in range(len(template))]


def test_feasible_states_4way_single_greens_present():
    gen = _TLSGenerator(40)
    gen.container_template = [{S: None} for _ in range(4)]
    feasible = gen._get_feasible_states()
    assert len(feasible) == 10
    for way in range(4):
        assert _single_green(gen.container_template, way) in feasible


def test_feasible_states_4way_left_straight_split():
    gen = _TLSGenerator(40)
    gen.container_template = [{L: None, SR: None} for _ in range(4)]
    feasible = gen._get_feasible_states()

    left_only = [{L: _TLS.GREEN, SR: _TLS.RED} if i in (0, 2) else {L: _TLS.RED, SR: _TLS.RED} for i in range(4)]
    straight_only = [{L: _TLS.RED, SR: _TLS.GREEN} if i in (0, 2) else {L: _TLS.RED, SR: _TLS.RED} for i in range(4)]
    assert left_only in feasible
    assert straight_only in feasible


def test_feasible_states_3way_counts_and_singles():
    gen = _TLSGenerator(40)
    gen.container_template = [{S: None} for _ in range(3)]
    feasible = gen._get_feasible_states()
    assert len(feasible) == 6
    for way in range(3):
        assert _single_green(gen.container_template, way) in feasible


def test_can_split_left():
    gen = _TLSGenerator(40)
    gen.container_template = [{L: None, S: None}]  # separate phases
    assert gen._can_split_left([0]) is True
    gen.container_template = [{(_Direction.L, _Direction.S): None}]  # combined phase
    assert gen._can_split_left([0]) is False


def test_make_candidate():
    gen = _TLSGenerator(40)
    gen.container_template = [{S: None}, {S: None}]
    candidate = gen._make_candidate(lambda i, _p: _TLS.GREEN if i == 0 else _TLS.RED)
    assert candidate == [{S: _TLS.GREEN}, {S: _TLS.RED}]


# ── _derive_imputed_state (the raw/estimated merge table) ───────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "est", "conf", "exp_state", "exp_weight"),
    [
        (None, None, 0.0, None, 0.0),  # nothing observed
        (None, _TLS.GREEN, 2.0, _TLS.GREEN, 2.0),  # estimated only
        (_TLS.RED, None, 5.0, _TLS.RED, 0.1),  # raw only -> w_small
        (_TLS.GREEN, _TLS.GREEN, 3.0, _TLS.GREEN, 100.0),  # agreement -> w_big
        (_TLS.RED, _TLS.GREEN, 0.9, _TLS.GREEN, 0.9),  # disagree, confident estimate wins
        (_TLS.RED, _TLS.GREEN, 0.5, _TLS.RED, 0.0),  # disagree, weak estimate -> keep raw
    ],
)
def test_derive_imputed_state(raw, est, conf, exp_state, exp_weight):
    gen = _TLSGenerator(40)
    gen.container_template = [{S: None}]
    imputed, weight = gen._derive_imputed_state([{S: raw}], [{S: est}], [{S: conf}])
    assert imputed[0][S] == exp_state
    assert weight[0][S] == pytest.approx(exp_weight)


# ── _derive_raw_state (most-recent observed detection in a 10-step lookback) ─────────────────


def _raw_for(record_tls, curr_step, horizon=40):
    tls = [_TLS.ABSENT] * horizon
    for idx, state in record_tls.items():
        tls[idx] = state
    approach = _approach([_inj(S_SHAPE, record_tls=tls)])
    gen = _TLSGenerator(horizon)
    gen.container_template = gen._gen_state_container([[approach]])
    return gen._derive_raw_state([[approach]], curr_step)[0][S]


@pytest.mark.parametrize(
    ("record_tls", "curr", "expected"),
    [
        ({5: _TLS.GREEN}, 5, _TLS.GREEN),
        ({5: _TLS.RED}, 5, _TLS.RED),
        ({5: _TLS.YELLOW}, 5, _TLS.GREEN),  # yellow generalised to green
        ({5: _TLS.GREEN}, 4, None),  # detection is in the future
        ({5: _TLS.GREEN}, 16, None),  # detection older than 10-step lookback
        ({5: _TLS.GREEN}, 15, _TLS.GREEN),  # detection exactly 10 steps back
        ({3: _TLS.RED, 5: _TLS.GREEN}, 5, _TLS.GREEN),  # most recent wins
        ({}, 5, None),  # nothing observed
    ],
)
def test_derive_raw_state(record_tls, curr, expected):
    assert _raw_for(record_tls, curr) == expected


# ── _derive_estimated_state (kinematic inference) ───────────────────────────────────────────


def test_estimated_state_must_green_from_vehicle_past_stop_line():
    horizon, curr = 40, 20
    inj_vehs = [{} for _ in range(horizon)]
    inj_vehs[curr] = {1: _VehicleState(3, 5.0, 0.0)}  # just past the stop line, moving
    approach = _approach([_inj(S_SHAPE, record_vehs=inj_vehs)])
    gen = _TLSGenerator(horizon)
    gen.container_template = gen._gen_state_container([[approach]])
    est, conf = gen._derive_estimated_state([[approach]], curr)
    assert est[0][S] == _TLS.GREEN
    assert conf[0][S] == gen.w_big


def test_estimated_state_green_from_fast_approaching_vehicles():
    horizon, curr = 40, 20
    vehs = [{} for _ in range(horizon)]
    vehs[curr] = {1: _VehicleState(10, 5.0, 0.0), 2: _VehicleState(10, 5.0, 0.0)}  # fast, at stop line
    approach = _approach([_inj(S_SHAPE)], shape=_straight_lane_shape(), record_vehs=vehs)
    gen = _TLSGenerator(horizon)
    gen.container_template = gen._gen_state_container([[approach]])
    est, conf = gen._derive_estimated_state([[approach]], curr)
    assert est[0][S] == _TLS.GREEN
    assert conf[0][S] == pytest.approx(np.log1p(2.0), rel=1e-3)


def test_estimated_state_red_from_stopped_vehicles():
    horizon, curr = 40, 20
    vehs = [{} for _ in range(horizon)]
    vehs[curr] = {1: _VehicleState(10, 0.5, 0.0), 2: _VehicleState(10, 0.5, 0.0)}  # crawling at stop line
    approach = _approach([_inj(S_SHAPE)], shape=_straight_lane_shape(), record_vehs=vehs)
    gen = _TLSGenerator(horizon)
    gen.container_template = gen._gen_state_container([[approach]])
    est, _ = gen._derive_estimated_state([[approach]], curr)
    assert est[0][S] == _TLS.RED


def test_estimated_state_none_without_vehicles():
    approach = _approach([_inj(S_SHAPE)], shape=_straight_lane_shape())
    gen = _TLSGenerator(40)
    gen.container_template = gen._gen_state_container([[approach]])
    est, conf = gen._derive_estimated_state([[approach]], 20)
    assert est[0][S] is None
    assert conf[0][S] is None


# ── _get_traj_metrics_at_phase (regression: window is [curr-dt, curr+dt), exclusive) ────────


def _metrics_with_vehicle_at(tt, curr, horizon=40):
    vehs = [{} for _ in range(horizon)]
    vehs[tt] = {1: _VehicleState(10, 5.0, 0.0)}  # at stop line, moving
    approach = _approach([_inj(S_SHAPE)], shape=_straight_lane_shape(), record_vehs=vehs)
    return _TLSGenerator(horizon)._get_traj_metrics_at_phase([approach], S, curr)


@pytest.mark.parametrize(
    ("tt", "counted"),
    [
        (10, True),  # curr - dt  (inclusive start)
        (29, True),  # curr + dt - 1 (last inclusive)
        (30, False),  # curr + dt  (exclusive end — the off-by-one regression)
        (9, False),  # before window
    ],
)
def test_traj_metric_window_bounds(tt, counted):
    _mean_acc, mean_spd, _sum_f, sum_g, must_green = _metrics_with_vehicle_at(tt, curr=20)
    assert must_green is False
    if counted:
        assert sum_g > 0.0
        assert mean_spd == pytest.approx(5.0)
    else:
        assert (sum_g, mean_spd) == (0.0, 0.0)


# ── _score_candidate_states ─────────────────────────────────────────────────────────────────


def test_score_candidate_picks_highest_match():
    gen = _TLSGenerator(40)
    cand_a, cand_b = [{S: _TLS.GREEN}], [{S: _TLS.RED}]
    result = gen._score_candidate_states([cand_a, cand_b], [{S: _TLS.GREEN}], [{S: 5.0}])
    assert result == [cand_a]


def test_score_candidate_returns_all_ties():
    gen = _TLSGenerator(40)
    cand_a = [{S: _TLS.GREEN, L: _TLS.RED}]
    cand_b = [{S: _TLS.GREEN, L: _TLS.GREEN}]
    imputed = [{S: _TLS.GREEN, L: None}]  # only S is observed -> both tie
    result = gen._score_candidate_states([cand_a, cand_b], imputed, [{S: 5.0, L: 0.0}])
    assert len(result) == 2


# ── _fill_right_turn_signal ─────────────────────────────────────────────────────────────────


def test_fill_right_turn_follows_straight():
    state = [{S: _TLS.GREEN, R: _TLS.RED}]
    assert _TLSGenerator(40)._fill_right_turn_signal(state)[0][R] == _TLS.GREEN


def test_fill_right_turn_follows_left_when_no_straight():
    state = [{L: _TLS.GREEN, R: _TLS.RED}]
    assert _TLSGenerator(40)._fill_right_turn_signal(state)[0][R] == _TLS.GREEN


def test_fill_right_turn_prefers_straight_over_left():
    state = [{S: _TLS.RED, L: _TLS.GREEN, R: _TLS.UNKNOWN}]
    assert _TLSGenerator(40)._fill_right_turn_signal(state)[0][R] == _TLS.RED


def test_fill_right_turn_noop_without_right_phase():
    state = [{S: _TLS.GREEN}]
    assert _TLSGenerator(40)._fill_right_turn_signal(state) == [{S: _TLS.GREEN}]


# ── sequence smoothing & yellow insertion ───────────────────────────────────────────────────


def _buff(states_per_step):
    return [[{S: state}] for state in states_per_step]


def _gen_for_smoothing(**kwargs):
    gen = _TLSGenerator(40, **kwargs)
    gen.container_template = [{S: None}]
    return gen


def test_find_short_intervals_detects_brief_red():
    gen = _gen_for_smoothing()
    buff = _buff([_TLS.GREEN] * 5 + [_TLS.RED] * 3 + [_TLS.GREEN] * 5)
    assert gen._find_short_intervals(buff, 0, S) == [(5, 7)]


def test_find_short_intervals_ignores_long_red():
    gen = _gen_for_smoothing()  # smoothing_width = 30
    buff = _buff([_TLS.GREEN] * 5 + [_TLS.RED] * 30 + [_TLS.GREEN] * 5)
    assert gen._find_short_intervals(buff, 0, S) == []


def test_find_short_intervals_symmetric_for_green_blip():
    gen = _gen_for_smoothing()
    buff = _buff([_TLS.RED] * 5 + [_TLS.GREEN] * 2 + [_TLS.RED] * 5)
    assert gen._find_short_intervals(buff, 0, S) == [(5, 6)]


def test_smooth_sequence_overwrites_short_blip():
    gen = _gen_for_smoothing()
    buff = _buff([_TLS.GREEN] * 5 + [_TLS.RED] * 3 + [_TLS.GREEN] * 5)
    smoothed = gen._smooth_sequence(buff)
    assert all(step[0][S] == _TLS.GREEN for step in smoothed)


def test_add_yellow_light_inserts_before_red():
    gen = _gen_for_smoothing(yellow_duration=3)
    buff = _buff([_TLS.GREEN] * 5 + [_TLS.RED] * 5)
    result = gen._add_yellow_light(buff)
    states = [step[0][S] for step in result]
    assert states == [_TLS.GREEN, _TLS.GREEN] + [_TLS.YELLOW] * 3 + [_TLS.RED] * 5


# ── gen_period (sequence assembly over a horizon) ────────────────────────────────────────────


def test_gen_period_short_horizon_fallback_is_constant():
    horizon = 5  # < 2 * delta_t -> no interior steps, falls back to a single broadcast moment
    intersection = [_approach([_inj(S_SHAPE)], id=i, horizon=horizon) for i in range(4)]
    intersection = [[lane] for lane in intersection]
    sequence = _TLSGenerator(horizon).gen_period(intersection)
    assert len(sequence) == horizon
    greens = [sum(way[S] == _TLS.GREEN for way in state) for state in sequence]
    assert greens == [1] * horizon  # exactly one way green, identical every step


def test_gen_period_two_way_returns_imputed_shape():
    horizon = 5
    intersection = [[_approach([_inj(S_SHAPE)], id=i, horizon=horizon)] for i in range(2)]
    sequence = _TLSGenerator(horizon).gen_period(intersection)
    assert len(sequence) == horizon
    assert all(len(state) == 2 and S in state[0] for state in sequence)


def test_gen_period_rejects_non_intersection_sizes():
    one_way = [[_approach([_inj(S_SHAPE)], horizon=5)]]
    assert _TLSGenerator(5).gen_period(one_way) == []
    five_way = [[_approach([_inj(S_SHAPE)], id=i, horizon=5)] for i in range(5)]
    assert _TLSGenerator(5).gen_period(five_way) == []


# ── _assign_vehicle_states_to_lanes (regression: acceleration anchored at tt-5) ─────────────


def _vehicle_track(horizon, *, speed_fn, position=(5.0, 0.0, 0.0), heading=0.0, agent_type=None):
    speeds = np.array([speed_fn(t) for t in range(horizon)])
    return schema.Track(
        type=puffer_types.AgentType.VEHICLE if agent_type is None else agent_type,
        position=np.tile(np.array(position, float), (horizon, 1)),
        heading=np.full(horizon, heading),
        velocity=np.column_stack([speeds, np.zeros(horizon)]),  # speed along +x
        valid=np.ones(horizon, bool),
        length=np.zeros(horizon),
        width=np.zeros(horizon),
        height=np.zeros(horizon),
    )


def _single_lane_matrix(n=11):
    pts = np.column_stack([np.arange(float(n)), np.zeros(n), np.zeros(n)])
    return pts.reshape(1, n, 3)


def test_assignment_acceleration_uses_five_step_window():
    horizon = 15
    track = _vehicle_track(horizon, speed_fn=lambda t: 0.01 * t * t)  # nonlinear -> window matters
    result = _assign_vehicle_states_to_lanes({7: track}, _single_lane_matrix(), {0: 0}, dt=0.1, horizon=horizon)
    state = result[0][10][7]
    assert state.lane_pos_idx == 5  # nearest lane point to x=5
    assert state.speed == pytest.approx(1.0)  # 0.01 * 100
    # (v[10] - v[5]) / (5 * 0.1) = (1.0 - 0.25) / 0.5 = 1.5  (a tt-1 window would give 1.9)
    assert state.acceleration == pytest.approx(1.5)


def test_assignment_acceleration_zero_when_no_prior_window():
    horizon = 15
    track = _vehicle_track(horizon, speed_fn=lambda t: 0.01 * t * t)
    result = _assign_vehicle_states_to_lanes({7: track}, _single_lane_matrix(), {0: 0}, dt=0.1, horizon=horizon)
    assert result[0][3][7].acceleration == 0.0  # tt-5 < 0 -> no anchor


def test_assignment_clamps_excessive_acceleration():
    horizon = 15
    # speed jumps 0 -> 30 at t=10: |a| = 30 / 0.5 = 60 > limit -> clamped to 0
    track = _vehicle_track(horizon, speed_fn=lambda t: 30.0 if t >= 10 else 0.0)
    result = _assign_vehicle_states_to_lanes({7: track}, _single_lane_matrix(), {0: 0}, dt=0.1, horizon=horizon)
    assert result[0][10][7].acceleration == 0.0


def test_assignment_skips_non_vehicles():
    horizon = 15
    track = _vehicle_track(horizon, speed_fn=lambda t: 1.0, agent_type=puffer_types.AgentType.PEDESTRIAN)
    result = _assign_vehicle_states_to_lanes({7: track}, _single_lane_matrix(), {0: 0}, dt=0.1, horizon=horizon)
    assert all(step == {} for step in result[0])


def test_assignment_rejects_far_vehicle():
    horizon = 5
    track = _vehicle_track(horizon, speed_fn=lambda t: 1.0, position=(5.0, 100.0, 0.0))  # 100 m off lane
    result = _assign_vehicle_states_to_lanes({7: track}, _single_lane_matrix(), {0: 0}, dt=0.1, horizon=horizon)
    assert all(step == {} for step in result[0])


def test_assignment_rejects_misaligned_heading():
    horizon = 5
    track = _vehicle_track(horizon, speed_fn=lambda t: 1.0, heading=np.pi / 2)  # perpendicular to lane
    result = _assign_vehicle_states_to_lanes({7: track}, _single_lane_matrix(), {0: 0}, dt=0.1, horizon=horizon)
    assert all(step == {} for step in result[0])


def test_assignment_skips_invalid_step():
    horizon = 15
    track = _vehicle_track(horizon, speed_fn=lambda t: 0.01 * t * t)
    track.valid[10] = False
    result = _assign_vehicle_states_to_lanes({7: track}, _single_lane_matrix(), {0: 0}, dt=0.1, horizon=horizon)
    assert result[0][10] == {}


# ── signalized-intersection detection ───────────────────────────────────────────────────────


def _lane(type_, polyline, *, entry=(), exit=(), left=(), right=()):
    return schema.MapElement(
        type=type_,
        polyline=np.array(polyline, float),
        entry_lanes=list(entry),
        exit_lanes=list(exit),
        left_neighbor=list(left),
        right_neighbor=list(right),
    )


def _diverge_clique_scenario(horizon=40):
    """One incoming lane (100) fanning into four junction lanes (1-4), each exiting to 11-14."""
    surface = puffer_types.LaneType.SURFACE_STREET
    ends = {1: (3, 3), 2: (4, 0), 3: (3, -3), 4: (3, 1)}
    lanes = {100: _lane(surface, [[-5, 0, 0], [0, 0, 0]], exit=[1, 2, 3, 4])}
    for jid, (ex, ey) in ends.items():
        oid = 10 + jid
        lanes[jid] = _lane(surface, [[0, 0, 0], [ex, ey, 0]], entry=[100], exit=[oid])
        lanes[oid] = _lane(surface, [[ex, ey, 0], [2 * ex, 2 * ey, 0]], entry=[jid])
    scenario = schema.PufferScenario(
        agents={},
        objects={},
        map=lanes,
        metadata=schema.ScenarioMetadata(id="t", dataset="d", scenario_length=horizon, dt=0.1),
    )
    states = [puffer_types.TLState.UNKNOWN] * horizon
    states[5] = puffer_types.TLState.GREEN  # at least one observed signal in the group
    extras = schema.ExtractionExtras(
        traffic_lights={1: schema.TrafficLightTrack(position=np.zeros(3), states=states, controlled_lane=1)}
    )
    return scenario, extras


def test_find_signalized_intersection_diverge_clique():
    scenario, extras = _diverge_clique_scenario()
    interp = _TrafficLightInterpolator(scenario, extras)
    interp._clean_lanes()
    groups = interp._find_signalized_intersections()
    assert len(groups) == 1
    assert sorted(groups[0]) == [1, 2, 3, 4]


def test_find_signalized_intersection_requires_observed_signal():
    scenario, extras = _diverge_clique_scenario()
    extras.traffic_lights = {}  # no observed signal anywhere -> criterion 2 fails
    interp = _TrafficLightInterpolator(scenario, extras)
    interp._clean_lanes()
    assert interp._find_signalized_intersections() == []


# ── interpolate_traffic_lights (top-level guards) ───────────────────────────────────────────


def _empty_scenario(length):
    return schema.PufferScenario(
        agents={},
        objects={},
        map={},
        metadata=schema.ScenarioMetadata(id="t", dataset="d", scenario_length=length, dt=0.1),
    )


def test_interpolate_noop_on_empty_horizon():
    extras = schema.ExtractionExtras()
    interpolate_traffic_lights(_empty_scenario(0), extras)
    assert extras.traffic_lights == {}


def test_interpolate_noop_with_too_few_lanes():
    extras = schema.ExtractionExtras()
    interpolate_traffic_lights(_empty_scenario(20), extras)  # empty map -> < 4 lanes
    assert extras.traffic_lights == {}


def test_interpolate_runs_on_detected_intersection_without_error():
    # Single approach (lane 100) -> the junction is detected but groups to one way and is skipped;
    # exercises the full detection + form_intersection path and leaves the input signal in place.
    scenario, extras = _diverge_clique_scenario()
    interpolate_traffic_lights(scenario, extras)
    assert set(extras.traffic_lights) == {1}
