import numpy as np

from bin_factory.transforms.traffic_light_interpolation import (
    _TLS,
    _ApproachingLane,
    _Direction,
    _InJunctionLane,
    _TLSGenerator,
)


def _straight_way(horizon):
    """One approaching lane feeding one straight in-junction lane (direction S)."""
    shape = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    inj = _InJunctionLane(
        id=0,
        shape=shape,
        record_tls=[_TLS.ABSENT] * horizon,
        record_vehs=[{} for _ in range(horizon)],
        direction=_Direction.S,
    )
    approach = _ApproachingLane(
        id=0,
        shape=shape,
        record_vehs=[{} for _ in range(horizon)],
        injunction_lanes=[inj],
    )
    return [approach]


def _straight_phase(way_state):
    return way_state[(_Direction.S,)]


def test_gen_period_4way_assigns_exactly_one_green_per_step():
    horizon = 5
    intersection = [_straight_way(horizon) for _ in range(4)]

    sequence = _TLSGenerator(horizon).gen_period(intersection)

    assert len(sequence) == horizon
    for state in sequence:
        assert len(state) == 4
        greens = sum(_straight_phase(way) == _TLS.GREEN for way in state)
        reds = sum(_straight_phase(way) == _TLS.RED for way in state)
        assert greens == 1
        assert reds == 3


def test_gen_period_3way_shape():
    horizon = 5
    intersection = [_straight_way(horizon) for _ in range(3)]

    sequence = _TLSGenerator(horizon).gen_period(intersection)

    assert len(sequence) == horizon
    assert all(len(state) == 3 for state in sequence)


def test_gen_period_rejects_non_intersection_sizes():
    assert _TLSGenerator(5).gen_period([_straight_way(5)]) == []  # 1 way
    assert _TLSGenerator(5).gen_period([_straight_way(5) for _ in range(5)]) == []  # 5 ways


def test_relevance_weights():
    assert _TLSGenerator._f(0, 1.0) == 1.0  # at stop line → fully relevant
    assert _TLSGenerator._f(-20, 1.0) == 0.0  # far behind stop line → irrelevant
    assert _TLSGenerator._g(0, 0.0) == 1.0  # at stop line → fully relevant
    assert _TLSGenerator._g(-30, 0.0) == 0.0  # far behind → irrelevant
