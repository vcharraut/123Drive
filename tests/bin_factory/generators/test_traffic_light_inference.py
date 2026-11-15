import numpy as np

from bin_factory import puffer_types, schema
from bin_factory.generators.traffic_light_inference import _fill_short_reds, _min_dwell_path, _vehicle_evidence


def _switch_cost(cost):
    return np.array([[0.0, -cost], [-cost, 0.0]])


def test_evidence_shorter_than_the_dwell_does_not_switch_the_state():
    # a vehicle crossing mid-red: 4 frames of GO evidence inside a long RED
    log_emit = np.zeros((2, 40))
    log_emit[0] = 1.0
    log_emit[1, 18:22] = 5.0

    assert not _min_dwell_path(log_emit, _switch_cost(6.0), 10).any()
    np.testing.assert_array_equal(np.flatnonzero(_min_dwell_path(log_emit, _switch_cost(6.0), 4)), np.arange(18, 22))


def test_first_and_last_states_may_be_shorter_than_the_dwell():
    # the scene starts and ends inside a phase
    log_emit = np.zeros((2, 40))
    log_emit[0, 3:37] = 1.0
    log_emit[1, :3] = log_emit[1, 37:] = 10.0

    path = _min_dwell_path(log_emit, _switch_cost(5.0), 10)

    np.testing.assert_array_equal(np.flatnonzero(path == 0), np.arange(3, 37))


def test_short_red_between_two_gos_is_filled():
    path = np.array(
        [[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )

    filled = _fill_short_reds(path, 3)

    np.testing.assert_array_equal(filled[0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(filled[1:], path[1:])  # REDs at the scene edges stay


def test_pull_away_from_a_stop_supports_go_before_the_crossing():
    dt, length = 0.1, 40
    t = np.arange(length)
    moving = np.clip(t - 9, 0, None) * dt  # stopped 3 m before the line, then accelerating at 3 m/s2
    x, speed = -3 + 1.5 * moving**2, 3 * moving
    track = schema.Track(
        type=int(puffer_types.AgentType.VEHICLE),
        position=np.stack([x, np.zeros(length), np.zeros(length)], 1),
        heading=np.zeros(length),
        velocity=np.stack([speed, np.zeros(length)], 1),
        valid=np.ones(length, dtype=np.int32),
        length=np.ones(length),
        width=np.ones(length),
        height=np.ones(length),
    )
    line = np.array([[0.0, 0.0], [10.0, 0.0]])

    llr = _vehicle_evidence({1: track}, [line], line[:1], np.array([[1.0, 0.0]]), np.array([False]), length, dt)

    crossing = int(np.argmax(x >= 0))
    assert llr[0, 5] == 0  # standing still
    assert (llr[0, 11 : crossing - 5] > 0).all()  # pulling away (over 0.5 m/s), before the crossing's own evidence
