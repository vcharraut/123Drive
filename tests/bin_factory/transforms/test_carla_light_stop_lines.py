"""Tests for moving CARLA traffic-light stop lines onto the line the leaderboard scores.

Layout: approach lane 1 runs along +x from x=0 to the junction entry at x=20, junction lane 2 continues to x=40.
The exported stop line sits 0.25 m inside the junction (x=20.25), like the stop-zone lines of the CARLA bins.
"""

import math

import numpy as np
import pytest

from bin_factory import puffer_types, schema
from bin_factory.transforms.carla_light_stop_lines import align_carla_light_stop_lines, align_light_stop_lines


def _lane(polyline, entry_lanes=()):
    return schema.MapElement(
        type=int(puffer_types.LaneType.SURFACE_STREET),
        polyline=np.array(polyline, dtype=np.float64),
        entry_lanes=list(entry_lanes),
    )


def _control(control_type, x, half_length=1.75, controlled_lanes=(2,)):
    return {
        "id": 0,
        "type": control_type,
        "controlled_lanes": list(controlled_lanes),
        "stop_line": np.array([[x, -half_length, 0.0], [x, half_length, 0.0]]),
        "heading": 0.0,
        "states": [],
        "junction_id": -1,
        "phase_idx": -1,
    }


def _scenario(control, junction_lane=((20, 0, 0), (40, 0, 0)), location="Town99"):
    lanes = {1: _lane([[0, 0, 0], [20, 0, 0]]), 2: _lane(junction_lane, entry_lanes=[1])}
    scenario = schema.PufferScenario(
        agents={},
        objects={},
        map=lanes,
        metadata=schema.ScenarioMetadata(id=location, dataset="opendrive", scenario_length=1, dt=0.1, location=location),
    )
    scenario.traffic_controls = [control]
    return scenario


def _point(x, y=0.0, in_junction=False, light=7):
    return [x, y, 0.0, float(in_junction), light]


def _line_x(scenario):
    return float(np.mean(scenario.traffic_controls[0]["stop_line"][:, 0]))


def test_line_moves_onto_a_carla_point_further_upstream_than_the_setback():
    scenario = _scenario(_control(puffer_types.TCType.TRAFFIC_LIGHT, 20.25))
    align_light_stop_lines(scenario, np.array([_point(19.4)]))
    assert _line_x(scenario) == pytest.approx(19.4)


def test_line_stops_at_the_setback_when_carla_scores_closer_to_the_junction():
    scenario = _scenario(_control(puffer_types.TCType.TRAFFIC_LIGHT, 20.25))
    align_light_stop_lines(scenario, np.array([_point(19.9)]))
    assert _line_x(scenario) == pytest.approx(19.75)  # junction entry 20.0 minus the 0.25 m setback


def test_line_follows_a_trigger_inside_the_junction():
    scenario = _scenario(_control(puffer_types.TCType.TRAFFIC_LIGHT, 20.25))
    align_light_stop_lines(scenario, np.array([_point(33.0, in_junction=True)]))
    assert _line_x(scenario) == pytest.approx(33.0)


def test_multi_lane_line_takes_the_most_upstream_point_of_its_own_light():
    scenario = _scenario(_control(puffer_types.TCType.TRAFFIC_LIGHT, 20.25, half_length=3.5))
    points = np.array([_point(19.8, y=1.75), _point(19.6, y=-1.75), _point(5.0, light=9)])
    align_light_stop_lines(scenario, points)
    assert _line_x(scenario) == pytest.approx(19.6)


def test_approach_lanes_join_the_controlled_lanes():
    scenario = _scenario(_control(puffer_types.TCType.TRAFFIC_LIGHT, 20.25))
    align_light_stop_lines(scenario, np.array([_point(19.6)]))
    assert scenario.traffic_controls[0]["controlled_lanes"] == [2, 1]


def test_heading_follows_the_lanes_at_the_line():
    bent = (20, 0, 0), (20 + 20 * math.cos(math.radians(10)), 20 * math.sin(math.radians(10)), 0)
    scenario = _scenario(_control(puffer_types.TCType.TRAFFIC_LIGHT, 20.25), junction_lane=bent)
    align_light_stop_lines(scenario, np.array([_point(19.6)]))
    control = scenario.traffic_controls[0]
    assert control["heading"] == pytest.approx(math.radians(5.0))  # approach lane 0 deg, junction lane 10 deg
    across = control["stop_line"][1, :2] - control["stop_line"][0, :2]
    assert across @ np.array([math.cos(control["heading"]), math.sin(control["heading"])]) == pytest.approx(0.0)
    assert np.linalg.norm(across) == pytest.approx(3.5)


def test_turned_multi_lane_line_ends_at_or_before_every_scored_point_of_its_light():
    bent = (20, 0, 0), (20 + 20 * math.cos(math.radians(10)), 20 * math.sin(math.radians(10)), 0)
    scenario = _scenario(_control(puffer_types.TCType.TRAFFIC_LIGHT, 20.25, half_length=5.25), junction_lane=bent)
    points = np.array([_point(19.6, y=3.5), _point(19.7, y=-3.5)])
    align_light_stop_lines(scenario, points)
    control = scenario.traffic_controls[0]
    along_dir = np.array([math.cos(control["heading"]), math.sin(control["heading"])])
    along = (points[:, :2] - control["stop_line"][:, :2].mean(axis=0)) @ along_dir
    assert along.min() == pytest.approx(0.0, abs=1e-9)  # on the most upstream point, the other one lies ahead


def test_stop_sign_keeps_its_position():
    scenario = _scenario(_control(puffer_types.TCType.STOP_SIGN, 20.25))
    align_light_stop_lines(scenario, np.array([_point(19.4)]))
    assert _line_x(scenario) == pytest.approx(20.25)
    assert scenario.traffic_controls[0]["controlled_lanes"] == [2]


def test_town_without_scored_points_is_rejected():
    scenario = _scenario(_control(puffer_types.TCType.TRAFFIC_LIGHT, 20.25), location="Town99")
    with pytest.raises(ValueError, match="no CARLA scored stop points"):
        align_carla_light_stop_lines(scenario, schema.ExtractionExtras(centroid=[0.0, 0.0, 0.0]))
