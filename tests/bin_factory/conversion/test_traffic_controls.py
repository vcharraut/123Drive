"""Tests for the traffic-controls processor.

Focus: a traffic light whose controlled lane is a bike lane must not be emitted as a
traffic control. In the py123d data a signal detection can reference a bike lane, which
`_extract_traffic_lights` keeps; rendered, its stop line lands at the bike lane's first
point ("the beginning of the bike lane"). Those signals are irrelevant to the vehicle
sim and are dropped here. Vehicle/bus-lane signals are kept.
"""

import numpy as np
import pytest

from bin_factory import puffer_types, schema
from bin_factory.conversion.traffic_controls import process_traffic_controls


def _lane(lane_type, polyline):
    return schema.RoadPolyline(
        layer=int(puffer_types.MapLayer.LANE), type=int(lane_type), polyline=np.array(polyline, dtype=np.float64)
    )


def _scenario(lanes, length=5):
    return schema.PufferScenario(
        agents={},
        objects={},
        polylines=lanes,
        metadata=schema.ScenarioMetadata(id="t", dataset="d", scenario_length=length, dt=0.1),
    )


def _tl(controlled_lane, length=5):
    return schema.TrafficLightTrack(
        position=np.zeros(3, dtype=np.float64),
        states=[puffer_types.TLState.GREEN] * length,
        controlled_lane=controlled_lane,
    )


def test_bike_lane_traffic_light_is_dropped():
    # lane 1 = vehicle approach, lane 2 = parallel bike lane; both carry an input signal.
    scenario = _scenario(
        {
            1: _lane(puffer_types.LaneType.SURFACE_STREET, [[0, 0, 0], [10, 0, 0]]),
            2: _lane(puffer_types.LaneType.BIKE_LANE, [[0, 5, 0], [10, 5, 0]]),
        }
    )
    extras = schema.ExtractionExtras(traffic_lights={1: _tl(1), 2: _tl(2)})

    process_traffic_controls(scenario, extras)

    controlled = {lid for tc in scenario.traffic_controls for lid in tc.controlled_lanes}
    assert controlled == {1}  # the bike-lane signal (lane 2) is dropped
    assert all(tc.type == puffer_types.TCType.TRAFFIC_LIGHT for tc in scenario.traffic_controls)
    assert 2 in scenario.polylines  # the bike lane itself stays in the map


@pytest.mark.parametrize(
    ("lane_type", "kept"),
    [
        (puffer_types.LaneType.SURFACE_STREET, True),
        (puffer_types.LaneType.BUS_LANE, True),  # scope is bike-lane only: bus-lane signals are not dropped
        (puffer_types.LaneType.BIKE_LANE, False),
    ],
)
def test_lane_traffic_light_is_kept_at_lane_start_unless_on_a_bike_lane(lane_type, kept):
    scenario = _scenario({1: _lane(lane_type, [[0, 0, 0], [10, 0, 0]])})
    extras = schema.ExtractionExtras(traffic_lights={1: _tl(1)})

    process_traffic_controls(scenario, extras)

    assert [tc.controlled_lanes for tc in scenario.traffic_controls] == ([[1]] if kept else [])
    for tc in scenario.traffic_controls:  # stop line straddles the controlled lane's first point
        np.testing.assert_allclose(np.asarray(tc.stop_line, dtype=np.float64).mean(axis=0), [0.0, 0.0, 0.0], atol=1e-9)


def test_static_traffic_light_keeps_only_uncovered_lanes():
    scenario = _scenario(
        {
            1: _lane(puffer_types.LaneType.SURFACE_STREET, [[0, 0, 0], [10, 0, 0]]),
            2: _lane(puffer_types.LaneType.SURFACE_STREET, [[0, 5, 0], [10, 5, 0]]),
        }
    )
    zone = schema.StopZone(
        type=int(puffer_types.TCType.TRAFFIC_LIGHT),
        polygon=np.array([[0, -1, 0], [0, 6, 0], [1, 6, 0], [1, -1, 0]], dtype=np.float64),
        controlled_lanes=[1, 2],
    )
    extras = schema.ExtractionExtras(traffic_lights={1: _tl(1)}, stop_zones=[zone])

    process_traffic_controls(scenario, extras)

    assert [control.controlled_lanes for control in scenario.traffic_controls] == [[1], [2]]
    assert scenario.traffic_controls[1].states == [puffer_types.TLState.UNKNOWN] * 5


def test_static_traffic_light_leaves_out_bike_lanes():
    scenario = _scenario(
        {
            1: _lane(puffer_types.LaneType.SURFACE_STREET, [[0, 0, 0], [10, 0, 0]]),
            2: _lane(puffer_types.LaneType.BIKE_LANE, [[0, 5, 0], [10, 5, 0]]),
            3: _lane(puffer_types.LaneType.BIKE_LANE, [[0, 20, 0], [10, 20, 0]]),
        }
    )
    light = int(puffer_types.TCType.TRAFFIC_LIGHT)
    shared = np.array([[0, -1, 0], [0, 6, 0], [1, 6, 0], [1, -1, 0]], dtype=np.float64)
    bike_only = np.array([[0, 19, 0], [0, 21, 0], [1, 21, 0], [1, 19, 0]], dtype=np.float64)
    zones = [schema.StopZone(light, shared, controlled_lanes=[1, 2]), schema.StopZone(light, bike_only, [3])]
    stop_sign = schema.StopZone(int(puffer_types.TCType.STOP_SIGN), bike_only, controlled_lanes=[3])

    process_traffic_controls(scenario, schema.ExtractionExtras(stop_zones=[*zones, stop_sign]))

    controls = [(control.type, control.controlled_lanes) for control in scenario.traffic_controls]
    assert controls == [(light, [1]), (int(puffer_types.TCType.STOP_SIGN), [3])]  # signs keep their bike lanes


def _approach_light_scenario(zone_x0):
    """nuPlan: the light is detected on connector 2, its stop polygon spans x0..x0+1 across approach lane 1."""
    approach = _lane(puffer_types.LaneType.SURFACE_STREET, [[0, 0, 0], [10, 0, 0]])
    connector = _lane(puffer_types.LaneType.SURFACE_STREET, [[10, 0, 0], [20, 0, 0]])
    approach.exit_lanes, connector.entry_lanes = [2], [1]
    scenario = _scenario({1: approach, 2: connector})
    zone = schema.StopZone(
        type=int(puffer_types.TCType.TRAFFIC_LIGHT),
        polygon=np.array(
            [[zone_x0, -2, 0], [zone_x0, 2, 0], [zone_x0 + 1, 2, 0], [zone_x0 + 1, -2, 0]], dtype=np.float64
        ),
        controlled_lanes=[1],
    )
    process_traffic_controls(scenario, schema.ExtractionExtras(traffic_lights={2: _tl(2)}, stop_zones=[zone]))
    return scenario


# The polygon lies on the approach lane or up to 9 m into the connector; the line sits at its far edge, since
# vehicles stop before the polygon. A zone just short of the approach start counts too.
@pytest.mark.parametrize(("zone_x0", "expected_x"), [(4.0, 5.0), (13.0, 14.0), (-1.2, 0.0)])
def test_light_stop_line_lies_on_its_approach_stop_zone(zone_x0, expected_x):
    scenario = _approach_light_scenario(zone_x0)

    assert [tc.controlled_lanes for tc in scenario.traffic_controls] == [[2]]
    np.testing.assert_allclose(scenario.traffic_controls[0].stop_line.mean(axis=0), [expected_x, 0, 0], atol=1e-9)
