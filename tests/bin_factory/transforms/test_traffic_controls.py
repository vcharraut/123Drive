"""Tests for the traffic-controls processor.

Focus: a traffic light whose controlled lane is a bike lane must not be emitted as a
traffic control. In the py123d data a signal detection can reference a bike lane, which
`_extract_traffic_lights` keeps; rendered, its stop line lands at the bike lane's first
point ("the beginning of the bike lane"). Those signals are irrelevant to the vehicle
sim and are dropped here. Vehicle/bus-lane signals are kept.
"""

import numpy as np

from bin_factory import puffer_types, schema
from bin_factory.transforms.traffic_controls import process_traffic_controls


def _lane(lane_type, polyline):
    return schema.MapElement(type=int(lane_type), polyline=np.array(polyline, dtype=np.float64))


def _scenario(lanes, length=5):
    return schema.PufferScenario(
        agents={},
        objects={},
        map=lanes,
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

    controlled = {lid for tc in scenario.traffic_controls for lid in tc["controlled_lanes"]}
    assert controlled == {1}  # the bike-lane signal (lane 2) is dropped
    assert all(tc["type"] == puffer_types.TCType.TRAFFIC_LIGHT for tc in scenario.traffic_controls)
    assert 2 in scenario.map  # the bike lane itself stays in the map


def test_vehicle_lane_traffic_light_is_kept_at_lane_start():
    scenario = _scenario({1: _lane(puffer_types.LaneType.SURFACE_STREET, [[0, 0, 0], [10, 0, 0]])})
    extras = schema.ExtractionExtras(traffic_lights={1: _tl(1)})

    process_traffic_controls(scenario, extras)

    assert [tc["controlled_lanes"] for tc in scenario.traffic_controls] == [[1]]
    # stop line straddles the controlled lane's first point
    stop_line = np.asarray(scenario.traffic_controls[0]["stop_line"], dtype=np.float64)
    np.testing.assert_allclose(stop_line.mean(axis=0), [0.0, 0.0, 0.0], atol=1e-9)


def test_bus_lane_traffic_light_is_kept():
    # scope is bike-lane only: bus-lane signals are not dropped
    scenario = _scenario({1: _lane(puffer_types.LaneType.BUS_LANE, [[0, 0, 0], [10, 0, 0]])})
    extras = schema.ExtractionExtras(traffic_lights={1: _tl(1)})

    process_traffic_controls(scenario, extras)

    assert [tc["controlled_lanes"] for tc in scenario.traffic_controls] == [[1]]


def test_only_bike_lane_signal_yields_no_traffic_controls():
    scenario = _scenario({2: _lane(puffer_types.LaneType.BIKE_LANE, [[0, 0, 0], [10, 0, 0]])})
    extras = schema.ExtractionExtras(traffic_lights={2: _tl(2)})

    process_traffic_controls(scenario, extras)

    assert scenario.traffic_controls == []
