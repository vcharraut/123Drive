import numpy as np

from bin_factory import puffer_types, schema
from bin_factory.transforms.sanitize import prune_invalid_map_elements


def _lane(points, exit_lanes=()):
    return schema.MapElement(
        type=int(puffer_types.LaneType.SURFACE_STREET),
        polyline=np.zeros((points, 3), dtype=np.float64),
        exit_lanes=list(exit_lanes),
    )


def _scenario(map_):
    return schema.PufferScenario(
        agents={},
        objects={},
        map=map_,
        metadata=schema.ScenarioMetadata(id="x", dataset="d", scenario_length=0, dt=0.0),
    )


def test_prune_drops_undersized_polylines_and_filters_dangling_refs():
    scenario = _scenario({1: _lane(2, exit_lanes=(2,)), 2: _lane(1)})  # lane 2 has < 2 points

    prune_invalid_map_elements(scenario, schema.ExtractionExtras())

    assert set(scenario.map) == {1}
    assert scenario.map[1].exit_lanes == []  # ref to pruned lane removed


def test_prune_filters_extras_against_surviving_lanes():
    scenario = _scenario({1: _lane(2), 2: _lane(1)})
    extras = schema.ExtractionExtras(
        traffic_lights={
            7: schema.TrafficLightTrack(position=np.zeros(3), states=[], controlled_lane=2),
            8: schema.TrafficLightTrack(position=np.zeros(3), states=[], controlled_lane=1),
        },
        stop_zones=[schema.StopZone(type=int(puffer_types.TCType.STOP_SIGN), polygon=np.zeros((4, 3)), controlled_lanes=[1, 2])],
    )

    prune_invalid_map_elements(scenario, extras)

    assert set(extras.traffic_lights) == {8}  # TL on pruned lane 2 dropped
    assert extras.stop_zones[0].controlled_lanes == [1]
