import numpy as np

from bin_factory import puffer_types, schema
from bin_factory.conversion.sanitize import clean_map, prune_invalid_map_elements


def _road(points, layer=puffer_types.MapLayer.LANE, kind=puffer_types.LaneType.SURFACE_STREET, **fields):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    polyline = np.column_stack([points, np.zeros(len(points))])
    return schema.RoadPolyline(layer=int(layer), type=int(kind), polyline=polyline, **fields)


def _lane(points, exit_lanes=()):
    """A lane of `points` points at the origin."""
    return _road(np.zeros((points, 2)), exit_lanes=list(exit_lanes))


def _scenario(map_):
    return schema.PufferScenario(
        agents={},
        objects={},
        polylines=map_,
        metadata=schema.ScenarioMetadata(id="x", dataset="d", scenario_length=0, dt=0.0),
    )


def test_prune_drops_undersized_polylines_and_filters_dangling_refs():
    scenario = _scenario({1: _lane(2, exit_lanes=(2,)), 2: _lane(1)})  # lane 2 has < 2 points

    prune_invalid_map_elements(scenario, schema.ExtractionExtras())

    assert set(scenario.polylines) == {1}
    assert scenario.polylines[1].exit_lanes == []  # ref to pruned lane removed


def test_prune_filters_extras_against_surviving_lanes():
    scenario = _scenario({1: _lane(2), 2: _lane(1)})
    extras = schema.ExtractionExtras(
        traffic_lights={
            7: schema.TrafficLightTrack(position=np.zeros(3), states=[], controlled_lane=2),
            8: schema.TrafficLightTrack(position=np.zeros(3), states=[], controlled_lane=1),
        },
        stop_zones=[
            schema.StopZone(type=int(puffer_types.TCType.STOP_SIGN), polygon=np.zeros((4, 3)), controlled_lanes=[1, 2])
        ],
    )

    prune_invalid_map_elements(scenario, extras)

    assert set(extras.traffic_lights) == {8}  # TL on pruned lane 2 dropped
    assert extras.stop_zones[0].controlled_lanes == [1]


def test_clean_map_drops_distant_or_unknown_links_and_infers_undefined_types():
    polylines = {
        1: _road([[0, 0], [20, 0]], exit_lanes=[2, 3, 99]),
        2: _road([[21, 0], [40, 0]], kind=puffer_types.LaneType.UNDEFINED, entry_lanes=[1]),
        3: _road([[120, 0], [140, 0]], entry_lanes=[1]),  # a WOMD link to a lane 100 m away
    }

    clean_map(polylines)

    assert polylines[1].exit_lanes == [2]
    assert polylines[3].entry_lanes == []
    assert polylines[2].type == puffer_types.LaneType.SURFACE_STREET


def test_clean_map_keeps_only_neighbours_that_run_alongside():
    polylines = {
        1: _road([[0, 0], [40, 0]], left_neighbor=[2], right_neighbor=[3]),
        2: _road([[0, 3.5], [40, 3.5]]),
        3: _road([[0, -3.5], [3, -3.5], [40, -40.5]]),  # alongside for 3 m, then diverging
    }

    clean_map(polylines)

    assert polylines[1].left_neighbor == [2]
    assert polylines[1].right_neighbor == []


def test_clean_map_spreads_speed_limits_along_lane_links():
    polylines = {
        1: _road([[0, 0], [20, 0]], exit_lanes=[2], speed_limit_mps=13.9),
        2: _road([[20, 0], [40, 0]], entry_lanes=[1], exit_lanes=[3]),
        3: _road([[40, 0], [60, 0]], entry_lanes=[2]),
    }

    clean_map(polylines)

    assert [lane.speed_limit_mps for lane in polylines.values()] == [13.9, 13.9, 13.9]


def test_clean_map_trims_a_lane_end_hook_and_keeps_the_endpoints():
    hooked = [[0, 0], [0.2, 0.15], [0.4, 0], [5, 0], [10, 0]]
    polylines = {1: _road(hooked)}

    clean_map(polylines)

    np.testing.assert_array_equal(polylines[1].polyline[:, :2], np.array(hooked)[[0, 3, 4]])


def test_clean_map_drops_duplicate_road_lines_even_reversed_across_dash_solid_pairs():
    line = [[0, 5], [10, 5]]
    road_line, types = puffer_types.MapLayer.ROAD_LINE, puffer_types.RoadLineType
    polylines = {
        10: _road(line, layer=road_line, kind=types.DASH_SOLID_WHITE),
        11: _road(line[::-1], layer=road_line, kind=types.SOLID_DASH_WHITE),
        12: _road(line, layer=road_line, kind=types.SOLID_WHITE),
    }

    clean_map(polylines)

    assert list(polylines) == [10, 12]
