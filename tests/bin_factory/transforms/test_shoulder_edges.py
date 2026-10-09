import numpy as np

from bin_factory import puffer_types, schema
from bin_factory.transforms.shoulder_edges import add_shoulder_edges


def _scenario(in_junction=False):
    centerline = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    left = np.array([[0.0, 1.75, 0.0], [20.0, 1.75, 0.0]])
    right = np.array([[0.0, -1.75, 0.0], [20.0, -1.75, 0.0]])
    lane = schema.MapElement(
        type=int(puffer_types.LaneType.SURFACE_STREET),
        polyline=centerline,
        left_boundary=left,
        right_boundary=right,
        in_junction=in_junction,
    )
    metadata = schema.ScenarioMetadata(id="s", dataset="d", scenario_length=1, dt=0.1)
    return schema.PufferScenario(agents={}, objects={}, map={0: lane}, metadata=metadata)


def _band(x0=5.0, x1=15.0, y0=-1.75, y1=-4.0):
    return np.array([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0], [x0, y0, 0.0]])


def _junction(x0, x1, y0=-6.0, y1=6.0):
    return np.array([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0], [x0, y0, 0.0]])


def _edges(scenario):
    return [e for e in scenario.map.values() if e.is_edge]


def _caps(scenario):
    return [e for e in _edges(scenario) if len(e.polyline) == 2 and np.ptp(e.polyline[:, 0]) < 1e-6]


def _two_lane_scenario(second_in_junction):
    scenario = _scenario()
    scenario.map[0].exit_lanes = [1]
    scenario.map[1] = schema.MapElement(
        type=int(puffer_types.LaneType.SURFACE_STREET),
        polyline=np.array([[20.0, 0.0, 0.0], [30.0, 0.0, 0.0], [40.0, 0.0, 0.0]]),
        left_boundary=np.array([[20.0, 1.75, 0.0], [40.0, 1.75, 0.0]]),
        right_boundary=np.array([[20.0, -1.75, 0.0], [40.0, -1.75, 0.0]]),
        in_junction=second_in_junction,
        entry_lanes=[0],
    )
    return scenario


def test_edge_added_along_right_boundary_with_lane_on_its_right():
    scenario = _scenario()
    assert add_shoulder_edges(scenario, [_band()], []) == 1
    edge = _edges(scenario)[0]
    assert edge.type == int(puffer_types.RoadEdgeType.BOUNDARY)
    xs = edge.polyline[:, 0]
    assert 3.0 <= xs.min() <= 6.0 and 14.0 <= xs.max() <= 17.0
    assert np.all(np.diff(xs) < 0)  # from high to low x: the lane (y=0) is on the right of that direction
    np.testing.assert_allclose(edge.polyline[:, 1], -1.75)


def test_left_boundary_edge_keeps_lane_direction():
    scenario = _scenario()
    assert add_shoulder_edges(scenario, [_band() * np.array([1.0, -1.0, 1.0])], []) == 1
    edge = _edges(scenario)[0]
    assert np.all(np.diff(edge.polyline[:, 0]) > 0)
    np.testing.assert_allclose(edge.polyline[:, 1], 1.75)


def test_ignored_cases():
    assert add_shoulder_edges(_scenario(), [_band() + np.array([0.0, -3.0, 0.0])], []) == 0  # 3 m gap
    assert add_shoulder_edges(_scenario(), [_band(y1=-2.15)], []) == 0  # 0.4 m hairline strip
    assert add_shoulder_edges(_scenario(), [_band(x0=9.0, x1=11.0)], []) == 0  # too short
    assert add_shoulder_edges(_scenario(), [], []) == 0


def test_layered_strips_count_as_one_band():
    scenario = _scenario()
    assert add_shoulder_edges(scenario, [_band(y0=-1.75, y1=-2.4), _band(y0=-2.4, y1=-4.5)], []) == 1


def test_band_inside_a_junction_surface_is_ignored_and_no_edge_crosses_it():
    inside = _scenario(in_junction=True)
    assert add_shoulder_edges(inside, [_band(x0=0.0, x1=20.0)], [_junction(-5.0, 25.0)]) == 0
    half = _scenario(in_junction=True)  # the band leaves the junction polygon at x = 10
    assert add_shoulder_edges(half, [_band(x0=0.0, x1=20.0)], [_junction(-5.0, 10.0)]) == 1
    edge = _edges(half)[0]
    assert edge.polyline[:, 0].min() >= 9.0  # nothing drawn inside the junction


def test_junction_lane_kerb_band_outside_the_polygon_gets_an_edge():
    scenario = _scenario(in_junction=True)
    assert add_shoulder_edges(scenario, [_band(x0=0.0, x1=20.0)], [_junction(-5.0, 25.0, y0=0.0, y1=6.0)]) == 1


def test_band_continuing_into_a_junction_interior_gets_a_cap():
    scenario = _two_lane_scenario(second_in_junction=True)
    assert add_shoulder_edges(scenario, [_band(x0=4.0, x1=40.0)], [_junction(20.0, 45.0)]) == 2  # run on lane 0 + cap
    cap = _caps(scenario)[0].polyline
    np.testing.assert_allclose(cap[:, 0], 20.0)
    assert cap[0, 1] < cap[1, 1]  # from the kerb (y=-4) to the lane boundary: the continuing side (+x) on its right
    np.testing.assert_allclose(sorted(cap[:, 1]), [-4.0, -1.75], atol=0.1)


def test_band_continuing_onto_a_lane_with_its_own_run_is_not_capped():
    for second_in_junction in (False, True):
        scenario = _two_lane_scenario(second_in_junction)
        assert add_shoulder_edges(scenario, [_band(x0=4.0, x1=40.0)], []) == 2  # one run per lane, no caps
        assert not _caps(scenario)


def test_band_ending_inside_a_lane_is_closed_by_the_kerb_not_a_cap():
    scenario = _scenario()
    assert add_shoulder_edges(scenario, [_band(x0=0.0, x1=12.0)], []) == 1
    assert not _caps(scenario)


def test_narrowing_within_hysteresis_continues_the_run_without_a_cap():
    scenario = _scenario()
    bands = [_band(x0=0.0, x1=10.0), _band(x0=10.0, x1=20.0, y1=-2.55)]  # 2.25 m band, then a 0.8 m strip
    assert add_shoulder_edges(scenario, bands, []) == 1
    edge = _edges(scenario)[0]
    assert edge.polyline[:, 0].min() <= 1.0 and edge.polyline[:, 0].max() >= 19.0


def test_narrowing_to_a_hairline_ends_the_run_without_a_cap():
    scenario = _scenario()
    bands = [_band(x0=0.0, x1=12.0), _band(x0=12.0, x1=20.0, y1=-2.05)]  # then a 0.3 m strip
    assert add_shoulder_edges(scenario, bands, []) == 1
    assert not _caps(scenario)


def test_short_dip_does_not_split_a_run():
    scenario = _scenario()
    assert add_shoulder_edges(scenario, [_band(x0=0.0, x1=9.0), _band(x0=12.0, x1=20.0)], []) == 1  # 3 m gap = one sample


def test_dead_end_with_band_going_on_gets_a_cap():
    scenario = _scenario()  # lane 0 has no exits; the band continues past its end
    assert add_shoulder_edges(scenario, [_band(x0=4.0, x1=30.0)], []) == 2
    assert len(_caps(scenario)) == 1


def test_band_reaching_the_lane_start_is_capped_even_if_the_first_sample_is_narrow():
    scenario = _two_lane_scenario(second_in_junction=True)
    # the shoulder starts 1 m into lane 0: first sample narrow, run snaps to the lane start; lane 1's band is junction interior
    assert add_shoulder_edges(scenario, [_band(x0=1.0, x1=40.0)], [_junction(20.0, 45.0)]) == 2
    assert len(_caps(scenario)) == 1


def test_short_junction_connector_fully_along_a_band_gets_an_edge():
    scenario = _scenario(in_junction=True)
    scenario.map[0].polyline = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    scenario.map[0].left_boundary = np.array([[0.0, 1.75, 0.0], [3.0, 1.75, 0.0]])
    scenario.map[0].right_boundary = np.array([[0.0, -1.75, 0.0], [3.0, -1.75, 0.0]])
    assert add_shoulder_edges(scenario, [_band(x0=-10.0, x1=10.0)], []) == 3  # the run + a cap at each dead end
    runs = [e for e in _edges(scenario) if np.allclose(e.polyline[:, 1], -1.75) and len(e.polyline) >= 2]
    assert len(runs) == 1 and np.ptp(runs[0].polyline[:, 0]) >= 2.9


def test_two_metre_stub_between_two_lanes_gets_its_edge_and_no_caps():
    scenario = _scenario()
    first = scenario.map[0]
    first.polyline = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    first.exit_lanes = [1]
    stub = schema.MapElement(
        type=int(puffer_types.LaneType.SURFACE_STREET),
        polyline=np.array([[20.0, 0.0, 0.0], [21.6, 0.0, 0.0]]),
        left_boundary=np.array([[20.0, 1.75, 0.0], [21.6, 1.75, 0.0]]),
        right_boundary=np.array([[20.0, -1.75, 0.0], [21.6, -1.75, 0.0]]),
        entry_lanes=[0], exit_lanes=[2],
    )
    third = schema.MapElement(
        type=int(puffer_types.LaneType.SURFACE_STREET),
        polyline=np.array([[21.6, 0.0, 0.0], [30.0, 0.0, 0.0], [40.0, 0.0, 0.0]]),
        left_boundary=np.array([[21.6, 1.75, 0.0], [40.0, 1.75, 0.0]]),
        right_boundary=np.array([[21.6, -1.75, 0.0], [40.0, -1.75, 0.0]]),
        entry_lanes=[1],
    )
    scenario.map[1], scenario.map[2] = stub, third
    # the shoulder is split at the stub's ends, so the stub's end samples sit on polygon corners
    bands = [_band(x0=0.0, x1=20.0), _band(x0=20.0, x1=21.6), _band(x0=21.6, x1=40.0)]
    assert add_shoulder_edges(scenario, bands, []) == 3  # one run per lane, including the stub
    assert not _caps(scenario)


def test_surfaces_on_another_level_are_ignored():
    scenario = _scenario()  # lane at z = 0
    bridge_band = _band(x0=5.0, x1=15.0) + np.array([0.0, 0.0, 6.0])  # same footprint, 6 m up
    assert add_shoulder_edges(scenario, [bridge_band], []) == 0
    assert add_shoulder_edges(_scenario(in_junction=True), [_band(x0=0.0, x1=20.0)], [_junction(-5.0, 25.0) + np.array([0.0, 0.0, 6.0])]) == 1


def test_run_ends_inside_the_lane_are_refined_to_the_band_limits():
    scenario = _scenario()  # samples every 2 m; the band spans x = 5..15, between samples
    assert add_shoulder_edges(scenario, [_band(x0=5.0, x1=15.0)], []) == 1
    xs = _edges(scenario)[0].polyline[:, 0]
    assert abs(xs.min() - 5.0) <= 0.25 and abs(xs.max() - 15.0) <= 0.25


def test_band_starting_just_past_the_lane_start_corner_covers_the_lane_start():
    scenario = _scenario()  # the first sample sits 0.1 m before the shoulder polygon corner
    assert add_shoulder_edges(scenario, [_band(x0=0.1, x1=10.0)], []) == 1
    xs = _edges(scenario)[0].polyline[:, 0]
    assert xs.min() == 0.0 and abs(xs.max() - 10.0) <= 0.25


def _lane(polyline, left, right, **fields):
    return schema.MapElement(
        type=int(puffer_types.LaneType.SURFACE_STREET),
        polyline=np.array(polyline, dtype=np.float64),
        left_boundary=np.array(left, dtype=np.float64),
        right_boundary=np.array(right, dtype=np.float64),
        **fields,
    )


def test_band_covered_by_another_lane_is_blocked_there():
    scenario = _scenario()  # a ramp connector overlaps the band beside lane 0 for x < 10
    scenario.map[1] = _lane([[0.0, -2.75, 0.0], [10.0, -2.75, 0.0]], [[0.0, -1.0, 0.0], [10.0, -1.0, 0.0]],
                            [[0.0, -4.5, 0.0], [10.0, -4.5, 0.0]], in_junction=True)
    assert add_shoulder_edges(scenario, [_band(x0=0.0, x1=20.0)], []) == 1
    xs = _edges(scenario)[0].polyline[:, 0]
    assert 9.5 <= xs.min() <= 10.5 and xs.max() >= 19.0  # nothing drawn across the connector


def test_crossing_lane_splits_the_run_instead_of_being_bridged():
    scenario = _scenario()  # a 3 m wide lane crosses the band at x = 10: one blocked sample, not a bridgeable dip
    scenario.map[1] = _lane([[10.0, -6.0, 0.0], [10.0, 0.0, 0.0], [10.0, 6.0, 0.0]], [[8.5, -6.0, 0.0], [8.5, 6.0, 0.0]],
                            [[11.5, -6.0, 0.0], [11.5, 6.0, 0.0]], in_junction=True)
    add_shoulder_edges(scenario, [_band(x0=0.0, x1=20.0)], [])
    runs = [e.polyline[:, 0] for e in _edges(scenario) if np.allclose(e.polyline[:, 1], -1.75)]
    assert len(runs) == 2
    assert all(xs.max() <= 8.6 or xs.min() >= 11.4 for xs in runs)


def test_cap_spans_the_continuing_band_when_the_end_sample_measured_a_sliver():
    scenario = _two_lane_scenario(second_in_junction=True)
    bands = [_band(x0=0.0, x1=19.5), _band(x0=19.5, x1=20.0, y1=-2.0), _band(x0=20.0, x1=40.0)]  # 0.25 m at the lane end
    assert add_shoulder_edges(scenario, bands, [_junction(20.0, 45.0)]) == 2  # run on lane 0 + cap
    cap = _caps(scenario)[0].polyline
    assert np.ptp(cap[:, 1]) >= 1.5


def test_zero_length_stub_between_two_lanes_is_looked_through_for_caps():
    scenario = _scenario()
    scenario.map[0].exit_lanes = [1]
    scenario.map[1] = _lane([[20.0, 0.0, 0.0], [20.0, 0.0, 0.0]], [[20.0, 1.75, 0.0], [20.0, 1.75, 0.0]],
                            [[20.0, -1.75, 0.0], [20.0, -1.75, 0.0]], entry_lanes=[0], exit_lanes=[2])
    scenario.map[2] = _lane([[20.0, 0.0, 0.0], [30.0, 0.0, 0.0], [40.0, 0.0, 0.0]], [[20.0, 1.75, 0.0], [40.0, 1.75, 0.0]],
                            [[20.0, -1.75, 0.0], [40.0, -1.75, 0.0]], entry_lanes=[1])
    assert add_shoulder_edges(scenario, [_band(x0=0.0, x1=40.0)], []) == 2  # one run per real lane, no caps
    assert not _caps(scenario)
