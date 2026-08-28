import numpy as np

from bin_factory import puffer_types, schema
from bin_factory.transforms.geometry import DEFAULT_LANE_WIDTH_M, arc_length, compute_lane_widths, polyline_length


def test_arc_length_cumulative():
    line = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    np.testing.assert_allclose(arc_length(line), [0.0, 3.0, 7.0])


def test_arc_length_uses_all_columns():
    line = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])  # vertical segment
    np.testing.assert_allclose(arc_length(line), [0.0, 5.0])


def test_arc_length_degenerate():
    np.testing.assert_array_equal(arc_length(np.zeros((1, 3))), [0.0])
    assert arc_length(np.zeros((0, 3))).shape == (0,)


def test_polyline_length_matches_arc_length_tail():
    line = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    assert polyline_length(line) == 7.0
    assert polyline_length(line) == arc_length(line)[-1]


def test_polyline_length_degenerate():
    assert polyline_length(np.zeros((1, 3))) == 0.0


def _lane_scenario(**kwargs):
    lane = schema.MapElement(type=int(puffer_types.LaneType.SURFACE_STREET), **kwargs)
    metadata = schema.ScenarioMetadata(id="s", dataset="d", scenario_length=1, dt=0.1)
    return schema.PufferScenario(agents={}, objects={}, map={0: lane}, metadata=metadata), lane


def test_compute_lane_widths_sums_boundary_distances():
    centerline = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    left = np.array([[0.0, 1.0, 0.0], [10.0, 1.0, 0.0]])
    right = np.array([[0.0, -1.0, 0.0], [5.0, -1.0, 0.0], [5.0, -3.0, 0.0], [10.0, -3.0, 0.0]])  # steps wider
    scenario, lane = _lane_scenario(polyline=centerline, left_boundary=left, right_boundary=right)
    compute_lane_widths(scenario)
    np.testing.assert_allclose(lane.width, [2.0, 2.0, 4.0])


def test_compute_lane_widths_falls_back_without_boundaries():
    centerline = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    scenario, lane = _lane_scenario(polyline=centerline)
    compute_lane_widths(scenario)
    np.testing.assert_array_equal(lane.width, [DEFAULT_LANE_WIDTH_M, DEFAULT_LANE_WIDTH_M])


def test_compute_lane_widths_skips_non_lanes():
    scenario, lane = _lane_scenario(polyline=np.zeros((2, 3)))
    lane.type = int(puffer_types.RoadLineType.SOLID_SINGLE_WHITE)
    compute_lane_widths(scenario)
    assert lane.width is None
