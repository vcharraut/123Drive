import numpy as np

from bin_factory import puffer_types, schema
from bin_factory.transforms.geometry import (
    DEFAULT_LANE_WIDTH_M,
    _simplify_polyline,
    arc_length,
    compute_lane_widths,
    polyline_length,
)


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


def test_simplification_recovers_z_from_original_station():
    line = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 10.0], [2.0, 0.0, 20.0], [3.0, 0.0, 30.0], [3.0, 1.0, 40.0]])

    simplified = _simplify_polyline(line, 0.5)

    # the vertex at (2, 0) is flat in the plane but 1.7 m off the height line, so it survives
    np.testing.assert_allclose(simplified[:, 2], [0.0, 10.0, 20.0, 30.0, 40.0])


def test_simplification_preserves_z_when_xy_revisits():
    line = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 10.0], [2.0, 0.0, 20.0], [1.0, 1.0, 30.0], [0.0, 2.0, 40.0]])

    simplified = _simplify_polyline(line, 0.1)

    np.testing.assert_array_equal(simplified, line)


def test_simplify_keeps_the_height_profile_of_a_straight_kerb():
    # Straight in the plane: flat at 7.2 m for 60 m, then a ramp down to 0 m over 40 m.
    y = np.linspace(0.0, 100.0, 101)
    z = np.where(y <= 60.0, 7.2, 7.2 - (y - 60.0) * 7.2 / 40.0)
    line = np.column_stack([np.full_like(y, 116.0), y, z])
    simplified = _simplify_polyline(line, 0.1)
    assert len(simplified) == 3
    assert np.allclose(simplified[1], [116.0, 60.0, 7.2])
    resampled_z = np.interp(y, simplified[:, 1], simplified[:, 2])
    assert np.max(np.abs(resampled_z - z)) <= 0.1


def test_simplify_of_a_flat_polyline_matches_the_plane_simplification():
    rng = np.random.default_rng(0)
    xy = np.cumsum(rng.normal(size=(40, 2)), axis=0)
    line = np.column_stack([xy, np.full(40, 3.0)])
    simplified = _simplify_polyline(line, 0.5)
    from shapely.geometry import LineString

    assert np.allclose(simplified[:, :2], np.array(LineString(xy).simplify(0.5).coords))
    assert np.all(simplified[:, 2] == 3.0)
