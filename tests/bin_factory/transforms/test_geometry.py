import numpy as np

from bin_factory.transforms.geometry import _simplify_polyline, arc_length, polyline_length


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


def test_simplification_recovers_z_from_original_station():
    line = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 10.0], [2.0, 0.0, 20.0], [3.0, 0.0, 30.0], [3.0, 1.0, 40.0]]
    )

    simplified = _simplify_polyline(line, 0.5)

    np.testing.assert_allclose(simplified[:, 2], [0.0, 10.0, 30.0, 40.0])


def test_simplification_preserves_z_when_xy_revisits():
    line = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 10.0], [2.0, 0.0, 20.0], [1.0, 1.0, 30.0], [0.0, 2.0, 40.0]]
    )

    simplified = _simplify_polyline(line, 0.1)

    np.testing.assert_array_equal(simplified, line)
