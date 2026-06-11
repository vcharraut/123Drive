import numpy as np
import pytest

from bin_factory import puffer_types, schema


def _pts(n):
    return np.zeros((n, 3), dtype=np.float64)


@pytest.mark.parametrize(
    ("road_type", "uses_polyline", "min_points"),
    [
        (puffer_types.LaneType.SURFACE_STREET, True, 2),
        (puffer_types.LaneType.FREEWAY, True, 2),
        (puffer_types.RoadLineType.SOLID_SINGLE_WHITE, True, 2),
        (puffer_types.RoadEdgeType.BOUNDARY, True, 2),
        (puffer_types.MiscRoadType.CROSSWALK, False, 3),
    ],
)
def test_geometry_key_and_min_points(road_type, uses_polyline, min_points):
    polyline = _pts(4) if uses_polyline else None
    polygon = None if uses_polyline else _pts(4)
    elem = schema.MapElement(type=int(road_type), polyline=polyline, polygon=polygon)

    assert elem.uses_polyline is uses_polyline
    assert elem.min_points == min_points
    assert elem.geometry is (polyline if uses_polyline else polygon)


def test_type_predicates_match_puffer_ranges():
    assert schema.MapElement(type=int(puffer_types.LaneType.FREEWAY)).is_lane
    assert schema.MapElement(type=int(puffer_types.RoadLineType.UNKNOWN)).is_line
    assert schema.MapElement(type=int(puffer_types.RoadEdgeType.MEDIAN)).is_edge
    assert schema.MapElement(type=int(puffer_types.MiscRoadType.CROSSWALK)).is_crosswalk


def test_lane_is_not_line_or_edge():
    lane = schema.MapElement(type=int(puffer_types.LaneType.SURFACE_STREET))
    assert lane.is_lane
    assert not lane.is_line
    assert not lane.is_edge
    assert not lane.is_crosswalk
