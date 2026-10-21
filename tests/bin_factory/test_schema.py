import pytest
from py123d.datatypes import map_objects

from bin_factory import puffer_types, schema


@pytest.mark.parametrize("name", ["MapLayer", "LaneType", "RoadLineType", "RoadEdgeType"])
def test_map_enums_mirror_py123d(name):
    ours, theirs = getattr(puffer_types, name), getattr(map_objects, name)
    assert [(m.name, m.value) for m in ours] == [(m.name, m.value) for m in theirs]


def test_polyline_predicates_follow_layer():
    def polyline(layer):
        return schema.RoadPolyline(layer=int(layer), type=1)

    assert polyline(puffer_types.MapLayer.LANE).is_lane
    assert polyline(puffer_types.MapLayer.ROAD_LINE).is_line
    assert polyline(puffer_types.MapLayer.ROAD_EDGE).is_edge
    edge = polyline(puffer_types.MapLayer.ROAD_EDGE)
    assert not edge.is_lane
    assert not edge.is_line
