"""Junction membership of shoulders in the extractor's shoulder context."""

import types

import numpy as np
import pytest
from py123d.datatypes import map_objects

from bin_factory.loader import extractor


SHOULDER_LAYER = getattr(map_objects.MapLayer, "SHOULDER", None)
SHOULDER_TYPE = getattr(map_objects.LaneType, "SHOULDER", None)


def _surface(points_xy, layer, lane_type=None):
    ring = [*points_xy, points_xy[0]]
    outline = np.array([[x, y, 0.0] for x, y in ring])
    obj = types.SimpleNamespace(
        layer=layer,
        outline_2d=types.SimpleNamespace(array=outline[:, :2]),
        outline_3d=types.SimpleNamespace(array=outline),
    )
    if lane_type is not None:
        obj.lane_type = lane_type
    return obj


def _rect(x0, x1, y0, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


class _MapApi:
    def __init__(self, objects_by_layer):
        self._objects = objects_by_layer

    @property
    def available_map_layers(self):
        return list(self._objects)

    def get_all_map_objects_in_layer(self, layer):
        return list(self._objects[layer])


@pytest.mark.skipif(SHOULDER_LAYER is None, reason="py123d without a SHOULDER layer")
def test_shoulder_mostly_inside_an_intersection_is_a_junction_shoulder_even_with_its_centre_outside():
    intersection = _surface(_rect(-8, 4, -4, 20), map_objects.MapLayer.INTERSECTION)
    mostly_inside = _surface(_rect(-8, 12, -4, 0), SHOULDER_LAYER)  # 60 % inside, representative point outside
    barely_inside = _surface(_rect(2, 22, -4, 0), SHOULDER_LAYER)  # 10 % inside
    far_away = _surface(_rect(30, 50, -4, 0), SHOULDER_LAYER)
    api = _MapApi(
        {map_objects.MapLayer.INTERSECTION: [intersection], SHOULDER_LAYER: [mostly_inside, barely_inside, far_away]}
    )
    non_junction, junction = extractor._extract_shoulder_context(api, np.zeros(3))
    assert len(junction) == 2  # the intersection outline and the shoulder that is 60 % inside it
    assert sorted(round(float(outline[:, 0].min()), 1) for outline in non_junction) == [2.0, 30.0]


@pytest.mark.skipif(SHOULDER_LAYER is None, reason="py123d without a SHOULDER layer")
def test_without_intersections_every_shoulder_is_outside():
    api = _MapApi({SHOULDER_LAYER: [_surface(_rect(0, 10, -4, 0), SHOULDER_LAYER)]})
    non_junction, junction = extractor._extract_shoulder_context(api, np.zeros(3))
    assert len(non_junction) == 1
    assert junction == []


@pytest.mark.skipif(SHOULDER_TYPE is None, reason="py123d without shoulder-typed lanes")
def test_shoulder_typed_lanes_are_read_when_there_is_no_shoulder_layer():
    driving = _surface(_rect(0, 10, 0, 3.5), map_objects.MapLayer.LANE, map_objects.LaneType.SURFACE_STREET)
    shoulder = _surface(_rect(0, 10, -3.5, 0), map_objects.MapLayer.LANE, SHOULDER_TYPE)
    api = _MapApi({map_objects.MapLayer.LANE: [driving, shoulder]})
    non_junction, junction = extractor._extract_shoulder_context(api, np.zeros(3))
    assert len(non_junction) == 1
    assert junction == []
    assert float(non_junction[0][:, 1].min()) == -3.5
