"""Which py123d lanes become bin lanes, and which ones shape the map centroid."""

import types

import numpy as np
import pytest
from py123d.datatypes import map_objects

from bin_factory.loader import extractor


SHOULDER_TYPE = getattr(map_objects.LaneType, "SHOULDER", None)


def _lane(lane_type, lane_group_id, y_m=0.0):
    centerline = np.array([[0.0, y_m, 0.0], [10.0, y_m, 0.0]])
    return types.SimpleNamespace(
        layer=map_objects.MapLayer.LANE,
        lane_type=lane_type,
        lane_group_id=lane_group_id,
        centerline_3d=types.SimpleNamespace(array=centerline),
    )


class _MapApi:
    def __init__(self, objects_by_layer):
        self._objects = objects_by_layer

    @property
    def available_map_layers(self):
        return list(self._objects)

    def get_all_map_objects_in_layer(self, layer):
        return list(self._objects[layer])


def test_undefined_lane_outside_every_lane_group_is_a_surface_not_a_lane():
    assert extractor._is_routable_lane(_lane(map_objects.LaneType.SURFACE_STREET, 7), has_lane_groups=True)
    # nuPlan: undefined lane types, but every lane belongs to a lane group
    assert extractor._is_routable_lane(_lane(map_objects.LaneType.UNDEFINED, 7), has_lane_groups=True)
    # OpenDRIVE none/restricted lanes: undefined type and no lane group
    assert not extractor._is_routable_lane(_lane(map_objects.LaneType.UNDEFINED, None), has_lane_groups=True)
    # datasets without lane groups keep their undefined lanes
    assert extractor._is_routable_lane(_lane(map_objects.LaneType.UNDEFINED, None), has_lane_groups=False)


@pytest.mark.skipif(SHOULDER_TYPE is None, reason="py123d without shoulder-typed lanes")
def test_shoulder_lanes_are_never_bin_lanes():
    assert not extractor._is_routable_lane(_lane(SHOULDER_TYPE, None), has_lane_groups=True)
    assert not extractor._is_routable_lane(_lane(SHOULDER_TYPE, None), has_lane_groups=False)


@pytest.mark.skipif(SHOULDER_TYPE is None, reason="py123d without shoulder-typed lanes")
def test_centroid_ignores_shoulders_and_none_lanes():
    api = _MapApi(
        {
            map_objects.MapLayer.LANE: [
                _lane(map_objects.LaneType.SURFACE_STREET, 1, y_m=0.0),
                _lane(SHOULDER_TYPE, None, y_m=-100.0),
                _lane(map_objects.LaneType.UNDEFINED, None, y_m=100.0),
            ],
            map_objects.MapLayer.LANE_GROUP: [],
        }
    )
    centroid = extractor._compute_centroid(None, api)
    assert np.allclose(centroid, [5.0, 0.0, 0.0])
