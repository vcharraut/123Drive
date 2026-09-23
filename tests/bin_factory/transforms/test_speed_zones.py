"""Speed zone construction on a synthetic two-road map.

Layout (x in metres, eastbound unless noted):

    lanes 1,2 (group 10, 0..100)  ->  junction lane 4 (100..120)  ->  lane 5 (group 12, 120..220)
    lane 3 westbound (group 11, 0..100) shares the road centre line with lane 1
    lane 5 -> lane 6 (group 14, 220..320, direct link, different limit)
    lane 5 -> junction lane 8 (220..240) -> lane 7 (group 13, 240..270, short stub)
    lane 5 -> junction lane 9 (turns south) -> lane 10 (group 15, southbound stub at x=230)
"""

import numpy as np
import pytest

from bin_factory import puffer_types, schema
from bin_factory.transforms import speed_zones


HALF_WIDTH_M = 1.75
LIMIT_MPS = 13.9
OTHER_LIMIT_MPS = 25.0


def _lane(x0, x1, y, group, *, limit=LIMIT_MPS, entry=(), exit_=(), westbound=False, junction=False, left=()):
    xs = np.linspace(x0, x1, 6)
    if westbound:
        xs = xs[::-1]
    polyline = np.column_stack([xs, np.full_like(xs, y), np.zeros_like(xs)])
    return _element(polyline, group, limit, entry, exit_, westbound, junction, left)


def _element(polyline, group, limit=LIMIT_MPS, entry=(), exit_=(), westbound=False, junction=False, left=()):
    left_offset = np.array([0.0, -HALF_WIDTH_M if westbound else HALF_WIDTH_M, 0.0])
    return schema.MapElement(
        type=int(puffer_types.LaneType.SURFACE_STREET),
        polyline=polyline,
        left_boundary=polyline + left_offset,
        right_boundary=polyline - left_offset,
        speed_limit_mps=limit,
        entry_lanes=list(entry),
        exit_lanes=list(exit_),
        left_neighbor=list(left),
        lane_group_id=group,
        in_junction=junction,
    )


def _scenario():
    lanes = {
        1: _lane(0, 100, 0.0, 10, exit_=[4]),
        2: _lane(0, 100, -3.5, 10, exit_=[4], left=[1]),
        3: _lane(0, 100, 3.5, 11, westbound=True),
        4: _lane(100, 120, 0.0, 20, entry=[1, 2], exit_=[5], junction=True),
        5: _lane(120, 220, 0.0, 12, entry=[4], exit_=[6, 8, 9]),
        6: _lane(220, 320, 0.0, 14, limit=OTHER_LIMIT_MPS, entry=[5]),
        7: _lane(240, 270, 0.0, 13, entry=[8]),
        8: _lane(220, 240, 0.0, 21, entry=[5], exit_=[7], junction=True),
        9: _element(_turn_south(), 22, entry=[5], exit_=[10], junction=True),
        10: _element(np.array([[230.0, -10.0, 0.0], [230.0, -25.0, 0.0], [230.0, -40.0, 0.0]]), 15, entry=[9]),
    }
    metadata = schema.ScenarioMetadata(id="zones", dataset="test", scenario_length=0, dt=0.1)
    return schema.PufferScenario(agents={}, objects={}, map=lanes, metadata=metadata)


def _turn_south():
    return np.array(
        [[220.0, 0.0, 0.0], [224.0, -1.0, 0.0], [227.5, -3.0, 0.0], [229.5, -6.0, 0.0], [230.0, -10.0, 0.0]]
    )


def _zones(scenario):
    return {eid: e.speed_zone_idx for eid, e in scenario.map.items()}


def test_parallel_lanes_and_opposite_direction_share_a_zone():
    scenario = _scenario()
    speed_zones.compute_speed_zones(scenario, min_zone_extent_m=0.0)
    zones = _zones(scenario)
    assert zones[1] == zones[2] == zones[3]


def test_opposite_direction_pairing_can_be_disabled():
    scenario = _scenario()
    speed_zones.compute_speed_zones(scenario, min_zone_extent_m=0.0, opposite_gap_m=0.0)
    zones = _zones(scenario)
    assert zones[1] == zones[2]
    assert zones[3] != zones[1]


def test_junction_lanes_carry_no_zone_and_split_roads():
    scenario = _scenario()
    speed_zones.compute_speed_zones(scenario, min_zone_extent_m=0.0)
    zones = _zones(scenario)
    assert zones[4] == -1
    assert zones[8] == -1
    assert zones[1] != zones[5]


def test_direct_link_with_different_limit_splits_zone():
    scenario = _scenario()
    speed_zones.compute_speed_zones(scenario, min_zone_extent_m=0.0)
    zones = _zones(scenario)
    assert zones[5] != zones[6]


def test_short_zone_merges_through_junction_into_same_limit_neighbour():
    scenario = _scenario()
    speed_zones.compute_speed_zones(scenario, min_zone_extent_m=50.0)
    zones = _zones(scenario)
    assert zones[7] == zones[5]
    assert zones[6] != zones[5]
    assert zones[1] != zones[5]


def test_short_zone_never_merges_around_a_corner():
    scenario = _scenario()
    speed_zones.compute_speed_zones(scenario, min_zone_extent_m=50.0)
    zones = _zones(scenario)
    assert zones[10] != zones[5]
    assert zones[10] >= 0


def test_zone_ids_are_compact_and_ordered_by_lowest_lane_id():
    scenario = _scenario()
    speed_zones.compute_speed_zones(scenario, min_zone_extent_m=0.0)
    zones = _zones(scenario)
    assert zones[1] == 0
    assert zones[5] == 1
    assert zones[6] == 2
    assert zones[7] == 3
    assert zones[10] == 4
    assert sorted(set(zones.values())) == [-1, 0, 1, 2, 3, 4]


def test_map_without_lane_groups_gets_no_zones():
    scenario = _scenario()
    for lane in scenario.map.values():
        lane.lane_group_id = None
        lane.speed_zone_idx = 7
    speed_zones.compute_speed_zones(scenario)
    assert all(z == -1 for z in _zones(scenario).values())


def test_direct_link_within_limit_tolerance_still_joins():
    scenario = _scenario()
    scenario.map[6].speed_limit_mps = LIMIT_MPS + 0.5 * speed_zones.SPEED_LIMIT_MATCH_TOLERANCE_MPS
    speed_zones.compute_speed_zones(scenario, min_zone_extent_m=0.0)
    zones = _zones(scenario)
    assert zones[5] == zones[6]


@pytest.mark.parametrize("gap_m", [1.0, 8.0])
def test_opposite_pairing_requires_close_inner_boundaries(gap_m):
    scenario = _scenario()
    scenario.map[3].polyline[:, 1] += 20.0
    scenario.map[3].left_boundary[:, 1] += 20.0
    scenario.map[3].right_boundary[:, 1] += 20.0
    speed_zones.compute_speed_zones(scenario, min_zone_extent_m=0.0, opposite_gap_m=gap_m)
    zones = _zones(scenario)
    assert zones[3] != zones[1]
