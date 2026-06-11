import numpy as np

from bin_factory import puffer_types, schema
from bin_factory.transforms.reindex import reindex_scenario


def _lane(entry=(), exit_lanes=()):
    return schema.MapElement(
        type=int(puffer_types.LaneType.SURFACE_STREET),
        polyline=np.zeros((2, 3), dtype=np.float64),
        entry_lanes=list(entry),
        exit_lanes=list(exit_lanes),
    )


def _scenario(map_):
    return schema.PufferScenario(
        agents={},
        objects={},
        map=map_,
        metadata=schema.ScenarioMetadata(id="x", dataset="d", scenario_length=0, dt=0.0),
    )


def test_reindex_remaps_ids_to_contiguous_range_and_topology():
    scenario = _scenario({10: _lane(exit_lanes=(20,)), 20: _lane(entry=(10,))})

    reindex_scenario(scenario)

    assert set(scenario.map) == {0, 1}
    assert scenario.map[0].exit_lanes == [1]
    assert scenario.map[1].entry_lanes == [0]


def test_reindex_drops_refs_to_absent_lanes():
    scenario = _scenario({10: _lane(exit_lanes=(99,))})  # 99 does not exist

    reindex_scenario(scenario)

    assert scenario.map[0].exit_lanes == []
