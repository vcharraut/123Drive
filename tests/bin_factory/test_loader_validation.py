from types import SimpleNamespace

import numpy as np
from py123d.datatypes import detections, map_objects

from bin_factory import puffer_types, schema
from bin_factory.loader import extractor, mapping, validate_scenario


def _track(position, valid):
    length = len(valid)
    return schema.Track(
        type=int(puffer_types.AgentType.VEHICLE),
        position=np.asarray(position, dtype=np.float64),
        heading=np.zeros(length),
        velocity=np.zeros((length, 2)),
        valid=np.asarray(valid, dtype=np.int32),
        length=np.ones(length),
        width=np.ones(length),
        height=np.ones(length),
    )


def test_other_dynamic_labels_are_preserved_as_agents():
    assert mapping.AGENT_TYPE_MAP[detections.DefaultBoxDetectionLabel.TRAIN] == puffer_types.AgentType.OTHER
    assert mapping.AGENT_TYPE_MAP[detections.DefaultBoxDetectionLabel.ANIMAL] == puffer_types.AgentType.OTHER
    assert mapping.AGENT_TYPE_MAP[detections.DefaultBoxDetectionLabel.OTHER] == puffer_types.AgentType.OTHER


def test_missing_velocity_uses_real_frame_gap_and_keeps_observed():
    track = extractor._make_empty_track(5)
    track.position[:, 0] = [0, 0, 2, 0, 8]
    track.valid[:] = [1, 0, 1, 0, 1]
    track.velocity[2] = [9, 3]

    extractor._fill_missing_velocities(track, 0.5)

    np.testing.assert_allclose(track.velocity[0], [2, 0])
    np.testing.assert_allclose(track.velocity[2], [9, 3])
    np.testing.assert_allclose(track.velocity[4], [6, 0])
    np.testing.assert_array_equal(track.velocity[[1, 3]], 0)


def test_singleton_ego_corridor_is_valid():
    state = SimpleNamespace(center_se3=SimpleNamespace(x=1.0, y=2.0))
    map_api = SimpleNamespace(
        available_map_layers=[],
        map_is_per_log=False,
        query=lambda corridor, layers, predicate: {},
    )

    elements, stop_zones, lane_ids = extractor._extract_map(map_api, np.zeros(3), [state], False)

    assert elements == {}
    assert stop_zones == []
    assert lane_ids == set()


def test_stop_zones_drop_lanes_outside_corridor():
    state = SimpleNamespace(center_se3=SimpleNamespace(x=1.0, y=2.0))
    zone = SimpleNamespace(
        layer=map_objects.MapLayer.STOP_ZONE,
        stop_zone_type=map_objects.StopZoneType.STOP_SIGN,
        outline_3d=SimpleNamespace(array=np.zeros((4, 3))),
        lane_ids=[42],
    )
    map_api = SimpleNamespace(
        available_map_layers=[map_objects.MapLayer.STOP_ZONE],
        map_is_per_log=False,
        query=lambda corridor, layers, predicate: {map_objects.MapLayer.STOP_ZONE: [zone]},
    )

    _elements, stop_zones, _lane_ids = extractor._extract_map(map_api, np.zeros(3), [state], False)

    assert stop_zones == []


def test_validation_rejects_non_xyz_geometry_and_nonfinite_dt():
    scenario = schema.PufferScenario(
        agents={},
        objects={},
        map={
            1: schema.MapElement(
                type=int(puffer_types.LaneType.SURFACE_STREET),
                polyline=np.zeros((2, 2)),
            )
        },
        metadata=schema.ScenarioMetadata(id="x", dataset="d", scenario_length=0, dt=np.nan),
    )

    errors = validate_scenario(scenario, level=1)

    assert "metadata.dt must be finite" in errors
    assert any("invalid shape" in error for error in errors)


def test_validation_reports_malformed_controls_and_lane_graph():
    scenario = schema.PufferScenario(
        agents={},
        objects={},
        map={},
        traffic_controls=["invalid"],
        lane_graph=[],
        metadata=schema.ScenarioMetadata(id="x", dataset="d", scenario_length=0, dt=0.0),
    )

    errors = validate_scenario(scenario, level=1)

    assert "TrafficControl 0 must be a dict" in errors
    assert "lane_graph must be a dict" in errors


def test_teleport_threshold_scales_with_dt():
    scenario = schema.PufferScenario(
        agents={0: _track([[0, 0, 0], [6, 0, 0]], [1, 1])},
        objects={},
        map={},
        metadata=schema.ScenarioMetadata(id="x", dataset="d", scenario_length=2, dt=0.1),
    )

    assert any("teleports" in error for error in validate_scenario(scenario, level=2))
