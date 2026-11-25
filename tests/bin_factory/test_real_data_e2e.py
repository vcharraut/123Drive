"""End-to-end conversion tests using committed 123D scenarios.

``tests/py123d_data`` contains two WOD-motion logs with per-log maps. Tests run discovery,
extraction, validation, transforms, and serialization, then check the binary layout and data.
"""

import dataclasses
import pathlib

import numpy as np
import pytest
import shapely

from bin_factory import loader, main, puffer_types, serialize
from bin_factory.preparation import ConversionSettings, prepare_scenario


DATA_ROOT = pathlib.Path(__file__).parent.parent / "py123d_data"
SPLIT = "wod-motion_train"
LOG_NAMES = ["4124fb4053fd031d", "7a4bd197a1dcf0e3"]
TL_LOG = "7a4bd197a1dcf0e3"  # log with traffic-light detections + crosswalk + all line/edge types
MAP_LOCATIONS = ["Town02", "Town10HD"]  # opendrive maps under tests/py123d_data/maps/opendrive


def _config(**overrides):
    return dataclasses.replace(ConversionSettings(validate_level=2), **overrides)


def _discover(log_name):
    scenes = loader.discover_scenes(
        py123d_data_root=str(DATA_ROOT), workers=1, split_names=[SPLIT], log_names=[log_name]
    )
    assert len(scenes) == 1, f"expected exactly one scene for {log_name}, got {len(scenes)}"
    return scenes[0]


def _discover_map(location):
    maps = loader.discover_scenes(py123d_data_root=str(DATA_ROOT), workers=1, map_only=True, datasets=["opendrive"])
    match = [m for m in maps if m.location == location]
    assert len(match) == 1, f"expected one opendrive map for {location}, got {len(match)}"
    return match[0]


def _round_trip(scenario):
    # The reader rejects any unread or misplaced byte, so decoding at all proves the layout is consistent.
    return serialize.binary_to_scenario(serialize.scenario_to_binary(scenario))


@pytest.fixture(scope="module", params=LOG_NAMES)
def converted(request):
    """(log name, discovered scene, converted scenario, scenario read back from its binary), converted once."""
    scene = _discover(request.param)
    scenario = prepare_scenario(scene, _config())
    return request.param, scene, scenario, _round_trip(scenario)


@pytest.fixture(scope="module", params=MAP_LOCATIONS)
def converted_map(request):
    """(location, converted map-only scenario, scenario read back from its binary), converted once."""
    scenario = prepare_scenario(_discover_map(request.param), _config())
    return request.param, scenario, _round_trip(scenario)


# Discovery


def test_discovers_both_fixtures():
    scenes = loader.discover_scenes(py123d_data_root=str(DATA_ROOT), workers=1, split_names=[SPLIT])
    assert {s.log_name for s in scenes} == set(LOG_NAMES)
    assert all(s.dataset == "wod-motion" for s in scenes)


# Binary layout


def test_binary_layout_and_counts_match_processed_scenario(converted):
    _, _, scenario, decoded = converted

    assert list(decoded.agents) == list(scenario.agents)
    assert list(decoded.polylines) == list(scenario.polylines)
    assert list(decoded.surfaces) == list(scenario.surfaces)
    assert [tc.id for tc in decoded.traffic_controls] == [tc.id for tc in scenario.traffic_controls]
    assert list(decoded.objects) == list(scenario.objects)
    for agent_id, track in scenario.agents.items():
        assert decoded.agents[agent_id].route == track.route
        np.testing.assert_allclose(decoded.agents[agent_id].route_entry_s, track.route_entry_s, rtol=1e-6)
        np.testing.assert_allclose(decoded.agents[agent_id].route_exit_s, track.route_exit_s, rtol=1e-6)
    assert any(track.route for track in scenario.agents.values())


def test_source_data_restores_coordinates_and_preserves_identities_and_timestamps(converted):
    log_name, scene, scenario, decoded = converted
    source = decoded.metadata

    assert source.scene_uuid == scene.scene_uuid
    assert source.log_name == log_name
    assert source.split == SPLIT
    assert source.log_version == scene.log_metadata.version
    assert source.map_version == scene.get_map_api().version
    assert source.map_has_z is True
    assert source.timestamps_are_relative is True
    assert source.timestamps_us == [timestamp.time_us for timestamp in scene.get_all_iteration_timestamps()]
    assert source.origin_xyz == scenario.metadata.origin_xyz

    ego = scene.get_ego_state_se3_at_iteration(0)
    world_position = scenario.agents[0].position[0] + source.origin_xyz
    np.testing.assert_allclose(
        world_position,
        [ego.center_se3.x, ego.center_se3.y, ego.center_se3.z - ego.bounding_box_se3.height / 2],
    )
    aux = scene.get_all_custom_modality_metadatas()["aux"].metadata
    assert decoded.agents[0].source_id == aux["track_index_to_token"][aux["sdc_track_index"]]
    detections = scene.get_box_detections_se3_at_iteration(0)
    token = detections[0].attributes.track_token
    assert token in {agent.source_id for agent in decoded.agents.values()}
    track = next(track for track in scenario.agents.values() if track.source_id == token)
    box = detections[0].bounding_box_se3
    np.testing.assert_allclose(
        track.position[0] + source.origin_xyz,
        [box.center_se3.x, box.center_se3.y, box.center_se3.z - box.height / 2],
    )


# Cross-references


def test_references_stay_within_bounds(converted):
    _, _, scenario, decoded = converted
    lane_ids = {eid for eid, road in decoded.polylines.items() if road.is_lane}
    road_ids = set(decoded.polylines)

    for road in decoded.polylines.values():
        assert len(road.polyline) >= 2  # undersized polylines are dropped before serializing
        assert set(road.entry_lanes) <= lane_ids
        assert set(road.exit_lanes) <= lane_ids

    for agent in decoded.agents.values():
        assert set(agent.route) <= lane_ids  # routes reference lanes only

    for tc in decoded.traffic_controls:
        assert set(tc.controlled_lanes) <= road_ids
        # Traffic-light controls carry a per-frame state vector; other controls (e.g. stop signs) don't.
        if tc.type == int(puffer_types.TCType.TRAFFIC_LIGHT):
            assert len(tc.states) == scenario.metadata.scenario_length
        else:
            assert tc.states == []


def test_map_types_and_geometry_are_supported(converted):
    scenario = converted[2]
    for elem in scenario.polylines.values():
        assert elem.type in puffer_types.POLYLINE_TYPES[elem.layer]
        assert elem.polyline is not None
        assert len(elem.polyline) >= 2
    for surface in scenario.surfaces.values():
        assert surface.layer in puffer_types.SURFACE_LAYERS
        assert len(surface.polygon) >= 3


# Metadata round-trip


def test_metadata_round_trips(converted):
    _, _, scenario, decoded_scenario = converted
    decoded, meta = decoded_scenario.metadata, scenario.metadata
    assert decoded.id == meta.id
    assert decoded.dataset == "wod-motion"
    assert decoded.scenario_length == meta.scenario_length
    assert abs(decoded.dt - meta.dt) < 1e-6
    assert decoded.tracks_to_predict == list(meta.tracks_to_predict)
    assert decoded.objects_of_interest == list(meta.objects_of_interest)
    assert set(decoded.tracks_to_predict) <= set(scenario.agents)


# Traffic lights


@pytest.mark.parametrize("converted", [TL_LOG], indirect=True)
def test_traffic_lights_present(converted):
    assert converted[3].traffic_controls, "expected traffic controls in the TL fixture"


@pytest.mark.parametrize("converted", [TL_LOG], indirect=True)
def test_inferred_lights_cover_more_frames_without_conflicting_greens(converted):
    inferred = _round_trip(prepare_scenario(_discover(TL_LOG), _config(infer_traffic_lights=True)))

    def lights(scenario):
        return {
            tc.id: np.array(tc.states)
            for tc in scenario.traffic_controls
            if tc.type == puffer_types.TCType.TRAFFIC_LIGHT
        }

    raw_lights, inferred_lights = lights(converted[3]), lights(inferred)
    assert sum((s != 0).sum() for s in inferred_lights.values()) > 1.5 * sum(
        (s != 0).sum() for s in raw_lights.values()
    )
    lanes = {tc.id: inferred.polylines[tc.controlled_lanes[0]].polyline[:, :2] for tc in inferred.traffic_controls}
    go = {
        tid: np.isin(s, (puffer_types.TLState.GREEN, puffer_types.TLState.YELLOW)) for tid, s in inferred_lights.items()
    }
    crossing_pairs = 0
    for a in inferred_lights:
        for b in inferred_lights:
            first, second = lanes[a], lanes[b]
            start_a, start_b = (np.diff(line[:2], axis=0)[0] for line in (first, second))
            cos = abs(start_a @ start_b) / (np.linalg.norm(start_a) * np.linalg.norm(start_b))
            if a < b and cos < np.cos(np.radians(45)) and shapely.LineString(first).crosses(shapely.LineString(second)):
                crossing_pairs += 1
                assert not (go[a] & go[b]).any(), f"lights {a} and {b} cross and are both green"
    assert crossing_pairs > 0


# Determinism and CLI output


def test_serialization_is_deterministic(converted):
    log_name, _, scenario, _ = converted
    again = prepare_scenario(_discover(log_name), _config())
    assert serialize.scenario_to_binary(again) == serialize.scenario_to_binary(scenario)


def test_convert_one_writes_parseable_bin(tmp_path, converted):
    main._convert_one(_discover(converted[0]), tmp_path, _config())

    bins = list(tmp_path.glob("*.bin"))
    assert len(bins) == 1
    assert bins[0].name.startswith("wod-motion__")
    assert serialize.binary_to_scenario(bins[0].read_bytes()).agents


# OpenDRIVE map-only conversion


def test_discovers_opendrive_maps():
    maps = loader.discover_scenes(py123d_data_root=str(DATA_ROOT), workers=1, map_only=True, datasets=["opendrive"])
    assert {m.location for m in maps} == set(MAP_LOCATIONS)
    assert all(m.dataset == "opendrive" for m in maps)


@pytest.mark.parametrize("location", MAP_LOCATIONS)
def test_map_only_binary_walk_consumes_whole_buffer_with_reversed_edges(location):
    scenario = prepare_scenario(_discover_map(location), _config(reverse_road_edges=True))  # as the opendrive preset
    assert list(_round_trip(scenario).polylines) == list(scenario.polylines)


def test_map_only_has_map_but_no_dynamic_entities(converted_map):
    _, scenario, decoded = converted_map
    assert scenario.agents == {}
    assert scenario.objects == {}
    assert scenario.metadata.scenario_length == 0
    assert scenario.metadata.dt == 0.0
    assert len(scenario.polylines) > 0

    # No agents, no objects; map + (stop-zone-derived) traffic controls remain.
    assert decoded.agents == {}
    assert decoded.objects == {}
    assert list(decoded.polylines) == list(scenario.polylines)
    assert list(decoded.surfaces) == list(scenario.surfaces)
    assert len(decoded.traffic_controls) == len(scenario.traffic_controls)


def test_map_only_references(converted_map):
    decoded = converted_map[2]
    lane_ids = {eid for eid, road in decoded.polylines.items() if road.is_lane}
    road_ids = set(decoded.polylines)
    for road in decoded.polylines.values():
        assert len(road.polyline) >= 2
        assert set(road.entry_lanes) <= lane_ids
        assert set(road.exit_lanes) <= lane_ids
    for tc in decoded.traffic_controls:
        assert set(tc.controlled_lanes) <= road_ids


def test_map_only_metadata_uses_location_as_id(converted_map):
    location, _, decoded = converted_map
    assert decoded.metadata.id == location  # opendrive identity is the map location
    assert decoded.metadata.dataset == "opendrive"
    assert decoded.metadata.scenario_length == 0
    assert decoded.metadata.tracks_to_predict == []
    assert decoded.metadata.objects_of_interest == []


def test_map_only_convert_one_writes_parseable_bin(tmp_path, converted_map):
    location = converted_map[0]
    main._convert_one(_discover_map(location), tmp_path, _config(reverse_road_edges=True))

    bins = list(tmp_path.glob("*.bin"))
    assert len(bins) == 1
    assert bins[0].name == f"opendrive__{location}.bin"
    assert serialize.binary_to_scenario(bins[0].read_bytes()).polylines
