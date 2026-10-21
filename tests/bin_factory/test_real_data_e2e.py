"""End-to-end conversion tests using committed 123D scenarios.

``tests/py123d_data`` contains two WOD-motion logs with per-log maps. Tests run discovery,
extraction, validation, transforms, and serialization, then check the binary layout and data.
"""

import pathlib

import numpy as np
import pytest
from binary_reader import parse_binary as _parse

from bin_factory import loader, main, puffer_types, serialize
from bin_factory.preparation import prepare_scenario


DATA_ROOT = pathlib.Path(__file__).parent.parent / "py123d_data"
SPLIT = "wod-motion_train"
LOG_NAMES = ["4124fb4053fd031d", "7a4bd197a1dcf0e3"]
TL_LOG = "7a4bd197a1dcf0e3"  # log with traffic-light detections + crosswalk + all line/edge types
MAP_LOCATIONS = ["Town02", "Town10HD"]  # opendrive maps under tests/py123d_data/maps/opendrive


def _config(**overrides):
    cfg = main.build_parser().parse_args(["--py123d_path", str(DATA_ROOT)])
    cfg.validate_level = 2
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _discover(log_name):
    scenes = loader.discover_scenes(
        py123d_data_root=str(DATA_ROOT), workers=1, split_names=[SPLIT], log_names=[log_name]
    )
    assert len(scenes) == 1, f"expected exactly one scene for {log_name}, got {len(scenes)}"
    return scenes[0]


def _process(log_name, **cfg_overrides):
    return prepare_scenario(_discover(log_name), _config(**cfg_overrides))


def _discover_map(location):
    maps = loader.discover_scenes(py123d_data_root=str(DATA_ROOT), workers=1, map_only=True, datasets=["opendrive"])
    match = [m for m in maps if m.location == location]
    assert len(match) == 1, f"expected one opendrive map for {location}, got {len(match)}"
    return match[0]


def _process_map(location, **cfg_overrides):
    return prepare_scenario(_discover_map(location), _config(map_only=True, **cfg_overrides))


# Discovery


def test_discovers_both_fixtures():
    scenes = loader.discover_scenes(py123d_data_root=str(DATA_ROOT), workers=1, split_names=[SPLIT])
    assert {s.log_name for s in scenes} == set(LOG_NAMES)
    assert all(s.dataset == "wod-motion" for s in scenes)


# Binary layout


@pytest.mark.parametrize("log_name", LOG_NAMES)
@pytest.mark.parametrize("infer_traffic_lights", [False, True])
def test_binary_layout_and_counts_match_processed_scenario(log_name, infer_traffic_lights):
    scenario = _process(log_name, infer_traffic_lights=infer_traffic_lights)
    data = serialize.scenario_to_binary(scenario)
    parsed = _parse(data)
    # Walking every field and landing exactly on the buffer end proves layout consistency
    # against real, variable-shaped data (mixed map types, routes, TL state vectors).
    assert parsed["consumed"] == len(data)

    assert parsed["counts"] == (
        len(scenario.agents),
        len(scenario.polylines),
        len(scenario.surfaces),
        len(scenario.traffic_controls),
        len(scenario.objects),
    )


@pytest.mark.parametrize("log_name", LOG_NAMES)
def test_source_data_restores_coordinates_and_preserves_identities_and_timestamps(log_name):
    scene = _discover(log_name)
    scenario = prepare_scenario(scene, _config())
    parsed = _parse(serialize.scenario_to_binary(scenario))
    source = parsed["source"]

    assert source["scene_uuid"] == scene.scene_uuid
    assert source["log_name"] == log_name
    assert source["split"] == SPLIT
    assert source["log_version"] == scene.log_metadata.version
    assert source["map_version"] == scene.get_map_api().version
    assert source["map_has_z"] is True
    assert source["timestamps_are_relative"] is True
    assert source["timestamps_us"] == [timestamp.time_us for timestamp in scene.get_all_iteration_timestamps()]
    assert source["origin_xyz"] == scenario.metadata.origin_xyz

    ego = scene.get_ego_state_se3_at_iteration(0)
    world_position = scenario.agents[0].position[0] + source["origin_xyz"]
    np.testing.assert_allclose(
        world_position,
        [ego.center_se3.x, ego.center_se3.y, ego.center_se3.z - ego.bounding_box_se3.height / 2],
    )
    aux = scene.get_all_custom_modality_metadatas()["aux"].metadata
    assert parsed["agents"][0]["source_id"] == aux["track_index_to_token"][aux["sdc_track_index"]]
    detections = scene.get_box_detections_se3_at_iteration(0)
    token = detections[0].attributes.track_token
    assert token in {agent["source_id"] for agent in parsed["agents"]}
    track = next(track for track in scenario.agents.values() if track.source_id == token)
    box = detections[0].bounding_box_se3
    np.testing.assert_allclose(
        track.position[0] + source["origin_xyz"],
        [box.center_se3.x, box.center_se3.y, box.center_se3.z - box.height / 2],
    )


# Cross-references


@pytest.mark.parametrize("log_name", LOG_NAMES)
def test_references_stay_within_bounds(log_name):
    scenario = _process(log_name)
    parsed = _parse(serialize.scenario_to_binary(scenario))
    lane_ids = {r["id"] for r in parsed["roads"] if r["lane"]}
    road_ids = {r["id"] for r in parsed["roads"]}

    for road in parsed["roads"]:
        assert road["npts"] >= 2  # undersized polylines are dropped by serialize
        assert set(road["entry"]) <= lane_ids
        assert set(road["exit"]) <= lane_ids

    for agent in parsed["agents"]:
        assert set(agent["route"]) <= lane_ids  # routes reference lanes only

    for tc in parsed["tcs"]:
        assert set(tc["lanes"]) <= road_ids
        # Traffic-light controls carry a per-frame state vector; other controls (e.g. stop signs) don't.
        if tc["type"] == int(puffer_types.TCType.TRAFFIC_LIGHT):
            assert len(tc["states"]) == scenario.metadata.scenario_length
        else:
            assert tc["states"] == []

    assert set(parsed["lane_graph_ids"]) <= lane_ids


@pytest.mark.parametrize("log_name", LOG_NAMES)
def test_map_types_and_geometry_are_supported(log_name):
    scenario = _process(log_name)
    for elem in scenario.polylines.values():
        assert elem.type in puffer_types.POLYLINE_TYPES[elem.layer]
        assert elem.polyline is not None
        assert len(elem.polyline) >= 2
    for surface in scenario.surfaces.values():
        assert surface.layer in puffer_types.SURFACE_LAYERS
        assert surface.polygon is not None
        assert len(surface.polygon) >= 3


# Metadata round-trip


@pytest.mark.parametrize("log_name", LOG_NAMES)
def test_metadata_round_trips(log_name):
    scenario = _process(log_name)
    parsed = _parse(serialize.scenario_to_binary(scenario))
    meta = scenario.metadata
    assert parsed["id"] == meta.id
    assert parsed["dataset"] == "wod-motion"
    assert parsed["scenario_length"] == meta.scenario_length
    assert abs(parsed["dt"] - meta.dt) < 1e-6
    assert parsed["ttp"] == list(meta.tracks_to_predict)
    assert parsed["ooi"] == list(meta.objects_of_interest)
    assert set(parsed["ttp"]) <= set(scenario.agents)


# Traffic lights


def test_traffic_lights_present_and_inferred():
    scenario = _process(TL_LOG)
    parsed = _parse(serialize.scenario_to_binary(scenario))
    assert parsed["counts"][3] > 0, "expected traffic controls in the TL fixture"

    inferred = _process(TL_LOG, infer_traffic_lights=True)
    parsed_inferred = _parse(serialize.scenario_to_binary(inferred))
    assert parsed_inferred["counts"][3] >= parsed["counts"][3]


# Determinism and CLI output


@pytest.mark.parametrize("log_name", LOG_NAMES)
def test_serialization_is_deterministic(log_name):
    s1 = _process(log_name)
    s2 = _process(log_name)
    assert serialize.scenario_to_binary(s1) == serialize.scenario_to_binary(s2)


@pytest.mark.parametrize("log_name", LOG_NAMES)
def test_convert_one_writes_parseable_bin(tmp_path, log_name):
    config = _config(validate_level=2)
    scene = _discover(log_name)
    main._convert_one(scene, tmp_path, config)

    bins = list(tmp_path.glob("*.bin"))
    assert len(bins) == 1
    assert bins[0].name.startswith("wod-motion__")
    data = bins[0].read_bytes()
    assert len(data) > 0
    assert _parse(data)["consumed"] == len(data)
    assert all(count >= 0 for count in _parse(data)["counts"])


# OpenDRIVE map-only conversion


def test_discovers_opendrive_maps():
    maps = loader.discover_scenes(py123d_data_root=str(DATA_ROOT), workers=1, map_only=True, datasets=["opendrive"])
    assert {m.location for m in maps} == set(MAP_LOCATIONS)
    assert all(m.dataset == "opendrive" for m in maps)


@pytest.mark.parametrize("location", MAP_LOCATIONS)
@pytest.mark.parametrize("reverse_road_edges", [False, True])  # opendrive preset reverses edges
def test_map_only_binary_walk_consumes_whole_buffer(location, reverse_road_edges):
    scenario = _process_map(location, reverse_road_edges=reverse_road_edges)
    data = serialize.scenario_to_binary(scenario)
    assert _parse(data)["consumed"] == len(data)


@pytest.mark.parametrize("location", MAP_LOCATIONS)
def test_map_only_has_map_but_no_dynamic_entities(location):
    scenario = _process_map(location)
    assert scenario.agents == {}
    assert scenario.objects == {}
    assert scenario.metadata.scenario_length == 0
    assert scenario.metadata.dt == 0.0
    assert len(scenario.polylines) > 0

    parsed = _parse(serialize.scenario_to_binary(scenario))
    # No agents, no objects; map + (stop-zone-derived) traffic controls remain.
    assert parsed["counts"] == (0, len(scenario.polylines), len(scenario.surfaces), len(scenario.traffic_controls), 0)
    assert parsed["agents"] == []
    assert parsed["objects"] == []


@pytest.mark.parametrize("location", MAP_LOCATIONS)
def test_map_only_references(location):
    scenario = _process_map(location)
    parsed = _parse(serialize.scenario_to_binary(scenario))
    lane_ids = {r["id"] for r in parsed["roads"] if r["lane"]}
    road_ids = {r["id"] for r in parsed["roads"]}
    for road in parsed["roads"]:
        assert road["npts"] >= 2
        assert set(road["entry"]) <= lane_ids
        assert set(road["exit"]) <= lane_ids
    for tc in parsed["tcs"]:
        assert set(tc["lanes"]) <= road_ids
    assert set(parsed["lane_graph_ids"]) <= lane_ids


@pytest.mark.parametrize("location", MAP_LOCATIONS)
def test_map_only_metadata_uses_location_as_id(location):
    scenario = _process_map(location)
    parsed = _parse(serialize.scenario_to_binary(scenario))
    assert parsed["id"] == location  # opendrive identity is the map location
    assert parsed["dataset"] == "opendrive"
    assert parsed["scenario_length"] == 0
    assert parsed["ttp"] == []
    assert parsed["ooi"] == []


@pytest.mark.parametrize("location", MAP_LOCATIONS)
def test_map_only_convert_one_writes_parseable_bin(tmp_path, location):
    config = _config(map_only=True, validate_level=2, reverse_road_edges=True)
    main._convert_one(_discover_map(location), tmp_path, config)

    bins = list(tmp_path.glob("*.bin"))
    assert len(bins) == 1
    assert bins[0].name == f"opendrive__{location}.bin"
    data = bins[0].read_bytes()
    assert _parse(data)["consumed"] == len(data)
