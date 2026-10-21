import copy
import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
from binary_reader import parse_binary as _parse

from bin_factory import puffer_types, schema, serialize


def _track(length, agent_type, valid, route=(), base=0.0):
    position = np.arange(length * 3, dtype=np.float64).reshape(length, 3) + base
    return schema.Track(
        type=int(agent_type),
        position=position,
        heading=np.linspace(0.0, 1.0, length),
        velocity=np.ones((length, 2), dtype=np.float64),
        valid=np.asarray(valid, dtype=np.int32),
        length=np.full(length, 4.5),
        width=np.full(length, 2.0),
        height=np.full(length, 1.6),
        route=list(route),
        route_entry_s=[0.0] * len(route),
        route_exit_s=[1.0] * len(route),
    )


def _lane(road_type, points, entry=(), exit_=(), speed=-1.0, layer=puffer_types.MapLayer.LANE):
    polyline = np.arange(points * 3, dtype=np.float64).reshape(points, 3)
    cum = np.linalg.norm(np.diff(polyline, axis=0), axis=1).cumsum()
    cum_length = np.concatenate([[0.0], cum]) if points > 1 else np.zeros(points)
    return schema.RoadPolyline(
        layer=int(layer),
        type=int(road_type),
        polyline=polyline,
        speed_limit_mps=speed,
        entry_lanes=list(entry),
        exit_lanes=list(exit_),
        length=float(cum_length[-1]) if points else 0.0,
        cum_length=cum_length,
    )


def _build_scenario():
    metadata = schema.ScenarioMetadata(
        id="scn_42",
        dataset="nuplan",
        scenario_length=3,
        dt=0.1,
        objects_of_interest=[1],
        tracks_to_predict=[0, 1],
        origin_xyz=(600000.123456789, 4000000.987654321, -5.25),
        map_has_z=True,
        scene_uuid="source-scene",
        log_name="log-" + "é" * 160,
        split="nuplan-mini_val",
        log_version="0.5.1",
        map_version="0.8.0",
        timestamps_us=[9007199254740993, 9007199254840994, 9007199254940992],
    )
    agents = {
        0: _track(3, puffer_types.AgentType.VEHICLE, [1, 1, 1], route=[10, 11]),
        1: _track(3, puffer_types.AgentType.PEDESTRIAN, [0, 1, 0], base=100.0),
    }
    crosswalk = schema.RoadSurface(
        layer=int(puffer_types.MapLayer.CROSSWALK),
        polygon=np.arange(12, dtype=np.float64).reshape(4, 3),
    )
    polylines = {
        10: _lane(puffer_types.LaneType.SURFACE_STREET, 4, exit_=[11], speed=13.0),
        11: _lane(puffer_types.LaneType.FREEWAY, 2, entry=[10]),
        12: _lane(puffer_types.RoadLineType.SOLID_WHITE, 3, layer=puffer_types.MapLayer.ROAD_LINE),
    }
    objects = {0: _track(3, puffer_types.ObjectType.GENERIC_OBJECT, [1, 1, 1])}
    agents[0].source_id = "ego-source"
    agents[1].source_id = "pedestrian-source"
    objects[0].source_id = "object-source"
    traffic_controls = [
        schema.TrafficControl(
            id=5,
            type=int(puffer_types.TCType.TRAFFIC_LIGHT),
            stop_line=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            heading=0.5,
            states=[int(puffer_types.TLState.GREEN)] * 3,
            controlled_lanes=[10],
        )
    ]
    lane_graph = {"lane_ids": [10, 11], "distances": np.array([[0.0, 3.0], [np.inf, 0.0]])}
    return schema.PufferScenario(
        agents=agents,
        objects=objects,
        polylines=polylines,
        surfaces={13: crosswalk},
        metadata=metadata,
        traffic_controls=traffic_controls,
        lane_graph=lane_graph,
    )


def test_round_trip_high_level_fields():
    scenario = _build_scenario()
    data = serialize.scenario_to_binary(scenario)
    assert serialize.scenario_to_binary(scenario) == data
    parsed = _parse(data)

    # Header counts: agents, road polylines, road surfaces, traffic controls, objects.
    assert parsed["counts"] == (2, 3, 1, 1, 1)

    ego = next(a for a in parsed["agents"] if a["id"] == 0)
    assert ego["route"] == [10, 11]

    lane10 = next(ln for ln in parsed["lanes"] if ln["id"] == 10)
    assert lane10["exit"] == [11]
    assert lane10["speed"] == 13.0
    assert lane10["npts"] == 4

    assert parsed["id"] == "scn_42"
    assert parsed["dataset"] == "nuplan"
    assert parsed["scenario_length"] == 3
    assert abs(parsed["dt"] - 0.1) < 1e-6  # float32 round-trip
    assert parsed["ooi"] == [1]
    assert parsed["ttp"] == [0, 1]
    assert parsed["source"] == {
        name: getattr(scenario.metadata, name)
        for name in (
            "origin_xyz",
            "map_has_z",
            "scene_uuid",
            "log_name",
            "split",
            "log_version",
            "map_version",
            "timestamps_us",
            "timestamps_are_relative",
        )
    }
    assert [agent["source_id"] for agent in parsed["agents"]] == ["ego-source", "pedestrian-source"]
    assert parsed["object_sources"] == ["object-source"]

    # The parser walks every field; ending exactly at the buffer end proves layout consistency.
    assert parsed["consumed"] == len(data)


def test_trajectory_stores_only_valid_span():
    scenario = _build_scenario()
    scenario.objects[0].valid[:] = 0
    parsed = _parse(serialize.scenario_to_binary(scenario))
    # (T, first, n): frames from the first to the last valid one; none for an all-invalid track.
    assert [a["span"] for a in parsed["agents"]] == [(3, 0, 3), (3, 1, 1)]
    assert parsed["object_spans"] == [(3, 0, 0)]


def test_big_endian_inputs_serialize_identically():
    little = _build_scenario()
    big = copy.deepcopy(little)
    for track in [*big.agents.values(), *big.objects.values()]:
        for field in ("position", "heading", "velocity", "valid", "length", "width", "height"):
            array = getattr(track, field)
            setattr(track, field, array.astype(">i4" if field == "valid" else ">f8"))
    for element in big.polylines.values():
        element.polyline = element.polyline.astype(">f8")
        if element.cum_length is not None:
            element.cum_length = element.cum_length.astype(">f8")
    for surface in big.surfaces.values():
        surface.polygon = surface.polygon.astype(">f8")
    big.lane_graph["distances"] = big.lane_graph["distances"].astype(">f8")

    assert serialize.scenario_to_binary(big) == serialize.scenario_to_binary(little)


def test_static_reader_preserves_source_data():
    scenario = _build_scenario()
    scenario.agents = {}
    data = serialize.scenario_to_binary(scenario)
    decoded = serialize.static_binary_to_scenario(data)

    assert decoded.metadata.origin_xyz == scenario.metadata.origin_xyz
    assert decoded.metadata.timestamps_us == scenario.metadata.timestamps_us
    assert decoded.metadata.log_name == scenario.metadata.log_name
    assert decoded.objects[0].source_id == "object-source"
    assert serialize.scenario_to_binary(decoded) == data


def test_static_reader_skips_unknown_section():
    scenario = _build_scenario()
    scenario.agents = {}
    data = bytearray(serialize.scenario_to_binary(scenario))
    section_count = struct.unpack_from("<I", data, 12)[0]
    data[16 + (section_count - 1) * 32 : 24 + (section_count - 1) * 32] = b"UNKNOWN!"
    decoded = serialize.static_binary_to_scenario(bytes(data))

    assert decoded.metadata.id == scenario.metadata.id
    assert decoded.metadata.timestamps_us == []
    assert decoded.metadata.log_name == ""
    assert decoded.objects[0].source_id == ""


@pytest.mark.parametrize("valid", [[0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1]])
def test_compact_channels_preserve_values_and_validity(valid):
    scenario = _build_scenario()
    scenario.agents = {}
    track = scenario.objects[0]
    track.valid[:] = valid
    track.position[:, 2] = 0.0
    track.width[:] = [2.0, 2.1, 2.2]
    data = serialize.scenario_to_binary(scenario)
    decoded = serialize.static_binary_to_scenario(data).objects[0]
    indices = np.flatnonzero(valid)
    first, stop = (indices[0], indices[-1] + 1) if len(indices) else (0, 0)
    for name in ("position", "heading", "velocity", "length", "width", "height"):
        expected = np.zeros_like(getattr(track, name))
        expected[first:stop] = getattr(track, name)[first:stop].astype(np.float32)
        np.testing.assert_array_equal(getattr(decoded, name), expected)
    np.testing.assert_array_equal(decoded.valid, valid)
    object_data, _count, _frames = serialize.read_sections(data)[b"OBJECTS"]
    record_offset = struct.unpack_from("<Q", object_data)[0]
    flags = struct.unpack_from("<i", object_data, record_offset + 16)[0]
    assert flags & 8
    assert bool(flags & 1) == bool(len(indices))
    assert not flags & 2 or stop - first == 1


def test_track_frames_can_differ_from_section_default():
    scenario = _build_scenario()
    scenario.agents = {}
    scenario.objects[1] = _track(1, puffer_types.ObjectType.BARRIER, [1])
    decoded = serialize.static_binary_to_scenario(serialize.scenario_to_binary(scenario))
    assert len(decoded.objects[0].position) == 3
    assert len(decoded.objects[1].position) == 1
    np.testing.assert_array_equal(decoded.objects[1].valid, [1])


def test_metadata_strings_are_not_truncated():
    scenario = _build_scenario()
    scenario.agents = {}
    scenario.metadata.id = "é" * 200
    scenario.metadata.location = "long-location-" * 20
    decoded = serialize.static_binary_to_scenario(serialize.scenario_to_binary(scenario))
    assert decoded.metadata.id == scenario.metadata.id
    assert decoded.metadata.location == scenario.metadata.location


def test_zero_geometry_elevation_is_implicit_without_losing_nonzero_z():
    scenario = _build_scenario()
    scenario.agents = {}
    scenario.metadata.map_has_z = False
    scenario.polylines[10].polyline[:, 2] = 0.0
    decoded = serialize.static_binary_to_scenario(serialize.scenario_to_binary(scenario))
    for eid, element in scenario.polylines.items():
        np.testing.assert_array_equal(decoded.polylines[eid].polyline, element.polyline.astype(np.float32))
    for eid, surface in scenario.surfaces.items():
        np.testing.assert_array_equal(decoded.surfaces[eid].polygon, surface.polygon.astype(np.float32))


@pytest.mark.parametrize("version", [0, 2, 3, 4])
def test_unsupported_version_is_rejected(version):
    data = bytearray(serialize.scenario_to_binary(_build_scenario()))
    struct.pack_into("<I", data, 8, version)
    with pytest.raises(serialize.StaticBinaryError, match="Unsupported binary format"):
        serialize.static_binary_to_scenario(data)


def test_record_offsets_cannot_point_outside_section():
    scenario = _build_scenario()
    scenario.agents = {}
    data = bytearray(serialize.scenario_to_binary(scenario))
    road_start = struct.unpack_from("<Q", data, 16 + 32 + 8)[0]
    struct.pack_into("<Q", data, road_start, len(data) + 8)
    with pytest.raises(serialize.StaticBinaryError, match="Invalid binary record offsets"):
        serialize.static_binary_to_scenario(data)


@pytest.mark.skipif(shutil.which("node") is None, reason="Browser parser verification needs Node.js")
def test_browser_reader_preserves_tracks_sources_and_curvature(tmp_path):
    scenario = _build_scenario()
    scenario.agents[0].valid[:] = [1, 0, 1]
    scenario.agents[0].width[:] = [2.0, 2.1, 2.2]
    scenario.agents[1].position[:, 2] = 0.0
    scenario.objects[1] = _track(1, puffer_types.ObjectType.BARRIER, [1])
    scenario.polylines[10].polyline = np.array([[0, 0, 0], [5, 0, 0], [5, 5, 0], [10, 5, 0]], dtype=float)
    scenario.polylines[10].cum_length = np.array([0, 5, 10, 15], dtype=float)
    scenario.traffic_controls[0].intersection_id = 42
    scenario.traffic_controls[0].phase_idx = 2
    binary = tmp_path / "scenario.bin"
    binary.write_bytes(serialize.scenario_to_binary(scenario))
    parser = Path(__file__).parents[2] / "src/viz/web/static/binary_parser.js"
    result = subprocess.run(
        [
            "node",
            "-e",
            """
const fs = require('fs'), assert = require('assert');
global.window = {};
require(process.argv[1]);
const bytes = fs.readFileSync(process.argv[2]);
const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const scene = window.parsePufferBinary(buffer);
assert.equal(window.parsePufferBinary(buffer, {laneGraph: false}).lane_graph, null);
assert(scene.agents[0].heading instanceof Float32Array);
assert(scene.agents[0].valid instanceof Uint8Array);
assert(scene.metadata.timestamps_us instanceof BigInt64Array);
assert(scene.road_polylines[0].curvature.some(value => value !== 0));
assert.equal(scene.road_polylines[0].length, 15);
assert.equal(scene.road_polylines[2].layer, 9);
assert.equal(scene.road_surfaces[0].layer, 3);
assert.equal(scene.road_surfaces[0].xyz.length, 4);
assert.equal(scene.traffic_control_elements[0].junction_id, 42);
assert.equal(scene.traffic_control_elements[0].phase_idx, 2);
const version = buffer.slice(0);
new DataView(version).setUint32(8, 99, true);
assert.throws(() => window.parsePufferBinary(version), /Unsupported binary format/);
const unknown = buffer.slice(0);
const lastEntry = 16 + 32 * (new DataView(unknown).getUint32(12, true) - 1);
new Uint8Array(unknown, lastEntry, 8).set(new TextEncoder().encode('UNKNOWN!'));
assert.equal(window.parsePufferBinary(unknown).metadata.timestamps_us, undefined);
console.log(JSON.stringify(scene, (_, value) => {
  if (typeof value === 'bigint') return value.toString();
  if (ArrayBuffer.isView(value)) return Array.from(value);
  return value;
}));
""",
            str(parser),
            str(binary),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    parsed = json.loads(result.stdout)
    assert parsed["metadata"]["origin_xyz"] == list(scenario.metadata.origin_xyz)
    assert parsed["metadata"]["timestamps_us"] == [str(value) for value in scenario.metadata.timestamps_us]
    assert parsed["metadata"]["log_name"] == scenario.metadata.log_name
    for actual, expected in zip(
        parsed["agents"] + parsed["objects"], [*scenario.agents.values(), *scenario.objects.values()], strict=True
    ):
        valid = np.flatnonzero(expected.valid)
        first, stop = (valid[0], valid[-1] + 1) if len(valid) else (0, 0)
        for field, key in [
            ("position", "xyz"),
            ("velocity", "velocity"),
            ("heading", "heading"),
            ("length", "length"),
            ("width", "width"),
            ("height", "height"),
        ]:
            channel = np.zeros_like(getattr(expected, field))
            channel[first:stop] = getattr(expected, field)[first:stop].astype(np.float32)
            np.testing.assert_array_equal(actual[key], channel)
        assert actual["valid"] == expected.valid.tolist()
        assert actual["source_id"] == expected.source_id
    assert [agent["route"] for agent in parsed["agents"]] == [track.route for track in scenario.agents.values()]
