"""Little-endian 123Drive binary format, version 3.

The header stores magic[8], version as u32, and section_count as u32. Each directory entry
packs tag[8], offset as u64, byte_length as u64, record_count as u32, and frame_count as u32.
Sections and indexed records start on 8-byte boundaries. Indexed sections begin
with record_count + 1 relative u64 offsets, and each record ends with zero padding.

AGENTS and OBJECTS store id, type, first, n, and flags as i32, followed by float32 channels
for x, y, optional z, heading, vx, vy, length, width, and height, then valid[n] as u8 padded
to 4 bytes. Agents append an int list route, then route_entry_s[count] and route_exit_s[count] as float32: the
cum_length span driven on each route lane. Flags 0..2 store constant dimensions as scalars,
bit 3 omits zero z, and bit 4 adds an overriding frame count after flags.
ROADS store id, type, n, and flags as i32, then x, y, optional z, and heading as float32[n].
Bit 0 omits zero z. Lanes append four topology int lists, speed as f32, cum_length[n], and curvature[n].
TRAFFIC stores id and type as i32, stop_line as 6xf32, heading as f32, and state and lane int lists.
LANEGRPH stores lane_ids as i32 and distances as f32 of shape count by count without record indexing.
METADATA stores id, dataset, and location strings, scenario_length as i32, dt as f32, and target int lists.
TLPHASE1 stores junction_id and phase_idx as i32 per traffic control.
SOURCE01 stores origin_xyz as 3xf64, map_has_z and timestamps_are_relative as i32, count and reserved as u32,
timestamps_us[count] as i64, five source strings, and source IDs per agent and object.
Strings store byte_length as i32, UTF-8 bytes, and padding to 4 bytes. Int lists store count and values as i32.
Readers can skip unknown sections by length. Enum values live in puffer_types.py.
"""

import dataclasses
import itertools
import struct
from collections.abc import Sequence
from typing import Any

import numpy as np

from bin_factory import schema


BINARY_MAGIC = b"123DRIVE"
BINARY_VERSION = 3
DIRECTORY_ENTRY = struct.Struct("<8sQQII")
TRAFFIC_PHASE_SECTION_TAG = b"TLPHASE1"
SOURCE_SECTION_TAG = b"SOURCE01"
_SOURCE_STRINGS = ("scene_uuid", "log_name", "split", "log_version", "map_version")
CURVATURE_HALF_WINDOW = 5.0  # metres of lane on each side of a point averaged into its curvature
_TRACK_FIELDS = (("position", 3), ("heading", 1), ("velocity", 2), ("length", 1), ("width", 1), ("height", 1))
BufferLike = bytes | bytearray | memoryview


def _write_int_list(buf: bytearray, values: Sequence[int]) -> None:
    buf.extend(struct.pack("<i", len(values)))
    buf.extend(np.asarray(values, dtype="<i4").tobytes())


def _write_string(buf: bytearray, value: str) -> None:
    encoded = value.encode("utf-8")
    buf.extend(struct.pack("<i", len(encoded)))
    buf.extend(encoded)
    buf.extend(bytes(-len(buf) % 4))


def _index_records(records: Sequence[BufferLike]) -> bytearray:
    buf = bytearray(8 * (len(records) + 1))
    for index, record in enumerate(records):
        struct.pack_into("<Q", buf, 8 * index, len(buf))
        buf.extend(record)
        buf.extend(bytes(-len(buf) % 8))
    struct.pack_into("<Q", buf, 8 * len(records), len(buf))
    return buf


def _write_tracks(tracks: dict[int, schema.Track], has_route: bool, frame_count: int) -> bytearray:
    records = []
    for eid, track in tracks.items():
        valid = np.flatnonzero(track.valid)
        first, n = (int(valid[0]), int(valid[-1] - valid[0] + 1)) if len(valid) else (0, 0)
        position = np.asarray(track.position, dtype="<f4")[first : first + n]
        dimensions = [
            np.asarray(getattr(track, name), dtype="<f4")[first : first + n] for name in ("length", "width", "height")
        ]
        constant = [n > 0 and bool(np.all(values == values[0])) for values in dimensions]
        omit_z = not np.any(position[:, 2])
        flags = sum(int(value) << bit for bit, value in enumerate(constant)) | (int(omit_z) << 3)
        flags |= int(len(track.position) != frame_count) << 4
        buf = bytearray(struct.pack("<5i", int(eid), int(track.type), first, n, flags))
        if flags & 16:
            buf.extend(struct.pack("<i", len(track.position)))
        buf.extend(position[:, : 2 if omit_z else 3].T.tobytes())
        buf.extend(np.asarray(track.heading, dtype="<f4")[first : first + n].tobytes())
        buf.extend(np.asarray(track.velocity, dtype="<f4")[first : first + n].T.tobytes())
        for values, scalar in zip(dimensions, constant, strict=False):
            buf.extend(values[:1].tobytes() if scalar else values.tobytes())
        buf.extend(np.asarray(track.valid, dtype="u1")[first : first + n].tobytes())
        buf.extend(bytes(-len(buf) % 4))
        if has_route:
            _write_int_list(buf, track.route)
            buf.extend(np.asarray(track.route_entry_s, dtype="<f4").tobytes())
            buf.extend(np.asarray(track.route_exit_s, dtype="<f4").tobytes())
        records.append(buf)
    return _index_records(records)


def _compute_curvature(pts: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Compute signed curvature in 1/m with left turns positive.

    Measure heading change over +-CURVATURE_HALF_WINDOW of 2D arc length clipped to the lane,
    then divide by that window length. Interpolate heading between chord midpoints and merge
    collinear resampled points so they do not split a chord.
    """
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))])
    unwrapped = np.unwrap(seg)
    corner = np.concatenate([[True], np.abs(np.diff(unwrapped)) > 1e-4, [True]])  # rad, float32 jitter is ~1e-5
    chord_s = s[corner]
    chord_heading = unwrapped[np.flatnonzero(corner)[:-1]]
    lo = np.maximum(s - CURVATURE_HALF_WINDOW, 0.0)
    hi = np.minimum(s + CURVATURE_HALF_WINDOW, s[-1])
    chord_mid = (chord_s[:-1] + chord_s[1:]) / 2
    turn = np.interp(hi, chord_mid, chord_heading) - np.interp(lo, chord_mid, chord_heading)
    return (turn / (hi - lo)).astype("<f4")


def _write_roads(road_map: dict[int, schema.MapElement]) -> bytearray:
    records = []
    for eid, elem in road_map.items():
        xyz = elem.geometry
        xyz_f = np.asarray(xyz, dtype="<f4")
        pts = np.asarray(xyz, dtype=np.float64)
        seg = np.arctan2(np.diff(pts[:, 1]), np.diff(pts[:, 0]))
        heading = np.append(seg, seg[-1]).astype("<f4")
        omit_z = not np.any(xyz_f[:, 2])
        buf = bytearray(struct.pack("<4i", int(eid), int(elem.type), len(xyz_f), int(omit_z)))
        buf.extend(xyz_f[:, : 2 if omit_z else 3].T.tobytes())
        buf.extend(heading.tobytes())
        if elem.is_lane:
            for lane_list in (elem.entry_lanes, elem.exit_lanes, elem.left_neighbor, elem.right_neighbor):
                _write_int_list(buf, lane_list)
            buf.extend(struct.pack("<f", elem.speed_limit_mps))
            buf.extend(np.asarray(elem.cum_length, dtype="<f4").tobytes())
            buf.extend(_compute_curvature(pts, seg).tobytes())
        records.append(buf)
    return _index_records(records)


def _write_traffic_controls(tcs: Sequence[schema.TrafficControl]) -> bytearray:
    records = []
    for tc in tcs:
        buf = bytearray(struct.pack("<ii", int(tc.id), int(tc.type)))
        buf.extend(np.asarray(tc.stop_line, dtype="<f4").tobytes())
        buf.extend(struct.pack("<f", float(tc.heading)))
        for values in (tc.states, tc.controlled_lanes):
            _write_int_list(buf, values)
        records.append(buf)
    return _index_records(records)


def _write_lane_graph(lane_graph: dict[str, Any] | None) -> tuple[bytearray, int]:
    buf = bytearray()
    if not lane_graph:
        return buf, 0
    buf.extend(np.asarray(lane_graph["lane_ids"], dtype="<i4").tobytes())
    buf.extend(np.asarray(lane_graph["distances"], dtype="<f4").tobytes())
    return buf, len(lane_graph["lane_ids"])


def _write_metadata(metadata: schema.ScenarioMetadata) -> bytearray:
    buf = bytearray()
    for name in ("id", "dataset", "location"):
        _write_string(buf, str(getattr(metadata, name)))
    buf.extend(struct.pack("<if", int(metadata.scenario_length), float(metadata.dt)))
    for values in (metadata.objects_of_interest, metadata.tracks_to_predict):
        _write_int_list(buf, values)
    return buf


def _write_traffic_phases(tcs: Sequence[schema.TrafficControl]) -> bytearray:
    buf = bytearray()
    for tc in tcs:
        buf.extend(struct.pack("<ii", tc.intersection_id if tc.phase_idx >= 0 else -1, tc.phase_idx))
    return buf


def _write_source(
    metadata: schema.ScenarioMetadata,
    agents: dict[int, schema.Track],
    objects: dict[int, schema.Track],
) -> bytearray:
    buf = bytearray(
        struct.pack(
            "<3diiII",
            *metadata.origin_xyz,
            metadata.map_has_z,
            metadata.timestamps_are_relative,
            len(metadata.timestamps_us),
            0,
        )
    )
    buf.extend(np.asarray(metadata.timestamps_us, dtype="<i8").tobytes())
    source_strings = [getattr(metadata, name) for name in _SOURCE_STRINGS]
    source_strings.extend(track.source_id for track in (*agents.values(), *objects.values()))
    for value in source_strings:
        _write_string(buf, value)
    return buf


def scenario_to_binary(scenario: schema.PufferScenario) -> bytes:
    agents, road_map, tcs, objects = scenario.agents, scenario.map, scenario.traffic_controls, scenario.objects
    frame_count = scenario.metadata.scenario_length
    object_frames = len(next(iter(objects.values())).position) if objects else frame_count
    graph_buf, graph_count = _write_lane_graph(scenario.lane_graph)
    sections = [
        (b"AGENTS", _write_tracks(agents, True, frame_count), len(agents), frame_count),
        (b"ROADS", _write_roads(road_map), len(road_map), 0),
        (b"TRAFFIC", _write_traffic_controls(tcs), len(tcs), 0),
        (b"OBJECTS", _write_tracks(objects, False, object_frames), len(objects), object_frames),
        (b"LANEGRPH", graph_buf, graph_count, 0),
        (b"METADATA", _write_metadata(scenario.metadata), 1, 0),
        (TRAFFIC_PHASE_SECTION_TAG, _write_traffic_phases(tcs), len(tcs), 0),
        (SOURCE_SECTION_TAG, _write_source(scenario.metadata, agents, objects), 1, 0),
    ]

    result = bytearray(struct.pack("<8sII", BINARY_MAGIC, BINARY_VERSION, len(sections)))
    result.extend(bytes(DIRECTORY_ENTRY.size * len(sections)))
    for index, (tag, payload, count, frames) in enumerate(sections):
        result.extend(bytes(-len(result) % 8))
        DIRECTORY_ENTRY.pack_into(
            result, 16 + index * DIRECTORY_ENTRY.size, tag, len(result), len(payload), count, frames
        )
        result.extend(payload)
    return bytes(result)


class StaticBinaryError(ValueError):
    """Raised when a .bin payload is not a supported static map binary."""


def read_sections(data: BufferLike) -> dict[bytes, tuple[memoryview, int, int]]:
    magic, version, count = struct.unpack_from("<8sII", data)
    if magic != BINARY_MAGIC or version != BINARY_VERSION:
        raise ValueError("Unsupported binary format; regenerate bins with the current converter")
    end = 16 + count * DIRECTORY_ENTRY.size
    sections: dict[bytes, tuple[memoryview, int, int]] = {}
    for index in range(count):
        tag, offset, size, records, frames = DIRECTORY_ENTRY.unpack_from(data, 16 + index * DIRECTORY_ENTRY.size)
        if offset % 8 or offset < end or offset + size > len(data) or tag.rstrip(b"\0") in sections:
            raise ValueError("Invalid binary section directory")
        sections[tag.rstrip(b"\0")] = (memoryview(data)[offset : offset + size], records, frames)
        end = offset + size
    if end != len(data):
        raise ValueError("Binary has trailing bytes after sections")
    return sections


@dataclasses.dataclass
class _Reader:
    data: BufferLike
    offset: int = 0

    def i32(self) -> int:
        value = int(struct.unpack_from("<i", self.data, self.offset)[0])
        self.offset += 4
        return value

    def f32(self) -> float:
        value = float(struct.unpack_from("<f", self.data, self.offset)[0])
        self.offset += 4
        return value

    def i32_array(self, n: int) -> np.ndarray:
        arr = np.frombuffer(self.data, dtype="<i4", count=n, offset=self.offset)
        self.offset += 4 * n
        return arr

    def f32_array(self, n: int) -> np.ndarray:
        arr = np.frombuffer(self.data, dtype="<f4", count=n, offset=self.offset)
        self.offset += 4 * n
        return arr

    def int_list(self) -> list[int]:
        return self.i32_array(self.i32()).tolist()

    def string(self) -> str:
        n = self.i32()
        value = bytes(self.data[self.offset : self.offset + n]).decode("utf-8")
        self.offset = (self.offset + n + 3) & ~3
        return value

    def records(self, count: int) -> list["_Reader"]:
        offsets = np.frombuffer(self.data, dtype="<u8", count=count + 1)
        if offsets[0] != 8 * (count + 1) or offsets[-1] != len(self.data) or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("Invalid binary record offsets")
        return [_Reader(self.data[int(start) : int(stop)]) for start, stop in itertools.pairwise(offsets)]


def static_binary_to_scenario(data: BufferLike, source: str = "<bytes>") -> schema.PufferScenario:
    try:
        sections = read_sections(data)
        if sections[b"AGENTS"][1]:
            raise StaticBinaryError(f"{source} is not static: found {sections[b'AGENTS'][1]} agents")
        road_data, road_count, _ = sections[b"ROADS"]
        road_map = _read_roads(_Reader(road_data), road_count)
        traffic_data, traffic_count, _ = sections[b"TRAFFIC"]
        traffic_controls = _read_traffic_controls(_Reader(traffic_data), traffic_count)
        object_data, object_count, object_frames = sections[b"OBJECTS"]
        objects = _read_objects(_Reader(object_data), object_count, object_frames)
        graph_data, graph_count, _ = sections[b"LANEGRPH"]
        lane_graph = _read_lane_graph(_Reader(graph_data), graph_count)
        metadata = _read_metadata(_Reader(sections[b"METADATA"][0]))
        if TRAFFIC_PHASE_SECTION_TAG in sections:
            _read_traffic_phases(_Reader(sections[TRAFFIC_PHASE_SECTION_TAG][0]), traffic_controls)
        if SOURCE_SECTION_TAG in sections:
            _read_source(_Reader(sections[SOURCE_SECTION_TAG][0]), metadata, objects)
    except (struct.error, ValueError, KeyError) as exc:
        if isinstance(exc, StaticBinaryError):
            raise
        raise StaticBinaryError(f"{source}: {exc}") from exc

    return schema.PufferScenario(
        agents={},
        objects=objects,
        map=road_map,
        traffic_controls=traffic_controls,
        lane_graph=lane_graph,
        metadata=metadata,
    )


def _read_roads(reader: _Reader, n_roads: int) -> dict[int, schema.MapElement]:
    road_map = {}
    for record in reader.records(n_roads):
        element_id, road_type, n_points, flags = (record.i32() for _ in range(4))
        x = record.f32_array(n_points)
        y = record.f32_array(n_points)
        z = np.zeros(n_points) if flags & 1 else record.f32_array(n_points)
        record.f32_array(n_points)  # heading is recomputed when serializing
        element = schema.MapElement(type=road_type)
        xyz = np.column_stack([x, y, z]).astype(np.float64)
        setattr(element, "polyline" if element.uses_polyline else "polygon", xyz)
        if element.is_lane:
            element.entry_lanes = record.int_list()
            element.exit_lanes = record.int_list()
            element.left_neighbor = record.int_list()
            element.right_neighbor = record.int_list()
            element.speed_limit_mps = record.f32()
            element.cum_length = record.f32_array(n_points).astype(np.float64)
            element.length = float(element.cum_length[-1])
            record.f32_array(n_points)  # curvature is recomputed when serializing
        road_map[element_id] = element
    return road_map


def _read_traffic_controls(reader: _Reader, n_traffic: int) -> list[schema.TrafficControl]:
    traffic_controls = []
    for record in reader.records(n_traffic):
        control_id = record.i32()
        control_type = record.i32()
        stop_line = record.f32_array(6).reshape(2, 3).astype(np.float64)
        heading = record.f32()
        states = record.int_list()
        controlled_lanes = record.int_list()
        traffic_controls.append(
            schema.TrafficControl(
                id=control_id,
                type=control_type,
                stop_line=stop_line,
                heading=heading,
                states=states,
                controlled_lanes=controlled_lanes,
            )
        )
    return traffic_controls


def _read_objects(reader: _Reader, n_objects: int, frame_count: int) -> dict[int, schema.Track]:
    objects = {}
    for record in reader.records(n_objects):
        object_id, object_type, first, n_stored, flags = (record.i32() for _ in range(5))
        n_frames = record.i32() if flags & 16 else frame_count
        channels: dict[str, Any] = {}
        for index, (name, width) in enumerate(_TRACK_FIELDS):
            values = np.zeros((width, n_frames), dtype=np.float64)
            if name == "position" and flags & 8:
                values[:2, first : first + n_stored] = record.f32_array(2 * n_stored).reshape(2, n_stored)
            else:
                scalar = index >= 3 and flags & (1 << (index - 3))
                stored = record.f32_array(1 if scalar else width * n_stored)
                values[:, first : first + n_stored] = (
                    stored.reshape(width, -1) if n_stored else stored.reshape(width, 0)
                )
            channels[name] = values.T.copy() if width > 1 else values[0]
        valid = np.zeros(n_frames, dtype=np.int32)
        valid[first : first + n_stored] = np.frombuffer(record.data, dtype="u1", count=n_stored, offset=record.offset)
        objects[object_id] = schema.Track(type=object_type, valid=valid, **channels)
    return objects


def _read_traffic_phases(reader: _Reader, traffic_controls: list[schema.TrafficControl]) -> None:
    for traffic_control in traffic_controls:
        traffic_control.intersection_id = reader.i32()
        traffic_control.phase_idx = reader.i32()


def _read_lane_graph(reader: _Reader, n_lanes: int) -> dict[str, Any] | None:
    if n_lanes == 0:
        return None
    lane_ids = reader.i32_array(n_lanes).tolist()
    distances = reader.f32_array(n_lanes * n_lanes).reshape(n_lanes, n_lanes)
    return {"lane_ids": lane_ids, "distances": distances}


def _read_metadata(reader: _Reader) -> schema.ScenarioMetadata:
    scenario_id = reader.string()
    dataset = reader.string()
    location = reader.string()
    scenario_length = reader.i32()
    dt = reader.f32()
    objects_of_interest = reader.int_list()
    tracks_to_predict = reader.int_list()
    return schema.ScenarioMetadata(
        id=scenario_id,
        dataset=dataset,
        location=location,
        scenario_length=scenario_length,
        dt=dt,
        objects_of_interest=objects_of_interest,
        tracks_to_predict=tracks_to_predict,
    )


def _read_source(reader: _Reader, metadata: schema.ScenarioMetadata, objects: dict[int, schema.Track]) -> None:
    metadata.origin_xyz = struct.unpack_from("<3d", reader.data)
    reader.offset = 24
    metadata.map_has_z = bool(reader.i32())
    metadata.timestamps_are_relative = bool(reader.i32())
    count = reader.i32()
    reader.i32()  # reserved, keeps int64 timestamps aligned
    metadata.timestamps_us = list(struct.unpack_from(f"<{count}q", reader.data, reader.offset))
    reader.offset += 8 * count
    for name in _SOURCE_STRINGS:
        setattr(metadata, name, reader.string())
    for track in objects.values():
        track.source_id = reader.string()
