"""Little-endian 123Drive binary format, version 0.4.

The header stores magic[8], version as major and minor u16, and section_count as u32. Each directory entry
packs tag[8], offset as u64, byte_length as u64, record_count as u32, and frame_count as u32.
Sections and indexed records start on 8-byte boundaries. Indexed sections begin
with record_count + 1 relative u64 offsets, and each record ends with zero padding.

AGENTS and OBJECTS store id, type, first, n, and flags as i32, followed by float32 channels
for x, y, optional z, heading, vx, vy, length, width, and height, then valid[n] as u8 padded
to 4 bytes. Agents append an int list route, then route_entry_s[count] and route_exit_s[count] as float32: the
cum_length span driven on each route lane. Flags 0..2 store constant dimensions as scalars,
bit 3 omits zero z, and bit 4 adds an overriding frame count after flags.
ROADPOLY stores lanes, road lines, and road edges: id, layer, type, n, and flags as i32, then x, y, optional z,
and heading as float32[n]. Layer and type are py123d MapLayer and subtype values. Bit 0 omits zero z.
Lanes append four topology int lists, speed as f32, cum_length[n], and curvature[n].
ROADSURF stores surface outlines: id, layer, n, and flags as i32, then x, y, and optional z as float32[n].
Polylines and surfaces share one id space.
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
from bin_factory.generators.graph import build_lane_distance_matrix


BINARY_MAGIC = b"123DRIVE"
BINARY_VERSION = (0, 4)
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


def _write_polylines(polylines: dict[int, schema.RoadPolyline]) -> bytearray:
    records = []
    for eid, elem in polylines.items():
        xyz = elem.polyline
        xyz_f = np.asarray(xyz, dtype="<f4")
        pts = np.asarray(xyz, dtype=np.float64)
        seg = np.arctan2(np.diff(pts[:, 1]), np.diff(pts[:, 0]))
        heading = np.append(seg, seg[-1]).astype("<f4")
        omit_z = not np.any(xyz_f[:, 2])
        buf = bytearray(struct.pack("<5i", int(eid), int(elem.layer), int(elem.type), len(xyz_f), int(omit_z)))
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


def _write_surfaces(surfaces: dict[int, schema.RoadSurface]) -> bytearray:
    records = []
    for eid, elem in surfaces.items():
        xyz_f = np.asarray(elem.polygon, dtype="<f4")
        omit_z = not np.any(xyz_f[:, 2])
        buf = bytearray(struct.pack("<4i", int(eid), int(elem.layer), len(xyz_f), int(omit_z)))
        buf.extend(xyz_f[:, : 2 if omit_z else 3].T.tobytes())
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


def _write_lane_graph(polylines: dict[int, schema.RoadPolyline]) -> tuple[bytearray, int]:
    lane_graph = build_lane_distance_matrix(polylines)
    if not lane_graph:
        return bytearray(), 0
    buf = bytearray(np.asarray(lane_graph["lane_ids"], dtype="<i4").tobytes())
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
    return bytearray().join(
        struct.pack("<ii", tc.intersection_id if tc.phase_idx >= 0 else -1, tc.phase_idx) for tc in tcs
    )


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
    agents, tcs, objects = scenario.agents, scenario.traffic_controls, scenario.objects
    frame_count = scenario.metadata.scenario_length
    object_frames = len(next(iter(objects.values())).position) if objects else frame_count
    graph_buf, graph_count = _write_lane_graph(scenario.polylines)
    sections = [
        (b"AGENTS", _write_tracks(agents, True, frame_count), len(agents), frame_count),
        (b"ROADPOLY", _write_polylines(scenario.polylines), len(scenario.polylines), 0),
        (b"ROADSURF", _write_surfaces(scenario.surfaces), len(scenario.surfaces), 0),
        (b"TRAFFIC", _write_traffic_controls(tcs), len(tcs), 0),
        (b"OBJECTS", _write_tracks(objects, False, object_frames), len(objects), object_frames),
        (b"LANEGRPH", graph_buf, graph_count, 0),
        (b"METADATA", _write_metadata(scenario.metadata), 1, 0),
        (TRAFFIC_PHASE_SECTION_TAG, _write_traffic_phases(tcs), len(tcs), 0),
        (SOURCE_SECTION_TAG, _write_source(scenario.metadata, agents, objects), 1, 0),
    ]

    result = bytearray(struct.pack("<8sHHI", BINARY_MAGIC, *BINARY_VERSION, len(sections)))
    result.extend(bytes(DIRECTORY_ENTRY.size * len(sections)))
    for index, (tag, payload, count, frames) in enumerate(sections):
        result.extend(bytes(-len(result) % 8))
        DIRECTORY_ENTRY.pack_into(
            result, 16 + index * DIRECTORY_ENTRY.size, tag, len(result), len(payload), count, frames
        )
        result.extend(payload)
    return bytes(result)


class BinaryFormatError(ValueError):
    """Raised when bytes are not a well-formed binary of the supported version."""


def read_sections(data: BufferLike) -> dict[bytes, tuple[memoryview, int, int]]:
    magic, major, minor, count = struct.unpack_from("<8sHHI", data)
    if magic != BINARY_MAGIC or (major, minor) != BINARY_VERSION:
        raise ValueError("Unsupported binary format; regenerate bins with the current converter")
    end = 16 + count * DIRECTORY_ENTRY.size
    sections: dict[bytes, tuple[memoryview, int, int]] = {}
    view = memoryview(data)
    for index in range(count):
        tag, offset, size, records, frames = DIRECTORY_ENTRY.unpack_from(data, 16 + index * DIRECTORY_ENTRY.size)
        if (
            offset % 8
            or offset < end
            or offset + size > len(data)
            or tag.rstrip(b"\0") in sections
            or any(view[end:offset])
        ):
            raise ValueError("Invalid binary section directory")
        sections[tag.rstrip(b"\0")] = (view[offset : offset + size], records, frames)
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
        if (
            offsets[0] != 8 * (count + 1)
            or offsets[-1] != len(self.data)
            or np.any(offsets[1:] < offsets[:-1])
            or np.any(offsets % 8)
        ):
            raise ValueError("Invalid binary record offsets")
        return [_Reader(self.data[int(start) : int(stop)]) for start, stop in itertools.pairwise(offsets)]

    def finish(self) -> None:
        rest = bytes(self.data[self.offset :])
        if len(rest) >= 8 or any(rest):
            raise ValueError("Binary record has unread bytes")


def binary_to_scenario(data: BufferLike, source: str = "<bytes>") -> schema.PufferScenario:
    """Decode a scene or map-only binary into the scenario it was written from.

    The writer derives polyline heading, lane cum_length and curvature, and the lane graph from the polylines,
    so the reader skips them. Raise BinaryFormatError for another version or any layout violation,
    including unread bytes.
    """
    try:
        sections = read_sections(data)

        def records(tag: bytes) -> list[_Reader]:
            section, count, _ = sections[tag]
            return _Reader(section).records(count)

        agents = _read_tracks(records(b"AGENTS"), sections[b"AGENTS"][2], has_route=True)
        polylines = _read_polylines(records(b"ROADPOLY"))
        surfaces = _read_surfaces(records(b"ROADSURF"))
        traffic_controls = _read_traffic_controls(records(b"TRAFFIC"))
        objects = _read_tracks(records(b"OBJECTS"), sections[b"OBJECTS"][2], has_route=False)
        graph_data, graph_count, _ = sections[b"LANEGRPH"]
        _skip_lane_graph(_Reader(graph_data), graph_count)
        metadata = _read_metadata(_Reader(sections[b"METADATA"][0]))
        if TRAFFIC_PHASE_SECTION_TAG in sections:
            _read_traffic_phases(_Reader(sections[TRAFFIC_PHASE_SECTION_TAG][0]), traffic_controls)
        if SOURCE_SECTION_TAG in sections:
            _read_source(_Reader(sections[SOURCE_SECTION_TAG][0]), metadata, [*agents.values(), *objects.values()])
    except (struct.error, ValueError, KeyError) as exc:
        raise BinaryFormatError(f"{source}: {exc}") from exc

    return schema.PufferScenario(
        agents=agents,
        objects=objects,
        polylines=polylines,
        surfaces=surfaces,
        traffic_controls=traffic_controls,
        metadata=metadata,
    )


def _read_xyz(record: _Reader, n_points: int, flags: int) -> np.ndarray:
    x = record.f32_array(n_points)
    y = record.f32_array(n_points)
    z = np.zeros(n_points) if flags & 1 else record.f32_array(n_points)
    return np.column_stack([x, y, z]).astype(np.float64)


def _read_polylines(records: list[_Reader]) -> dict[int, schema.RoadPolyline]:
    polylines = {}
    for record in records:
        element_id, layer, road_type, n_points, flags = (record.i32() for _ in range(5))
        element = schema.RoadPolyline(layer=layer, type=road_type, polyline=_read_xyz(record, n_points, flags))
        record.f32_array(n_points)  # heading is recomputed when serializing
        if element.is_lane:
            element.entry_lanes, element.exit_lanes, element.left_neighbor, element.right_neighbor = (
                record.int_list() for _ in range(4)
            )
            element.speed_limit_mps = record.f32()
            record.f32_array(2 * n_points)  # cum_length derives from the polyline, curvature is recomputed
        record.finish()
        polylines[element_id] = element
    return polylines


def _read_surfaces(records: list[_Reader]) -> dict[int, schema.RoadSurface]:
    surfaces = {}
    for record in records:
        element_id, layer, n_points, flags = (record.i32() for _ in range(4))
        surfaces[element_id] = schema.RoadSurface(layer=layer, polygon=_read_xyz(record, n_points, flags))
        record.finish()
    return surfaces


def _read_traffic_controls(records: list[_Reader]) -> list[schema.TrafficControl]:
    traffic_controls = []
    for record in records:  # keyword arguments evaluate in order, so they read the record's fields in turn
        traffic_controls.append(
            schema.TrafficControl(
                id=record.i32(),
                type=record.i32(),
                stop_line=record.f32_array(6).reshape(2, 3).astype(np.float64),
                heading=record.f32(),
                states=record.int_list(),
                controlled_lanes=record.int_list(),
            )
        )
        record.finish()
    return traffic_controls


def _read_tracks(records: list[_Reader], frame_count: int, has_route: bool) -> dict[int, schema.Track]:
    tracks = {}
    for record in records:
        track_id, track_type, first, n_stored, flags = (record.i32() for _ in range(5))
        n_frames = record.i32() if flags & 16 else frame_count
        if first < 0 or n_stored < 0 or first + n_stored > n_frames:
            raise ValueError("Invalid track span")
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
        record.offset = (record.offset + n_stored + 3) & ~3
        track = schema.Track(type=track_type, valid=valid, **channels)
        if has_route:
            track.route = record.int_list()
            track.route_entry_s = record.f32_array(len(track.route)).astype(np.float64).tolist()
            track.route_exit_s = record.f32_array(len(track.route)).astype(np.float64).tolist()
        record.finish()
        tracks[track_id] = track
    return tracks


def _read_traffic_phases(reader: _Reader, traffic_controls: list[schema.TrafficControl]) -> None:
    for traffic_control in traffic_controls:
        traffic_control.intersection_id = reader.i32()
        traffic_control.phase_idx = reader.i32()
    reader.finish()


def _skip_lane_graph(reader: _Reader, n_lanes: int) -> None:
    reader.i32_array(n_lanes)
    reader.f32_array(n_lanes * n_lanes)
    reader.finish()


def _read_metadata(reader: _Reader) -> schema.ScenarioMetadata:
    metadata = schema.ScenarioMetadata(
        id=reader.string(),
        dataset=reader.string(),
        location=reader.string(),
        scenario_length=reader.i32(),
        dt=reader.f32(),
        objects_of_interest=reader.int_list(),
        tracks_to_predict=reader.int_list(),
    )
    reader.finish()
    return metadata


def _read_source(reader: _Reader, metadata: schema.ScenarioMetadata, tracks: list[schema.Track]) -> None:
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
    for track in tracks:
        track.source_id = reader.string()
    reader.finish()
