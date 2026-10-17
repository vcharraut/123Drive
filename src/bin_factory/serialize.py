"""Encode scenarios and decode static PufferDrive binaries.

Full binary layout is specified in ``docs/binary-format.rst``; enum values are in ``puffer_types.py``.
"""

import dataclasses
import struct

import numpy as np

from bin_factory import schema


METADATA_ID_BYTES = 128
METADATA_DATASET_BYTES = 32
METADATA_LOCATION_BYTES = 32
TRAFFIC_PHASE_SECTION_TAG = b"TLPHASE1"
CURVATURE_HALF_WINDOW = 5.0  # metres of lane on each side of a point averaged into its curvature
_TRACK_FIELDS = (("position", 3), ("heading", 1), ("velocity", 2), ("length", 1), ("width", 1), ("height", 1))
_METADATA_STRINGS = (
    ("id", METADATA_ID_BYTES),
    ("dataset", METADATA_DATASET_BYTES),
    ("location", METADATA_LOCATION_BYTES),
)


def _write_int_list(buf: bytearray, values: list[int]) -> None:
    buf.extend(struct.pack("<i", len(values)))
    if values:
        buf.extend(struct.pack(f"<{len(values)}i", *map(int, values)))


def _write_dynamic_states(buf: bytearray, track: schema.Track) -> None:
    """Write a per-track trajectory block (T, first, n + xyz + heading + velocity + bbox + valid).

    Only frames [first, first + n), from the first to the last valid frame, are stored; readers fill the others with
    zeros (invalid).
    """
    valid = np.flatnonzero(np.asarray(track.valid))
    first, n = (int(valid[0]), int(valid[-1] - valid[0] + 1)) if len(valid) else (0, 0)
    buf.extend(struct.pack("<iii", len(track.position), first, n))
    for name, _ in _TRACK_FIELDS:
        buf.extend(np.asarray(getattr(track, name), dtype="<f4")[first : first + n].T.tobytes())
    buf.extend(np.asarray(track.valid, dtype="<i4")[first : first + n].tobytes())


def scenario_to_binary(scenario: schema.PufferScenario) -> bytes:
    """Return the scenario bytes in the layout specified in ``docs/binary-format.rst``.

    Run ``prune_invalid_map_elements`` before reindexing so every map element has at least
    ``min_points`` geometry points.
    """
    buf = bytearray()
    agents, road_map, tcs, objects = scenario.agents, scenario.map, scenario.traffic_controls, scenario.objects

    buf.extend(struct.pack("<iiii", len(agents), len(road_map), len(tcs), len(objects)))

    # Agents: id, type, trajectory, route, route_gt_len, control_state
    for eid, track in agents.items():
        buf.extend(struct.pack("<ii", int(eid), int(track.type)))
        _write_dynamic_states(buf, track)

        # Route
        _write_int_list(buf, track.route)
        buf.extend(struct.pack("<i", int(track.route_gt_len)))
        buf.extend(struct.pack("<i", int(track.control_state)))

    # Road map: id, type, geometry, heading; lanes get topology (entry/exit/left/right),
    # speed limit, lengths and curvature
    for eid, elem in road_map.items():
        road_type = elem.type
        xyz = elem.geometry

        xyz_f = np.asarray(xyz, dtype="<f4")
        pts = np.asarray(xyz, dtype=np.float64)
        seg = np.arctan2(np.diff(pts[:, 1]), np.diff(pts[:, 0]))
        heading = np.append(seg, seg[-1]).astype("<f4")

        buf.extend(struct.pack("<ii", int(eid), int(road_type)))
        buf.extend(struct.pack("<i", len(xyz_f)))
        for col in [xyz_f[:, 0], xyz_f[:, 1], xyz_f[:, 2]]:
            buf.extend(col.tobytes())
        buf.extend(heading.tobytes())

        if elem.is_lane:
            for lane_list in [elem.entry_lanes, elem.exit_lanes, elem.left_neighbor, elem.right_neighbor]:
                _write_int_list(buf, lane_list)
            buf.extend(struct.pack("<f", elem.speed_limit_mps))
            buf.extend(struct.pack("<f", float(elem.length)))
            buf.extend(np.asarray(elem.cum_length, dtype="<f4").tobytes())
            # Signed curvature (1/m, left positive): heading change over +-CURVATURE_HALF_WINDOW of 2D arc length
            # (clipped to the lane) divided by that length. Heading is interpolated between the midpoints of the
            # polyline's chords; collinear points inserted by resampling are merged so they don't split a chord.
            s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))])
            unwrapped = np.unwrap(seg)
            corner = np.concatenate([[True], np.abs(np.diff(unwrapped)) > 1e-4, [True]])  # rad; float32 jitter is ~1e-5
            chord_s = s[corner]
            chord_heading = unwrapped[np.flatnonzero(corner)[:-1]]
            lo = np.maximum(s - CURVATURE_HALF_WINDOW, 0.0)
            hi = np.minimum(s + CURVATURE_HALF_WINDOW, s[-1])
            chord_mid = (chord_s[:-1] + chord_s[1:]) / 2
            turn = np.interp(hi, chord_mid, chord_heading) - np.interp(lo, chord_mid, chord_heading)
            buf.extend((turn / (hi - lo)).astype("<f4").tobytes())

    # Traffic controls: id, type, stop line endpoints, heading, states, controlled lanes
    for tc in tcs:
        buf.extend(struct.pack("<ii", int(tc.id), int(tc.type)))
        sl = np.asarray(tc.stop_line, dtype="<f4")
        buf.extend(struct.pack("<fff", *sl[0]))
        buf.extend(struct.pack("<fff", *sl[1]))
        buf.extend(struct.pack("<f", float(tc.heading)))
        for int_list in [tc.states, tc.controlled_lanes]:
            _write_int_list(buf, int_list)

    # Objects: id, type, trajectory (same layout as agents but no route)
    for eid, track in objects.items():
        buf.extend(struct.pack("<ii", int(eid), int(track.type)))
        _write_dynamic_states(buf, track)

    # Lane graph: pairwise distance matrix between lanes (Dijkstra-precomputed)
    if lg := scenario.lane_graph:
        n = len(lg["lane_ids"])
        buf.extend(struct.pack("<i", n))
        buf.extend(struct.pack(f"<{n}i", *lg["lane_ids"]))
        buf.extend(np.asarray(lg["distances"], dtype="<f4").tobytes())
    else:
        buf.extend(struct.pack("<i", 0))

    # Metadata: scenario id, dataset, location, timing, prediction targets
    for name, width in _METADATA_STRINGS:
        buf.extend(str(getattr(scenario.metadata, name)).encode("utf-8")[:width].ljust(width, b"\0"))
    buf.extend(struct.pack("<i", int(scenario.metadata.scenario_length)))
    buf.extend(struct.pack("<f", float(scenario.metadata.dt)))
    for int_list in [scenario.metadata.objects_of_interest, scenario.metadata.tracks_to_predict]:
        _write_int_list(buf, int_list)

    # Traffic light phases (optional tagged trailing section, readers stopping after the metadata ignore it):
    # junction id and phase per traffic control, both -1 if the control is not in a light cycle
    buf.extend(TRAFFIC_PHASE_SECTION_TAG)
    for tc in tcs:
        buf.extend(struct.pack("<ii", tc.intersection_id if tc.phase_idx >= 0 else -1, tc.phase_idx))

    return bytes(buf)


class StaticBinaryError(ValueError):
    """Raised when a .bin payload is not a supported static map binary."""


@dataclasses.dataclass
class _Reader:
    data: bytes
    offset: int = 0

    def i32(self) -> int:
        value = struct.unpack_from("<i", self.data, self.offset)[0]
        self.offset += 4
        return int(value)

    def f32(self) -> float:
        value = struct.unpack_from("<f", self.data, self.offset)[0]
        self.offset += 4
        return float(value)

    def i32_array(self, n: int) -> np.ndarray:
        if n == 0:
            return np.zeros(0, dtype=np.int32)
        arr = np.frombuffer(self.data, dtype="<i4", count=n, offset=self.offset).copy()
        self.offset += 4 * n
        return arr

    def f32_array(self, n: int) -> np.ndarray:
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        arr = np.frombuffer(self.data, dtype="<f4", count=n, offset=self.offset).copy()
        self.offset += 4 * n
        return arr

    def int_list(self) -> list[int]:
        n = self.i32()
        return self.i32_array(n).astype(np.int64).tolist()

    def string(self, n: int) -> str:
        raw = self.data[self.offset : self.offset + n]
        self.offset += n
        return raw.split(b"\0", 1)[0].decode("utf-8")


def static_binary_to_scenario(data: bytes, source: str = "<bytes>") -> schema.PufferScenario:
    reader = _Reader(data)
    try:
        n_agents = reader.i32()
        n_roads = reader.i32()
        n_traffic = reader.i32()
        n_objects = reader.i32()

        if n_agents:
            raise StaticBinaryError(f"{source} is not static: found {n_agents} agents")

        road_map = _read_roads(reader, n_roads)
        traffic_controls = _read_traffic_controls(reader, n_traffic)
        objects = _read_objects(reader, n_objects)
        lane_graph = _read_lane_graph(reader)
        metadata = _read_metadata(reader)
        if reader.data[reader.offset : reader.offset + len(TRAFFIC_PHASE_SECTION_TAG)] == TRAFFIC_PHASE_SECTION_TAG:
            reader.offset += len(TRAFFIC_PHASE_SECTION_TAG)
            _read_traffic_phases(reader, traffic_controls)
    except (struct.error, ValueError) as exc:
        if isinstance(exc, StaticBinaryError):
            raise
        raise StaticBinaryError(f"{source} is not a valid static PufferDrive binary") from exc

    if reader.offset != len(data):
        raise StaticBinaryError(f"{source} has {len(data) - reader.offset} trailing bytes after metadata")

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
    for _ in range(n_roads):
        element_id = reader.i32()
        road_type = reader.i32()
        n_points = reader.i32()
        x = reader.f32_array(n_points)
        y = reader.f32_array(n_points)
        z = reader.f32_array(n_points)
        reader.f32_array(n_points)

        element = schema.MapElement(type=road_type)
        xyz = np.column_stack([x, y, z]).astype(np.float64)
        setattr(element, "polyline" if element.uses_polyline else "polygon", xyz)

        if element.is_lane:
            element.entry_lanes = reader.int_list()
            element.exit_lanes = reader.int_list()
            element.left_neighbor = reader.int_list()
            element.right_neighbor = reader.int_list()
            element.speed_limit_mps = reader.f32()
            element.length = reader.f32()
            element.cum_length = reader.f32_array(n_points).astype(np.float64)
            reader.f32_array(n_points)

        road_map[element_id] = element
    return road_map


def _read_traffic_controls(reader: _Reader, n_traffic: int) -> list[schema.TrafficControl]:
    traffic_controls = []
    for _ in range(n_traffic):
        control_id = reader.i32()
        control_type = reader.i32()
        stop_line = np.array(
            [
                [reader.f32(), reader.f32(), reader.f32()],
                [reader.f32(), reader.f32(), reader.f32()],
            ],
            dtype=np.float64,
        )
        heading = reader.f32()
        states = reader.int_list()
        controlled_lanes = reader.int_list()
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


def _read_objects(reader: _Reader, n_objects: int) -> dict[int, schema.Track]:
    objects = {}
    for _ in range(n_objects):
        object_id = reader.i32()
        object_type = reader.i32()
        n_frames = reader.i32()
        first = reader.i32()
        n_stored = reader.i32()
        channels = {}
        for name, width in _TRACK_FIELDS:
            values = np.zeros((width, n_frames), dtype=np.float64)
            values[:, first : first + n_stored] = reader.f32_array(width * n_stored).reshape(width, n_stored)
            channels[name] = values.T.copy() if width > 1 else values[0]
        valid = np.zeros(n_frames, dtype=np.int32)
        valid[first : first + n_stored] = reader.i32_array(n_stored)
        objects[object_id] = schema.Track(type=object_type, valid=valid, **channels)
    return objects


def _read_traffic_phases(reader: _Reader, traffic_controls: list[schema.TrafficControl]) -> None:
    for traffic_control in traffic_controls:
        traffic_control.intersection_id = reader.i32()
        traffic_control.phase_idx = reader.i32()


def _read_lane_graph(reader: _Reader) -> dict | None:
    n_lanes = reader.i32()
    if n_lanes == 0:
        return None
    lane_ids = reader.i32_array(n_lanes).astype(np.int64).tolist()
    distances = reader.f32_array(n_lanes * n_lanes).reshape(n_lanes, n_lanes).astype(np.float64)
    return {"lane_ids": lane_ids, "distances": distances}


def _read_metadata(reader: _Reader) -> schema.ScenarioMetadata:
    strings = {name: reader.string(width) for name, width in _METADATA_STRINGS}
    scenario_length = reader.i32()
    dt = reader.f32()
    objects_of_interest = reader.int_list()
    tracks_to_predict = reader.int_list()
    return schema.ScenarioMetadata(
        **strings,
        scenario_length=scenario_length,
        dt=dt,
        objects_of_interest=objects_of_interest,
        tracks_to_predict=tracks_to_predict,
    )
