"""Serialize a PufferScenario to PufferDrive binary format.

Full binary layout is specified in ``docs/binary-format.rst``; enum values are in ``puffer_types.py``.
"""

import struct

import numpy as np

from bin_factory import schema


METADATA_ID_BYTES = 128
METADATA_DATASET_BYTES = 32
CURVATURE_HALF_WINDOW = 5.0  # metres of lane on each side of a point averaged into its curvature


def _write_int_list(buf: bytearray, values: list[int]) -> None:
    buf.extend(struct.pack("<i", len(values)))
    if values:
        buf.extend(struct.pack(f"<{len(values)}i", *map(int, values)))


def _write_dynamic_states(buf: bytearray, track: schema.Track) -> np.ndarray:
    """Write a per-track trajectory block (T + xyz + heading + velocity + bbox + valid). Returns xyz."""
    xyz = np.asarray(track.position, dtype="<f4")
    buf.extend(struct.pack("<i", len(xyz)))
    for col in [
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        track.heading,
        track.velocity[:, 0],
        track.velocity[:, 1],
        track.length,
        track.width,
        track.height,
    ]:
        buf.extend(np.asarray(col, dtype="<f4").tobytes())
    buf.extend(np.asarray(track.valid, dtype="<i4").tobytes())
    return xyz


def scenario_to_binary(scenario: schema.PufferScenario) -> bytes:
    """Serialize a PufferScenario into the PufferDrive .bin format.

    See the module docstring for the full binary layout. Map elements must already be
    serializable (>= min_points geometry) — transforms.sanitize.prune_invalid_map_elements
    guarantees this before reindexing.

    Returns:
        ``bytes`` containing the serialized scenario.
    """
    buf = bytearray()
    agents, road_map, tcs, objects = scenario.agents, scenario.map, scenario.traffic_controls, scenario.objects

    buf.extend(struct.pack("<iiii", len(agents), len(road_map), len(tcs), len(objects)))

    # Agents: id, type, trajectory, route, route_gt_len, goal, control_state
    for eid, track in agents.items():
        buf.extend(struct.pack("<ii", int(eid), int(track.type)))
        xyz = _write_dynamic_states(buf, track)

        # Route
        _write_int_list(buf, track.route)
        buf.extend(struct.pack("<i", int(track.route_gt_len)))

        # Goal position from last valid frame
        valid_idx = np.where(np.asarray(track.valid) > 0)[0]
        if len(valid_idx) > 0:
            i = valid_idx[-1]
            buf.extend(struct.pack("<fff", float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])))
        else:
            buf.extend(struct.pack("<fff", 0.0, 0.0, 0.0))
        buf.extend(struct.pack("<i", int(track.control_state)))

    # Road map: id, type, geometry, heading; lanes get topology (entry/exit/left/right), speed limit, lengths and curvature
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

    # Objects: id, type, trajectory (same layout as agents but no route/goal)
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

    # Metadata: scenario id, dataset, timing, prediction targets
    buf.extend(str(scenario.metadata.id).encode("utf-8")[:METADATA_ID_BYTES].ljust(METADATA_ID_BYTES, b"\0"))
    buf.extend(
        str(scenario.metadata.dataset).encode("utf-8")[:METADATA_DATASET_BYTES].ljust(METADATA_DATASET_BYTES, b"\0")
    )
    buf.extend(struct.pack("<i", int(scenario.metadata.scenario_length)))
    buf.extend(struct.pack("<f", float(scenario.metadata.dt)))
    for int_list in [scenario.metadata.objects_of_interest, scenario.metadata.tracks_to_predict]:
        _write_int_list(buf, int_list)

    # Traffic control groups (optional trailing section, readers stopping after the metadata ignore it):
    # intersection id, signal group id, signal sequence per traffic control, -1 if unknown
    buf.extend(struct.pack("<i", len(tcs)))
    for tc in tcs:
        buf.extend(struct.pack("<iii", tc.intersection_id, tc.signal_group_id, tc.signal_sequence))

    return bytes(buf)
