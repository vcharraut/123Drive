"""Serialize a PufferScenario to PufferDrive binary format.

Binary layout (all little-endian):

  HEADER
    int32 x4  — counts: [num_agents, num_road_elements, num_traffic_controls, num_objects]

  AGENTS (repeated num_agents)
    int32 x2          — agent_id, agent_type (AgentType enum)
    DYNAMIC_STATES    — trajectory block (see below)
    int_list          — lane route (ordered lane IDs the agent follows)
    int32             — route_gt_len (number of leading route lanes supported by GT)
    float32 x3        — goal position (x, y, z) from last valid frame
    int32             — off_road flag (1 if no route assigned, else 0)

  ROAD MAP ELEMENTS (repeated num_road_elements)
    int32 x2          — element_id, road_type (LaneType/RoadLineType/RoadEdgeType/MiscRoadType)
    int32             — num_points
    float32[N] x3     — x, y, z columns (polyline for lines/edges, polygon for crosswalks)
    float32[N]        — per-point heading (radians, from segment tangent)
    if road_type is lane:
      int_list        — entry_lane IDs (predecessors)
      int_list        — exit_lane IDs (successors)
      float32         — speed_limit_mps (-1 if unknown)

  TRAFFIC CONTROLS (repeated num_traffic_controls)
    int32 x2          — control_id, control_type (TCType enum)
    float32 x3        — stop_line start point (x, y, z)
    float32 x3        — stop_line end point (x, y, z)
    float32           — heading (radians)
    int_list          — per-frame states (TLState ints; empty for non-light controls)
    int_list          — controlled_lane IDs

  OBJECTS (repeated num_objects)
    int32 x2          — object_id, object_type (ObjectType enum)
    DYNAMIC_STATES    — trajectory block

  LANE GRAPH
    int32             — N (number of lanes in graph; 0 = no graph)
    if N > 0:
      int32[N]        — lane_ids
      float32[N]      — lane_lengths
      float32[N*N]    — pairwise distance matrix (row-major)

  METADATA
    char[128]         — scenario_id (utf-8, null-padded)
    char[32]          — dataset name (utf-8, null-padded)
    int32             — scenario_length (number of timesteps)
    float32           — dt (seconds per timestep)
    int_list          — objects_of_interest (currently empty)
    int_list          — tracks_to_predict (currently empty)

  Sub-structures:

  DYNAMIC_STATES — per-track trajectory over T timesteps:
    int32             — T (trajectory length)
    float32[T] x3     — x, y, z position columns
    float32[T]        — heading
    float32[T] x2     — velocity_x, velocity_y
    float32[T]        — length, width, height (per-frame bounding box)
    int32[T]          — valid mask (1 = observed, 0 = missing)

  int_list — length-prefixed int32 array:
    int32             — N (count)
    int32[N]          — values
"""

import struct

import numpy as np

from bin_factory import puffer_types


METADATA_ID_BYTES = 128
METADATA_DATASET_BYTES = 32


def scenario_to_binary(scenario):
    """Serialize a PufferScenario into the PufferDrive .bin format. See module docstring for layout."""
    buf = bytearray()
    agents, road_map, tcs, objects = scenario.agents, scenario.map, scenario.traffic_controls, scenario.objects

    # Header — road_count placeholder at offset 4, patched after filtering
    buf.extend(struct.pack("<iiii", len(agents), 0, len(tcs), len(objects)))

    # Agents: id, type, trajectory, route, route_gt_len, goal, off_road flag
    for eid, track in agents.items():
        buf.extend(struct.pack("<ii", int(eid), int(track.type)))

        # Dynamic states
        xyz = np.asarray(track.position, dtype=np.float32)
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
            buf.extend(np.asarray(col, dtype=np.float32).tobytes())
        buf.extend(np.asarray(track.valid, dtype=np.int32).tobytes())

        # Route
        buf.extend(struct.pack("<i", len(track.route)))
        if track.route:
            buf.extend(struct.pack(f"<{len(track.route)}i", *map(int, track.route)))
        buf.extend(struct.pack("<i", int(track.route_gt_len)))

        # Goal position from last valid frame
        valid_idx = np.where(np.asarray(track.valid) > 0)[0]
        if len(valid_idx) > 0:
            i = valid_idx[-1]
            buf.extend(struct.pack("<fff", float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])))
        else:
            buf.extend(struct.pack("<fff", 0.0, 0.0, 0.0))

        buf.extend(struct.pack("<i", 0 if track.route else 1))  # off_road flag

    # Road map: id, type, geometry, heading; lanes get topology + speed limit
    road_count = 0
    for eid, elem in road_map.items():
        road_type = elem["type"]
        is_line = (
            puffer_types.is_road_lane(road_type)
            or puffer_types.is_road_line(road_type)
            or puffer_types.is_road_edge(road_type)
        )
        xyz = elem.get("polyline" if is_line else "polygon")
        if xyz is None or len(xyz) <= 1:
            continue
        road_count += 1

        xyz_f = np.asarray(xyz, dtype=np.float32)
        pts = np.asarray(xyz, dtype=np.float64)
        seg = np.arctan2(np.diff(pts[:, 1]), np.diff(pts[:, 0]))
        heading = np.append(seg, seg[-1]).astype(np.float32)

        buf.extend(struct.pack("<ii", int(eid), int(road_type)))
        buf.extend(struct.pack("<i", len(xyz_f)))
        for col in [xyz_f[:, 0], xyz_f[:, 1], xyz_f[:, 2]]:
            buf.extend(col.tobytes())
        buf.extend(heading.tobytes())

        if puffer_types.is_road_lane(road_type):
            for lane_list in [elem["entry_lanes"], elem["exit_lanes"]]:
                buf.extend(struct.pack("<i", len(lane_list)))
                if lane_list:
                    buf.extend(struct.pack(f"<{len(lane_list)}i", *map(int, lane_list)))
            buf.extend(struct.pack("<f", elem["speed_limit_mps"]))

    struct.pack_into("<i", buf, 4, road_count)  # patch actual road count in header

    # Traffic controls: id, type, stop line endpoints, heading, states, controlled lanes
    for tc in tcs:
        buf.extend(struct.pack("<ii", int(tc["id"]), int(tc["type"])))
        sl = np.asarray(tc["stop_line"], dtype=np.float32)
        buf.extend(struct.pack("<fff", *sl[0]))
        buf.extend(struct.pack("<fff", *sl[1]))
        buf.extend(struct.pack("<f", float(tc["heading"])))
        for int_list in [tc["states"], tc["controlled_lanes"]]:
            buf.extend(struct.pack("<i", len(int_list)))
            if int_list:
                buf.extend(struct.pack(f"<{len(int_list)}i", *map(int, int_list)))

    # Objects: id, type, trajectory (same layout as agents but no route/goal)
    for eid, track in objects.items():
        buf.extend(struct.pack("<ii", int(eid), int(track.type)))
        xyz = np.asarray(track.position, dtype=np.float32)
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
            buf.extend(np.asarray(col, dtype=np.float32).tobytes())
        buf.extend(np.asarray(track.valid, dtype=np.int32).tobytes())

    # Lane graph: pairwise distance matrix between lanes (Dijkstra-precomputed)
    if lg := scenario.lane_graph:
        n = len(lg["lane_ids"])
        buf.extend(struct.pack("<i", n))
        buf.extend(struct.pack(f"<{n}i", *lg["lane_ids"]))
        buf.extend(np.asarray(lg["lane_lengths"], dtype=np.float32).tobytes())
        buf.extend(np.asarray(lg["distances"], dtype=np.float32).tobytes())
    else:
        buf.extend(struct.pack("<i", 0))

    # Metadata: scenario id, dataset, timing, prediction targets
    buf.extend(str(scenario.metadata.id).encode("utf-8")[:METADATA_ID_BYTES].ljust(METADATA_ID_BYTES, b"\0"))
    buf.extend(
        str(scenario.metadata.dataset).encode("utf-8")[:METADATA_DATASET_BYTES].ljust(METADATA_DATASET_BYTES, b"\0")
    )
    buf.extend(struct.pack("<i", int(scenario.metadata.scenario_length)))
    buf.extend(struct.pack("<f", float(scenario.metadata.dt)))
    buf.extend(struct.pack("<i", 0))  # objects_of_interest (empty)
    buf.extend(struct.pack("<i", 0))  # tracks_to_predict (empty)

    return bytes(buf)
