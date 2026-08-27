import struct

import numpy as np

from bin_factory import puffer_types, schema, serialize


def _track(length, agent_type, valid, route=(), route_gt_len=0, control_state=2, base=0.0):
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
        route_gt_len=route_gt_len,
        control_state=control_state,
    )


def _lane(road_type, points, entry=(), exit_=(), speed=-1.0):
    polyline = np.arange(points * 3, dtype=np.float64).reshape(points, 3)
    cum = np.linalg.norm(np.diff(polyline, axis=0), axis=1).cumsum()
    cum_length = np.concatenate([[0.0], cum]) if points > 1 else np.zeros(points)
    return schema.MapElement(
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
    )
    agents = {
        0: _track(3, puffer_types.AgentType.VEHICLE, [1, 1, 1], route=[10, 11], route_gt_len=1, control_state=0),
        1: _track(3, puffer_types.AgentType.PEDESTRIAN, [0, 1, 0], base=100.0),
    }
    crosswalk = schema.MapElement(
        type=int(puffer_types.MiscRoadType.CROSSWALK),
        polygon=np.arange(12, dtype=np.float64).reshape(4, 3),
    )
    road_map = {
        10: _lane(puffer_types.LaneType.SURFACE_STREET, 4, exit_=[11], speed=13.0),
        11: _lane(puffer_types.LaneType.FREEWAY, 2, entry=[10]),
        12: _lane(puffer_types.RoadLineType.SOLID_SINGLE_WHITE, 3),
        13: crosswalk,
    }
    objects = {0: _track(3, puffer_types.ObjectType.GENERIC_OBJECT, [1, 1, 1])}
    traffic_controls = [
        {
            "id": 5,
            "type": int(puffer_types.TCType.TRAFFIC_LIGHT),
            "stop_line": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "heading": 0.5,
            "states": [int(puffer_types.TLState.GREEN)] * 3,
            "controlled_lanes": [10],
            "junction_id": 20,
            "phase_idx": 2,
        }
    ]
    lane_graph = {"lane_ids": [10, 11], "distances": np.array([[0.0, 3.0], [np.inf, 0.0]])}
    return schema.PufferScenario(
        agents=agents,
        objects=objects,
        map=road_map,
        metadata=metadata,
        traffic_controls=traffic_controls,
        lane_graph=lane_graph,
    )


class _Reader:
    def __init__(self, data):
        self.data = data
        self.off = 0

    def ints(self, n):
        vals = struct.unpack_from(f"<{n}i", self.data, self.off)
        self.off += 4 * n
        return list(vals)

    def floats(self, n):
        vals = struct.unpack_from(f"<{n}f", self.data, self.off)
        self.off += 4 * n
        return list(vals)

    def raw(self, n):
        chunk = self.data[self.off : self.off + n]
        self.off += n
        return chunk

    def skip_dynamic(self):
        (t,) = self.ints(1)
        self.floats(9 * t)  # xyz, heading, vx, vy, length, width, height
        self.ints(t)  # valid
        return t


def _parse(data):
    r = _Reader(data)
    n_agents, n_road, n_tc, n_objects = r.ints(4)

    agents = []
    for _ in range(n_agents):
        eid, _type = r.ints(2)
        r.skip_dynamic()
        (n_route,) = r.ints(1)
        route = r.ints(n_route)
        (route_gt_len,) = r.ints(1)
        goal = r.floats(3)
        (control_state,) = r.ints(1)
        agents.append({"id": eid, "route": route, "route_gt_len": route_gt_len, "goal": goal, "cs": control_state})

    lanes = []
    for _ in range(n_road):
        eid, road_type = r.ints(2)
        (npts,) = r.ints(1)
        r.floats(4 * npts)  # xyz + heading
        if 0 <= road_type <= 9:
            entry = r.ints(r.ints(1)[0])
            exit_ = r.ints(r.ints(1)[0])
            (speed,) = r.floats(1)
            r.floats(1)  # length
            r.floats(npts)  # cum_length
            lanes.append({"id": eid, "entry": entry, "exit": exit_, "speed": speed, "npts": npts})

    for _ in range(n_tc):
        r.ints(2)
        r.floats(7)  # stop line (2x3) + heading
        r.ints(r.ints(1)[0])  # states
        r.ints(r.ints(1)[0])  # controlled lanes

    for _ in range(n_objects):
        r.ints(2)
        r.skip_dynamic()

    (lg_n,) = r.ints(1)
    if lg_n:
        r.ints(lg_n)
        r.floats(lg_n * lg_n)

    meta_id = r.raw(serialize.METADATA_ID_BYTES).rstrip(b"\0").decode()
    meta_dataset = r.raw(serialize.METADATA_DATASET_BYTES).rstrip(b"\0").decode()
    (scenario_length,) = r.ints(1)
    (dt,) = r.floats(1)
    ooi = r.ints(r.ints(1)[0])
    ttp = r.ints(r.ints(1)[0])

    assert r.raw(len(serialize.TRAFFIC_PHASE_SECTION_TAG)) == serialize.TRAFFIC_PHASE_SECTION_TAG
    phases = [tuple(r.ints(2)) for _ in range(n_tc)]

    return {
        "phases": phases,
        "counts": (n_agents, n_road, n_tc, n_objects),
        "agents": agents,
        "lanes": lanes,
        "id": meta_id,
        "dataset": meta_dataset,
        "scenario_length": scenario_length,
        "dt": dt,
        "ooi": ooi,
        "ttp": ttp,
        "consumed": r.off,
    }


def test_round_trip_high_level_fields():
    data = serialize.scenario_to_binary(_build_scenario())
    parsed = _parse(data)

    # Header counts: agents, road elements, traffic controls, objects.
    assert parsed["counts"] == (2, 4, 1, 1)

    ego = next(a for a in parsed["agents"] if a["id"] == 0)
    assert ego["route"] == [10, 11]
    assert ego["route_gt_len"] == 1
    assert ego["cs"] == int(puffer_types.ControlState.CONTROLLABLE)
    assert ego["goal"] == [6.0, 7.0, 8.0]  # last valid frame position

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
    assert parsed["phases"] == [(20, 2)]

    # The parser walks every field; ending exactly at the buffer end proves layout consistency.
    assert parsed["consumed"] == len(data)


def test_serialization_is_deterministic():
    scenario = _build_scenario()
    assert serialize.scenario_to_binary(scenario) == serialize.scenario_to_binary(scenario)
