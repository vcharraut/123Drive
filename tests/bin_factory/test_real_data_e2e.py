"""End-to-end pipeline tests on real (committed) 123D scenarios.

Fixtures live in ``tests/py123d_data`` — two self-contained WOD-motion logs with per-log
maps. Each test drives the real conversion path (discover -> extract -> validate -> transform
-> serialize) and walks the resulting binary to assert layout + invariants survive on real data.
"""

import pathlib
import struct

import numpy as np
import pytest

from bin_factory import loader, main, puffer_types, serialize, transforms


DATA_ROOT = pathlib.Path(__file__).parent.parent / "py123d_data"
SPLIT = "wod-motion_train"
LOG_NAMES = ["4124fb4053fd031d", "7a4bd197a1dcf0e3"]
TL_LOG = "7a4bd197a1dcf0e3"  # log with traffic-light detections + crosswalk + all line/edge types
MAP_LOCATIONS = ["Town02", "Town10HD"]  # opendrive maps under tests/py123d_data/maps/opendrive


def _config(**overrides):
    cfg = main.build_parser().parse_args(["--py123d_path", str(DATA_ROOT)])
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
    """Run extract -> validate(level 2) -> pipeline. Returns (scenario, extras)."""
    config = _config(**cfg_overrides)
    scenario, extras = loader.extract_scenario(_discover(log_name))
    errors = loader.validate_scenario(scenario, extras=extras, level=2)
    assert errors == [], f"{log_name} failed semantic validation: {errors}"
    transforms.run(scenario, extras, config)
    return scenario, extras


def _discover_map(location):
    maps = loader.discover_scenes(py123d_data_root=str(DATA_ROOT), workers=1, map_only=True, datasets=["opendrive"])
    match = [m for m in maps if m.location == location]
    assert len(match) == 1, f"expected one opendrive map for {location}, got {len(match)}"
    return match[0]


def _process_map(location, **cfg_overrides):
    """Map-only flow: extract -> validate(level 2) -> pipeline on a bare MapAPI."""
    config = _config(map_only=True, **cfg_overrides)
    scenario, extras = loader.extract_scenario(_discover_map(location))
    errors = loader.validate_scenario(scenario, extras=extras, level=2)
    assert errors == [], f"{location} failed semantic validation: {errors}"
    transforms.run(scenario, extras, config)
    return scenario, extras


# ── Binary walker ───────────────────────
# Walks every field of the serialized buffer; ending exactly at len(data) proves layout.


class _Reader:
    def __init__(self, data):
        self.data, self.off = data, 0

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
        n_points = r.skip_dynamic()
        route = r.ints(r.ints(1)[0])
        (route_gt_len,) = r.ints(1)
        r.floats(3)  # goal
        (control_state,) = r.ints(1)
        agents.append(
            {
                "id": eid,
                "type": _type,
                "n_points": n_points,
                "route": route,
                "route_gt_len": route_gt_len,
                "cs": control_state,
            }
        )

    roads = []
    for _ in range(n_road):
        eid, road_type = r.ints(2)
        (npts,) = r.ints(1)
        r.floats(4 * npts)  # xyz + heading
        is_lane = 0 <= road_type <= 9
        if is_lane:
            entry = r.ints(r.ints(1)[0])
            exit_ = r.ints(r.ints(1)[0])
            r.floats(1)  # speed
            r.floats(1)  # length
            r.floats(npts)  # cum_length
        else:
            entry, exit_ = [], []
        roads.append({"id": eid, "type": road_type, "npts": npts, "lane": is_lane, "entry": entry, "exit": exit_})

    tcs = []
    for _ in range(n_tc):
        eid, tc_type = r.ints(2)
        r.floats(7)  # stop line (2x3) + heading
        states = r.ints(r.ints(1)[0])
        lanes = r.ints(r.ints(1)[0])
        tcs.append({"id": eid, "type": tc_type, "states": states, "lanes": lanes})

    objects = []
    for _ in range(n_objects):
        eid, _type = r.ints(2)
        r.skip_dynamic()
        objects.append(eid)

    (lg_n,) = r.ints(1)
    lane_graph_ids = []
    if lg_n:
        lane_graph_ids = r.ints(lg_n)
        r.floats(lg_n * lg_n)

    meta_id = r.raw(serialize.METADATA_ID_BYTES).rstrip(b"\0").decode()
    meta_dataset = r.raw(serialize.METADATA_DATASET_BYTES).rstrip(b"\0").decode()
    (scenario_length,) = r.ints(1)
    (dt,) = r.floats(1)
    ooi = r.ints(r.ints(1)[0])
    ttp = r.ints(r.ints(1)[0])
    assert r.raw(len(serialize.TRAFFIC_PHASE_SECTION_TAG)) == serialize.TRAFFIC_PHASE_SECTION_TAG
    phases = [tuple(r.ints(2)) for _ in range(n_tc)]
    assert r.raw(len(serialize.LANE_WIDTH_SECTION_TAG)) == serialize.LANE_WIDTH_SECTION_TAG
    widths = {road["id"]: r.floats(road["npts"]) for road in roads if road["lane"]}

    return {
        "widths": widths,
        "counts": (n_agents, n_road, n_tc, n_objects),
        "agents": agents,
        "roads": roads,
        "tcs": tcs,
        "objects": objects,
        "lane_graph_ids": lane_graph_ids,
        "id": meta_id,
        "dataset": meta_dataset,
        "scenario_length": scenario_length,
        "dt": dt,
        "ooi": ooi,
        "ttp": ttp,
        "phases": phases,
        "consumed": r.off,
    }


# ── Discovery ───────────────────────


def test_discovers_both_fixtures():
    scenes = loader.discover_scenes(py123d_data_root=str(DATA_ROOT), workers=1, split_names=[SPLIT])
    assert {s.log_name for s in scenes} == set(LOG_NAMES)
    assert all(s.dataset == "wod-motion" for s in scenes)


# ── Full pipeline: buffer layout integrity ───────────────────────


@pytest.mark.parametrize("log_name", LOG_NAMES)
@pytest.mark.parametrize("interpolate_tl", [False, True])
def test_binary_walk_consumes_whole_buffer(log_name, interpolate_tl):
    scenario, _ = _process(log_name, interpolate_tl=interpolate_tl)
    data = serialize.scenario_to_binary(scenario)
    parsed = _parse(data)
    # Walking every field and landing exactly on the buffer end proves layout consistency
    # against real, variable-shaped data (mixed map types, routes, TL state vectors).
    assert parsed["consumed"] == len(data)


@pytest.mark.parametrize("log_name", LOG_NAMES)
def test_header_counts_match_processed_scenario(log_name):
    scenario, _ = _process(log_name)
    parsed = _parse(serialize.scenario_to_binary(scenario))
    n_road = sum(1 for e in scenario.map.values() if e.geometry is not None and len(e.geometry) > 1)
    assert parsed["counts"] == (
        len(scenario.agents),
        n_road,
        len(scenario.traffic_controls),
        len(scenario.objects),
    )


# ── Reindex invariants on real data ───────────────────────


@pytest.mark.parametrize("log_name", LOG_NAMES)
def test_reindex_makes_ids_contiguous(log_name):
    scenario, _ = _process(log_name)
    assert list(scenario.agents) == list(range(len(scenario.agents)))
    assert list(scenario.map) == list(range(len(scenario.map)))
    assert list(scenario.objects) == list(range(len(scenario.objects)))

    parsed = _parse(serialize.scenario_to_binary(scenario))
    assert [a["id"] for a in parsed["agents"]] == list(range(len(parsed["agents"])))


@pytest.mark.parametrize("log_name", LOG_NAMES)
def test_no_reindex_preserves_source_ids(log_name):
    scenario, _ = _process(log_name, no_reindex=True)
    # Source map IDs from WOD are not a contiguous 0..n range; keeping them proves reindex was skipped.
    assert list(scenario.map) != list(range(len(scenario.map)))


# ── Cross-reference integrity on real data ───────────────────────


@pytest.mark.parametrize("log_name", LOG_NAMES)
def test_references_stay_within_bounds(log_name):
    scenario, _ = _process(log_name)
    parsed = _parse(serialize.scenario_to_binary(scenario))
    lane_ids = {r["id"] for r in parsed["roads"] if r["lane"]}
    road_ids = {r["id"] for r in parsed["roads"]}

    for road in parsed["roads"]:
        assert road["npts"] >= 2  # undersized polylines are dropped by serialize
        assert set(road["entry"]) <= lane_ids
        assert set(road["exit"]) <= lane_ids

    for agent in parsed["agents"]:
        assert set(agent["route"]) <= lane_ids  # routes reference lanes only
        assert agent["route_gt_len"] <= len(agent["route"])

    for tc in parsed["tcs"]:
        assert set(tc["lanes"]) <= road_ids
        # Traffic-light controls carry a per-frame state vector; other controls (e.g. stop signs) don't.
        if tc["type"] == int(puffer_types.TCType.TRAFFIC_LIGHT):
            assert len(tc["states"]) == scenario.metadata.scenario_length
        else:
            assert tc["states"] == []

    assert set(parsed["lane_graph_ids"]) <= lane_ids


@pytest.mark.parametrize("log_name", LOG_NAMES)
def test_map_types_in_valid_ranges(log_name):
    scenario, _ = _process(log_name)
    # 0-9 lane, 10-19 line, 20-29 edge, 31 crosswalk (see puffer_types / CLAUDE.md).
    for elem in scenario.map.values():
        assert 0 <= elem.type <= 31
        assert elem.is_lane or elem.is_line or elem.is_edge or elem.is_crosswalk


# ── Metadata round-trip ───────────────────────


@pytest.mark.parametrize("log_name", LOG_NAMES)
def test_metadata_round_trips(log_name):
    scenario, _ = _process(log_name)
    parsed = _parse(serialize.scenario_to_binary(scenario))
    meta = scenario.metadata
    assert parsed["id"] == meta.id
    assert parsed["dataset"] == "wod-motion"
    assert parsed["scenario_length"] == meta.scenario_length
    assert abs(parsed["dt"] - meta.dt) < 1e-6
    assert parsed["ttp"] == list(meta.tracks_to_predict)
    assert parsed["ooi"] == list(meta.objects_of_interest)
    # Prediction targets must point at real agents after reindex.
    assert set(parsed["ttp"]) <= set(scenario.agents)


# ── Traffic lights ───────────────────────


def test_traffic_lights_present_and_imputed():
    scenario, _ = _process(TL_LOG)
    parsed = _parse(serialize.scenario_to_binary(scenario))
    assert parsed["counts"][2] > 0, "expected traffic controls in the TL fixture"

    interp, _ = _process(TL_LOG, interpolate_tl=True)
    parsed_interp = _parse(serialize.scenario_to_binary(interp))
    # Imputation fills gaps from neighbour trajectories, never dropping controls.
    assert parsed_interp["counts"][2] >= parsed["counts"][2]


# ── Determinism + real entry point (writes a .bin to disk) ───────────────────────


@pytest.mark.parametrize("log_name", LOG_NAMES)
def test_serialization_is_deterministic(log_name):
    s1, _ = _process(log_name)
    s2, _ = _process(log_name)
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
    assert (np.frombuffer(data[:16], dtype="<i4") >= 0).all()  # header counts non-negative


# ── Map-only pipeline (opendrive, no logs) ───────────────────────


def test_discovers_opendrive_maps():
    maps = loader.discover_scenes(py123d_data_root=str(DATA_ROOT), workers=1, map_only=True, datasets=["opendrive"])
    assert {m.location for m in maps} == set(MAP_LOCATIONS)
    assert all(m.dataset == "opendrive" for m in maps)


@pytest.mark.parametrize("location", MAP_LOCATIONS)
@pytest.mark.parametrize("reverse_road_edges", [False, True])  # opendrive preset reverses edges
def test_map_only_binary_walk_consumes_whole_buffer(location, reverse_road_edges):
    scenario, _ = _process_map(location, reverse_road_edges=reverse_road_edges)
    data = serialize.scenario_to_binary(scenario)
    assert _parse(data)["consumed"] == len(data)


@pytest.mark.parametrize("location", MAP_LOCATIONS)
def test_map_only_has_map_but_no_dynamic_entities(location):
    scenario, _ = _process_map(location)
    assert scenario.agents == {}
    assert scenario.objects == {}
    assert scenario.metadata.scenario_length == 0
    assert scenario.metadata.dt == 0.0
    assert len(scenario.map) > 0

    parsed = _parse(serialize.scenario_to_binary(scenario))
    n_road = sum(1 for e in scenario.map.values() if e.geometry is not None and len(e.geometry) > 1)
    # No agents, no objects; map + (stop-zone-derived) traffic controls remain.
    assert parsed["counts"] == (0, n_road, len(scenario.traffic_controls), 0)
    assert parsed["agents"] == []
    assert parsed["objects"] == []


@pytest.mark.parametrize("location", MAP_LOCATIONS)
def test_map_only_reindex_and_references(location):
    scenario, _ = _process_map(location)
    assert list(scenario.map) == list(range(len(scenario.map)))

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
    scenario, _ = _process_map(location)
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
