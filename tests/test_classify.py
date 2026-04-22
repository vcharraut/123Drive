"""Tests for the classify pipeline.

The new design bridges py123d UUIDs to nuPlan .db tokens via timestamps:
  sync.arrow uuid → (log_name, timestamp_us)  [build_uuid_index]
  log_name → db_path                           [build_log_db_index]
  db query: scenario_tag WHERE timestamp IN [start, start+duration)
  continuity filter: type kept only if longest consecutive run > 30% of frames

All tests are self-contained — no real .db files or arrow files required.
"""

import json
import sqlite3
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pytest

from bin_factory.classify import (
    NUPLAN_14_SCENARIO_TYPES,
    _20S_US,
    build_log_db_index,
    build_uuid_index,
    classify_bins,
    discover_db_files,
    get_scenario_types_in_window,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UID1 = "818c6675-ce85-5c96-a2d6-1c8b03f45ca5"
_UID2 = "0166016b-897c-58de-b027-a4ff54f04787"
_UID3 = "017ecaf6-522c-5925-b219-1f230faa1b58"  # not in any index
_LOG1 = "2021.07.24.20.37.45_veh-17_00015_00375"
_LOG2 = "2021.06.07.18.53.26_veh-26_00005_00427"
_T0 = 1_627_159_200_000_000  # arbitrary base timestamp (μs)

# Real bin in nuplan_bins_test/
_BIN_UID = "0ab82aa6-0e1c-508d-96fa-183876ed0dfa"

NUPLAN_BINS_TEST_DIR = Path(__file__).parent.parent / "nuplan_bins_test"
NUPLAN_TEST_LABELS_JSON = Path(__file__).parent / "nuplan_test_scenario_labels.json"


def _make_sync_arrow(log_dir: Path, log_name: str, rows: list[tuple[str, int]]) -> None:
    """Write a minimal sync.arrow with (uuid, timestamp_us) rows and log_name in metadata."""
    import msgpack
    log_dir.mkdir(parents=True, exist_ok=True)

    uuid_type = pa.string()  # simplified — real py123d uses extension type
    schema = pa.schema(
        [pa.field("sync.uuid", uuid_type), pa.field("sync.timestamp_us", pa.int64())],
        metadata={b"metadata": msgpack.packb({"log_name": log_name, "dataset": "nuplan"})},
    )
    table = pa.table(
        {"sync.uuid": [r[0] for r in rows], "sync.timestamp_us": [r[1] for r in rows]},
        schema=schema,
    )
    sink = pa.BufferOutputStream()
    with pa_ipc.new_file(sink, schema) as writer:
        writer.write_table(table)
    (log_dir / "sync.arrow").write_bytes(sink.getvalue().to_pybytes())


def _make_db(path: Path, rows: list[tuple[int, str]]) -> None:
    """Create a minimal nuPlan-schema .db.

    rows: list of (timestamp_us, scenario_type)
    """
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE lidar_pc (
            token     BLOB NOT NULL PRIMARY KEY,
            timestamp INTEGER NOT NULL
        );
        CREATE TABLE scenario_tag (
            token          BLOB NOT NULL PRIMARY KEY,
            type           TEXT NOT NULL,
            lidar_pc_token BLOB NOT NULL,
            FOREIGN KEY (lidar_pc_token) REFERENCES lidar_pc(token)
        );
    """)
    for ts, stype in rows:
        lp_token = uuid.uuid4().bytes
        tag_token = uuid.uuid4().bytes
        con.execute("INSERT OR IGNORE INTO lidar_pc VALUES (?, ?)", (lp_token, ts))
        con.execute("INSERT INTO scenario_tag VALUES (?, ?, ?)", (tag_token, stype, lp_token))
    con.commit()
    con.close()


def _make_bin(directory: Path, uid: str, dataset: str = "nuplan") -> Path:
    path = directory / f"{dataset}__{uid}.bin"
    path.write_bytes(b"")
    return path


# ---------------------------------------------------------------------------
# discover_db_files
# ---------------------------------------------------------------------------

def test_discover_db_files(tmp_path):
    (tmp_path / "a.db").touch()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.db").touch()
    (tmp_path / "not_a_db.txt").touch()

    found = {p.name for p in discover_db_files(tmp_path)}
    assert found == {"a.db", "b.db"}


# ---------------------------------------------------------------------------
# build_uuid_index
# ---------------------------------------------------------------------------

def test_build_uuid_index_single_log(tmp_path):
    log_dir = tmp_path / "logs" / "split" / _LOG1
    _make_sync_arrow(log_dir, _LOG1, [(_UID1, _T0), (_UID2, _T0 + 100_000)])

    index = build_uuid_index(tmp_path)
    assert index[_UID1] == (_LOG1, _T0)
    assert index[_UID2] == (_LOG1, _T0 + 100_000)


def test_build_uuid_index_multiple_logs(tmp_path):
    _make_sync_arrow(tmp_path / "logs" / "val" / _LOG1, _LOG1, [(_UID1, _T0)])
    _make_sync_arrow(tmp_path / "logs" / "train" / _LOG2, _LOG2, [(_UID2, _T0 + 5_000_000)])

    index = build_uuid_index(tmp_path)
    assert index[_UID1][0] == _LOG1
    assert index[_UID2][0] == _LOG2


def test_build_uuid_index_bad_arrow_skipped(tmp_path):
    good_dir = tmp_path / "logs" / "val" / _LOG1
    good_dir.mkdir(parents=True)
    (good_dir / "sync.arrow").write_bytes(b"not an arrow file")

    index = build_uuid_index(tmp_path)
    assert _UID1 not in index  # bad file skipped gracefully


# ---------------------------------------------------------------------------
# build_log_db_index
# ---------------------------------------------------------------------------

def test_build_log_db_index(tmp_path):
    db1 = tmp_path / f"{_LOG1}.db"
    db2 = tmp_path / f"{_LOG2}.db"
    db1.touch()
    db2.touch()

    index = build_log_db_index([db1, db2])
    assert index[_LOG1] == db1
    assert index[_LOG2] == db2


# ---------------------------------------------------------------------------
# get_scenario_types_in_window
# ---------------------------------------------------------------------------

def test_window_captures_types_within_range(tmp_path):
    db = tmp_path / "log.db"
    _make_db(db, [
        (_T0,              "changing_lane"),        # t=0   → inside
        (_T0 + 10_000_000, "high_magnitude_speed"), # t=10s → inside
        (_T0 + 25_000_000, "on_intersection"),      # t=25s → outside
    ])

    types = set(get_scenario_types_in_window(db, _T0, duration_us=_20S_US))
    assert types == {"changing_lane", "high_magnitude_speed"}


def test_window_boundary_is_exclusive(tmp_path):
    db = tmp_path / "log.db"
    _make_db(db, [
        (_T0,         "changing_lane"),
        (_T0 + _20S_US, "on_intersection"),  # exactly at end → excluded
    ])

    types = get_scenario_types_in_window(db, _T0, duration_us=_20S_US)
    assert "changing_lane" in types
    assert "on_intersection" not in types


def test_window_deduplicates_types(tmp_path):
    db = tmp_path / "log.db"
    _make_db(db, [
        (_T0,             "changing_lane"),
        (_T0 + 1_000_000, "changing_lane"),
        (_T0 + 2_000_000, "changing_lane"),
    ])

    types = get_scenario_types_in_window(db, _T0, duration_us=_20S_US)
    assert types == ["changing_lane"]


# ---------------------------------------------------------------------------
# classify_bins
# ---------------------------------------------------------------------------

def test_classify_bins_full_pipeline(tmp_path):
    db = tmp_path / f"{_LOG1}.db"
    _make_db(db, [
        (_T0,              "changing_lane"),
        (_T0 + 10_000_000, "high_magnitude_speed"),
    ])

    uuid_index = {_UID1: (_LOG1, _T0)}
    log_db_index = {_LOG1: db}
    bins_dir = tmp_path / "bins"
    bins_dir.mkdir()
    _make_bin(bins_dir, _UID1)

    results = classify_bins(bins_dir, uuid_index, log_db_index)
    assert set(results[f"nuplan__{_UID1}.bin"]) == {"changing_lane", "high_magnitude_speed"}


def test_classify_bins_uuid_not_in_arrow_index(tmp_path):
    bins_dir = tmp_path / "bins"
    bins_dir.mkdir()
    _make_bin(bins_dir, _UID1)

    results = classify_bins(bins_dir, uuid_index={}, log_db_index={})
    assert results[f"nuplan__{_UID1}.bin"] == []


def test_classify_bins_log_has_no_db(tmp_path):
    bins_dir = tmp_path / "bins"
    bins_dir.mkdir()
    _make_bin(bins_dir, _UID1)

    uuid_index = {_UID1: (_LOG1, _T0)}
    results = classify_bins(bins_dir, uuid_index, log_db_index={})
    assert results[f"nuplan__{_UID1}.bin"] == []


def test_classify_bins_malformed_filename(tmp_path):
    bins_dir = tmp_path / "bins"
    bins_dir.mkdir()
    (bins_dir / "no_double_underscore.bin").write_bytes(b"")

    results = classify_bins(bins_dir, uuid_index={}, log_db_index={})
    assert results["no_double_underscore.bin"] == []


def test_classify_bins_type_filter(tmp_path):
    db = tmp_path / f"{_LOG1}.db"
    _make_db(db, [
        (_T0,             "changing_lane"),
        (_T0 + 5_000_000, "high_magnitude_speed"),
    ])
    bins_dir = tmp_path / "bins"
    bins_dir.mkdir()
    _make_bin(bins_dir, _UID1)

    results = classify_bins(
        bins_dir,
        uuid_index={_UID1: (_LOG1, _T0)},
        log_db_index={_LOG1: db},
        filter_types=["changing_lane"],
    )
    assert results[f"nuplan__{_UID1}.bin"] == ["changing_lane"]


# ---------------------------------------------------------------------------
# main() CLI — full end-to-end
# ---------------------------------------------------------------------------

def test_main_end_to_end(tmp_path, monkeypatch):
    # Two logs, two bins; _UID3 intentionally absent
    log1_dir = tmp_path / "py123d" / "logs" / "val" / _LOG1
    log2_dir = tmp_path / "py123d" / "logs" / "train" / _LOG2
    _make_sync_arrow(log1_dir, _LOG1, [(_UID1, _T0)])
    _make_sync_arrow(log2_dir, _LOG2, [(_UID2, _T0)])

    db_dir = tmp_path / "dbs"
    db_dir.mkdir()
    _make_db(db_dir / f"{_LOG1}.db", [(_T0, "starting_left_turn")])
    _make_db(db_dir / f"{_LOG2}.db", [(_T0, "starting_right_turn")])

    bins_dir = tmp_path / "bins"
    bins_dir.mkdir()
    _make_bin(bins_dir, _UID1)
    _make_bin(bins_dir, _UID2)
    _make_bin(bins_dir, _UID3)

    output = tmp_path / "labels.json"
    monkeypatch.setattr("sys.argv", [
        "classify",
        "--bins_dir", str(bins_dir),
        "--nuplan_db_dir", str(db_dir),
        "--py123d_data_root", str(tmp_path / "py123d"),
        "--output", str(output),
    ])

    assert main() == 0
    labels = json.loads(output.read_text())
    assert labels[f"nuplan__{_UID1}.bin"] == ["starting_left_turn"]
    assert labels[f"nuplan__{_UID2}.bin"] == ["starting_right_turn"]
    assert labels[f"nuplan__{_UID3}.bin"] == []


# ---------------------------------------------------------------------------
# Continuity threshold
# ---------------------------------------------------------------------------

def test_continuous_threshold_excludes_scattered(tmp_path):
    """Types without a sufficient continuous run are excluded."""
    db = tmp_path / "log.db"
    # 8 frames total:
    #   "changing_lane"            : frames 0-3 → run=4, 4/8=50% > 25% → included
    #   "on_intersection"          : frames 4-5 → run=2, 2/8=25% (not > 25%) → excluded
    #   "following_lane_with_lead" : frames 6,8s (gap) → max run=1, 12.5% → excluded
    _make_db(db, [
        (_T0 + 0,          "changing_lane"),
        (_T0 + 1_000_000,  "changing_lane"),
        (_T0 + 2_000_000,  "changing_lane"),
        (_T0 + 3_000_000,  "changing_lane"),
        (_T0 + 4_000_000,  "on_intersection"),
        (_T0 + 5_000_000,  "on_intersection"),
        (_T0 + 6_000_000,  "following_lane_with_lead"),
        (_T0 + 8_000_000,  "following_lane_with_lead"),
    ])

    types = set(get_scenario_types_in_window(db, _T0, duration_us=_20S_US))
    assert types == {"changing_lane"}


def test_continuous_threshold_strict_greater_than(tmp_path):
    """A run equal to exactly 25% of frames does not qualify (strict >)."""
    db = tmp_path / "log.db"
    # 8 frames: "changing_lane" in frames 0-1 → run=2, 2/8=25% (not > 25%) → excluded
    _make_db(db, [(_T0 + i * 1_000_000, "changing_lane") for i in range(2)] +
                 [(_T0 + (2 + i) * 1_000_000, "on_intersection") for i in range(6)])

    types = get_scenario_types_in_window(db, _T0, duration_us=_20S_US)
    assert "changing_lane" not in types


# ---------------------------------------------------------------------------
# nuplan_bins_test integration — classifies the real bin against mock DB data
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
_NUPLAN_DB_DIR = _PROJECT_ROOT / "mini"
_PY123D_ROOT = _PROJECT_ROOT / "py123d_output"


@pytest.mark.skipif(
    not (_NUPLAN_DB_DIR.exists() and _PY123D_ROOT.exists()),
    reason="requires local mini/ and py123d_output/ data",
)
def test_classify_nuplan_bins_test():
    """Classify the single real bin in nuplan_bins_test against the real mini DB.

    nuplan_test_scenario_labels.json is the ground truth generated once by running
    classify against mini/ + py123d_output/ at 25% threshold. This test ensures the
    pipeline produces the same output on re-run.
    """
    from bin_factory.classify import build_log_db_index, build_uuid_index, discover_db_files

    uuid_index = build_uuid_index(_PY123D_ROOT)
    log_db_index = build_log_db_index(discover_db_files(_NUPLAN_DB_DIR))

    results = classify_bins(NUPLAN_BINS_TEST_DIR, uuid_index, log_db_index)
    bin_name = f"nuplan__{_BIN_UID}.bin"

    expected = json.loads(NUPLAN_TEST_LABELS_JSON.read_text())
    assert sorted(results[bin_name]) == sorted(expected[bin_name])


# ---------------------------------------------------------------------------
# Constant sanity
# ---------------------------------------------------------------------------

def test_14_scenario_types():
    assert len(NUPLAN_14_SCENARIO_TYPES) == 14
    assert "changing_lane" in NUPLAN_14_SCENARIO_TYPES
    assert "starting_left_turn" in NUPLAN_14_SCENARIO_TYPES
