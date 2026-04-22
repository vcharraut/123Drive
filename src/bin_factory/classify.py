"""Classify nuPlan .bin files into scenario categories via nuPlan SQLite databases.

Pipeline:
  1. Scan py123d sync.arrow files to build: uuid → (log_name, start_timestamp_us)
     (py123d generates UUIDs; the original nuPlan .db uses 8-byte tokens — the only
     shared key between the two systems is the microsecond timestamp)
  2. Scan .db files to build: log_name → db_path
  3. For each .bin, extract its scene UUID, resolve (log_name, start_ts), then
     query ALL scenario_tag types in [start_ts, start_ts + duration_us] from
     that log's .db — a 20-second bin spans ~200 frames, each tagged independently
  4. Write results to a JSON file and print a distribution summary

Usage:
  uv run classify \\
    --bins_dir ./nuplan_mini_val_bins \\
    --nuplan_db_dir ./mini \\
    --py123d_data_root ./py123d_output \\
    --output nuplan_mini_scenario_labels.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow as pa

logger = logging.getLogger(__name__)

NUPLAN_BENCHMARK_14_SCENARIO_TYPES = [
    "accelerating_at_crosswalk",
    "accelerating_at_stop_sign",
    "accelerating_at_traffic_light",
    "changing_lane",
    "following_lane_with_lead",
    "high_magnitude_jerk",
    "high_magnitude_speed",
    "low_magnitude_speed",
    "near_pedestrian_at_pickup_dropoff",
    "on_intersection",
    "on_pickup_dropoff",
    "starting_left_turn",
    "starting_right_turn",
    "starting_straight_traffic_light_intersection_traversal",
    "starting_unprotected_noncross_turn",
]

NUPLAN_15_SCENARIO_TYPES = [
    "accelerating_at_crosswalk",
    "accelerating_at_stop_sign",
    "accelerating_at_traffic_light",
    "changing_lane",
    "following_lane_with_lead",
    "high_magnitude_jerk",
    "high_magnitude_speed",
    "low_magnitude_speed",
    "near_pedestrian_at_pickup_dropoff",
    "on_intersection",
    "on_pickup_dropoff",
    "starting_left_turn",
    "starting_right_turn",
    "starting_straight_traffic_light_intersection_traversal",
    "starting_unprotected_noncross_turn",
]

_20S_US = 20_000_000  # 20 seconds in microseconds

# uuid string → (log_name, start_timestamp_us)
_UuidIndex = Dict[str, Tuple[str, int]]
# log_name → db_path
_LogDbIndex = Dict[str, Path]


def discover_db_files(db_root: Path) -> List[Path]:
    return sorted(db_root.rglob("*.db"))


def build_uuid_index(py123d_data_root: Path) -> _UuidIndex:
    """Scan all sync.arrow files and return uuid → (log_name, timestamp_us).

    The py123d UUID in each bin filename is the uuid of the scene's first frame.
    The log_name is stored in the arrow schema metadata and matches the .db stem.
    """
    index: _UuidIndex = {}
    logs_root = py123d_data_root / "logs"

    for sync_path in sorted(logs_root.rglob("sync.arrow")):
        try:
            table = pa.ipc.open_file(sync_path).read_all()
            # log_name is in the arrow schema metadata (msgpack-encoded)
            raw_meta = table.schema.metadata or {}
            meta_bytes = raw_meta.get(b"metadata", b"")
            log_name = _extract_log_name_from_meta(meta_bytes, sync_path)

            uuids = table.column("sync.uuid").to_pylist()
            timestamps = table.column("sync.timestamp_us").to_pylist()
            for uid, ts in zip(uuids, timestamps):
                index[str(uid)] = (log_name, int(ts))
        except Exception as e:
            logger.warning("Failed to read %s: %s", sync_path, e)

    logger.info("Built uuid index: %d frames across %d logs", len(index), len({v[0] for v in index.values()}))
    return index


def _extract_log_name_from_meta(meta_bytes: bytes, sync_path: Path) -> str:
    """Extract log_name from msgpack-encoded arrow schema metadata, falling back to directory name."""
    try:
        import msgpack
        meta = msgpack.unpackb(meta_bytes, raw=False)
        if "log_name" in meta:
            return meta["log_name"]
    except Exception:
        pass
    return sync_path.parent.name


def build_log_db_index(db_paths: List[Path]) -> _LogDbIndex:
    """Map log_name → db_path using the .db filename stem."""
    return {db_path.stem: db_path for db_path in db_paths}


def find_scenario_type_in_windows(
    bin_name: str,
    db_path: Path,
    start_ts: int,
    duration_us: int = _20S_US,
) -> List[str]:
    """Return scenario types with a continuous run > min_continuous_fraction of frames.

    A type qualifies only if its longest uninterrupted streak of consecutive frames
    exceeds min_continuous_fraction * total_frames in [start_ts, start_ts + duration_us].
    """
    end_ts = start_ts + duration_us
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(
            """
            SELECT lp.token, st.type
            FROM lidar_pc lp
            LEFT JOIN scenario_tag st ON st.lidar_pc_token = lp.token
            WHERE lp.timestamp >= ? AND lp.timestamp < ?
            AND st.type = ?
            """,
            (start_ts, end_ts, NUPLAN_15_SCENARIO_TYPES[-1]),
        )
        rows = cur.fetchall()
        con.close()
        if len(rows) > 0:
            print(f"Queried {len(rows)} frames in {bin_name} from {db_path.name}")
    except Exception as e:
        logger.warning("Failed to query window in %s: %s", db_path.name, e)

def _find_city_db(db_path: Path) -> str:
    """Return the location/city of the log (e.g. 'las_vegas', 'singapore'), or '' on failure."""
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = con.cursor()
        cur.execute("SELECT location FROM log LIMIT 1")
        row = cur.fetchone()
        con.close()
        if row:
            return row[0].lower()
    except Exception:
        pass
    return ""


def _is_singapore_db(db_path: Path) -> bool:
    """Return True if the log was recorded in Singapore (left-hand traffic)."""
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = con.cursor()
        cur.execute("SELECT location FROM log LIMIT 1")
        row = cur.fetchone()
        con.close()
        if row:
            return "singapore" in row[0].lower() or "sg" in row[0].lower()
    except Exception:
        pass
    return False


def get_scenario_types_in_window(
    bin_name: str,
    db_path: Path,
    start_ts: int,
    duration_us: int = _20S_US,
    min_continuous_fraction: float = 0.25,
) -> List[str]:
    """Return scenario types with a continuous run > min_continuous_fraction of frames.

    A type qualifies only if its longest uninterrupted streak of consecutive frames
    exceeds min_continuous_fraction * total_frames in [start_ts, start_ts + duration_us].
    """
    end_ts = start_ts + duration_us
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(
            """
            SELECT lp.token, st.type
            FROM lidar_pc lp
            LEFT JOIN scenario_tag st ON st.lidar_pc_token = lp.token
            WHERE lp.timestamp >= ? AND lp.timestamp < ?
            ORDER BY lp.timestamp, lp.token
            """,
            (start_ts, end_ts),
        )
        rows = cur.fetchall()
        con.close()

        # Build ordered unique-frame list and per-frame type sets
        frame_order: List[bytes] = []
        seen: set = set()
        frame_types: Dict[bytes, set] = {}
        for row in rows:
            token = bytes(row["token"])
            stype = row["type"]
            # if stype is NUPLAN_15_SCENARIO_TYPES[0]:
            #     print(f"Found type {stype} in frame {token.hex()} of {bin_name}")
            if token not in seen:
                frame_order.append(token)
                seen.add(token)
                frame_types[token] = set()
            if stype is not None:
                frame_types[token].add(stype)

        total_frames = len(frame_order)
        if total_frames == 0:
            return []

        all_types: set = set()
        for types in frame_types.values():
            all_types.update(types)
        min_run = total_frames * min_continuous_fraction
        result = []
        for stype in sorted(all_types):
            # if stype == NUPLAN_15_SCENARIO_TYPES[-1]:
            #     print(f"Found {stype} in {bin_name} with {total_frames} frames and min_run={min_run:.1f}")
            max_run = 0
            current_run = 0
            for token in frame_order:
                if stype in frame_types[token]:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            if max_run > min_run:
                result.append(stype)
            elif stype == NUPLAN_15_SCENARIO_TYPES[-1]:
                # A noncross turn doesn't cross oncoming traffic.
                # In right-hand traffic (US cities) that's a left turn;
                # in left-hand traffic (Singapore) that's a right turn.
                noncross_tag = "unprotected_right_turn" if _is_singapore_db(db_path) else "unprotected_left_turn"
                print(f'Found {noncross_tag} in {bin_name} of city {_find_city_db(db_path)} with {"left-hand" if _is_singapore_db(db_path) else "right-hand"} traffic')
                if noncross_tag in all_types:
                    result.append(stype)

        return result
    except Exception as e:
        logger.warning("Failed to query window in %s: %s", db_path.name, e)
        return []


def classify_bins(
    bins_dir: Path,
    uuid_index: _UuidIndex,
    log_db_index: _LogDbIndex,
    duration_us: int = _20S_US,
    filter_types: Optional[List[str]] = None,
    min_continuous_fraction: float = 0.25,
) -> Dict[str, List[str]]:
    """Return bin_filename → [scenario_types] collected across the full scenario window."""
    filter_set = set(filter_types) if filter_types else None
    results: Dict[str, List[str]] = {}

    for bin_path in sorted(bins_dir.glob("*.bin")):
        parts = bin_path.stem.split("__", 1)
        if len(parts) != 2:
            logger.warning("Skipping %s: cannot extract UUID from filename", bin_path.name)
            results[bin_path.name] = []
            continue

        uid = parts[1]
        location = uuid_index.get(uid)
        if location is None:
            logger.debug("UUID %s not found in any sync.arrow", uid)
            results[bin_path.name] = []
            continue

        log_name, start_ts = location
        db_path = log_db_index.get(log_name)
        if db_path is None:
            logger.debug("No .db found for log %s", log_name)
            results[bin_path.name] = []
            continue

        # find_scenario_type_in_windows(bin_path.name, db_path, start_ts, duration_us)
        types = get_scenario_types_in_window(bin_path.name, db_path, start_ts, duration_us, min_continuous_fraction)
        if filter_set:
            types = [t for t in types if t in filter_set]
        results[bin_path.name] = types

    found = sum(1 for v in results.values() if v)
    logger.info("Classified %d / %d bins", found, len(results))
    return results


def print_summary(results: Dict[str, List[str]]) -> None:
    type_counts: Counter = Counter()
    for types in results.values():
        for t in types:
            type_counts[t] += 1

    print("\nScenario type distribution:")
    for stype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {stype}: {count}")

    unclassified = sum(1 for v in results.values() if not v)
    print(f"\nTotal bins   : {len(results)}")
    print(f"Classified   : {len(results) - unclassified}")
    print(f"Unclassified : {unclassified}")


def main() -> int:
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Classify nuPlan .bin files into scenario categories using nuPlan SQLite databases"
    )
    parser.add_argument("--bins_dir", required=True, help="Directory containing .bin files")
    parser.add_argument(
        "--nuplan_db_dir",
        required=True,
        help="Root directory containing nuPlan .db files (searched recursively)",
    )
    parser.add_argument(
        "--py123d_data_root",
        required=True,
        help="py123d data root (contains logs/); used to map bin UUIDs to timestamps via sync.arrow",
    )
    parser.add_argument("--output", default="scenario_labels.json", help="Output JSON path")
    parser.add_argument(
        "--duration_s",
        type=float,
        default=20.0,
        help="Scenario window duration in seconds (must match --duration_s used during conversion)",
    )
    parser.add_argument(
        "--scenario_types",
        nargs="+",
        default=None,
        help="Restrict output to specific scenario types (default: all)",
    )
    parser.add_argument(
        "--min_continuous_fraction",
        type=float,
        default=0.25,
        help="Minimum fraction of frames a type must appear continuously to be classified (default: 0.25)",
    )
    args = parser.parse_args()

    bins_dir = Path(args.bins_dir)
    db_root = Path(args.nuplan_db_dir)
    py123d_root = Path(args.py123d_data_root)

    for p, name in [(bins_dir, "--bins_dir"), (db_root, "--nuplan_db_dir"), (py123d_root, "--py123d_data_root")]:
        if not p.exists():
            parser.error(f"{name} does not exist: {p}")

    db_paths = discover_db_files(db_root)
    if not db_paths:
        parser.error(f"No .db files found under {db_root}")
    logger.info("Found %d .db file(s)", len(db_paths))

    duration_us = int(args.duration_s * 1_000_000)
    uuid_index = build_uuid_index(py123d_root)
    log_db_index = build_log_db_index(db_paths)
    results = classify_bins(
        bins_dir,
        uuid_index,
        log_db_index,
        duration_us=duration_us,
        filter_types=args.scenario_types,
        min_continuous_fraction=args.min_continuous_fraction,
    )

    output_path = Path(args.output)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    logger.info("Saved labels to %s", output_path)

    # print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
