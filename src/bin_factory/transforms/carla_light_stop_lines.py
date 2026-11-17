"""Move CARLA traffic-light stop lines onto the line CARLA's leaderboard scores.

RunningRedLightTest walks every lane of a light's trigger volume forward in 0.5 m steps and scores the last waypoint
before the junction. Trigger volumes live in the CARLA level, not in the OpenDRIVE file, so their scored points ship as
data (data/carla_light_stop_points.json, written by tools/dump_carla_light_stop_points.py).
"""

from __future__ import annotations

import json
import math
from functools import cache
from importlib import resources

import numpy as np

from bin_factory import puffer_types, schema


SCORED_POINTS_FILE = "data/carla_light_stop_points.json"
ENTRY_SETBACK_M = 0.25  # a line ends up at least this far before the junction lanes it controls
MATCH_HEADING_MAX_RAD = math.radians(45.0)
MATCH_LATERAL_MARGIN_M = 0.5  # beyond the line's half length
MATCH_ALONG_MAX_M = 25.0  # Town04 light 515 scores 13.4 m inside its junction
POINT_X, POINT_Y, POINT_HEADING, POINT_IN_JUNCTION, POINT_LIGHT = range(5)


@cache
def _scored_points_by_town() -> dict[str, np.ndarray]:
    towns = json.loads(resources.files("bin_factory").joinpath(SCORED_POINTS_FILE).read_text())["towns"]
    return {
        town: np.array(
            [[p["x"], p["y"], p["heading"], float(p["in_junction"]), p["light"]] for p in points], dtype=np.float64
        ).reshape(-1, 5)
        for town, points in towns.items()
    }


def align_carla_light_stop_lines(scenario: schema.PufferScenario, extras: schema.ExtractionExtras) -> None:
    """Places each traffic light's stop line at CARLA's scored line (no closer than ENTRY_SETBACK_M to the junction
    unless CARLA scores it inside the junction), adds the approach lanes to its controlled lanes, and points every
    control's heading along its lanes at the line."""
    town = scenario.metadata.location or scenario.metadata.id
    points_by_town = _scored_points_by_town()
    if town not in points_by_town:
        raise ValueError(
            f"{town}: no CARLA scored stop points in {SCORED_POINTS_FILE}; run tools/dump_carla_light_stop_points.py"
        )
    points = points_by_town[town].copy()
    points[:, [POINT_X, POINT_Y]] -= np.asarray(extras.centroid, dtype=np.float64)[:2]
    align_light_stop_lines(scenario, points)


def align_light_stop_lines(scenario: schema.PufferScenario, points: np.ndarray) -> None:
    """points: (N, 5) scored stop points in the scenario frame (x, y, heading, in_junction, light id)."""
    lanes = {element_id: element for element_id, element in scenario.map.items() if element.is_lane}
    for control in scenario.traffic_controls:
        if control["type"] != puffer_types.TCType.TRAFFIC_LIGHT:
            _align_heading_with_lanes(control, lanes)
            continue
        junction_lanes = list(control["controlled_lanes"])
        _move_onto_scored_line(control, points, junction_lanes, lanes)
        _add_approach_lanes(control, lanes)
        _align_heading_with_lanes(control, lanes)
        # the heading update turned the line about its middle, which moves the ends of a multi-lane line along the road
        _move_onto_scored_line(control, points, junction_lanes, lanes)


def _move_onto_scored_line(
    control: dict, points: np.ndarray, junction_lanes: list[int], lanes: dict[int, schema.MapElement]
) -> None:
    line = np.array(control["stop_line"], dtype=np.float64)
    along_dir = np.array([math.cos(control["heading"]), math.sin(control["heading"])])
    across_dir = np.array([-along_dir[1], along_dir[0]])
    middle = line[:, :2].mean(axis=0)
    offsets = points[:, [POINT_X, POINT_Y]] - middle
    along = offsets @ along_dir
    heading_gap = np.abs((points[:, POINT_HEADING] - control["heading"] + math.pi) % (2 * math.pi) - math.pi)
    half_length = np.linalg.norm(line[1, :2] - line[0, :2]) / 2
    candidates = np.flatnonzero(
        (heading_gap < MATCH_HEADING_MAX_RAD)
        & (np.abs(offsets @ across_dir) <= half_length + MATCH_LATERAL_MARGIN_M)
        & (np.abs(along) <= MATCH_ALONG_MAX_M)
    )
    if not len(candidates):
        raise ValueError(f"traffic light {control['id']}: no CARLA scored stop point at its stop line")
    nearest_light = points[candidates[np.argmin(np.abs(along[candidates]))], POINT_LIGHT]
    light_points = candidates[points[candidates, POINT_LIGHT] == nearest_light]
    most_upstream = light_points[np.argmin(along[light_points])]
    shift = along[most_upstream]
    if not points[most_upstream, POINT_IN_JUNCTION]:
        junction_entry = min(
            float((np.asarray(lanes[lane_id].polyline)[0, :2] - middle) @ along_dir) for lane_id in junction_lanes
        )
        shift = min(shift, junction_entry - ENTRY_SETBACK_M)
    line[:, :2] += shift * along_dir
    control["stop_line"] = line


def _add_approach_lanes(control: dict, lanes: dict[int, schema.MapElement]) -> None:
    controlled = list(control["controlled_lanes"])
    approach = {entry for lane_id in controlled for entry in lanes[lane_id].entry_lanes if entry in lanes}
    control["controlled_lanes"] = controlled + sorted(approach - set(controlled))


def _align_heading_with_lanes(control: dict, lanes: dict[int, schema.MapElement]) -> None:
    line = np.array(control["stop_line"], dtype=np.float64)
    middle = line[:, :2].mean(axis=0)
    directions = [
        _nearest_segment_direction(np.asarray(lanes[lane_id].polyline)[:, :2], middle)
        for lane_id in control["controlled_lanes"]
        if lane_id in lanes and len(lanes[lane_id].polyline) >= 2
    ]
    if not directions:
        return
    heading = math.atan2(float(np.mean(np.sin(directions))), float(np.mean(np.cos(directions))))
    old_across = line[1, :2] - line[0, :2]
    half_length = np.linalg.norm(old_across) / 2
    across = np.array([-math.sin(heading), math.cos(heading)])
    if old_across @ across < 0:  # keep the endpoint order the line had
        across = -across
    line[0, :2] = middle - half_length * across
    line[1, :2] = middle + half_length * across
    control["stop_line"] = line
    control["heading"] = heading


def _nearest_segment_direction(polyline: np.ndarray, point: np.ndarray) -> float:
    starts, ends = polyline[:-1], polyline[1:]
    segments = ends - starts
    lengths_sq = np.maximum((segments**2).sum(axis=1), 1e-12)
    t = np.clip(((point - starts) * segments).sum(axis=1) / lengths_sq, 0.0, 1.0)
    distances = np.linalg.norm(starts + t[:, None] * segments - point, axis=1)
    nearest = segments[int(np.argmin(distances))]
    return math.atan2(nearest[1], nearest[0])
