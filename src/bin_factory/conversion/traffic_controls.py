import dataclasses
import math

import numpy as np

from bin_factory import puffer_types, schema
from bin_factory.log_context import log


SAME_APPROACH_RAD = 0.35  # ~20 deg: lights this close in heading serve the same approach
FACING_APPROACH_RAD = 0.785  # up to ~45 deg off exactly opposite, two approaches still face each other


def process_traffic_controls(scenario: schema.PufferScenario, extras: schema.ExtractionExtras) -> None:
    lanes_by_id = {eid: edata for eid, edata in scenario.polylines.items() if edata.is_lane}
    light_controls, covered_lanes = _build_traffic_light_controls(extras.traffic_lights, lanes_by_id)
    next_id = max((tc.id for tc in light_controls), default=-1) + 1
    zone_controls = _build_stop_zone_controls(
        extras.stop_zones,
        lanes_by_id,
        covered_lanes,
        scenario.metadata.scenario_length,
        next_id,
    )
    scenario.traffic_controls = group_facing_phases(light_controls + zone_controls)


def _build_traffic_light_controls(
    traffic_lights: dict[int, schema.TrafficLightTrack],
    lanes_by_id: dict[int, schema.RoadPolyline],
) -> tuple[list[schema.TrafficControl], set[int]]:
    controls: list[schema.TrafficControl] = []
    covered_lanes: set[int] = set()

    for element_id, traffic_light in traffic_lights.items():
        controlled_lane_id = traffic_light.controlled_lane
        if (controlled_lane := lanes_by_id.get(controlled_lane_id)) is None:
            log.debug(
                "lane=%s tl=%s: controlled lane not in map, skipping",
                controlled_lane_id,
                element_id,
            )
            continue
        if controlled_lane.type == puffer_types.LaneType.BIKE_LANE:
            log.debug(
                "lane=%s tl=%s: controlled lane is a bike lane, skipping",
                controlled_lane_id,
                element_id,
            )
            continue
        try:
            heading = _compute_heading_from_incoming_lanes(controlled_lane, lanes_by_id)
            stop_line = _stop_line_from_position(heading, controlled_lane_id, lanes_by_id)
        except ValueError as exc:
            log.warning("lane=%s tl=%s: malformed traffic light, skipping: %s", controlled_lane_id, element_id, exc)
            continue

        controls.append(
            schema.TrafficControl(
                id=int(element_id),
                type=puffer_types.TCType.TRAFFIC_LIGHT,
                controlled_lanes=[controlled_lane_id],
                stop_line=stop_line,
                heading=heading,
                states=traffic_light.states,
            ),
        )
        covered_lanes.add(controlled_lane_id)

    return controls, covered_lanes


def _build_stop_zone_controls(
    stop_zones: list[schema.StopZone],
    lanes_by_id: dict[int, schema.RoadPolyline],
    covered_lanes: set[int],
    scenario_length: int,
    next_id: int,
) -> list[schema.TrafficControl]:
    controls: list[schema.TrafficControl] = []
    light_lanes = set(covered_lanes)

    for element_data in stop_zones:
        stop_zone_type = element_data.type
        # PY123D-REPORT[nuplan]: TL stop polygons reference the approach lanes while TL detections reference the
        # connectors after them, so each signalized approach came out twice with the polygon copy stuck at UNKNOWN.
        # Drop approach lanes whose successor already carries a light. Skip lanes that already have a control so
        # overlapping nuPlan TL polygons and WOMD stop signs do not duplicate controls.
        controlled_lanes = [
            lid
            for lid in element_data.controlled_lanes
            if lid not in covered_lanes
            and not (
                stop_zone_type == puffer_types.TCType.TRAFFIC_LIGHT
                and light_lanes.intersection(lanes_by_id[lid].exit_lanes)
            )
        ]
        if not controlled_lanes:
            continue

        controlled_lane_id = controlled_lanes[0]
        if lanes_by_id.get(controlled_lane_id) is None:
            log.debug(
                "lane=%s: controlled lane for stop zone (type=%s) not in map, skipping",
                controlled_lane_id,
                stop_zone_type,
            )
            continue
        try:
            # The zone may lie on the controlled lanes or on the lanes entering them.
            near_lanes = controlled_lanes + [e for lid in controlled_lanes for e in lanes_by_id[lid].entry_lanes]
            heading = _heading_near(np.asarray(element_data.polygon)[:, :2].mean(axis=0), near_lanes, lanes_by_id)
            stop_line = _stop_line_from_polygon(element_data.polygon, heading)
        except ValueError as exc:
            log.warning("lane=%s: malformed stop zone, skipping: %s", controlled_lane_id, exc)
            continue

        states = (
            [puffer_types.TLState.UNKNOWN] * scenario_length
            if stop_zone_type == puffer_types.TCType.TRAFFIC_LIGHT
            else []
        )
        controls.append(
            schema.TrafficControl(
                id=next_id,
                type=stop_zone_type,
                controlled_lanes=controlled_lanes,
                stop_line=stop_line,
                heading=heading,
                states=states,
                intersection_id=element_data.intersection_id,
                phase_idx=element_data.phase_idx if element_data.intersection_id >= 0 else -1,
            ),
        )
        covered_lanes.update(controlled_lanes)
        next_id += 1

    return controls


def group_facing_phases(controls: list[schema.TrafficControl]) -> list[schema.TrafficControl]:
    """Give each junction light the first phase whose lights all serve its approach or face it, compact 0..N-1."""
    junctions: dict[int, list[list[float]]] = {}
    grouped = []
    for tc in controls:
        if tc.phase_idx < 0:
            grouped.append(tc)
            continue
        phases = junctions.setdefault(tc.intersection_id, [])
        phase = next(
            (
                index
                for index, headings in enumerate(phases)
                if not any(
                    SAME_APPROACH_RAD
                    <= abs(math.remainder(tc.heading - heading, 2 * math.pi))
                    <= math.pi - FACING_APPROACH_RAD
                    for heading in headings
                )
            ),
            len(phases),
        )
        if phase == len(phases):
            phases.append([])
        phases[phase].append(tc.heading)
        grouped.append(dataclasses.replace(tc, phase_idx=phase))
    return grouped


def _stop_line_from_position(heading: float, lane_id: int, lanes_by_id: dict[int, schema.RoadPolyline]) -> np.ndarray:
    lane = lanes_by_id.get(lane_id)
    if lane is None:
        raise ValueError("controlled lane is missing")
    polyline = np.asarray(lane.polyline, dtype=np.float64)
    if polyline.ndim != 2 or len(polyline) == 0:
        raise ValueError("controlled lane polyline is empty")
    center = polyline[0]
    left_boundary = np.asarray(lane.left_boundary if lane.left_boundary is not None else [], dtype=np.float64)
    right_boundary = np.asarray(lane.right_boundary if lane.right_boundary is not None else [], dtype=np.float64)
    width = _lane_width_from_boundaries(left_boundary, right_boundary)
    half_width = (width / 2) * np.array([-np.sin(heading), np.cos(heading), 0.0])
    return np.array([center - half_width, center + half_width], dtype=np.float64)


def _heading_near(point: np.ndarray, lane_ids: list[int], lanes_by_id: dict[int, schema.RoadPolyline]) -> float:
    """Return the direction of the nearest lane segment at ``point``.

    Stop zones may sit at an approach lane's end in nuPlan or inside a CARLA junction.
    Use the direction at the zone to orient its stop line.
    """
    candidates = []
    for lane_id in lane_ids:
        polyline = np.asarray(lanes_by_id[lane_id].polyline, dtype=np.float64)[:, :2]
        a, ab = polyline[:-1], np.diff(polyline, axis=0)
        length_sq = (ab**2).sum(axis=1)
        t = np.clip(((point - a) * ab).sum(axis=1) / np.maximum(length_sq, 1e-9), 0.0, 1.0)
        degenerate = np.where(length_sq > 0, 0.0, np.inf)
        distance = np.linalg.norm(point - (a + t[:, None] * ab), axis=1) + degenerate
        k = int(np.argmin(distance))
        candidates.append((float(distance[k]), float(np.arctan2(ab[k, 1], ab[k, 0]))))
    return min(candidates)[1]


def _compute_heading_from_incoming_lanes(
    lane: schema.RoadPolyline, lanes_by_id: dict[int, schema.RoadPolyline]
) -> float:
    headings = []
    for entry_id in lane.entry_lanes:
        entry_lane = lanes_by_id.get(entry_id)
        if entry_lane is None:
            continue
        polyline = np.asarray(entry_lane.polyline, dtype=np.float64)
        if polyline.ndim != 2 or len(polyline) < 2:
            continue
        dxy = polyline[-1] - polyline[-2]
        headings.append(np.arctan2(dxy[1], dxy[0]))

    if not headings:
        polyline = np.asarray(lane.polyline, dtype=np.float64)
        if polyline.ndim != 2 or len(polyline) < 2:
            raise ValueError("lane polyline needs at least two points")
        dxy = polyline[1] - polyline[0]
        return float(np.arctan2(dxy[1], dxy[0]))

    return float(np.arctan2(np.mean(np.sin(headings)), np.mean(np.cos(headings))))


def _lane_width_from_boundaries(left_boundary: np.ndarray, right_boundary: np.ndarray) -> float:
    if left_boundary.ndim != 2 or right_boundary.ndim != 2:
        return 3.5
    if len(left_boundary) == 0 or len(right_boundary) == 0:
        return 3.5
    width = float(np.linalg.norm(right_boundary[0] - left_boundary[0]))
    if not np.isfinite(width) or width <= 0:
        return 3.5
    return width


def _stop_line_from_polygon(polygon: np.ndarray, heading: float) -> np.ndarray:
    polygon = np.asarray(polygon, dtype=np.float64)
    if polygon.ndim != 2 or len(polygon) < 2:
        raise ValueError("stop zone polygon needs at least two points")
    if len(polygon) > 1 and np.allclose(polygon[0], polygon[-1]):
        polygon = polygon[:-1]
    if len(polygon) < 2:
        raise ValueError("stop zone polygon collapses to fewer than two points")

    center = polygon.mean(axis=0)
    perp_xy = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float64)
    intersections = []

    for idx, start in enumerate(polygon):
        end = polygon[(idx + 1) % len(polygon)]
        edge_xy = end[:2] - start[:2]
        transform = np.column_stack((perp_xy, -edge_xy))
        if np.isclose(np.linalg.det(transform), 0.0):
            continue

        t, u = np.linalg.solve(transform, start[:2] - center[:2])
        if 0.0 <= u <= 1.0:
            point = start + u * (end - start)
            intersections.append((t, point))

    if len(intersections) >= 2:
        intersections.sort(key=lambda item: item[0])
        return np.array([intersections[0][1], intersections[-1][1]], dtype=np.float64)

    half = np.ptp(polygon[:, :2] @ perp_xy) / 2
    perp = np.array([perp_xy[0], perp_xy[1], 0.0], dtype=np.float64)
    return np.array([center - perp * half, center + perp * half], dtype=np.float64)
