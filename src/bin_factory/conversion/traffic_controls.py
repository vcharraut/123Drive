import dataclasses
import math

import numpy as np
import shapely

from bin_factory import puffer_types, schema
from bin_factory.conversion.geometry import locate_on_polyline
from bin_factory.log_context import log


SAME_APPROACH_RAD = 0.35  # ~20 deg: lights this close in heading serve the same approach
FACING_APPROACH_RAD = 0.785  # up to ~45 deg off exactly opposite, two approaches still face each other
ZONE_MAX_GAP = 1.5  # metres between a light's approach path and a stop zone of that approach


def process_traffic_controls(scenario: schema.PufferScenario, extras: schema.ExtractionExtras) -> None:
    lanes_by_id = {eid: edata for eid, edata in scenario.polylines.items() if edata.is_lane}
    light_zones = [zone for zone in extras.stop_zones if zone.type == puffer_types.TCType.TRAFFIC_LIGHT]
    controls: list[schema.TrafficControl] = []

    # Logged lights first: each controls its lane
    for element_id, traffic_light in extras.traffic_lights.items():
        controlled_lane_id = traffic_light.controlled_lane
        if (controlled_lane := lanes_by_id.get(controlled_lane_id)) is None:
            log.debug("lane=%s tl=%s: controlled lane not in map, skipping", controlled_lane_id, element_id)
            continue
        if controlled_lane.type == puffer_types.LaneType.BIKE_LANE:
            log.debug("lane=%s tl=%s: controlled lane is a bike lane, skipping", controlled_lane_id, element_id)
            continue
        try:
            heading, stop_line = light_stop_line(controlled_lane, lanes_by_id, light_zones)
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

    # Then stop zones, on the lanes no control covers yet
    light_lanes = {control.controlled_lanes[0] for control in controls}
    covered_lanes = set(light_lanes)
    next_id = max((tc.id for tc in controls), default=-1) + 1
    for element_data in extras.stop_zones:
        stop_zone_type = element_data.type
        is_light = stop_zone_type == puffer_types.TCType.TRAFFIC_LIGHT
        # PY123D-REPORT[nuplan]: TL stop polygons reference the approach lanes while TL detections reference the
        # connectors after them, so each signalized approach came out twice with the polygon copy stuck at UNKNOWN.
        # Drop approach lanes whose successor already carries a light. Skip lanes that already have a control so
        # overlapping nuPlan TL polygons and WOMD stop signs do not duplicate controls. A light leaves out bike
        # lanes, as logged lights do.
        controlled_lanes = [
            lid
            for lid in element_data.controlled_lanes
            if lid not in covered_lanes
            and not (
                is_light
                and (
                    lanes_by_id[lid].type == puffer_types.LaneType.BIKE_LANE
                    or light_lanes.intersection(lanes_by_id[lid].exit_lanes)
                )
            )
        ]
        if not controlled_lanes:
            continue

        controlled_lane_id = controlled_lanes[0]
        if controlled_lane_id not in lanes_by_id:
            log.debug("lane=%s: stop zone (type=%s) lane not in map, skipping", controlled_lane_id, stop_zone_type)
            continue
        try:
            # The zone may lie on the controlled lanes or on the lanes entering them.
            near_lanes = controlled_lanes + [e for lid in controlled_lanes for e in lanes_by_id[lid].entry_lanes]
            heading = _heading_near(np.asarray(element_data.polygon)[:, :2].mean(axis=0), near_lanes, lanes_by_id)
            stop_line = _stop_line_from_polygon(element_data.polygon, heading)
        except ValueError as exc:
            log.warning("lane=%s: malformed stop zone, skipping: %s", controlled_lane_id, exc)
            continue

        controls.append(
            schema.TrafficControl(
                id=next_id,
                type=stop_zone_type,
                controlled_lanes=controlled_lanes,
                stop_line=stop_line,
                heading=heading,
                states=[puffer_types.TLState.UNKNOWN] * scenario.metadata.scenario_length if is_light else [],
                intersection_id=element_data.intersection_id,
                phase_idx=element_data.phase_idx if element_data.intersection_id >= 0 else -1,
            ),
        )
        covered_lanes.update(controlled_lanes)
        next_id += 1

    scenario.traffic_controls = group_facing_phases(controls)


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


def light_stop_line(
    lane: schema.RoadPolyline, lanes_by_id: dict[int, schema.RoadPolyline], light_zones: list[schema.StopZone]
) -> tuple[float, np.ndarray]:
    """Return the heading and stop line of a light on ``lane``: at its approach's stop zone, else across its start."""
    at_start = _stop_line_at_start(lane, lanes_by_id)
    return _stop_line_at_zone(lane, lanes_by_id, light_zones) or at_start


def _cross_line(lane: schema.RoadPolyline, center: np.ndarray, heading: float) -> np.ndarray:
    """Return a line through ``center`` across ``lane``'s width, perpendicular to ``heading``."""
    left = np.asarray(lane.left_boundary if lane.left_boundary is not None else [], dtype=np.float64)
    right = np.asarray(lane.right_boundary if lane.right_boundary is not None else [], dtype=np.float64)
    width = (
        float(np.linalg.norm(right[0] - left[0])) if left.ndim == right.ndim == 2 and len(left) and len(right) else 0.0
    )
    if not np.isfinite(width) or width <= 0:
        width = 3.5
    half_width = (width / 2) * np.array([-np.sin(heading), np.cos(heading), 0.0])
    return np.array([center - half_width, center + half_width], dtype=np.float64)


def _stop_line_at_start(
    lane: schema.RoadPolyline, lanes_by_id: dict[int, schema.RoadPolyline]
) -> tuple[float, np.ndarray]:
    """Return the mean heading of the lanes entering ``lane`` and a stop line across its start."""
    headings = []
    for entry_id in lane.entry_lanes:
        entry_lane = lanes_by_id.get(entry_id)
        polyline = np.asarray(entry_lane.polyline if entry_lane is not None else [], dtype=np.float64)
        if polyline.ndim == 2 and len(polyline) >= 2:
            dxy = polyline[-1] - polyline[-2]
            headings.append(np.arctan2(dxy[1], dxy[0]))

    polyline = np.asarray(lane.polyline, dtype=np.float64)
    if headings:
        heading = float(np.arctan2(np.mean(np.sin(headings)), np.mean(np.cos(headings))))
    elif polyline.ndim == 2 and len(polyline) >= 2:
        dxy = polyline[1] - polyline[0]
        heading = float(np.arctan2(dxy[1], dxy[0]))
    else:
        raise ValueError("lane polyline needs at least two points")
    if polyline.ndim != 2 or len(polyline) == 0:
        raise ValueError("controlled lane polyline is empty")
    return heading, _cross_line(lane, polyline[0], heading)


def _stop_line_at_zone(
    lane: schema.RoadPolyline, lanes_by_id: dict[int, schema.RoadPolyline], light_zones: list[schema.StopZone]
) -> tuple[float, np.ndarray] | None:
    """Return the heading and stop line at a traffic-light stop zone of the light's approach, if one lies on its path.

    The path is the approach lane and the controlled lane. The line spans the controlled lane's width where the zone's
    far edge projects onto the path; zones short of the approach lane start project onto it.
    """
    # PY123D-REPORT[nuplan]: lights are detected on junction connectors, but the stop polygon lies on the approach
    # lane before them or up to 9 m into the connector: a line at the connector start is off by over 3 m for 11% of
    # nuPlan lights. Vehicles stop before the polygon, and a line through its centre is crossed while still RED.
    for entry_id in lane.entry_lanes:
        zones = [zone for zone in light_zones if entry_id in zone.controlled_lanes]
        if not zones or (entry := lanes_by_id.get(entry_id)) is None:
            continue
        path = np.vstack([np.asarray(entry.polyline, dtype=np.float64), np.asarray(lane.polyline, dtype=np.float64)])
        line = shapely.LineString(path[:, :2])
        polygons = [shapely.Polygon(np.asarray(zone.polygon)[:, :2]).buffer(0) for zone in zones]
        near = [polygon for polygon in polygons if line.distance(polygon) <= ZONE_MAX_GAP]
        if not near:
            continue
        edges = [shapely.points(shapely.get_coordinates(polygon)) for polygon in near]
        s = min(shapely.line_locate_point(line, points).max() for points in edges)
        cum = schema.arc_length(path[:, :2])
        center = np.array([np.interp(s, cum, path[:, k]) for k in range(3)])
        ahead, behind = line.interpolate(min(s + 0.5, line.length)), line.interpolate(max(s - 0.5, 0.0))
        heading = float(np.arctan2(ahead.y - behind.y, ahead.x - behind.x))
        return heading, _cross_line(lane, center, heading)
    return None


def _heading_near(point: np.ndarray, lane_ids: list[int], lanes_by_id: dict[int, schema.RoadPolyline]) -> float:
    """Return the direction of the nearest lane segment at ``point``.

    Stop zones may sit at an approach lane's end in nuPlan or inside a CARLA junction.
    Use the direction at the zone to orient its stop line.
    """
    candidates = []
    for lane_id in lane_ids:
        polyline = np.asarray(lanes_by_id[lane_id].polyline, dtype=np.float64)
        distance, _, direction, _ = locate_on_polyline(point[None], polyline)
        candidates.append((float(distance[0]), float(direction[0])))
    return min(candidates)[1]


def _stop_line_from_polygon(polygon: np.ndarray, heading: float) -> np.ndarray:
    polygon = np.asarray(polygon, dtype=np.float64)
    if polygon.ndim != 2 or len(polygon) < 2:
        raise ValueError("stop zone polygon needs at least two points")
    if np.allclose(polygon[0], polygon[-1]):
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
