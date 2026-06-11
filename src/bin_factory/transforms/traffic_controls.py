import numpy as np

from bin_factory import puffer_types, schema
from bin_factory.log_context import log


def process_traffic_controls(scenario: schema.PufferScenario, extras: schema.ExtractionExtras) -> None:
    map_data = scenario.map
    scenario_length = scenario.metadata.scenario_length

    lanes_by_id = {eid: edata for eid, edata in map_data.items() if edata.is_lane}

    elements: list[dict] = []
    covered_lanes: set[int] = set()
    used_ids: set[int] = set()

    for element_id, traffic_light in extras.traffic_lights.items():
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
        control_id = int(element_id)

        elements.append(
            {
                "id": control_id,
                "type": puffer_types.TCType.TRAFFIC_LIGHT,
                "controlled_lanes": [traffic_light.controlled_lane],
                "stop_line": stop_line,
                "heading": heading,
                "states": traffic_light.states,
            },
        )
        used_ids.add(control_id)
        covered_lanes.add(traffic_light.controlled_lane)

    next_id = max(used_ids, default=-1) + 1

    for element_data in extras.stop_zones:
        stop_zone_type = element_data.type
        if stop_zone_type == puffer_types.TCType.TRAFFIC_LIGHT and scenario_length > 0:
            continue  # Skip stop zones if traffic lights are already defined, to avoid duplicates

        controlled_lanes = [lid for lid in element_data.controlled_lanes if lid not in covered_lanes]
        if not controlled_lanes:
            continue

        controlled_lane_id = controlled_lanes[0]
        if (controlled_lane := lanes_by_id.get(controlled_lane_id)) is None:
            log.debug(
                "lane=%s: controlled lane for stop zone (type=%s) not in map, skipping",
                controlled_lane_id,
                stop_zone_type,
            )
            continue
        try:
            heading = _compute_heading_from_incoming_lanes(controlled_lane, lanes_by_id)
            stop_line = _stop_line_from_polygon(element_data.polygon, heading)
        except ValueError as exc:
            log.warning("lane=%s: malformed stop zone, skipping: %s", controlled_lane_id, exc)
            continue

        if stop_zone_type == puffer_types.TCType.TRAFFIC_LIGHT:
            states = [puffer_types.TLState.UNKNOWN] * scenario_length
        else:
            states = []

        elements.append(
            {
                "id": next_id,
                "type": stop_zone_type,
                "controlled_lanes": controlled_lanes,
                "stop_line": stop_line,
                "heading": heading,
                "states": states,
            },
        )
        next_id += 1

    scenario.traffic_controls = elements


def _stop_line_from_position(heading: float, lane_id: int, lanes_by_id: dict[int, schema.MapElement]) -> np.ndarray:
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
    lane_vector = np.array([np.cos(heading), np.sin(heading), 0.0])
    return np.array(
        [
            center - (width / 2) * np.array([-lane_vector[1], lane_vector[0], 0.0]),
            center + (width / 2) * np.array([-lane_vector[1], lane_vector[0], 0.0]),
        ],
        dtype=np.float64,
    )


def _compute_heading_from_incoming_lanes(lane: schema.MapElement, lanes_by_id: dict[int, schema.MapElement]) -> float:
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
