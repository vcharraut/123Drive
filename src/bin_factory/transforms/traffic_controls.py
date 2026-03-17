import numpy as np

from bin_factory import types as puffer_types


DEFAULT_LANE_WIDTH = 3.7


def process_traffic_controls(scenario, extras):
    map_data = scenario.map
    scenario_length = scenario.metadata.scenario_length

    lanes_by_id = {eid: edata for eid, edata in map_data.items() if puffer_types.is_road_lane(edata["type"])}

    elements = []
    covered_lanes = set()

    for element_id, element_data in extras["traffic_lights"].items():
        controlled_lane_id = element_data.controlled_lane
        covered_lanes.add(controlled_lane_id)

        position = element_data.position
        lane = lanes_by_id.get(controlled_lane_id)
        heading = _heading_from_entry_lanes(controlled_lane_id, lanes_by_id)
        stop_line = _stop_line_from_position(position, lane, heading)

        states = [s if s is not None else puffer_types.TLState.UNKNOWN for s in element_data.states]
        elements.append(
            {
                "id": int(element_id),
                "controlled_lanes": [controlled_lane_id],
                "stop_line": stop_line,
                "heading": heading,
                "states": states,
                "type": puffer_types.TCType.TRAFFIC_LIGHT,
            }
        )

    if scenario_length <= 0:
        next_id = max((e["id"] for e in elements), default=-1) + 1
        for element_data in extras["stop_zones"]:
            stop_zone_type = element_data["type"]
            controlled_lanes = [lid for lid in element_data["controlled_lanes"] if lid not in covered_lanes]
            if not controlled_lanes:
                continue

            heading = _heading_from_entry_lanes(controlled_lanes[0], lanes_by_id)
            stop_line = _longest_polygon_edge(element_data["polygon"])

            elements.append(
                {
                    "id": next_id,
                    "controlled_lanes": controlled_lanes,
                    "stop_line": stop_line,
                    "heading": heading,
                    "states": [puffer_types.TLState.UNKNOWN] * scenario_length,
                    "type": stop_zone_type,
                }
            )
            next_id += 1

    scenario.traffic_controls = elements


def _stop_line_from_position(position, lane, heading):
    center = np.asarray(position, dtype=np.float64)
    width = DEFAULT_LANE_WIDTH
    if lane is not None:
        left_boundary = lane.get("left_boundary")
        right_boundary = lane.get("right_boundary")
        if left_boundary is not None and right_boundary is not None:
            width = float(np.linalg.norm(np.asarray(left_boundary[-1]) - np.asarray(right_boundary[-1])))

    d2 = np.array([np.cos(heading), np.sin(heading)])
    perp = np.array([-d2[1], d2[0], 0.0])
    half = width / 2.0
    return np.array([center - perp * half, center + perp * half], dtype=np.float64)


def _heading_from_entry_lanes(controlled_lane_id, lanes_by_id):
    lane = lanes_by_id.get(controlled_lane_id)
    if lane is None:
        return 0.0

    headings = []
    for entry_id in lane.get("entry_lanes", []):
        entry_lane = lanes_by_id.get(entry_id)
        if entry_lane is None:
            continue
        polyline = entry_lane["polyline"]
        if len(polyline) >= 2:
            d = polyline[-1] - polyline[-2]
            headings.append(np.arctan2(d[1], d[0]))
    if headings:
        return float(np.arctan2(np.mean(np.sin(headings)), np.mean(np.cos(headings))))

    polyline = lane["polyline"]
    if len(polyline) >= 2:
        d = polyline[1] - polyline[0]
        return float(np.arctan2(d[1], d[0]))

    return 0.0


def _longest_polygon_edge(polygon):
    polygon = np.asarray(polygon, dtype=np.float64)
    edges = np.roll(polygon, -1, axis=0) - polygon
    best = np.argmax(np.linalg.norm(edges, axis=1))
    return np.array([polygon[best], polygon[(best + 1) % len(polygon)]], dtype=np.float64)
