"""Helpers for Puffer web visualization."""

import numpy as np

from bin_factory.convert.types import (
    AGENT_TYPE_NAMES,
    ROAD_COLORS,
    ROAD_TYPE_NAMES,
    TL_STATE_COLORS,
    TL_STATE_NAMES,
    LaneType,
    MiscRoadType,
    RoadEdgeType,
    RoadLineType,
    is_road_edge,
    is_road_lane,
    is_road_line,
)

# Vehicle colors palette
VEHICLE_COLORS = [
    "#FF0000",  # Red (ego)
    "#1F77B4",  # Blue
    "#FF7F0E",  # Orange
    "#2CA02C",  # Green
    "#9467BD",  # Purple
    "#8C564B",  # Brown
    "#E377C2",  # Pink
    "#BCBD22",  # Yellow-green
    "#17BECF",  # Cyan
    "#AEC7E8",  # Light blue
    "#FFBB78",  # Light orange
    "#98DF8A",  # Light green
    "#FF9896",  # Light red
    "#C5B0D5",  # Light purple
    "#C49C94",  # Light brown
    "#F7B6D2",  # Light pink
    "#DBDB8D",  # Light yellow-green
    "#9EDAE5",  # Light cyan
]

def get_agent_color(agent_id, is_ego=False):
    """Get consistent color for an agent based on ID."""
    if is_ego:
        return VEHICLE_COLORS[0]
    return VEHICLE_COLORS[(agent_id % (len(VEHICLE_COLORS) - 1)) + 1]


def get_agent_type_name(type_id):
    return AGENT_TYPE_NAMES.get(type_id, f"unknown ({type_id})")


def get_road_type_name(type_id):
    return ROAD_TYPE_NAMES.get(type_id, f"unknown ({type_id})")


def get_traffic_state_name(state_id):
    return TL_STATE_NAMES.get(state_id, f"unknown ({state_id})")


def get_traffic_state_color(state_id):
    return TL_STATE_COLORS.get(state_id, "#808080")


is_yellow_line = lambda t: t in (
    RoadLineType.BROKEN_SINGLE_YELLOW, RoadLineType.BROKEN_DOUBLE_YELLOW,
    RoadLineType.SOLID_SINGLE_YELLOW, RoadLineType.SOLID_DOUBLE_YELLOW,
    RoadLineType.PASSING_DOUBLE_YELLOW,
)

is_broken_line = lambda t: t in (RoadLineType.BROKEN_SINGLE_WHITE, RoadLineType.BROKEN_SINGLE_YELLOW)


def get_road_styling(road_type):
    """Get color and dash style for road element."""
    if is_road_lane(road_type):
        if road_type == LaneType.UNKNOWN:
            return ROAD_COLORS["lane_unknown"], None, 2.0
        return ROAD_COLORS["lane"], None, 1.5
    if is_road_line(road_type):
        if road_type == RoadLineType.UNKNOWN:
            return ROAD_COLORS["road_line_unknown"], "dash", 2.0
        color = ROAD_COLORS["road_line_yellow"] if is_yellow_line(road_type) else ROAD_COLORS["road_line_white"]
        dash = "dot" if is_broken_line(road_type) else "solid"
        return color, dash, 1.5
    if is_road_edge(road_type):
        if road_type == RoadEdgeType.UNKNOWN:
            return ROAD_COLORS["road_edge_unknown"], "solid", 3.0
        return ROAD_COLORS["road_edge"], "solid", 2.5
    if road_type == MiscRoadType.CROSSWALK:
        return ROAD_COLORS["crosswalk"], "solid", 3
    if road_type == MiscRoadType.SPEED_BUMP:
        return ROAD_COLORS["speed_bump"], "solid", 3
    if road_type == MiscRoadType.DRIVEWAY:
        return ROAD_COLORS["stop_sign"], None, 3
    return "#888888", None, 1


def get_vehicle_corners(x, y, heading, length, width):
    """Calculate corner points of vehicle rectangle."""
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    hl, hw = length / 2, width / 2
    corners = [
        (-hl, -hw),
        (hl, -hw),
        (hl, hw),
        (-hl, hw),
        (-hl, -hw),  # close polygon
    ]
    xs, ys = [], []
    for dx, dy in corners:
        rx = dx * cos_h - dy * sin_h + x
        ry = dx * sin_h + dy * cos_h + y
        xs.append(rx)
        ys.append(ry)
    return xs, ys


def get_heading_arrow(x, y, heading, length):
    """Get arrow start/end for heading indicator."""
    arrow_len = length * 0.6
    return x + arrow_len * np.cos(heading), y + arrow_len * np.sin(heading)


def rdp_simplify(pts: np.ndarray, epsilon: float = 0.5) -> np.ndarray:
    """Iterative Ramer-Douglas-Peucker polyline simplification."""
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        seg = pts[j, :2] - pts[i, :2]
        norm = np.linalg.norm(seg)
        if norm < 1e-10:
            dists = np.linalg.norm(pts[i : j + 1, :2] - pts[i, :2], axis=1)
        else:
            t = np.dot(pts[i : j + 1, :2] - pts[i, :2], seg) / (norm * norm)
            proj = pts[i, :2] + np.outer(np.clip(t, 0, 1), seg)
            dists = np.linalg.norm(pts[i : j + 1, :2] - proj, axis=1)
        k = int(np.argmax(dists)) + i
        if dists[k - i] > epsilon:
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return pts[keep]


def build_lane_map(road_elements):
    """Build dict mapping lane ID to xyz array."""
    lane_map = {}
    for elem in road_elements:
        if is_road_lane(elem.get("type", 0)):
            lane_map[elem["id"]] = elem.get("xyz", np.array([]))
    return lane_map


def compute_route_polyline(route_lane_ids, lane_map, start_pos=None, max_jump_m=30.0):
    """Build route polyline from lane IDs, dropping inter-lane jumps > max_jump_m."""
    segments = []
    for lane_id in route_lane_ids:
        if lane_id in lane_map:
            xyz = lane_map[lane_id]
            if len(xyz) > 0:
                segments.append(xyz)

    if not segments:
        return None

    # Chain segments, dropping connector if the gap is too large
    kept = [segments[0]]
    for seg in segments[1:]:
        gap = np.linalg.norm(kept[-1][-1, :2] - seg[0, :2])
        if gap <= max_jump_m:
            kept.append(seg)

    points = np.vstack(kept)
    if len(points) < 2:
        return None

    # Crop to start near agent's initial position
    if start_pos is not None:
        dists = np.linalg.norm(points[:, :2] - np.array(start_pos[:2]), axis=1)
        start_idx = np.argmin(dists)
        points = points[start_idx:]

    return points if len(points) >= 2 else None


def format_velocity(vx, vy):
    """Format velocity as magnitude and direction."""
    mag = np.sqrt(vx**2 + vy**2)
    return f"{mag:.2f} m/s ({vx:.2f}, {vy:.2f})"


def format_heading(heading):
    """Format heading in radians to degrees."""
    deg = np.degrees(heading) % 360
    return f"{deg:.1f}°"
