"""Helpers for Puffer web visualization."""

import numpy as np


# Agent type mapping
AGENT_TYPE_NAMES = {
    0: "unset",
    1: "vehicle",
    2: "pedestrian",
    3: "cyclist",
    4: "other",
}

# Road map element type mapping (from src/encoder/roadgraph.py)
ROAD_ELEMENT_TYPES = {
    0: "lane_unknown",
    # Lanes (1-9)
    1: "lane_freeway",
    2: "lane_surface_street",
    3: "lane_bike",
    # Road lines (10-19)
    10: "road_line_unknown",
    11: "road_line_broken_white",
    12: "road_line_solid_white",
    13: "road_line_double_white",
    14: "road_line_broken_yellow",
    15: "road_line_double_yellow",
    16: "road_line_solid_yellow",
    17: "road_line_solid_double_yellow",
    18: "road_line_passing_yellow",
    # Road edges (20-29)
    20: "road_edge_unknown",
    21: "road_edge_boundary",
    22: "road_edge_median",
    23: "road_edge_sidewalk",
    # Other features (31+)
    31: "crosswalk",
    32: "speed_bump",
    33: "stop_sign",
}

# Traffic light state mapping
TRAFFIC_LIGHT_STATES = {
    0: "unknown",
    1: "arrow_stop",
    2: "arrow_caution",
    3: "arrow_go",
    4: "stop",
    5: "caution",
    6: "go",
    7: "flashing_stop",
    8: "flashing_caution",
}

# Traffic light colors
TRAFFIC_LIGHT_COLORS = {
    0: "#808080",  # Unknown - gray
    1: "#FF0000",  # Arrow red
    2: "#FFFF00",  # Arrow yellow
    3: "#00FF00",  # Arrow green
    4: "#FF0000",  # Red
    5: "#FFFF00",  # Yellow
    6: "#00FF00",  # Green
    7: "#FF6600",  # Flashing red
    8: "#FFFF00",  # Flashing yellow
}

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

# Road element styling
ROAD_COLORS = {
    "lane": "#E0E0E0",
    "lane_unknown": "#00BFFF",
    "road_line_white": "#AAAAAA",
    "road_line_yellow": "#D4AA00",
    "road_line_unknown": "#FF00FF",
    "road_edge": "#333333",
    "road_edge_unknown": "#00FFFF",
    "crosswalk": "#FFD700",
    "stop_sign": "#FF0000",
    "speed_bump": "#FF69B4",
}


def get_agent_color(agent_id, is_ego=False):
    """Get consistent color for an agent based on ID."""
    if is_ego:
        return VEHICLE_COLORS[0]
    return VEHICLE_COLORS[(agent_id % (len(VEHICLE_COLORS) - 1)) + 1]


def get_agent_type_name(type_id):
    return AGENT_TYPE_NAMES.get(type_id, f"unknown ({type_id})")


def get_road_type_name(type_id):
    return ROAD_ELEMENT_TYPES.get(type_id, f"unknown ({type_id})")


def get_traffic_state_name(state_id):
    return TRAFFIC_LIGHT_STATES.get(state_id, f"unknown ({state_id})")


def get_traffic_state_color(state_id):
    return TRAFFIC_LIGHT_COLORS.get(state_id, "#808080")


def is_lane(road_type):
    return 0 <= road_type <= 9


def is_road_line(road_type):
    return 10 <= road_type <= 19


def is_road_edge(road_type):
    return 20 <= road_type <= 29


def is_unknown_lane(road_type):
    return road_type == 0


def is_unknown_road_line(road_type):
    return road_type == 10


def is_unknown_road_edge(road_type):
    return road_type == 20


def is_yellow_line(road_type):
    return road_type in [14, 15, 16, 17, 18]


def is_broken_line(road_type):
    return road_type in [11, 14]


def get_road_styling(road_type):
    """Get color and dash style for road element."""
    if is_lane(road_type):
        if is_unknown_lane(road_type):
            return ROAD_COLORS["lane_unknown"], None, 2.0
        return ROAD_COLORS["lane"], None, 1.5
    elif is_road_line(road_type):
        if is_unknown_road_line(road_type):
            return ROAD_COLORS["road_line_unknown"], "dash", 2.0
        color = ROAD_COLORS["road_line_yellow"] if is_yellow_line(road_type) else ROAD_COLORS["road_line_white"]
        dash = "dot" if is_broken_line(road_type) else "solid"
        return color, dash, 1.5
    elif is_road_edge(road_type):
        if is_unknown_road_edge(road_type):
            return ROAD_COLORS["road_edge_unknown"], "solid", 3.0
        return ROAD_COLORS["road_edge"], "solid", 2.5
    elif road_type == 31:
        return ROAD_COLORS["crosswalk"], "solid", 3
    elif road_type == 32:
        return ROAD_COLORS["speed_bump"], "solid", 3
    elif road_type == 33:
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
            dists = np.linalg.norm(pts[i:j+1, :2] - pts[i, :2], axis=1)
        else:
            t = np.dot(pts[i:j+1, :2] - pts[i, :2], seg) / (norm * norm)
            proj = pts[i, :2] + np.outer(np.clip(t, 0, 1), seg)
            dists = np.linalg.norm(pts[i:j+1, :2] - proj, axis=1)
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
        if is_lane(elem.get("type", 0)):
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
