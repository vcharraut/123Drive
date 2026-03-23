"""Helpers for Puffer web visualization."""

from bin_factory.puffer_types import (
    AGENT_TYPE_NAMES,
    LANE_RANGE,
    OBJECT_TYPE_NAMES,
    ROAD_EDGE_RANGE,
    ROAD_LINE_RANGE,
    ROAD_TYPE_NAMES,
    TC_TYPE_NAMES,
    TL_STATE_NAMES,
    TLState,
)


TL_STATE_COLORS = {
    TLState.GREEN: "#00FF00",
    TLState.YELLOW: "#FFFF00",
    TLState.RED: "#FF0000",
    TLState.OFF: "#808080",
    TLState.UNKNOWN: "#808080",
}

ROAD_COLORS = {
    "lane": "#E0E0E0",
    "lane_unknown": "#00BFFF",
    "road_line_white": "#AAAAAA",
    "road_line_yellow": "#D4AA00",
    "road_line_unknown": "#FF00FF",
    "road_edge": "#333333",
    "road_edge_unknown": "#00FFFF",
    "crosswalk": "#FFD700",
    "speed_bump": "#FF69B4",
}


def as_json_dict():
    return {
        "AGENT_TYPE_NAMES": AGENT_TYPE_NAMES,
        "ROAD_TYPE_NAMES": ROAD_TYPE_NAMES,
        "TL_STATE_NAMES": TL_STATE_NAMES,
        "TL_STATE_COLORS": dict(TL_STATE_COLORS.items()),
        "TC_TYPE_NAMES": TC_TYPE_NAMES,
        "OBJECT_TYPE_NAMES": OBJECT_TYPE_NAMES,
        "LANE_RANGE": LANE_RANGE,
        "ROAD_LINE_RANGE": ROAD_LINE_RANGE,
        "ROAD_EDGE_RANGE": ROAD_EDGE_RANGE,
        "ROAD_COLORS": ROAD_COLORS,
    }
