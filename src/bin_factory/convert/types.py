from enum import IntEnum


class AgentType(IntEnum):
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    OTHER = 4


class LaneType(IntEnum):
    UNKNOWN = 0
    FREEWAY = 1
    SURFACE_STREET = 2
    BIKE_LANE = 3
    BUS_LANE = 4


class RoadLineType(IntEnum):
    UNKNOWN = 10
    BROKEN_SINGLE_WHITE = 11
    SOLID_SINGLE_WHITE = 12
    SOLID_DOUBLE_WHITE = 13
    BROKEN_SINGLE_YELLOW = 14
    BROKEN_DOUBLE_YELLOW = 15
    SOLID_SINGLE_YELLOW = 16
    SOLID_DOUBLE_YELLOW = 17
    PASSING_DOUBLE_YELLOW = 18


class RoadEdgeType(IntEnum):
    UNKNOWN = 20
    BOUNDARY = 21
    MEDIAN = 22


class MiscRoadType(IntEnum):
    CROSSWALK = 31
    SPEED_BUMP = 32
    DRIVEWAY = 33


class TLState(IntEnum):
    GREEN = 0
    YELLOW = 1
    RED = 2
    OFF = 3
    UNKNOWN = 4


class TCType(IntEnum):
    TRAFFIC_LIGHT = 1
    STOP_SIGN = 2
    YIELD_SIGN = 3


# Derived name dicts (int -> str)
AGENT_TYPE_NAMES = {0: "unset", **{t.value: t.name.lower() for t in AgentType}}
ROAD_TYPE_NAMES = {
    t.value: t.name.lower() for enum_cls in [LaneType, RoadLineType, RoadEdgeType, MiscRoadType] for t in enum_cls
}
TL_STATE_NAMES = {t.value: t.name.lower() for t in TLState}
TC_TYPE_NAMES = {t.value: t.name.lower() for t in TCType}

# Range checks
LANE_RANGE = (0, 9)
ROAD_LINE_RANGE = (10, 19)
ROAD_EDGE_RANGE = (20, 29)


def is_road_lane(t):
    return 0 <= t <= 9


def is_road_line(t):
    return 10 <= t <= 19


def is_road_edge(t):
    return 20 <= t <= 29


# Viz colors (single source)
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
    "stop_sign": "#FF0000",
}


def as_json_dict():
    return {
        "AGENT_TYPE_NAMES": AGENT_TYPE_NAMES,
        "ROAD_TYPE_NAMES": ROAD_TYPE_NAMES,
        "TL_STATE_NAMES": TL_STATE_NAMES,
        "TL_STATE_COLORS": dict(TL_STATE_COLORS.items()),
        "TC_TYPE_NAMES": TC_TYPE_NAMES,
        "LANE_RANGE": LANE_RANGE,
        "ROAD_LINE_RANGE": ROAD_LINE_RANGE,
        "ROAD_EDGE_RANGE": ROAD_EDGE_RANGE,
        "ROAD_COLORS": ROAD_COLORS,
    }
