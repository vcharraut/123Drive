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


class ObjectType(IntEnum):
    TRAFFIC_SIGN = 1
    TRAFFIC_CONE = 2
    TRAFFIC_LIGHT = 3
    BARRIER = 4
    GENERIC_OBJECT = 5


# Derived name dicts (int -> str)
AGENT_TYPE_NAMES = {0: "unset", **{t.value: t.name.lower() for t in AgentType}}
ROAD_TYPE_NAMES = {
    t.value: t.name.lower() for enum_cls in [LaneType, RoadLineType, RoadEdgeType, MiscRoadType] for t in enum_cls
}
TL_STATE_NAMES = {t.value: t.name.lower() for t in TLState}
TC_TYPE_NAMES = {t.value: t.name.lower() for t in TCType}
OBJECT_TYPE_NAMES = {t.value: t.name.lower() for t in ObjectType}

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
