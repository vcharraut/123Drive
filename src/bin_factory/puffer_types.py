import enum


class AgentType(enum.IntEnum):
    OTHER = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3


class ControlState(enum.IntEnum):
    CONTROLLABLE = 0
    NON_CONTROLLABLE_MOVING = 1
    NON_CONTROLLABLE_STATIC = 2


class LaneType(enum.IntEnum):
    UNKNOWN = 0
    FREEWAY = 1
    SURFACE_STREET = 2
    BIKE_LANE = 3
    BUS_LANE = 4


class RoadLineType(enum.IntEnum):
    UNKNOWN = 10
    BROKEN_SINGLE_WHITE = 11
    SOLID_SINGLE_WHITE = 12
    SOLID_DOUBLE_WHITE = 13
    BROKEN_SINGLE_YELLOW = 14
    BROKEN_DOUBLE_YELLOW = 15
    SOLID_SINGLE_YELLOW = 16
    SOLID_DOUBLE_YELLOW = 17
    PASSING_DOUBLE_YELLOW = 18


class RoadEdgeType(enum.IntEnum):
    UNKNOWN = 20
    BOUNDARY = 21
    MEDIAN = 22


class MiscRoadType(enum.IntEnum):
    UNKNOWN = 30
    CROSSWALK = 31
    SPEED_BUMP = 32


class TLState(enum.IntEnum):
    UNKNOWN = 0
    RED = 1
    YELLOW = 2
    GREEN = 3
    OFF = 4


class TCType(enum.IntEnum):
    TRAFFIC_LIGHT = 1
    STOP_SIGN = 2
    YIELD_SIGN = 3


class ObjectType(enum.IntEnum):
    TRAFFIC_SIGN = 1
    TRAFFIC_CONE = 2
    TRAFFIC_LIGHT = 3
    BARRIER = 4
    GENERIC_OBJECT = 5


# Derived name dicts (int -> str)
AGENT_TYPE_NAMES = {t.value: t.name.lower() for t in AgentType}
CONTROL_STATE_NAMES = {t.value: t.name.lower() for t in ControlState}
ROAD_TYPE_NAMES = {
    t.value: t.name.lower() for enum_cls in [LaneType, RoadLineType, RoadEdgeType, MiscRoadType] for t in enum_cls
}
TL_STATE_NAMES = {t.value: t.name.lower() for t in TLState}
TC_TYPE_NAMES = {t.value: t.name.lower() for t in TCType}
OBJECT_TYPE_NAMES = {t.value: t.name.lower() for t in ObjectType}


# Road type ranges: 0-9 = lane, 10-19 = line marking, 20-29 = edge, 30+ = misc
def is_road_lane(t):
    return 0 <= t <= 9


def is_road_line(t):
    return 10 <= t <= 19


def is_road_edge(t):
    return 20 <= t <= 29


def is_yellow_line(t):
    return t in (
        RoadLineType.BROKEN_SINGLE_YELLOW,
        RoadLineType.BROKEN_DOUBLE_YELLOW,
        RoadLineType.SOLID_SINGLE_YELLOW,
        RoadLineType.SOLID_DOUBLE_YELLOW,
        RoadLineType.PASSING_DOUBLE_YELLOW,
    )


def is_crosswalk(t):
    return t == MiscRoadType.CROSSWALK


def is_broken_line(t):
    return t in (RoadLineType.BROKEN_SINGLE_WHITE, RoadLineType.BROKEN_SINGLE_YELLOW)
