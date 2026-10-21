import enum


class AgentType(enum.IntEnum):
    OTHER = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3


# Map enums mirror py123d.datatypes.map_objects names and values. py123d's enums do not compare equal to ints.
class MapLayer(enum.IntEnum):
    LANE = 0
    LANE_GROUP = 1
    INTERSECTION = 2
    CROSSWALK = 3
    WALKWAY = 4
    CARPARK = 5
    GENERIC_DRIVABLE = 6
    STOP_ZONE = 7
    ROAD_EDGE = 8
    ROAD_LINE = 9
    SPEED_BUMP = 10
    TRAFFIC_LIGHT = 11
    SHOULDER = 12
    STATIC_OBJECT = 13


class LaneType(enum.IntEnum):
    UNDEFINED = 0
    FREEWAY = 1
    SURFACE_STREET = 2
    BIKE_LANE = 3
    BUS_LANE = 4
    SHOULDER = 5


class RoadLineType(enum.IntEnum):
    NONE = 0
    UNKNOWN = 1
    DASH_SOLID_YELLOW = 2
    DASH_SOLID_WHITE = 3
    DASHED_WHITE = 4
    DASHED_YELLOW = 5
    DOUBLE_SOLID_YELLOW = 6
    DOUBLE_SOLID_WHITE = 7
    DOUBLE_DASH_YELLOW = 8
    DOUBLE_DASH_WHITE = 9
    SOLID_YELLOW = 10
    SOLID_WHITE = 11
    SOLID_DASH_WHITE = 12
    SOLID_DASH_YELLOW = 13
    SOLID_BLUE = 14


class RoadEdgeType(enum.IntEnum):
    UNKNOWN = 0
    ROAD_EDGE_BOUNDARY = 1
    ROAD_EDGE_MEDIAN = 2


POLYLINE_LAYERS = frozenset({MapLayer.LANE, MapLayer.ROAD_LINE, MapLayer.ROAD_EDGE})
SURFACE_LAYERS = frozenset(
    {
        MapLayer.LANE_GROUP,
        MapLayer.INTERSECTION,
        MapLayer.CROSSWALK,
        MapLayer.WALKWAY,
        MapLayer.CARPARK,
        MapLayer.GENERIC_DRIVABLE,
        MapLayer.SPEED_BUMP,
        MapLayer.SHOULDER,
    }
)
POLYLINE_TYPES = {MapLayer.LANE: LaneType, MapLayer.ROAD_LINE: RoadLineType, MapLayer.ROAD_EDGE: RoadEdgeType}


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
MAP_LAYER_NAMES = {t.value: t.name.lower() for t in MapLayer}
POLYLINE_TYPE_NAMES = {layer.value: {t.value: t.name.lower() for t in types} for layer, types in POLYLINE_TYPES.items()}
TL_STATE_NAMES = {t.value: t.name.lower() for t in TLState}
TC_TYPE_NAMES = {t.value: t.name.lower() for t in TCType}
OBJECT_TYPE_NAMES = {t.value: t.name.lower() for t in ObjectType}
