from py123d.datatypes.detections import DefaultBoxDetectionLabel, TrafficLightStatus
from py123d.datatypes.map_objects import LaneType, MapLayer, RoadEdgeType, RoadLineType, StopZoneType

from bin_factory.convert import puffer_types


AGENT_TYPE_MAP = {
    DefaultBoxDetectionLabel.EGO: puffer_types.AgentType.VEHICLE,
    DefaultBoxDetectionLabel.VEHICLE: puffer_types.AgentType.VEHICLE,
    DefaultBoxDetectionLabel.PERSON: puffer_types.AgentType.PEDESTRIAN,
    DefaultBoxDetectionLabel.BICYCLE: puffer_types.AgentType.CYCLIST,
}

OBJECT_TYPE_MAP = {
    DefaultBoxDetectionLabel.TRAFFIC_SIGN: puffer_types.ObjectType.TRAFFIC_SIGN,
    DefaultBoxDetectionLabel.TRAFFIC_CONE: puffer_types.ObjectType.TRAFFIC_CONE,
    DefaultBoxDetectionLabel.TRAFFIC_LIGHT: puffer_types.ObjectType.TRAFFIC_LIGHT,
    DefaultBoxDetectionLabel.BARRIER: puffer_types.ObjectType.BARRIER,
    DefaultBoxDetectionLabel.GENERIC_OBJECT: puffer_types.ObjectType.GENERIC_OBJECT,
}

ROAD_TYPE_MAP = {
    MapLayer.LANE: {
        LaneType.UNDEFINED: puffer_types.LaneType.UNKNOWN,
        LaneType.FREEWAY: puffer_types.LaneType.FREEWAY,
        LaneType.SURFACE_STREET: puffer_types.LaneType.SURFACE_STREET,
        LaneType.BIKE_LANE: puffer_types.LaneType.BIKE_LANE,
    },
    MapLayer.ROAD_LINE: {
        RoadLineType.UNKNOWN: puffer_types.RoadLineType.UNKNOWN,
        RoadLineType.DASHED_WHITE: puffer_types.RoadLineType.BROKEN_SINGLE_WHITE,
        RoadLineType.SOLID_WHITE: puffer_types.RoadLineType.SOLID_SINGLE_WHITE,
        RoadLineType.DOUBLE_SOLID_WHITE: puffer_types.RoadLineType.SOLID_DOUBLE_WHITE,
        RoadLineType.DASHED_YELLOW: puffer_types.RoadLineType.BROKEN_SINGLE_YELLOW,
        RoadLineType.DOUBLE_DASH_YELLOW: puffer_types.RoadLineType.BROKEN_DOUBLE_YELLOW,
        RoadLineType.SOLID_YELLOW: puffer_types.RoadLineType.SOLID_SINGLE_YELLOW,
        RoadLineType.DOUBLE_SOLID_YELLOW: puffer_types.RoadLineType.SOLID_DOUBLE_YELLOW,
        RoadLineType.DASH_SOLID_YELLOW: puffer_types.RoadLineType.PASSING_DOUBLE_YELLOW,
        RoadLineType.SOLID_DASH_YELLOW: puffer_types.RoadLineType.PASSING_DOUBLE_YELLOW,
        RoadLineType.DOUBLE_DASH_WHITE: puffer_types.RoadLineType.BROKEN_SINGLE_WHITE,
        RoadLineType.DASH_SOLID_WHITE: puffer_types.RoadLineType.SOLID_SINGLE_WHITE,
        RoadLineType.SOLID_DASH_WHITE: puffer_types.RoadLineType.SOLID_SINGLE_WHITE,
        RoadLineType.SOLID_BLUE: puffer_types.RoadLineType.SOLID_SINGLE_WHITE,
    },
    MapLayer.ROAD_EDGE: {
        RoadEdgeType.UNKNOWN: puffer_types.RoadEdgeType.UNKNOWN,
        RoadEdgeType.ROAD_EDGE_BOUNDARY: puffer_types.RoadEdgeType.BOUNDARY,
        RoadEdgeType.ROAD_EDGE_MEDIAN: puffer_types.RoadEdgeType.MEDIAN,
    },
    MapLayer.CROSSWALK: {None: puffer_types.MiscRoadType.CROSSWALK},
}

STOP_ZONE_TYPE_MAP = {
    StopZoneType.TRAFFIC_LIGHT: puffer_types.TCType.TRAFFIC_LIGHT,
    StopZoneType.STOP_SIGN: puffer_types.TCType.STOP_SIGN,
    StopZoneType.YIELD_SIGN: puffer_types.TCType.YIELD_SIGN,
}

TL_STATE_MAP = {
    TrafficLightStatus.GREEN: puffer_types.TLState.GREEN,
    TrafficLightStatus.YELLOW: puffer_types.TLState.YELLOW,
    TrafficLightStatus.RED: puffer_types.TLState.RED,
    TrafficLightStatus.OFF: puffer_types.TLState.OFF,
    TrafficLightStatus.UNKNOWN: puffer_types.TLState.UNKNOWN,
}
