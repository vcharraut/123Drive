from py123d.datatypes import detections, map_objects

from bin_factory import types as puffer_types


AGENT_TYPE_MAP = {
    detections.DefaultBoxDetectionLabel.EGO: puffer_types.AgentType.VEHICLE,
    detections.DefaultBoxDetectionLabel.VEHICLE: puffer_types.AgentType.VEHICLE,
    detections.DefaultBoxDetectionLabel.PERSON: puffer_types.AgentType.PEDESTRIAN,
    detections.DefaultBoxDetectionLabel.BICYCLE: puffer_types.AgentType.CYCLIST,
}

OBJECT_TYPE_MAP = {
    detections.DefaultBoxDetectionLabel.TRAFFIC_SIGN: puffer_types.ObjectType.TRAFFIC_SIGN,
    detections.DefaultBoxDetectionLabel.TRAFFIC_CONE: puffer_types.ObjectType.TRAFFIC_CONE,
    detections.DefaultBoxDetectionLabel.TRAFFIC_LIGHT: puffer_types.ObjectType.TRAFFIC_LIGHT,
    detections.DefaultBoxDetectionLabel.BARRIER: puffer_types.ObjectType.BARRIER,
    detections.DefaultBoxDetectionLabel.GENERIC_OBJECT: puffer_types.ObjectType.GENERIC_OBJECT,
}

ROAD_TYPE_MAP = {
    map_objects.MapLayer.LANE: {
        map_objects.LaneType.UNDEFINED: puffer_types.LaneType.UNKNOWN,
        map_objects.LaneType.FREEWAY: puffer_types.LaneType.FREEWAY,
        map_objects.LaneType.SURFACE_STREET: puffer_types.LaneType.SURFACE_STREET,
        map_objects.LaneType.BIKE_LANE: puffer_types.LaneType.BIKE_LANE,
    },
    map_objects.MapLayer.ROAD_LINE: {
        map_objects.RoadLineType.UNKNOWN: puffer_types.RoadLineType.UNKNOWN,
        map_objects.RoadLineType.DASHED_WHITE: puffer_types.RoadLineType.BROKEN_SINGLE_WHITE,
        map_objects.RoadLineType.SOLID_WHITE: puffer_types.RoadLineType.SOLID_SINGLE_WHITE,
        map_objects.RoadLineType.DOUBLE_SOLID_WHITE: puffer_types.RoadLineType.SOLID_DOUBLE_WHITE,
        map_objects.RoadLineType.DASHED_YELLOW: puffer_types.RoadLineType.BROKEN_SINGLE_YELLOW,
        map_objects.RoadLineType.DOUBLE_DASH_YELLOW: puffer_types.RoadLineType.BROKEN_DOUBLE_YELLOW,
        map_objects.RoadLineType.SOLID_YELLOW: puffer_types.RoadLineType.SOLID_SINGLE_YELLOW,
        map_objects.RoadLineType.DOUBLE_SOLID_YELLOW: puffer_types.RoadLineType.SOLID_DOUBLE_YELLOW,
        map_objects.RoadLineType.DASH_SOLID_YELLOW: puffer_types.RoadLineType.PASSING_DOUBLE_YELLOW,
        map_objects.RoadLineType.SOLID_DASH_YELLOW: puffer_types.RoadLineType.PASSING_DOUBLE_YELLOW,
        map_objects.RoadLineType.DOUBLE_DASH_WHITE: puffer_types.RoadLineType.BROKEN_SINGLE_WHITE,
        map_objects.RoadLineType.DASH_SOLID_WHITE: puffer_types.RoadLineType.SOLID_SINGLE_WHITE,
        map_objects.RoadLineType.SOLID_DASH_WHITE: puffer_types.RoadLineType.SOLID_SINGLE_WHITE,
        map_objects.RoadLineType.SOLID_BLUE: puffer_types.RoadLineType.SOLID_SINGLE_WHITE,
    },
    map_objects.MapLayer.ROAD_EDGE: {
        map_objects.RoadEdgeType.UNKNOWN: puffer_types.RoadEdgeType.UNKNOWN,
        map_objects.RoadEdgeType.ROAD_EDGE_BOUNDARY: puffer_types.RoadEdgeType.BOUNDARY,
        map_objects.RoadEdgeType.ROAD_EDGE_MEDIAN: puffer_types.RoadEdgeType.MEDIAN,
    },
    map_objects.MapLayer.CROSSWALK: {None: puffer_types.MiscRoadType.CROSSWALK},
}

STOP_ZONE_TYPE_MAP = {
    map_objects.StopZoneType.TRAFFIC_LIGHT: puffer_types.TCType.TRAFFIC_LIGHT,
    map_objects.StopZoneType.STOP_SIGN: puffer_types.TCType.STOP_SIGN,
    map_objects.StopZoneType.YIELD_SIGN: puffer_types.TCType.YIELD_SIGN,
}

TL_STATE_MAP = {
    detections.TrafficLightStatus.GREEN: puffer_types.TLState.GREEN,
    detections.TrafficLightStatus.YELLOW: puffer_types.TLState.YELLOW,
    detections.TrafficLightStatus.RED: puffer_types.TLState.RED,
    detections.TrafficLightStatus.OFF: puffer_types.TLState.OFF,
    detections.TrafficLightStatus.UNKNOWN: puffer_types.TLState.UNKNOWN,
}
