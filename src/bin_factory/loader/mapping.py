from py123d.datatypes import detections, map_objects

from bin_factory import puffer_types


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

LANE_TYPE_MAP = {
    map_objects.LaneType.UNDEFINED: int(puffer_types.LaneType.UNKNOWN),
    map_objects.LaneType.FREEWAY: int(puffer_types.LaneType.FREEWAY),
    map_objects.LaneType.SURFACE_STREET: int(puffer_types.LaneType.SURFACE_STREET),
    map_objects.LaneType.BIKE_LANE: int(puffer_types.LaneType.BIKE_LANE),
}

CROSSWALK_TYPE = int(puffer_types.MiscRoadType.CROSSWALK)

STOP_ZONE_TYPE_MAP = {
    map_objects.StopZoneType.TRAFFIC_LIGHT: int(puffer_types.TCType.TRAFFIC_LIGHT),
    map_objects.StopZoneType.STOP_SIGN: int(puffer_types.TCType.STOP_SIGN),
    map_objects.StopZoneType.YIELD_SIGN: int(puffer_types.TCType.YIELD_SIGN),
    map_objects.StopZoneType.PEDESTRIAN_CROSSING: None,  # Not supported as a stop zone type, only as a crosswalk type
    map_objects.StopZoneType.TURN_STOP: int(puffer_types.TCType.STOP_SIGN), # No specific turn stop type in puffer
}

ROAD_EDGE_TYPE_MAP = {
    map_objects.RoadEdgeType.UNKNOWN: int(puffer_types.RoadEdgeType.UNKNOWN),
    map_objects.RoadEdgeType.ROAD_EDGE_BOUNDARY: int(puffer_types.RoadEdgeType.BOUNDARY),
    map_objects.RoadEdgeType.ROAD_EDGE_MEDIAN: int(puffer_types.RoadEdgeType.MEDIAN),
}

ROAD_LINE_TYPE_MAP = {
    map_objects.RoadLineType.UNKNOWN: int(puffer_types.RoadLineType.UNKNOWN),
    map_objects.RoadLineType.DASHED_WHITE: int(puffer_types.RoadLineType.BROKEN_SINGLE_WHITE),
    map_objects.RoadLineType.SOLID_WHITE: int(puffer_types.RoadLineType.SOLID_SINGLE_WHITE),
    map_objects.RoadLineType.DOUBLE_SOLID_WHITE: int(puffer_types.RoadLineType.SOLID_DOUBLE_WHITE),
    map_objects.RoadLineType.DASHED_YELLOW: int(puffer_types.RoadLineType.BROKEN_SINGLE_YELLOW),
    map_objects.RoadLineType.DOUBLE_DASH_YELLOW: int(puffer_types.RoadLineType.BROKEN_DOUBLE_YELLOW),
    map_objects.RoadLineType.SOLID_YELLOW: int(puffer_types.RoadLineType.SOLID_SINGLE_YELLOW),
    map_objects.RoadLineType.DOUBLE_SOLID_YELLOW: int(puffer_types.RoadLineType.SOLID_DOUBLE_YELLOW),
    map_objects.RoadLineType.DASH_SOLID_YELLOW: int(puffer_types.RoadLineType.PASSING_DOUBLE_YELLOW),
    map_objects.RoadLineType.SOLID_DASH_YELLOW: int(puffer_types.RoadLineType.PASSING_DOUBLE_YELLOW),
    map_objects.RoadLineType.DOUBLE_DASH_WHITE: int(puffer_types.RoadLineType.BROKEN_SINGLE_WHITE),
    map_objects.RoadLineType.DASH_SOLID_WHITE: int(puffer_types.RoadLineType.SOLID_SINGLE_WHITE),
    map_objects.RoadLineType.SOLID_DASH_WHITE: int(puffer_types.RoadLineType.SOLID_SINGLE_WHITE),
    map_objects.RoadLineType.SOLID_BLUE: int(puffer_types.RoadLineType.SOLID_SINGLE_WHITE),
}

MAP_TYPE_MAP = {
    map_objects.MapLayer.LANE: LANE_TYPE_MAP,
    map_objects.MapLayer.LANE_GROUP: None,  # Not Supported
    map_objects.MapLayer.INTERSECTION: None,  # Not Supported
    map_objects.MapLayer.CROSSWALK: {None: CROSSWALK_TYPE},
    map_objects.MapLayer.WALKWAY: None,  # Not Supported
    map_objects.MapLayer.CARPARK: None,  # Not Supported
    map_objects.MapLayer.GENERIC_DRIVABLE: None,  # Not Supported
    map_objects.MapLayer.STOP_ZONE: STOP_ZONE_TYPE_MAP,
    map_objects.MapLayer.ROAD_EDGE: ROAD_EDGE_TYPE_MAP,
    map_objects.MapLayer.ROAD_LINE: ROAD_LINE_TYPE_MAP,
}


TL_STATE_MAP = {
    detections.TrafficLightStatus.GREEN: puffer_types.TLState.GREEN,
    detections.TrafficLightStatus.YELLOW: puffer_types.TLState.YELLOW,
    detections.TrafficLightStatus.RED: puffer_types.TLState.RED,
    detections.TrafficLightStatus.OFF: puffer_types.TLState.OFF,
    detections.TrafficLightStatus.UNKNOWN: puffer_types.TLState.UNKNOWN,
}
