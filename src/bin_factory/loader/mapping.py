from py123d.datatypes import detections, map_objects

from bin_factory import puffer_types


SUPPORTED_MAP_LAYERS = frozenset(
    {
        map_objects.MapLayer.LANE,
        map_objects.MapLayer.ROAD_LINE,
        map_objects.MapLayer.ROAD_EDGE,
        map_objects.MapLayer.CROSSWALK,
        map_objects.MapLayer.STOP_ZONE,
    }
)


AGENT_TYPE_MAP = {
    detections.DefaultBoxDetectionLabel.EGO: puffer_types.AgentType.VEHICLE,
    detections.DefaultBoxDetectionLabel.VEHICLE: puffer_types.AgentType.VEHICLE,
    detections.DefaultBoxDetectionLabel.TRAIN: puffer_types.AgentType.OTHER,
    detections.DefaultBoxDetectionLabel.PERSON: puffer_types.AgentType.PEDESTRIAN,
    detections.DefaultBoxDetectionLabel.TWO_WHEELER: puffer_types.AgentType.CYCLIST,
    detections.DefaultBoxDetectionLabel.ANIMAL: puffer_types.AgentType.OTHER,
    detections.DefaultBoxDetectionLabel.OTHER: puffer_types.AgentType.OTHER,
    detections.DefaultBoxDetectionLabel.GENERIC_OBJECT: puffer_types.AgentType.OTHER,
}

OBJECT_TYPE_MAP = {
    detections.DefaultBoxDetectionLabel.TRAFFIC_SIGN: puffer_types.ObjectType.TRAFFIC_SIGN,
    detections.DefaultBoxDetectionLabel.TRAFFIC_CONE: puffer_types.ObjectType.TRAFFIC_CONE,
    detections.DefaultBoxDetectionLabel.TRAFFIC_LIGHT: puffer_types.ObjectType.TRAFFIC_LIGHT,
    detections.DefaultBoxDetectionLabel.BARRIER: puffer_types.ObjectType.BARRIER,
}

LANE_TYPE_MAP = {
    map_objects.LaneType.UNDEFINED: int(puffer_types.LaneType.UNKNOWN),
    map_objects.LaneType.FREEWAY: int(puffer_types.LaneType.FREEWAY),
    map_objects.LaneType.SURFACE_STREET: int(puffer_types.LaneType.SURFACE_STREET),
    map_objects.LaneType.BIKE_LANE: int(puffer_types.LaneType.BIKE_LANE),
    map_objects.LaneType.BUS_LANE: int(puffer_types.LaneType.BUS_LANE),
}

CROSSWALK_TYPE = int(puffer_types.MiscRoadType.CROSSWALK)

STOP_ZONE_TYPE_MAP = {
    map_objects.StopZoneType.TRAFFIC_LIGHT: int(puffer_types.TCType.TRAFFIC_LIGHT),
    map_objects.StopZoneType.STOP_SIGN: int(puffer_types.TCType.STOP_SIGN),
    map_objects.StopZoneType.YIELD_SIGN: int(puffer_types.TCType.YIELD_SIGN),
    map_objects.StopZoneType.PEDESTRIAN_CROSSING: None,  # Not supported as a stop zone type, only as a crosswalk type
    map_objects.StopZoneType.TURN_STOP: int(puffer_types.TCType.STOP_SIGN),  # No specific turn stop type in puffer
}

ROAD_EDGE_TYPE_MAP = {
    map_objects.RoadEdgeType.UNKNOWN: int(puffer_types.RoadEdgeType.UNKNOWN),
    map_objects.RoadEdgeType.ROAD_EDGE_BOUNDARY: int(puffer_types.RoadEdgeType.BOUNDARY),
    map_objects.RoadEdgeType.ROAD_EDGE_MEDIAN: int(puffer_types.RoadEdgeType.MEDIAN),
}

ROAD_LINE_TYPE_MAP = {
    map_objects.RoadLineType.NONE: int(puffer_types.RoadLineType.UNKNOWN),
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

TL_STATE_MAP = {
    detections.TrafficLightStatus.GREEN: puffer_types.TLState.GREEN,
    detections.TrafficLightStatus.YELLOW: puffer_types.TLState.YELLOW,
    detections.TrafficLightStatus.RED: puffer_types.TLState.RED,
    detections.TrafficLightStatus.OFF: puffer_types.TLState.OFF,
    detections.TrafficLightStatus.UNKNOWN: puffer_types.TLState.UNKNOWN,
}
