from py123d.datatypes import detections, map_objects

from bin_factory import puffer_types


SUPPORTED_MAP_LAYERS = frozenset(
    {
        map_objects.MapLayer.LANE,
        map_objects.MapLayer.ROAD_LINE,
        map_objects.MapLayer.ROAD_EDGE,
        map_objects.MapLayer.LANE_GROUP,
        map_objects.MapLayer.INTERSECTION,
        map_objects.MapLayer.CROSSWALK,
        map_objects.MapLayer.WALKWAY,
        map_objects.MapLayer.CARPARK,
        map_objects.MapLayer.GENERIC_DRIVABLE,
        map_objects.MapLayer.STOP_ZONE,
        map_objects.MapLayer.SPEED_BUMP,
        map_objects.MapLayer.SHOULDER,
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
}

OBJECT_TYPE_MAP = {
    getattr(detections.DefaultBoxDetectionLabel, member.name): member for member in puffer_types.ObjectType
}

STOP_ZONE_TYPE_MAP = {
    map_objects.StopZoneType.TRAFFIC_LIGHT: int(puffer_types.TCType.TRAFFIC_LIGHT),
    map_objects.StopZoneType.STOP_SIGN: int(puffer_types.TCType.STOP_SIGN),
    map_objects.StopZoneType.YIELD_SIGN: int(puffer_types.TCType.YIELD_SIGN),
    map_objects.StopZoneType.PEDESTRIAN_CROSSING: None,  # Not supported as a stop zone type, only as a crosswalk type
    map_objects.StopZoneType.TURN_STOP: int(puffer_types.TCType.STOP_SIGN),  # No specific turn stop type in puffer
}

TL_STATE_MAP = {
    detections.TrafficLightStatus.GREEN: puffer_types.TLState.GREEN,
    detections.TrafficLightStatus.YELLOW: puffer_types.TLState.YELLOW,
    detections.TrafficLightStatus.RED: puffer_types.TLState.RED,
    detections.TrafficLightStatus.OFF: puffer_types.TLState.OFF,
    detections.TrafficLightStatus.UNKNOWN: puffer_types.TLState.UNKNOWN,
}
