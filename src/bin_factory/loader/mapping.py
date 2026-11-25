from py123d.datatypes import detections, map_objects

from bin_factory import puffer_types


# Layers with a polyline or surface type in the schema, and stop zones, which become traffic controls
SUPPORTED_MAP_LAYERS = frozenset(
    map_objects.MapLayer(int(layer))
    for layer in puffer_types.POLYLINE_LAYERS | puffer_types.SURFACE_LAYERS | {puffer_types.MapLayer.STOP_ZONE}
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

TL_STATE_MAP = {status: puffer_types.TLState[status.name] for status in detections.TrafficLightStatus}
