import dataclasses

from bin_factory import schema
from bin_factory.schema import remap_element_refs


def prune_invalid_map_elements(scenario: schema.PufferScenario, extras: schema.ExtractionExtras | None = None) -> None:
    valid_ids = {
        element_id
        for element_id, element in scenario.polylines.items()
        if element.polyline is not None and len(element.polyline) >= 2
    }
    if len(valid_ids) != len(scenario.polylines):
        identity_map = {element_id: element_id for element_id in valid_ids}
        scenario.polylines = {
            element_id: remap_element_refs(element, identity_map)
            for element_id, element in scenario.polylines.items()
            if element_id in valid_ids
        }
    scenario.surfaces = {
        element_id: element
        for element_id, element in scenario.surfaces.items()
        if element.polygon is not None and len(element.polygon) >= 3
    }

    if extras is None:
        return

    lane_ids = {element_id for element_id, element in scenario.polylines.items() if element.is_lane}
    extras.traffic_lights = {
        element_id: light for element_id, light in extras.traffic_lights.items() if light.controlled_lane in lane_ids
    }
    extras.stop_zones = [
        dataclasses.replace(zone, controlled_lanes=[lid for lid in zone.controlled_lanes if lid in lane_ids])
        for zone in extras.stop_zones
        if any(lid in lane_ids for lid in zone.controlled_lanes)
    ]
