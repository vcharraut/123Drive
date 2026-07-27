import dataclasses

from bin_factory import schema
from bin_factory.schema import remap_element_refs


def prune_invalid_map_elements(scenario: schema.PufferScenario, extras: schema.ExtractionExtras | None = None) -> None:
    valid_ids = {element_id for element_id, element in scenario.map.items() if _is_serializable_map_element(element)}
    if len(valid_ids) != len(scenario.map):
        identity_map = {element_id: element_id for element_id in valid_ids}
        scenario.map = {
            element_id: remap_element_refs(element, identity_map)
            for element_id, element in scenario.map.items()
            if element_id in valid_ids
        }
        scenario.agents = {agent_id: _filter_track_route(track, valid_ids) for agent_id, track in scenario.agents.items()}
        scenario.objects = {
            object_id: _filter_track_route(track, valid_ids) for object_id, track in scenario.objects.items()
        }

    if extras is None:
        return

    lane_ids = {element_id for element_id, element in scenario.map.items() if element.is_lane}
    extras.traffic_lights = {
        element_id: light for element_id, light in extras.traffic_lights.items() if light.controlled_lane in lane_ids
    }
    extras.stop_zones = [
        dataclasses.replace(zone, controlled_lanes=[lid for lid in zone.controlled_lanes if lid in lane_ids])
        for zone in extras.stop_zones
        if any(lid in lane_ids for lid in zone.controlled_lanes)
    ]


def _is_serializable_map_element(element: schema.MapElement) -> bool:
    geometry = element.geometry
    return geometry is not None and len(geometry) >= element.min_points


def _filter_track_route(track: schema.Track, valid_ids: set[int]) -> schema.Track:
    track.route = [lane_id for lane_id in track.route if lane_id in valid_ids]
    track.route_gt_len = min(track.route_gt_len, len(track.route))
    return track
