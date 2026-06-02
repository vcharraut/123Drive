import dataclasses

from bin_factory.schema import MAP_REF_KEYS


def prune_invalid_map_elements(scenario, extras=None) -> None:
    valid_ids = {element_id for element_id, element in scenario.map.items() if _is_serializable_map_element(element)}
    if len(valid_ids) == len(scenario.map):
        return

    scenario.map = {
        element_id: _filter_map_refs(element, valid_ids)
        for element_id, element in scenario.map.items()
        if element_id in valid_ids
    }
    scenario.agents = {agent_id: _filter_track_route(track, valid_ids) for agent_id, track in scenario.agents.items()}
    scenario.objects = {
        object_id: _filter_track_route(track, valid_ids) for object_id, track in scenario.objects.items()
    }

    if extras is None:
        return

    extras.traffic_lights = {
        element_id: light for element_id, light in extras.traffic_lights.items() if light.controlled_lane in valid_ids
    }
    extras.stop_zones = [
        dataclasses.replace(zone, controlled_lanes=[lid for lid in zone.controlled_lanes if lid in valid_ids])
        for zone in extras.stop_zones
    ]


def _is_serializable_map_element(element):
    geometry = element.geometry
    return geometry is not None and len(geometry) >= element.min_points


def _filter_map_refs(element, valid_ids):
    return dataclasses.replace(
        element,
        **{key: [ref_id for ref_id in getattr(element, key) if ref_id in valid_ids] for key in MAP_REF_KEYS},
    )


def _filter_track_route(track, valid_ids):
    track.route = [lane_id for lane_id in track.route if lane_id in valid_ids]
    track.route_gt_len = min(track.route_gt_len, len(track.route))
    return track
