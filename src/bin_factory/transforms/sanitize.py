from bin_factory import puffer_types


_MAP_REF_KEYS = ("entry_lanes", "exit_lanes", "left_neighbor", "right_neighbor")


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

    extras["traffic_lights"] = {
        element_id: light
        for element_id, light in extras.get("traffic_lights", {}).items()
        if light.controlled_lane in valid_ids
    }
    extras["stop_zones"] = [
        {**zone, "controlled_lanes": [lane_id for lane_id in zone["controlled_lanes"] if lane_id in valid_ids]}
        for zone in extras.get("stop_zones", [])
    ]


def _is_serializable_map_element(element):
    geometry_key = "polyline" if _uses_polyline(element["type"]) else "polygon"
    geometry = element.get(geometry_key)
    min_points = 2 if geometry_key == "polyline" else 3
    return geometry is not None and len(geometry) >= min_points


def _uses_polyline(element_type):
    return (
        puffer_types.is_road_lane(element_type)
        or puffer_types.is_road_line(element_type)
        or puffer_types.is_road_edge(element_type)
    )


def _filter_map_refs(element, valid_ids):
    return {
        **element,
        **{
            key: [ref_id for ref_id in element.get(key, []) if ref_id in valid_ids]
            for key in _MAP_REF_KEYS
            if key in element
        },
    }


def _filter_track_route(track, valid_ids):
    track.route = [lane_id for lane_id in track.route if lane_id in valid_ids]
    track.route_gt_len = min(track.route_gt_len, len(track.route))
    return track
