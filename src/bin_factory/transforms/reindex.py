_MAP_LIST_REF_KEYS = ("entry_lanes", "exit_lanes")
_MAP_SINGLE_REF_KEYS = ("left_lane", "right_lane")


def reindex_scenario_and_extras(scenario, extras) -> None:
    map_id_map = {element_id: idx for idx, element_id in enumerate(scenario.map)}

    scenario.map = {
        map_id_map[element_id]: _remap_map_element(element, map_id_map) for element_id, element in scenario.map.items()
    }
    scenario.agents = {idx: _remap_track(track, map_id_map) for idx, (_, track) in enumerate(scenario.agents.items())}
    scenario.objects = {idx: _remap_track(track, map_id_map) for idx, (_, track) in enumerate(scenario.objects.items())}
    extras["traffic_lights"] = {
        map_id_map[element_id]: _remap_traffic_light(traffic_light, map_id_map)
        for element_id, traffic_light in extras["traffic_lights"].items()
    }
    extras["stop_zones"] = [_remap_stop_zone(stop_zone, map_id_map) for stop_zone in extras["stop_zones"]]


def _remap_map_element(element, map_id_map):
    remapped = {**element}
    for key in _MAP_LIST_REF_KEYS:
        if key in remapped:
            remapped[key] = [map_id_map[ref_id] for ref_id in remapped[key] if ref_id in map_id_map]
    for key in _MAP_SINGLE_REF_KEYS:
        ref_id = remapped.get(key)
        if ref_id is not None:
            remapped[key] = map_id_map.get(ref_id)
    return remapped


def _remap_track(track, map_id_map):
    track.route = [map_id_map[ref_id] for ref_id in track.route if ref_id in map_id_map]
    return track


def _remap_traffic_light(traffic_light, map_id_map):
    traffic_light.controlled_lane = map_id_map[traffic_light.controlled_lane]
    return traffic_light


def _remap_stop_zone(stop_zone, map_id_map):
    return {
        **stop_zone,
        "controlled_lanes": [map_id_map[ref_id] for ref_id in stop_zone["controlled_lanes"] if ref_id in map_id_map],
    }
