_MAP_LIST_REF_KEYS = ("entry_lanes", "exit_lanes", "left_neighbor", "right_neighbor")


def reindex_scenario(scenario) -> None:
    map_id_map = {element_id: idx for idx, element_id in enumerate(scenario.map)}

    scenario.map = {
        map_id_map[element_id]: _remap_map_element(element, map_id_map) for element_id, element in scenario.map.items()
    }
    scenario.agents = {idx: _remap_track(track, map_id_map) for idx, (_, track) in enumerate(scenario.agents.items())}
    scenario.objects = {idx: _remap_track(track, map_id_map) for idx, (_, track) in enumerate(scenario.objects.items())}
    scenario.traffic_controls = [_remap_traffic_control(tc, map_id_map) for tc in scenario.traffic_controls]
    if scenario.lane_graph:
        scenario.lane_graph["lane_ids"] = [
            map_id_map[lid] for lid in scenario.lane_graph["lane_ids"] if lid in map_id_map
        ]


def _remap_map_element(element, map_id_map):
    remapped = {**element}
    for key in _MAP_LIST_REF_KEYS:
        if key in remapped:
            remapped[key] = [map_id_map[ref_id] for ref_id in remapped[key] if ref_id in map_id_map]
    return remapped


def _remap_track(track, map_id_map):
    track.route = [map_id_map[ref_id] for ref_id in track.route if ref_id in map_id_map]
    track.route_gt_len = min(track.route_gt_len, len(track.route))
    return track


def _remap_traffic_control(tc, map_id_map):
    return {
        **tc,
        "controlled_lanes": [map_id_map[lid] for lid in tc["controlled_lanes"] if lid in map_id_map],
    }
