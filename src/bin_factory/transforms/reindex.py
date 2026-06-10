from bin_factory.schema import remap_element_refs


def reindex_scenario(scenario) -> None:
    map_id_map = _build_id_map(scenario.map)
    agent_id_map = _build_id_map(scenario.agents)
    object_id_map = _build_id_map(scenario.objects)
    traffic_control_id_map = _build_id_map([tc["id"] for tc in scenario.traffic_controls])

    scenario.map = {
        map_id_map[element_id]: remap_element_refs(element, map_id_map)
        for element_id, element in scenario.map.items()
    }
    scenario.agents = {
        agent_id_map[track_id]: _remap_track(track, map_id_map) for track_id, track in scenario.agents.items()
    }
    scenario.objects = {
        object_id_map[track_id]: _remap_track(track, map_id_map) for track_id, track in scenario.objects.items()
    }
    scenario.traffic_controls = [
        _remap_traffic_control(tc, map_id_map, traffic_control_id_map[tc["id"]]) for tc in scenario.traffic_controls
    ]
    if scenario.lane_graph:
        scenario.lane_graph["lane_ids"] = [
            map_id_map[lid] for lid in scenario.lane_graph["lane_ids"] if lid in map_id_map
        ]
    scenario.metadata.objects_of_interest = [
        agent_id_map[i] for i in scenario.metadata.objects_of_interest if i in agent_id_map
    ]
    scenario.metadata.tracks_to_predict = [
        agent_id_map[i] for i in scenario.metadata.tracks_to_predict if i in agent_id_map
    ]


def _build_id_map(ids):
    return {element_id: idx for idx, element_id in enumerate(ids)}


def _remap_track(track, map_id_map):
    track.route = [map_id_map[ref_id] for ref_id in track.route if ref_id in map_id_map]
    track.route_gt_len = min(track.route_gt_len, len(track.route))
    return track


def _remap_traffic_control(tc, map_id_map, control_id):
    return {
        **tc,
        "id": control_id,
        "controlled_lanes": [map_id_map[lid] for lid in tc["controlled_lanes"] if lid in map_id_map],
    }
