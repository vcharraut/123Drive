import numpy as np
from py123d.datatypes.map_objects import MapLayer

from bin_factory.convert import puffer_types, type_map


def build_puffer_dict(scenario, reindex_id=False):
    puffer_agents = _convert_dynamic_entities(
        scenario.agents, type_map.AGENT_TYPE_MAP, puffer_types.AgentType.OTHER, include_route=True
    )
    road_map_elements = _convert_road_map_elements(scenario.map)
    traffic_control_elements = _convert_traffic_control_elements(scenario.traffic_controls)
    puffer_objects = _convert_dynamic_entities(
        scenario.objects, type_map.OBJECT_TYPE_MAP, puffer_types.ObjectType.GENERIC_OBJECT
    )

    puffer_dict = {
        "agents": puffer_agents,
        "road_map_elements": road_map_elements,
        "traffic_control_elements": traffic_control_elements,
        "objects": puffer_objects,
        "metadata": {
            "id": scenario.metadata["id"],
            "dataset": scenario.metadata["dataset"],
            "scenario_length": scenario.metadata["scenario_length"],
            "timestep_seconds": scenario.metadata["timestep_seconds"],
            "objects_of_interest": [],
            "tracks_to_predict": [],
        },
    }

    if reindex_id:
        puffer_dict = _reindex_puffer_dict(puffer_dict)

    puffer_dict["lane_graph_distances"] = scenario.lane_graph

    return puffer_dict


def _convert_dynamic_entities(items, type_map_dict, default_type, include_route=False):
    return [
        {
            "id": eid,
            "type": type_map_dict.get(data["type"], default_type),
            "states": {k: data[k] for k in ("heading", "velocity", "length", "width", "height", "valid")}
            | {"xyz": data["position"]},
            **({"route": data.get("route", [])} if include_route else {}),
        }
        for eid, data in items.items()
    ]


def _convert_road_map_elements(static_map_elements):
    puffer_elements = []

    for element_id, element_data in static_map_elements.items():
        layer = element_data["layer"]
        element_type = element_data["type"]

        if element_type is None:
            raise ValueError(f"Map element {element_id} has unset type")

        element_type_int = type_map.ROAD_TYPE_MAP.get(layer, {}).get(element_type, -1)

        if element_type_int == -1:
            continue

        if layer in (MapLayer.LANE, MapLayer.ROAD_LINE, MapLayer.ROAD_EDGE):
            xyz = element_data["polyline"]
            if xyz is None or len(xyz) <= 1:
                continue
        else:
            xyz = element_data["polygon"]
            if xyz is None or len(xyz) <= 1:
                continue

        puffer_element = {
            "id": element_id,
            "type": element_type_int,
            "xyz": xyz,
        }

        if layer == MapLayer.LANE:
            puffer_element["speed_limit"] = element_data["speed_limit_mps"]
            puffer_element["entry_lanes"] = element_data["entry_lanes"]
            puffer_element["exit_lanes"] = element_data["exit_lanes"]

        puffer_elements.append(puffer_element)

    return puffer_elements


def _convert_traffic_control_elements(traffic_controls):
    puffer_elements = []

    for tc in traffic_controls:
        type_hint = tc["type_hint"]
        if type_hint == "observed_tl":
            puffer_type = puffer_types.TCType.TRAFFIC_LIGHT
        else:
            puffer_type = type_map.STOP_ZONE_TYPE_MAP.get(type_hint)
            if puffer_type is None:
                continue

        puffer_elements.append(
            {
                "id": tc["id"],
                "type": puffer_type,
                "stop_line": tc["stop_line"],
                "heading": tc["heading"],
                "states": np.array(
                    [type_map.TL_STATE_MAP.get(s, puffer_types.TLState.UNKNOWN) for s in tc["states"]],
                ),
                "controlled_lanes": tc["controlled_lanes"],
            }
        )

    return puffer_elements


def _reindex_puffer_dict(puffer_dict):
    road_id_map = {r["id"]: i for i, r in enumerate(puffer_dict["road_map_elements"])}

    for i, road in enumerate(puffer_dict["road_map_elements"]):
        road["id"] = i
        if "entry_lanes" in road:
            road["entry_lanes"] = [road_id_map[lid] for lid in road["entry_lanes"]]
        if "exit_lanes" in road:
            road["exit_lanes"] = [road_id_map[lid] for lid in road["exit_lanes"]]

    for i, agent in enumerate(puffer_dict["agents"]):
        agent["id"] = i
        agent["route"] = [road_id_map[lid] for lid in agent["route"]]

    for i, tc in enumerate(puffer_dict["traffic_control_elements"]):
        tc["id"] = i
        tc["controlled_lanes"] = [road_id_map[lid] for lid in tc["controlled_lanes"]]

    for i, obj in enumerate(puffer_dict["objects"]):
        obj["id"] = i

    return puffer_dict
