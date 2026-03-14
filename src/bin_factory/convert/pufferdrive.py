from bin_factory import logger_utils
from bin_factory.convert import agents, graph, objects, roadgraph, traffic_controls


logger = logger_utils.get_logger(__name__)


def convert_to_puffer_dict(
    py123d_dict: dict,
    min_route_valid_points: int = 0,
    route_check_timestep: int = 0,
    reindex_id: bool = False,
) -> dict:
    if not isinstance(py123d_dict, dict):
        raise TypeError(f"Expected dict, got {type(py123d_dict).__name__}")

    required_fields = ["metadata", "agents", "map", "objects", "traffic_lights"]
    missing = [f for f in required_fields if f not in py123d_dict]
    if missing:
        raise ValueError(f"py123d dict missing required fields: {missing}")

    metadata = py123d_dict["metadata"]
    if not isinstance(metadata, dict):
        raise TypeError(f"Expected metadata dict, got {type(metadata).__name__}")

    required_metadata_fields = ["id", "dataset", "scenario_length", "timestep_seconds"]
    missing_metadata = [f for f in required_metadata_fields if f not in metadata]
    if missing_metadata:
        raise ValueError(f"py123d metadata missing required fields: {missing_metadata}")

    puffer_agents, sdc_index = agents.convert_agents(
        py123d_dict["agents"],
        py123d_dict["map"],
        min_route_valid_points=min_route_valid_points,
        route_check_timestep=route_check_timestep,
    )

    road_map_elements = roadgraph.convert_road_map_elements(py123d_dict["map"], metadata["dataset"])

    traffic_control_elements = traffic_controls.convert_traffic_control_elements(
        py123d_dict["traffic_lights"],
        py123d_dict["map"],
        metadata["scenario_length"],
    )
    puffer_objects = objects.convert_objects(py123d_dict["objects"])

    puffer_metadata = {
        "id": metadata["id"],
        "dataset": metadata["dataset"],
        "scenario_length": metadata["scenario_length"],
        "sdc_index": sdc_index,
        "timestep_seconds": metadata["timestep_seconds"],
        "objects_of_interest": [],
        "tracks_to_predict": [],
    }

    puffer_dict = {
        "agents": puffer_agents,
        "road_map_elements": road_map_elements,
        "traffic_control_elements": traffic_control_elements,
        "objects": puffer_objects,
        "metadata": puffer_metadata,
    }

    if reindex_id:
        puffer_dict = _reindex_puffer_dict(puffer_dict)

    lane_graph_data = graph.build_lane_distance_matrix(puffer_dict["road_map_elements"])
    puffer_dict["lane_graph_distances"] = lane_graph_data

    return puffer_dict


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
