from bin_factory import logger_utils
from bin_factory.convert import agents, roadgraph, traffic_controls


logger = logger_utils.get_logger(__name__)


def convert_to_puffer_dict(scenario: dict, min_route_valid_points: int = 0, route_check_timestep: int = 0) -> dict:
    if not isinstance(scenario, dict):
        raise TypeError(f"Expected dict, got {type(scenario).__name__}")

    required_fields = ["id", "agents", "map", "traffic_lights"]
    missing = [f for f in required_fields if f not in scenario]
    if missing:
        raise ValueError(f"scenario missing required fields: {missing}")

    puffer_agents, sdc_index = agents.convert_agents(
        scenario["agents"],
        scenario["map"],
        min_route_valid_points=min_route_valid_points,
        route_check_timestep=route_check_timestep,
    )

    road_map_elements = roadgraph.convert_road_map_elements(scenario["map"], scenario.get("dataset_name", ""))

    traffic_control_elements = traffic_controls.convert_traffic_control_elements(
        scenario["traffic_lights"],
        scenario["map"],
        scenario.get("scenario_length", 0),
    )

    puffer_metadata = {
        "dataset_name": scenario.get("dataset_name", ""),
        "scenario_length": scenario.get("scenario_length", 0),
        "sdc_index": sdc_index,
        "timestep_seconds": scenario.get("timestep_seconds", 0.0),
        "objects_of_interests": [],
        "tracks_to_predict": [],
    }

    return {
        "scenario_id": scenario["id"],
        "agents": puffer_agents,
        "road_map_elements": road_map_elements,
        "traffic_control_elements": traffic_control_elements,
        "metadata": puffer_metadata,
    }
