# Based on: Yan, X., Liang, E., Wang, J., Zhu, H., & Liu, H. X. (2025).
# "Improving Traffic Signal Data Quality for the Waymo Open Motion Dataset."
# arXiv:2506.07150v1. University of Michigan.
# https://github.com/michigan-traffic-lab/WOMD-Traffic-Signal-Data-Improvement

from src.bin_factory.transforms.traffic_lights.pipeline import generate_tl_states
from src.bin_factory.transforms.traffic_lights.topology import build_lane_topology


def add_traffic_lights_to_scenario(scenario):
    if "map" not in scenario:
        raise ValueError("Scenario missing 'map' - cannot generate traffic lights")
    if "scenario_length" not in scenario:
        raise ValueError("Scenario missing 'scenario_length' - cannot generate traffic lights")
    if "agents" not in scenario:
        raise ValueError("Scenario missing 'agents' - cannot infer traffic light states")

    lane_data, signalized = build_lane_topology(scenario)
    scenario["traffic_lights"] = generate_tl_states(scenario, lane_data, signalized)
    return scenario
