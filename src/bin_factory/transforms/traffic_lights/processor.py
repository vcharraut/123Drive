from typing import Any

from src.bin_factory.transforms.traffic_lights.waymonic_tlsgen import WaymonicTLSGenerator
from src.bin_factory.transforms.traffic_lights.waymonizer import Waymonizer


class ScenarioProcessor(Waymonizer, WaymonicTLSGenerator):
    def __init__(self, scenario) -> None:
        self.scenario = scenario
        Waymonizer.__init__(self, scenario)
        WaymonicTLSGenerator.__init__(self, scenario, self.lanecenters, self.signalized_intersections)


def add_traffic_lights_to_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    # Validate required fields
    if "map" not in scenario:
        raise ValueError("Scenario missing 'map' - cannot generate traffic lights")
    if "scenario_length" not in scenario:
        raise ValueError("Scenario missing 'scenario_length' - cannot generate traffic lights")
    if "agents" not in scenario:
        raise ValueError("Scenario missing 'agents' - cannot infer traffic light states")

    sp = ScenarioProcessor(scenario)

    dynamic_map_states = sp.generate_waymonic_tls(
        return_data="dynamic_states",
        end_step=scenario["scenario_length"],
    )

    # Replace traffic_lights with generated traffic lights
    scenario["traffic_lights"] = {**dynamic_map_states}
    return scenario
