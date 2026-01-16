from src.processors.validation.processor import validate_scenario
from src.processors.polyline_interpolation.processor import interpolate_polylines
from src.processors.traffic_lights.processor import add_traffic_lights_to_scenario

PROCESSORS = {
    "validation": validate_scenario,
    "polyline_interpolation": interpolate_polylines,
    "traffic_lights": add_traffic_lights_to_scenario,
}


def apply_processors(scenario: dict, processor_names: list[str], configs: dict | None = None) -> dict:
    configs = configs or {}
    for name in processor_names or []:
        if name not in PROCESSORS:
            raise ValueError(f"Unknown processor: {name} (supported: {list(PROCESSORS.keys())})")
        scenario = PROCESSORS[name](scenario, **(configs.get(name) or {}))
    return scenario
