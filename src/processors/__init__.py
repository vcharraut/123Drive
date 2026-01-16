from src.processors.polyline_interpolation.processor import interpolate_polylines
from src.processors.traffic_lights.processor import add_traffic_lights_to_scenario


# Processors that run on unified format (before conversion to puffer)
# Note: validation now runs on puffer_dict format, handled separately in pipeline.py
PROCESSORS = {
    "polyline_interpolation": interpolate_polylines,
    "traffic_lights": add_traffic_lights_to_scenario,
}


def apply_processors(scenario: dict, processor_names: list[str], configs: dict | None = None) -> dict:
    configs = configs or {}
    for name in processor_names or []:
        if name == "validation":
            continue  # validation runs on puffer_dict, handled in pipeline.py
        if name not in PROCESSORS:
            raise ValueError(f"Unknown processor: {name} (supported: {list(PROCESSORS.keys())})")
        scenario = PROCESSORS[name](scenario, **(configs.get(name) or {}))
    return scenario
