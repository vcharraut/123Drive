def _get_processor(name: str):
    if name == "validation":
        from src.processors.validation.processor import validate_scenario

        return validate_scenario
    if name == "traffic_lights":
        from src.processors.traffic_lights.processor import add_traffic_lights_to_scenario

        return add_traffic_lights_to_scenario
    if name == "polyline_interpolation":
        from src.processors.polyline_interpolation.processor import interpolate_polylines

        return interpolate_polylines

    raise ValueError(
        f"Unknown processor: {name} (supported: validation|traffic_lights|polyline_interpolation)",
    )


def apply_processors(scenario: dict, processor_names: list[str], configs: dict | None = None) -> dict:
    configs = configs or {}
    for name in processor_names or []:
        scenario = _get_processor(name)(scenario, **(configs.get(name) or {}))
    return scenario
