import dataclasses
from typing import Any

from bin_factory import loader, schema
from bin_factory.conversion.geometry import interpolate_all_polygons, process_polylines, reverse_road_edges
from bin_factory.conversion.sanitize import clean_map, drop_duplicate_lines, prune_invalid_map_elements
from bin_factory.conversion.traffic_controls import process_traffic_controls
from bin_factory.generators.routes import process_agent_routes
from bin_factory.generators.shoulder_edges import add_shoulder_edges
from bin_factory.generators.traffic_light_inference import infer_traffic_lights
from bin_factory.log_context import log


@dataclasses.dataclass(frozen=True)
class ConversionSettings:
    """Options that shape one scenario's conversion, with their only defaults.

    The convert CLI fills them from presets and flags; tests build them directly.
    """

    scenario_id_field: str = "scene_uuid"
    dt: float = 0.1
    validate_level: int = 1
    max_segment_length: float = 10.0
    edge_max_segment_length: float = 1.0
    area_threshold: float = 0.1
    speed_limit_source: str = "road"
    static_objects: str = "onroad"
    shoulders: str = "drivable"
    ramp_widening_m: float = 0.0
    reverse_road_edges: bool = False
    infer_traffic_lights: bool = False


def prepare_scenario(py123d_data: Any, settings: ConversionSettings) -> schema.PufferScenario:
    scenario, extras = loader.extract_scenario(
        py123d_data, settings.scenario_id_field, settings.speed_limit_source, settings.static_objects
    )
    clean_map(scenario.polylines)
    # Store the nominal sampling step rather than py123d's measured mean.
    if scenario.metadata.scenario_length:
        if abs(scenario.metadata.dt - settings.dt) > 0.05 * settings.dt:
            raise ValueError(f"Scene step {scenario.metadata.dt:.4f} s is not the requested --dt {settings.dt} s")
        scenario.metadata.dt = settings.dt

    _validate(scenario, settings.validate_level, "Validation failed", extras)
    if settings.shoulders == "offroad":
        added = add_shoulder_edges(
            scenario, extras.non_junction_shoulders, extras.junction_drivables, settings.ramp_widening_m
        )
        log.info("offroad shoulders: %d road edges added along lane boundaries", added)
    if settings.reverse_road_edges:
        reverse_road_edges(scenario)
    process_polylines(scenario, settings.max_segment_length, settings.area_threshold, settings.edge_max_segment_length)
    # Simplification straightens small kinks, so road lines that differed by more than the duplicate distance can match
    drop_duplicate_lines(scenario.polylines)
    interpolate_all_polygons(scenario, tolerance=settings.area_threshold)
    prune_invalid_map_elements(scenario, extras)
    if settings.infer_traffic_lights:
        infer_traffic_lights(scenario, extras)
    process_traffic_controls(scenario, extras)
    process_agent_routes(scenario)
    _validate(scenario, settings.validate_level, "Post-transform validation failed")
    return scenario


def _validate(
    scenario: schema.PufferScenario, level: int, failure: str, extras: schema.ExtractionExtras | None = None
) -> None:
    if level <= 0:
        return
    errors = loader.validate_scenario(scenario, extras=extras, level=level)
    for error in errors:
        log.error(f"{scenario.metadata.id}: {error}")
    if errors:
        raise loader.ValidationError(f"{failure} for scenario {scenario.metadata.id} with {len(errors)} errors")
