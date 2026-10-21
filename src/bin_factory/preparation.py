from typing import Any

from bin_factory import loader, schema
from bin_factory.conversion.geometry import interpolate_all_polygons, process_polylines, reverse_road_edges
from bin_factory.conversion.sanitize import prune_invalid_map_elements
from bin_factory.conversion.traffic_controls import process_traffic_controls
from bin_factory.generators.graph import build_lane_distance_matrix, compute_lane_lengths
from bin_factory.generators.routes import process_agent_routes
from bin_factory.generators.shoulder_edges import add_shoulder_edges
from bin_factory.generators.traffic_light_inference import infer_traffic_lights
from bin_factory.log_context import log


def prepare_scenario(py123d_data: Any, config: Any) -> schema.PufferScenario:
    scenario, extras = loader.extract_scenario(
        py123d_data, config.scenario_id_field, config.speed_limit_source, config.static_objects
    )
    # Store the nominal sampling step rather than py123d's measured mean.
    if scenario.metadata.scenario_length:
        if abs(scenario.metadata.dt - config.dt) > 0.05 * config.dt:
            raise ValueError(f"Scene step {scenario.metadata.dt:.4f} s is not the requested --dt {config.dt} s")
        scenario.metadata.dt = config.dt

    _validate(scenario, config.validate_level, "Validation failed", extras)
    if config.infer_traffic_lights:
        infer_traffic_lights(scenario, extras)
    if config.shoulders == "offroad":
        added = add_shoulder_edges(
            scenario, extras.non_junction_shoulders, extras.junction_drivables, config.ramp_widening_m
        )
        log.info("offroad shoulders: %d road edges added along lane boundaries", added)
    if config.reverse_road_edges:
        reverse_road_edges(scenario)
    process_polylines(scenario, config.max_segment_length, config.area_threshold, config.edge_max_segment_length)
    # Simplification straightens small kinks, so road lines that differed by more than the duplicate distance can match
    loader.drop_duplicate_lines(scenario.polylines)
    interpolate_all_polygons(scenario, tolerance=config.area_threshold)
    prune_invalid_map_elements(scenario, extras)
    process_traffic_controls(scenario, extras)
    process_agent_routes(scenario)
    compute_lane_lengths(scenario)
    scenario.lane_graph = build_lane_distance_matrix(scenario.polylines)
    _validate(scenario, config.validate_level, "Post-transform validation failed")
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
