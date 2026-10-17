from bin_factory import loader
from bin_factory.log_context import log
from bin_factory.transforms.geometry import interpolate_all_polygons, process_polylines, reverse_road_edges
from bin_factory.transforms.graph import build_lane_distance_matrix, compute_lane_lengths
from bin_factory.transforms.reindex import reindex_scenario
from bin_factory.transforms.routes import process_agent_routes
from bin_factory.transforms.sanitize import prune_invalid_map_elements
from bin_factory.transforms.shoulder_edges import add_shoulder_edges
from bin_factory.transforms.traffic_controls import process_traffic_controls
from bin_factory.transforms.traffic_light_interpolation import interpolate_traffic_lights


def prepare_scenario(py123d_data, config):
    scenario, extras = loader.extract_scenario(
        py123d_data, config.scenario_id_field, config.speed_limit_source, config.static_objects
    )
    # Store the nominal sampling step rather than py123d's measured mean.
    if scenario.metadata.scenario_length:
        if abs(scenario.metadata.dt - config.dt) > 0.05 * config.dt:
            raise ValueError(f"Scene step {scenario.metadata.dt:.4f} s is not the requested --dt {config.dt} s")
        scenario.metadata.dt = config.dt

    _validate(scenario, config.validate_level, "Validation failed", extras)
    if config.interpolate_tl:
        interpolate_traffic_lights(scenario, extras)
    if config.shoulders == "offroad":
        added = add_shoulder_edges(
            scenario, extras.non_junction_shoulders, extras.junction_drivables, config.ramp_widening_m
        )
        log.info("offroad shoulders: %d road edges added along lane boundaries", added)
    if config.reverse_road_edges:
        reverse_road_edges(scenario)
    process_polylines(scenario, config.max_segment_length, config.area_threshold, config.edge_max_segment_length)
    interpolate_all_polygons(scenario)
    prune_invalid_map_elements(scenario, extras)
    process_traffic_controls(scenario, extras)
    process_agent_routes(scenario, config.route_check_timestep)
    compute_lane_lengths(scenario)
    scenario.lane_graph = build_lane_distance_matrix(scenario.map)
    if not config.no_reindex:
        reindex_scenario(scenario)
    _validate(scenario, config.validate_level, "Post-transform validation failed")
    return scenario


def _validate(scenario, level, failure, extras=None):
    if level <= 0:
        return
    errors = loader.validate_scenario(scenario, extras=extras, level=level)
    for error in errors:
        log.error(f"{scenario.metadata.id}: {error}")
    if errors:
        raise loader.ValidationError(f"{failure} for scenario {scenario.metadata.id} with {len(errors)} errors")
