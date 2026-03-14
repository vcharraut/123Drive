from bin_factory.processors import geometry, graph, routes, traffic_controls


__all__ = []


def process_scenario(
    scenario,
    *,
    max_segment_length=2.0,
    area_threshold=0.1,
    min_route_valid_points=0,
    route_check_timestep=0,
):
    geometry.process_polylines(scenario, max_segment_length, area_threshold)
    geometry.ensure_all_3d(scenario)
    geometry.interpolate_all_polygons(scenario)
    geometry.reverse_road_edges(scenario)
    traffic_controls.process_traffic_controls(scenario)
    routes.process_agent_routes(scenario, min_route_valid_points, route_check_timestep)
    scenario.lane_graph = graph.build_lane_distance_matrix(scenario.map)
    return scenario
