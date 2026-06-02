"""The scenario processing pipeline.

The conversion's processing stage is a fixed, order-dependent sequence of transforms with
heterogeneous signatures. This module gives that sequence a home: each transform is wrapped
in a uniform ``(scenario, extras, config) -> None`` adapter, and ``build_stages`` declares the
order — including the config-gated stages — as data. ``run`` is the single entry point
(the ``process_scenario`` the docs refer to).

Ordering is load-bearing:
- ``compute_lane_lengths`` must run after ``process_polylines`` (lengths match serialized geometry),
- ``build_lane_graph`` needs those lengths,
- ``invalid_agent_overlap`` needs routes from ``process_agent_routes``,
- ``reindex`` must run last.
"""

from .geometry import interpolate_all_polygons, process_polylines
from .graph import build_lane_distance_matrix, compute_lane_lengths
from .invalid_agents import invalid_agent_overlap
from .reindex import reindex_scenario
from .routes import process_agent_routes
from .sanitize import prune_invalid_map_elements
from .traffic_controls import process_traffic_controls
from .traffic_light_interpolation import interpolate_traffic_lights


def _interpolate_traffic_lights(scenario, extras, config) -> None:
    interpolate_traffic_lights(scenario, extras)


def _process_polylines(scenario, extras, config) -> None:
    process_polylines(scenario, config.max_segment_length, config.area_threshold)


def _interpolate_polygons(scenario, extras, config) -> None:
    interpolate_all_polygons(scenario)


def _prune_invalid_map_elements(scenario, extras, config) -> None:
    prune_invalid_map_elements(scenario, extras)


def _process_traffic_controls(scenario, extras, config) -> None:
    process_traffic_controls(scenario, extras)


def _process_agent_routes(scenario, extras, config) -> None:
    process_agent_routes(scenario, config.min_route_valid_points, config.route_check_timestep)


def _invalid_agent_overlap(scenario, extras, config) -> None:
    invalid_agent_overlap(scenario)


def _compute_lane_lengths(scenario, extras, config) -> None:
    compute_lane_lengths(scenario)


def _build_lane_graph(scenario, extras, config) -> None:
    scenario.lane_graph = build_lane_distance_matrix(scenario.map)


def _reindex_scenario(scenario, extras, config) -> None:
    reindex_scenario(scenario)


def build_stages(config) -> list:
    """Return the ordered processing stages for ``config``, gating the optional ones."""
    stages = []
    if config.interpolate_tl:
        stages.append(_interpolate_traffic_lights)
    stages += [
        _process_polylines,
        _interpolate_polygons,
        _prune_invalid_map_elements,
        _process_traffic_controls,
        _process_agent_routes,
    ]
    if config.invalid_agent_overlap:
        stages.append(_invalid_agent_overlap)
    stages += [_compute_lane_lengths, _build_lane_graph]
    if not config.no_reindex:
        stages.append(_reindex_scenario)
    return stages


def run(scenario, extras, config) -> None:
    """Run the processing pipeline in declared order, mutating ``scenario`` in place."""
    for stage in build_stages(config):
        stage(scenario, extras, config)
