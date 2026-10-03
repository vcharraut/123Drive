"""The scenario processing pipeline.

Ordering is load-bearing:
- ``compute_lane_lengths`` must run after ``process_polylines`` (lengths match serialized geometry),
- ``build_lane_distance_matrix`` needs those lengths,
- ``invalid_agent_overlap`` needs routes from ``process_agent_routes``,
- ``reindex_scenario`` must run last.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .geometry import interpolate_all_polygons, process_polylines, reverse_road_edges
from .graph import build_lane_distance_matrix, compute_lane_lengths
from .invalid_agents import invalid_agent_overlap
from .reindex import reindex_scenario
from .routes import process_agent_routes
from .sanitize import prune_invalid_map_elements
from .traffic_controls import process_traffic_controls
from .traffic_light_interpolation import interpolate_traffic_lights


if TYPE_CHECKING:
    import argparse

    from bin_factory import schema


def run(scenario: schema.PufferScenario, extras: schema.ExtractionExtras, config: argparse.Namespace) -> None:
    """Run the processing pipeline in order, mutating ``scenario`` in place."""
    if config.interpolate_tl:
        interpolate_traffic_lights(scenario, extras)
    if config.reverse_road_edges:
        reverse_road_edges(scenario)
    process_polylines(scenario, config.max_segment_length, config.area_threshold)
    interpolate_all_polygons(scenario)
    prune_invalid_map_elements(scenario, extras)
    process_traffic_controls(scenario, extras)
    process_agent_routes(scenario, config.route_check_timestep)
    if config.invalid_agent_overlap:
        invalid_agent_overlap(scenario)
    compute_lane_lengths(scenario)
    scenario.lane_graph = build_lane_distance_matrix(scenario.map)
    if not config.no_reindex:
        reindex_scenario(scenario)
