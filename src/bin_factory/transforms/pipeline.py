"""The scenario processing pipeline.

Compute lane lengths after processing polylines so they match the serialized geometry.
The lane-distance matrix needs those lengths. Reindex IDs last.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bin_factory.log_context import log

from .geometry import interpolate_all_polygons, process_polylines, reverse_road_edges
from .graph import build_lane_distance_matrix, compute_lane_lengths
from .reindex import reindex_scenario
from .routes import process_agent_routes
from .sanitize import prune_invalid_map_elements
from .shoulder_edges import add_shoulder_edges
from .traffic_controls import process_traffic_controls
from .traffic_light_interpolation import interpolate_traffic_lights


if TYPE_CHECKING:
    import argparse

    from bin_factory import schema


def run(scenario: schema.PufferScenario, extras: schema.ExtractionExtras, config: argparse.Namespace) -> None:
    """Run the processing pipeline in order, mutating ``scenario`` in place."""
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
