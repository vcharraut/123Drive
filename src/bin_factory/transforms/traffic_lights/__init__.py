"""
Traffic lights enhancement for ScenarioMax.

Based on:
    Yan, X., Liang, E., Wang, J., Zhu, H., & Liu, H. X. (2025).
    "Improving Traffic Signal Data Quality for the Waymo Open Motion Dataset."
    arXiv:2506.07150v1. University of Michigan.
    https://github.com/michigan-traffic-lab/WOMD-Traffic-Signal-Data-Improvement
"""

from src.bin_factory.transforms.traffic_lights.processor import add_traffic_lights_to_scenario


__all__ = ["add_traffic_lights_to_scenario"]
