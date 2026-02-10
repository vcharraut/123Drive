from src import logger_utils
from src.processors.polyline.utils import (
    distance_based_interpolate,
    remove_duplicate_points,
    simplify_polyline,
    validate_polyline,
)


logger = logger_utils.get_logger(__name__)


def process_polylines(scenario, max_segment_length=2.0, area_threshold=0.1, dist_threshold=10.0):
    map_elements = scenario.get("map", {})
    if not map_elements:
        return scenario

    for element_id, element in map_elements.items():
        if "polyline" not in element:
            continue

        polyline = element["polyline"]

        if len(polyline) < 2:
            continue

        if not validate_polyline(polyline, element_id=str(element_id)):
            continue

        polyline = remove_duplicate_points(polyline)

        if max_segment_length > 0:
            polyline = distance_based_interpolate(polyline, max_segment_length)

        if area_threshold > 0 and len(polyline) >= 3:
            polyline = simplify_polyline(polyline, area_threshold, dist_threshold)

        element["polyline"] = polyline

    return scenario
