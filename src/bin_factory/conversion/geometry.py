import numpy as np
from shapely import geometry as shapely_geom

from bin_factory import schema


def interpolate_all_polygons(scenario: schema.PufferScenario, spacing: float = 5.0, tolerance: float = 0.0) -> None:
    """Close each surface outline, simplify it within ``tolerance``, and densify it to ``spacing``."""
    for surface in scenario.surfaces.values():
        if len(surface.polygon) == 0:
            surface.polygon = np.zeros((0, 3), dtype=np.float64)
            continue
        xyz = np.asarray(surface.polygon, dtype=np.float64)
        if not np.allclose(xyz[0], xyz[-1]):
            xyz = np.vstack([xyz, xyz[0:1]])
        if tolerance > 0 and len(xyz) >= 4:
            xyz = _simplify_polyline(xyz, tolerance)  # topology-preserving, so the ring stays closed
        surface.polygon = np.asarray(shapely_geom.LineString(xyz).segmentize(spacing).coords, dtype=np.float64)


def reverse_road_edges(scenario: schema.PufferScenario) -> None:
    # PY123D-REPORT[nuplan,carla,opendrive,nuscenes]: py123d road edges run opposite to the WOMD orientation;
    # enabled per dataset via the `reverse_road_edges` preset key.
    for element in scenario.polylines.values():
        if element.is_edge and element.polyline is not None:
            element.polyline = element.polyline[::-1]


def process_polylines(
    scenario: schema.PufferScenario,
    max_segment_length: float = 10.0,
    area_threshold: float = 0.1,
    edge_max_segment_length: float = 1.0,
) -> None:
    for element in scenario.polylines.values():
        if element.polyline is None or len(element.polyline) < 2:
            continue
        polyline = _remove_duplicate_points(element.polyline)
        if area_threshold > 0 and len(polyline) >= 3:
            polyline = _simplify_polyline(polyline, area_threshold)
        # PufferDrive culls and grids each segment by its midpoint, so long edge segments drop
        # corners near the observation window border. Give road edges shorter segments.
        segment_length = edge_max_segment_length if element.is_edge else max_segment_length
        if segment_length > 0:
            polyline = resample_polyline(polyline, segment_length)
        element.polyline = polyline


def _remove_duplicate_points(polyline: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    diffs = np.diff(polyline, axis=0)
    keep_mask = np.concatenate([[True], (diffs * diffs).sum(axis=1) >= tol * tol])
    if keep_mask.all():
        return polyline
    return polyline[keep_mask]


def resample_polyline(polyline: np.ndarray, max_segment_length: float) -> np.ndarray:
    if len(polyline) < 2:
        return polyline.copy()

    diffs = np.diff(polyline, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    subdivisions = np.maximum(np.ceil(seg_lengths / max_segment_length).astype(int), 1)

    # Pre-allocate space for each segment's subdivisions plus the final endpoint.
    total_points = int(subdivisions.sum()) + 1
    if total_points == len(polyline):
        return polyline.copy()
    result = np.empty((total_points, polyline.shape[1]), dtype=polyline.dtype)

    segments = np.repeat(np.arange(len(diffs)), subdivisions)
    starts = np.cumsum(subdivisions) - subdivisions
    t = (np.arange(total_points - 1) - starts[segments]) / subdivisions[segments]
    result[:-1] = polyline[segments] + t[:, None] * diffs[segments]
    result[-1] = polyline[-1]

    return result


def _simplify_polyline(polyline: np.ndarray, tolerance: float) -> np.ndarray:
    # Shapely simplifies in 2D and keeps the retained vertices' z
    return np.array(shapely_geom.LineString(polyline).simplify(tolerance).coords)


def locate_on_polyline(
    points: np.ndarray, polyline: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return each xy point's distance to the xyz polyline, and its arc length, direction and height there."""
    start, step = polyline[:-1], np.diff(polyline, axis=0)
    squared = (step[:, :2] ** 2).sum(1)
    t = np.clip(((points[:, None] - start[:, :2]) * step[:, :2]).sum(-1) / np.maximum(squared, 1e-12), 0.0, 1.0)
    gap = np.linalg.norm(points[:, None] - (start[:, :2] + t[..., None] * step[:, :2]), axis=-1)
    gap[:, squared <= 1e-12] = np.inf
    segment = gap.argmin(1)
    rows = np.arange(len(points))
    t = t[rows, segment]
    arc = schema.arc_length(polyline[:, :2])[segment] + t * np.sqrt(squared[segment])
    direction = np.arctan2(step[segment, 1], step[segment, 0])
    return gap[rows, segment], arc, direction, start[segment, 2] + t * step[segment, 2]


def height_at(polyline: np.ndarray, point: np.ndarray) -> float:
    """Return the polyline's height where the xy ``point`` projects onto it."""
    return float(locate_on_polyline(np.asarray(point, dtype=np.float64)[None, :2], polyline)[3][0])
