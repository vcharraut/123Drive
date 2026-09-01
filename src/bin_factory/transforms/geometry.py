import numpy as np
import shapely
from shapely import geometry as shapely_geom

from bin_factory import schema


# ── Polyline length primitives ──────────────────────────────────────────────────────────


def arc_length(polyline: np.ndarray) -> np.ndarray:
    """Cumulative arc-length per point. cum[0]=0, cum[-1]=total length. Norms over all columns."""
    polyline = np.asarray(polyline)
    if len(polyline) < 2:
        return np.zeros(len(polyline), dtype=np.float64)
    seg = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float64)


DEFAULT_LANE_WIDTH_M = 3.7
# PufferDrive's loader rejects any lane point wider than its MAX_LANE_WIDTH_M (50 m)
MAX_LANE_WIDTH_M = 50.0


def compute_lane_widths(scenario: schema.PufferScenario, default_width_m: float = DEFAULT_LANE_WIDTH_M) -> None:
    """Annotate each lane with per-point `width` = centerline distance to left boundary + to right boundary.

    Must run AFTER process_polylines so values align with the serialized polyline.
    """
    for element in scenario.map.values():
        if not element.is_lane or element.polyline is None:
            continue
        element.width = _lane_widths_from_boundaries(
            element.polyline, element.left_boundary, element.right_boundary, default_width_m
        )


def _lane_widths_from_boundaries(
    polyline: np.ndarray,
    left_boundary: np.ndarray | None,
    right_boundary: np.ndarray | None,
    default_width_m: float,
) -> np.ndarray:
    widths = np.full(len(polyline), default_width_m, dtype=np.float64)
    if left_boundary is None or right_boundary is None or len(left_boundary) < 2 or len(right_boundary) < 2:
        return widths
    left = shapely_geom.LineString(np.asarray(left_boundary, dtype=np.float64)[:, :2])
    right = shapely_geom.LineString(np.asarray(right_boundary, dtype=np.float64)[:, :2])
    points = shapely.points(np.asarray(polyline, dtype=np.float64)[:, :2])
    measured = shapely.distance(left, points) + shapely.distance(right, points)
    valid = np.isfinite(measured) & (measured > 0.0)
    widths[valid] = measured[valid]
    return np.minimum(widths, MAX_LANE_WIDTH_M)


def polyline_length(polyline: np.ndarray) -> float:
    """Total arc-length of a polyline (sum of segment norms over all columns)."""
    polyline = np.asarray(polyline)
    if len(polyline) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(polyline, axis=0), axis=1)))


# ── Interpolate polygons to ensure they are all the same spacing ───────────────────────


def interpolate_all_polygons(scenario: schema.PufferScenario, spacing: float = 5.0) -> None:
    for element_data in scenario.map.values():
        if element_data.polygon is not None:
            element_data.polygon = _interpolate_polygon(element_data.polygon, spacing)


def _interpolate_polygon(xyz: np.ndarray | None, spacing: float) -> np.ndarray:
    if xyz is None or len(xyz) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    xyz = np.asarray(xyz, dtype=np.float64)
    if not np.allclose(xyz[0], xyz[-1]):
        xyz = np.vstack([xyz, xyz[0:1]])

    densified = shapely_geom.LineString(xyz).segmentize(spacing)
    return np.asarray(densified.coords, dtype=np.float64)


# ── Reverse road-edge heading  ─────────────────────────────────────────────────────────


def reverse_road_edges(scenario: schema.PufferScenario) -> None:
    for element in scenario.map.values():
        if element.is_edge and element.polyline is not None:
            element.polyline = element.polyline[::-1]


# ── Downsample and simplify polylines ──────────────────────────────────────────────────


def process_polylines(
    scenario: schema.PufferScenario, max_segment_length: float = 2.0, area_threshold: float = 0.1
) -> None:
    map_elements = scenario.map
    if not map_elements:
        return

    for element in map_elements.values():
        if element.polyline is None:
            continue

        polyline = element.polyline

        if len(polyline) < 2:
            continue

        polyline = _remove_duplicate_points(polyline)

        if area_threshold > 0 and len(polyline) >= 3:
            polyline = _simplify_polyline(polyline, area_threshold)

        if max_segment_length > 0:
            polyline = _distance_based_interpolate(polyline, max_segment_length)

        element.polyline = polyline


def _remove_duplicate_points(polyline: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    diffs = np.diff(polyline, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    keep_mask = np.concatenate([[True], distances >= tol])

    return polyline[keep_mask]


def _distance_based_interpolate(polyline: np.ndarray, max_segment_length: float) -> np.ndarray:
    diffs = np.diff(polyline, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    subdivisions = np.maximum(np.ceil(seg_lengths / max_segment_length).astype(int), 1)

    # Pre-allocate: each segment contributes `subdivisions[i]` points, plus the final endpoint
    total_points = subdivisions.sum() + 1
    result = np.empty((total_points, polyline.shape[1]), dtype=polyline.dtype)

    idx = 0
    for i in range(len(diffs)):
        n = subdivisions[i]
        t = np.arange(n).reshape(-1, 1) / n
        result[idx : idx + n] = polyline[i] + t * diffs[i]
        idx += n
    result[idx] = polyline[-1]

    return result


def _simplify_polyline(polyline: np.ndarray, tolerance: float) -> np.ndarray:
    simplified_2d = np.array(shapely_geom.LineString(polyline[:, :2]).simplify(tolerance).coords)

    # Re-interpolate z from original polyline at simplified 2D positions
    cum_orig = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(polyline[:, :2], axis=0), axis=1))])
    cum_simp = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(simplified_2d, axis=0), axis=1))])
    z_interp = np.interp(cum_simp, cum_orig, polyline[:, 2])

    return np.column_stack([simplified_2d, z_interp])
