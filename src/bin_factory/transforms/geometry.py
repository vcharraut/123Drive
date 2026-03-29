import logging

import numpy as np
from shapely import geometry as shapely_geom

from bin_factory import puffer_types


logger = logging.getLogger(__name__)


# ── Interpolate polygons to ensure they are all the same spacing ───────────────────────


def interpolate_all_polygons(scenario, spacing=3.0) -> None:
    for element_data in scenario.map.values():
        if "polygon" in element_data:
            element_data["polygon"] = _interpolate_polygon(element_data["polygon"], spacing)


def _interpolate_polygon(xyz, spacing=3.0):
    if xyz is None or len(xyz) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    if not np.allclose(xyz[0], xyz[-1]):
        xyz = np.vstack([xyz, xyz[0:1]])

    diffs = np.diff(xyz, axis=0)
    seg_lengths = np.linalg.norm(diffs[:, :2], axis=1)
    total_length = seg_lengths.sum()

    if total_length < spacing:
        return xyz

    cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
    num_points = max(int(total_length / spacing), 2)
    target_dists = np.linspace(0, total_length, num_points, endpoint=False)

    idx = np.clip(np.searchsorted(cum_lengths[1:], target_dists, side="right"), 0, len(seg_lengths) - 1)
    safe_lengths = np.where(seg_lengths[idx] > 0, seg_lengths[idx], 1.0)
    t = ((target_dists - cum_lengths[idx]) / safe_lengths)[:, np.newaxis]
    interpolated = xyz[idx] + t * diffs[idx]

    return np.vstack([interpolated, interpolated[0:1]])


# ── Reverse road edges for certain datasets to face opposite direction to lanes ────────

REVERSE_ROAD_EDGE_DATASETS = ["av2", "nuplan", "carla"]


def reverse_road_edges(scenario) -> None:
    dataset = scenario.metadata.dataset
    if dataset.split("-")[0] not in REVERSE_ROAD_EDGE_DATASETS:
        return
    for element_data in scenario.map.values():
        if puffer_types.is_road_edge(element_data["type"]) and "polyline" in element_data:
            element_data["polyline"] = element_data["polyline"][::-1]


# ── Rotate body-frame velocity to global frame for certain datasets ─────────────────────

BODY_FRAME_VELOCITY_DATASETS = ["nuplan"]


def rotate_body_to_global_velocity(scenario):
    if scenario.metadata.dataset.split("-")[0] not in BODY_FRAME_VELOCITY_DATASETS:
        return
    for track in [*scenario.agents.values(), *scenario.objects.values()]:
        cos_h = np.cos(track.heading)
        sin_h = np.sin(track.heading)
        vx = track.velocity[:, 0].copy()
        vy = track.velocity[:, 1].copy()
        track.velocity[:, 0] = vx * cos_h - vy * sin_h
        track.velocity[:, 1] = vx * sin_h + vy * cos_h


# ── Downsample and simplify polylines ──────────────────────────────────────────────────


def process_polylines(scenario, max_segment_length=2.0, area_threshold=0.1) -> None:
    map_elements = scenario.map
    if not map_elements:
        return

    for element in map_elements.values():
        if "polyline" not in element:
            continue

        polyline = element["polyline"]

        if len(polyline) < 2:
            continue

        polyline = _remove_duplicate_points(polyline)

        if area_threshold > 0 and len(polyline) >= 3:
            polyline = _simplify_polyline(polyline, area_threshold)

        if max_segment_length > 0:
            polyline = _distance_based_interpolate(polyline, max_segment_length)

        element["polyline"] = polyline


def _remove_duplicate_points(polyline, tol=1e-9):
    diffs = np.diff(polyline, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    keep_mask = np.concatenate([[True], distances >= tol])

    return polyline[keep_mask]


def _distance_based_interpolate(polyline, max_segment_length):
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


def _simplify_polyline(polyline, tolerance):
    simplified_2d = np.array(shapely_geom.LineString(polyline[:, :2]).simplify(tolerance).coords)

    # Re-interpolate z from original polyline at simplified 2D positions
    cum_orig = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(polyline[:, :2], axis=0), axis=1))])
    cum_simp = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(simplified_2d, axis=0), axis=1))])
    z_interp = np.interp(cum_simp, cum_orig, polyline[:, 2])

    return np.column_stack([simplified_2d, z_interp])
