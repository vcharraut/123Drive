import numpy as np

from src import logger_utils


logger = logger_utils.get_logger(__name__)


def validate_polyline(polyline, element_id=""):
    if not isinstance(polyline, np.ndarray):
        logger.warning(f"Element {element_id}: polyline is not ndarray (got {type(polyline).__name__})")
        return False
    if polyline.ndim != 2 or polyline.shape[1] != 3:
        logger.warning(f"Element {element_id}: polyline has invalid shape {polyline.shape}, expected (N, 3)")
        return False
    if np.any(np.isnan(polyline)):
        logger.warning(f"Element {element_id}: polyline contains NaN values")
        return False
    if np.any(np.isinf(polyline)):
        logger.warning(f"Element {element_id}: polyline contains infinite values")
        return False
    return True


def remove_duplicate_points(polyline, tol=1e-9):
    if len(polyline) < 2:
        return polyline
    diffs = np.diff(polyline, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    keep_mask = np.concatenate([[True], distances >= tol])
    return polyline[keep_mask]


def distance_based_interpolate(polyline, max_segment_length):
    if len(polyline) < 2:
        return polyline

    interpolated_points = [polyline[0]]

    for i in range(len(polyline) - 1):
        current_point = polyline[i]
        next_point = polyline[i + 1]

        segment_vector = next_point - current_point
        segment_length = np.linalg.norm(segment_vector)

        if segment_length > max_segment_length:
            num_subdivisions = int(np.ceil(segment_length / max_segment_length))
            for j in range(1, num_subdivisions):
                t = j / num_subdivisions
                intermediate_point = current_point + t * segment_vector
                interpolated_points.append(intermediate_point)

        interpolated_points.append(next_point)

    return np.array(interpolated_points, dtype=polyline.dtype)


def simplify_polyline(polyline, area_threshold, dist_threshold):
    # Visvalingam-Whyatt on (N,3) numpy array
    n = len(polyline)
    if n < 3:
        return polyline

    skip = [False] * n
    skip_changed = True

    while skip_changed:
        skip_changed = False
        k = 0
        while k < n - 1:
            k1 = k + 1
            while k1 < n - 1 and skip[k1]:
                k1 += 1
            if k1 >= n - 1:
                break

            k2 = k1 + 1
            while k2 < n and skip[k2]:
                k2 += 1
            if k2 >= n:
                break

            p1, p2, p3 = polyline[k], polyline[k1], polyline[k2]
            area = 0.5 * abs(
                (p1[0] - p3[0]) * (p2[1] - p1[1]) - (p1[0] - p2[0]) * (p3[1] - p1[1])
            )
            dist = np.linalg.norm(p1[:2] - p3[:2])

            if area < area_threshold and dist < dist_threshold:
                skip[k1] = True
                skip_changed = True
                k = k2
            else:
                k = k1

    return polyline[~np.array(skip)]
