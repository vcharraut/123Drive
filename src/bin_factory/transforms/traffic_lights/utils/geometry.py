import math

import numpy as np

from .generic import Direction, UnionFind


def distance_between_points(pt1, pt2):
    return float(np.linalg.norm(np.asarray(pt1) - np.asarray(pt2)))


def two_lines_parallel(line1, line2, LINE_PARALLEL_THRESHOLD=15):
    vec1 = (line1[1][0] - line1[0][0], line1[1][1] - line1[0][1])
    vec2 = (line2[1][0] - line2[0][0], line2[1][1] - line2[0][1])
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]

    mag_vec1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
    mag_vec2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)

    if mag_vec1 == 0 or mag_vec2 == 0:
        return False
    cos_angle = max(min(dot_product / (mag_vec1 * mag_vec2), 1), -1)
    angle = math.acos(cos_angle) * (180 / math.pi)

    return angle < LINE_PARALLEL_THRESHOLD


def polyline_length(polyline):
    return sum(distance_between_points(polyline[i - 1], polyline[i]) for i in range(1, len(polyline)))


def real_neighbor_type(polyline1, polyline2, POINT_CLOSE_THRESHOLD=5, LENGTH_DIFFERENCE_THRESHOLD=3):
    start_close = distance_between_points(polyline1[0], polyline2[0]) < POINT_CLOSE_THRESHOLD
    end_close = distance_between_points(polyline1[-1], polyline2[-1]) < POINT_CLOSE_THRESHOLD
    len1 = polyline_length(polyline1)
    len2 = polyline_length(polyline2)
    longer = len2 + LENGTH_DIFFERENCE_THRESHOLD < len1

    if start_close and end_close:
        return "complete"
    elif start_close and not end_close and longer:
        return "side-start"
    elif not start_close and end_close and longer:
        return "side-end"
    return "other"


def calculate_turning_angle(points):
    vectors = np.diff(points, axis=0)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angle_diffs = np.diff(angles)
    angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
    return np.sum(angle_diffs)


def classify_direction(shape):
    total_turn_angle = calculate_turning_angle(shape)
    if np.abs(total_turn_angle) < np.pi / 6:
        return Direction.S
    elif total_turn_angle > 0:
        return Direction.L
    return Direction.R


def points_to_vector(pt_start, pt_end):
    return np.array([pt_end[0] - pt_start[0], pt_end[1] - pt_start[1]])


def vector_heading(vec, unit="radian"):
    if unit == "radian":
        return np.arctan2(vec[1], vec[0])
    elif unit == "degree":
        return np.rad2deg(np.arctan2(vec[1], vec[0]))
    raise ValueError(f"Unknown unit: {unit}")


def angle_of_two_vectors(vec1, vec2, unit="radian"):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        raise ValueError("One of the vectors is a zero vector.")

    cos_theta = np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0)

    if unit == "radian":
        return np.arccos(cos_theta)
    elif unit == "degree":
        return np.degrees(np.arccos(cos_theta))
    raise Exception()


def angle_of_twoheadings(angle1, angle2, unit="radian"):
    if unit == "degree":
        angle1, angle2 = np.deg2rad(angle1), np.deg2rad(angle2)

    diff = np.abs(angle1 - angle2)
    while diff > 2 * np.pi:
        diff -= np.pi * 2
    if diff > np.pi:
        diff = 2 * np.pi - diff

    if unit == "radian":
        return diff
    elif unit == "degree":
        return np.rad2deg(diff)


def group_vectors_by_angles(lane_vectors, ANGLE_CRITERIA=np.pi / 6):
    uf = UnionFind(len(lane_vectors))
    for i in range(len(lane_vectors)):
        for j in range(len(lane_vectors)):
            if angle_of_two_vectors(lane_vectors[i], lane_vectors[j]) < ANGLE_CRITERIA:
                uf.union(i, j)
    return uf.form_groups()
