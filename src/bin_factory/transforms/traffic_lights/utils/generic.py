from enum import Enum

import numpy as np
from py123d.datatypes.detections import DefaultBoxDetectionLabel


class TLS(Enum):
    ABSENT = -1
    UNKNOWN = 0
    RED = 1
    YELLOW = 2
    GREEN = 3


class DetailedTLS(Enum):
    ABSENT = -1
    UNKNOWN = 0
    ARROW_STOP = 1
    ARROW_CAUTION = 2
    ARROW_GO = 3
    STOP = 4
    CAUTION = 5
    GO = 6
    FLASHING_STOP = 7
    FLASHING_CAUTION = 8

    def generalize(self):
        mapping = {
            DetailedTLS.ABSENT: TLS.ABSENT,
            DetailedTLS.UNKNOWN: TLS.UNKNOWN,
            DetailedTLS.ARROW_STOP: TLS.RED,
            DetailedTLS.ARROW_CAUTION: TLS.YELLOW,
            DetailedTLS.ARROW_GO: TLS.GREEN,
            DetailedTLS.STOP: TLS.RED,
            DetailedTLS.CAUTION: TLS.YELLOW,
            DetailedTLS.GO: TLS.GREEN,
            DetailedTLS.FLASHING_STOP: TLS.RED,
            DetailedTLS.FLASHING_CAUTION: TLS.YELLOW,
        }
        return mapping[self]


class Direction(Enum):
    L = 0
    S = 1
    R = 2


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size

    def find(self, p):
        if self.parent[p] != p:
            self.parent[p] = self.find(self.parent[p])
        return self.parent[p]

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)

        if rootP != rootQ:
            if self.rank[rootP] > self.rank[rootQ]:
                self.parent[rootQ] = rootP
            elif self.rank[rootP] < self.rank[rootQ]:
                self.parent[rootP] = rootQ
            else:
                self.parent[rootQ] = rootP
                self.rank[rootP] += 1

    def form_groups(self):
        root_to_elements = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in root_to_elements:
                root_to_elements[root] = []
            root_to_elements[root].append(i)
        return list(root_to_elements.values())


# Deferred imports to avoid circular dependencies
from .geometry import (  # noqa: E402
    angle_of_two_vectors,
    angle_of_twoheadings,
    group_vectors_by_angles,
    points_to_vector,
    vector_heading,
)
from .intersection import VehicleState  # noqa: E402


def group_lanes_into_ways(approaching_lanes):
    def _form_lane_vector(lane):
        start_point = lane.shape[-min(50, len(lane.shape))]
        end_point = lane.shape[-1]
        return points_to_vector(start_point, end_point)

    lane_vectors = [_form_lane_vector(lane) for lane in approaching_lanes]
    groups = group_vectors_by_angles(lane_vectors)
    ways = [[approaching_lanes[i] for i in group] for group in groups]

    if len(ways) == 3:
        angle01 = angle_of_two_vectors(_form_lane_vector(ways[1][0]), _form_lane_vector(ways[0][0]))
        angle02 = angle_of_two_vectors(_form_lane_vector(ways[2][0]), _form_lane_vector(ways[0][0]))
        angle12 = angle_of_two_vectors(_form_lane_vector(ways[2][0]), _form_lane_vector(ways[1][0]))
        if angle02 > angle01 and angle02 > angle12:
            ways = [ways[0], ways[2], ways[1]]
        elif angle12 > angle01 and angle12 > angle02:
            ways = [ways[1], ways[2], ways[0]]
    elif len(ways) == 4:
        ways.sort(key=lambda way: vector_heading(_form_lane_vector(way[0])), reverse=False)

    return ways


def has_unprotected_left_turns(tls_state):
    if len(tls_state) == 3:
        opposite_way_pairs = [(0, 1)]
    elif len(tls_state) == 4:
        opposite_way_pairs = [(0, 2), (1, 3)]
    else:
        opposite_way_pairs = []

    for i, j in opposite_way_pairs:
        phase_i_L = next((phase for phase in tls_state[i] if Direction.L in phase), None)
        phase_j_L = next((phase for phase in tls_state[j] if Direction.S in phase), None)
        if phase_i_L and phase_j_L and tls_state[i][phase_i_L] == tls_state[j][phase_j_L] == TLS.GREEN:
            return True
    return False


def assign_veh_states_to_lane(
    tracks,
    lane_center_matrix,
    row_to_lane_id=None,
    start_step=0,
    end_step=91,
    DISTANCE_CRITERIA=4,
    ANGLE_CRITERIA=np.pi / 12,
    ACCELERATION_MAXLIMIT=10,
):
    def _filter_criteria(row, heading):
        distance_criteria = min_distances[row] < DISTANCE_CRITERIA
        if min_dis_col[row] == 0:
            lane_start_point = lane_center_matrix[row][min_dis_col[row]]
            lane_end_point = lane_center_matrix[row][min_dis_col[row] + 1]
        else:
            lane_start_point = lane_center_matrix[row][min_dis_col[row] - 1]
            lane_end_point = lane_center_matrix[row][min_dis_col[row]]

        lane_vector = lane_end_point - lane_start_point
        lane_angle = vector_heading(lane_vector, unit="radian")
        angle_abs_diff = angle_of_twoheadings(heading, lane_angle, unit="radian")
        return distance_criteria and angle_abs_diff < ANGLE_CRITERIA

    veh_assignment_by_row = [[{} for _ in range(start_step, end_step)] for _ in range(lane_center_matrix.shape[0])]

    for _id, track in tracks.items():
        if track["type"] not in (DefaultBoxDetectionLabel.VEHICLE, DefaultBoxDetectionLabel.EGO):
            continue

        position = track["position"]
        valid = track["valid"]
        velocity = track["velocity"]
        heading = track["heading"]

        for tt in range(start_step, end_step):
            if not valid[tt]:
                continue

            veh_position = np.array([position[tt][0], position[tt][1], position[tt][2]])
            distances = np.linalg.norm(lane_center_matrix - veh_position, axis=2)
            min_distances = np.min(distances, axis=1)
            min_dis_col = np.argmin(distances, axis=1)

            candidate_rows = [row for row in range(min_distances.shape[0]) if _filter_criteria(row, heading[tt])]
            if not candidate_rows:
                continue

            best_row = candidate_rows[np.argmin([distances[row, min_dis_col[row]] for row in candidate_rows])]
            best_col = min_dis_col[best_row]

            absolute_speed = np.linalg.norm([velocity[tt][0], velocity[tt][1]])
            acceleration = 0
            j = tt - 5
            while j >= 0:
                if valid[j]:
                    absolute_speed_last = np.linalg.norm([velocity[j][0], velocity[j][1]])
                    acceleration = (absolute_speed - absolute_speed_last) / ((tt - j) / 10)
                    break
                j -= 1
            if abs(acceleration) > ACCELERATION_MAXLIMIT:
                acceleration = 0
            veh_assignment_by_row[best_row][tt][int(_id)] = VehicleState(
                int(_id),
                int(best_col),
                float(absolute_speed),
                float(acceleration),
            )

    if not row_to_lane_id:
        row_to_lane_id = {i: i for i in range(len(veh_assignment_by_row))}
    return {str(row_to_lane_id[row]): veh_assignment_by_row[row] for row in range(len(veh_assignment_by_row))}
