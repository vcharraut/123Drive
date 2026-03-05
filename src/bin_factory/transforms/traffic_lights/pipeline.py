# Based on: Yan, X., Liang, E., Wang, J., Zhu, H., & Liu, H. X. (2025).
# "Improving Traffic Signal Data Quality for the Waymo Open Motion Dataset."
# arXiv:2506.07150v1. University of Michigan.
# https://github.com/michigan-traffic-lab/WOMD-Traffic-Signal-Data-Improvement

import numpy as np

from src.bin_factory import types
from src.bin_factory.transforms.traffic_lights.tlsgenerator import TLSGenerator
from src.bin_factory.transforms.traffic_lights.utils import (
    TLS,
    DetailedTLS,
    Direction,
    assign_veh_states_to_lane,
    group_lanes_into_ways,
    has_unprotected_left_turns,
)
from src.bin_factory.transforms.traffic_lights.utils.intersection import ApproachingLane, InJunctionLane


def detailed_to_scenario_state(detailed_tls):
    mapping = {
        DetailedTLS.ABSENT: types.TRAFFIC_LIGHT_UNKNOWN,
        DetailedTLS.UNKNOWN: types.TRAFFIC_LIGHT_UNKNOWN,
        DetailedTLS.ARROW_STOP: types.TRAFFIC_LIGHT_ARROW_RED,
        DetailedTLS.ARROW_CAUTION: types.TRAFFIC_LIGHT_ARROW_YELLOW,
        DetailedTLS.ARROW_GO: types.TRAFFIC_LIGHT_ARROW_GREEN,
        DetailedTLS.STOP: types.TRAFFIC_LIGHT_RED,
        DetailedTLS.CAUTION: types.TRAFFIC_LIGHT_YELLOW,
        DetailedTLS.GO: types.TRAFFIC_LIGHT_GREEN,
        DetailedTLS.FLASHING_STOP: types.TRAFFIC_LIGHT_FLASHING_RED,
        DetailedTLS.FLASHING_CAUTION: types.TRAFFIC_LIGHT_FLASHING_YELLOW,
    }
    return mapping.get(detailed_tls, types.TRAFFIC_LIGHT_UNKNOWN)


def generate_tl_states(scenario, lane_data, signalized):
    scenario_length = scenario["scenario_length"]
    dynamic_map_states = {}

    if not signalized:
        return dynamic_map_states

    lane_center_matrix, _lc_id_to_row, row_to_lc_id = _form_lanecenter_matrix(lane_data)
    veh_assignment = assign_veh_states_to_lane(
        scenario["agents"],
        lane_center_matrix,
        row_to_lc_id,
        end_step=scenario_length,
    )

    for intersection_ids in signalized:
        intersection = _form_intersection(lane_data, intersection_ids, veh_assignment, scenario_length)
        if len(intersection) not in (3, 4):
            continue

        tls_generator = TLSGenerator(scenario_length)
        tls_sequence = tls_generator.gen_tls_period(intersection, start_step=0, end_step=scenario_length)

        _collect_dynamic_states(intersection, tls_sequence, dynamic_map_states, scenario_length)

    return dynamic_map_states


def _form_lanecenter_matrix(lane_data):
    max_len = max(len(lane["polyline"]) for lane in lane_data.values())
    lane_center_matrix = np.full(
        (len(lane_data), max_len, 3),
        fill_value=np.inf,
        dtype=np.float64,
    )

    lc_id_to_idx = {f_id: i for i, f_id in enumerate(lane_data.keys())}
    idx_to_lc_id = {i: f_id for i, f_id in enumerate(lane_data.keys())}

    for id, lane in lane_data.items():
        polyline = lane["polyline"]
        lane_center_matrix[lc_id_to_idx[id], : len(polyline)] = polyline

    return lane_center_matrix, lc_id_to_idx, idx_to_lc_id


def _form_intersection(lane_data, injunction_ids, veh_assignment, scenario_length):
    incoming_ids = {
        entry_id
        for lc_id in injunction_ids
        for entry_id in lane_data[lc_id]["entry_lanes"]
        if entry_id not in injunction_ids
    }

    approaching_lanes = []
    for lc_id in incoming_ids:
        approaching = ApproachingLane(
            shape=lane_data[lc_id]["polyline"],
            record_vehs=veh_assignment[str(lc_id)],
            id=lc_id,
            length=scenario_length,
        )
        for in_j_id in lane_data[lc_id]["exit_lanes"]:
            injunction = InJunctionLane(
                shape=lane_data[in_j_id]["polyline"],
                record_tls=lane_data[in_j_id]["record_tls"],
                record_vehs=veh_assignment[str(in_j_id)],
                id=in_j_id,
                length=scenario_length,
            )
            approaching.injunction_lanes.append(injunction)
        approaching_lanes.append(approaching)

    return group_lanes_into_ways(approaching_lanes)


def _collect_dynamic_states(intersection, tls_sequence, dynamic_map_states, scenario_length):
    for t, tls_state in enumerate(tls_sequence):
        if tls_state is None:
            continue
        unprotected_left = has_unprotected_left_turns(tls_state)

        for i, approach in enumerate(intersection):
            for lane in approach:
                direction_set = {conn.direction for conn in lane.injunction_lanes}
                phase = next(p for p in tls_state[i] if list(direction_set)[0] in p)
                state = tls_state[i][phase]

                arrow_ever = any(
                    st in (DetailedTLS.ARROW_GO, DetailedTLS.ARROW_CAUTION, DetailedTLS.ARROW_STOP)
                    for conn in lane.injunction_lanes
                    for st in conn.record_tls_detailed
                )

                detailed_state = _reinterpret_state(direction_set, state, arrow_ever, unprotected_left)

                for conn in lane.injunction_lanes:
                    tl_id = int(conn.id)
                    if tl_id not in dynamic_map_states:
                        dynamic_map_states[tl_id] = {
                            "type": types.TRAFFIC_LIGHT,
                            "position": np.array(conn.shape[0]),
                            "states": [None] * scenario_length,
                            "controlled_lane": tl_id,
                        }
                    dynamic_map_states[tl_id]["states"][t] = detailed_to_scenario_state(detailed_state)


def _reinterpret_state(direction_set, state, arrow_ever, unprotected_left):
    def _general(s):
        return {TLS.RED: DetailedTLS.STOP, TLS.YELLOW: DetailedTLS.CAUTION, TLS.GREEN: DetailedTLS.GO}[s]

    if Direction.S in direction_set or direction_set == {Direction.R}:
        return _general(state)
    elif direction_set == {Direction.L}:
        return {
            TLS.RED: DetailedTLS.ARROW_STOP if arrow_ever else DetailedTLS.STOP,
            TLS.YELLOW: DetailedTLS.ARROW_CAUTION if arrow_ever else DetailedTLS.CAUTION,
            TLS.GREEN: DetailedTLS.ARROW_GO if arrow_ever and unprotected_left else DetailedTLS.GO,
        }[state]
    elif direction_set == {Direction.L, Direction.R}:
        return {
            TLS.RED: DetailedTLS.ARROW_STOP if arrow_ever else DetailedTLS.STOP,
            TLS.YELLOW: DetailedTLS.ARROW_CAUTION if arrow_ever else DetailedTLS.CAUTION,
            TLS.GREEN: DetailedTLS.ARROW_GO if arrow_ever else DetailedTLS.GO,
        }[state]
    assert False
