from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field

import numpy as np

from bin_factory import puffer_types, schema


logger = logging.getLogger(__name__)


_SUPPORTED_DATASET = "wod-motion"
_LANE_TYPES = {puffer_types.LaneType.FREEWAY, puffer_types.LaneType.SURFACE_STREET}
_LANE_SHORT_THRESHOLD = 2.0
_POINT_CLOSE_THRESHOLD = 5.0
_LINE_PARALLEL_THRESHOLD = 15.0
_DISTANCE_CRITERIA = 4.0
_ANGLE_CRITERIA = np.pi / 12
_ACCELERATION_MAXLIMIT = 10.0
_TAIL_LANE_LENGTH_THRESHOLD = 8.0
_TAIL_ENTRY_LENGTH_THRESHOLD = 20.0
_TAIL_ALIGNMENT_THRESHOLD = np.pi / 9


class _TLS(enum.IntEnum):
    ABSENT = -1
    UNKNOWN = 0
    RED = 1
    YELLOW = 2
    GREEN = 3


def _copy_state(state):
    return [dict(d) for d in state]


class _Direction(enum.IntEnum):
    L = 0
    S = 1
    R = 2


@dataclass(slots=True)
class _VehicleState:
    lane_pos_idx: int
    speed: float
    acceleration: float


@dataclass(slots=True)
class _InJunctionLane:
    id: int
    shape: np.ndarray
    record_tls: list[_TLS]
    record_vehs: list[dict[int, _VehicleState]]
    direction: _Direction = field(init=False)
    new_tls: list[_TLS] = field(init=False)

    def __post_init__(self) -> None:
        self.direction = _classify_direction(self.shape[:, :2])
        self.new_tls = [_TLS.UNKNOWN for _ in self.record_tls]


@dataclass(slots=True)
class _ApproachingLane:
    id: int
    shape: np.ndarray
    record_vehs: list[dict[int, _VehicleState]]
    injunction_lanes: list[_InJunctionLane] = field(default_factory=list)


@dataclass(slots=True)
class _LaneRecord:
    id: int
    polyline: np.ndarray
    entry_lanes: list[int]
    exit_lanes: list[int]
    left_neighbors: list[int]
    right_neighbors: list[int]
    record_tls: list[_TLS]
    diverge_lanes: set[int] = field(default_factory=set)
    merge_lanes: set[int] = field(default_factory=set)


class _UnionFind:
    def __init__(self, items) -> None:
        self.parent = {item: item for item in items}
        self.rank = dict.fromkeys(items, 1)

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1

    def groups(self) -> list[list[int]]:
        root_to_items: dict[int, list[int]] = {}
        for item in self.parent:
            root_to_items.setdefault(self.find(item), []).append(item)
        return [sorted(items) for items in root_to_items.values()]


def impute_traffic_lights(scenario, extras) -> None:
    dataset = str(scenario.metadata.dataset)
    if dataset != _SUPPORTED_DATASET:
        logger.debug("Skipping traffic-light imputation for dataset %s", dataset)
        return
    if scenario.metadata.scenario_length <= 0:
        return

    imputer = _TrafficLightImputer(scenario, extras)
    imputer.impute()


class _TrafficLightImputer:
    def __init__(self, scenario, extras) -> None:
        self.scenario = scenario
        self.extras = extras
        self.length = scenario.metadata.scenario_length
        self.dt = max(float(scenario.metadata.dt), 1e-3)
        self.lanes = self._build_lanes()

    def impute(self) -> None:
        if len(self.lanes) < 4:
            return

        self._clean_lanes()
        signalized_intersections = self._find_signalized_intersections()
        if not signalized_intersections:
            logger.debug("Scenario %s: no signalized intersections found", self.scenario.metadata.id)
            return

        lane_center_matrix, row_to_lane_id = self._form_lane_center_matrix()
        veh_assignment = _assign_vehicle_states_to_lanes(
            self.scenario.agents,
            lane_center_matrix,
            row_to_lane_id,
            self.dt,
            self.length,
        )

        traffic_lights = dict(self.extras.get("traffic_lights", {}))
        generator = _TLSGenerator(self.length)
        updated_lanes = 0

        for intersection_ids in signalized_intersections:
            intersection = self._form_intersection(intersection_ids, veh_assignment)
            if len(intersection) not in (3, 4):
                continue
            tls_sequence = generator.gen_period(intersection)
            if not tls_sequence:
                continue
            updated_lanes += self._write_generated_states(intersection, tls_sequence, traffic_lights)

        if updated_lanes == 0:
            logger.debug("Scenario %s: traffic-light imputation produced no updates", self.scenario.metadata.id)
            return

        self.extras["traffic_lights"] = traffic_lights
        logger.info(
            "Scenario %s: imputed traffic lights for %d lanes across %d intersections",
            self.scenario.metadata.id,
            updated_lanes,
            len(signalized_intersections),
        )

    def _build_lanes(self) -> dict[int, _LaneRecord]:
        lanes: dict[int, _LaneRecord] = {}
        for lane_id, element in self.scenario.map.items():
            if element.get("type") not in _LANE_TYPES:
                continue
            polyline = _as_xyz_array(element.get("polyline"))
            if len(polyline) < 2:
                continue
            lanes[int(lane_id)] = _LaneRecord(
                id=int(lane_id),
                polyline=polyline,
                entry_lanes=[int(ref) for ref in element.get("entry_lanes", [])],
                exit_lanes=[int(ref) for ref in element.get("exit_lanes", [])],
                left_neighbors=[int(ref) for ref in element.get("left_neighbor", [])],
                right_neighbors=[int(ref) for ref in element.get("right_neighbor", [])],
                record_tls=[_TLS.ABSENT for _ in range(self.length)],
            )

        for traffic_light in self.extras.get("traffic_lights", {}).values():
            lane_id = int(traffic_light.controlled_lane)
            lane = lanes.get(lane_id)
            if lane is None:
                continue
            for idx, state in enumerate(traffic_light.states[: self.length]):
                lane.record_tls[idx] = _from_puffer_tls(state)

        return lanes

    def _clean_lanes(self) -> None:
        to_delete = [
            lane_id
            for lane_id, lane in self.lanes.items()
            if (len(lane.entry_lanes) == 0 or len(lane.exit_lanes) == 0)
            and _polyline_length(lane.polyline) < _LANE_SHORT_THRESHOLD
        ]
        for lane_id in to_delete:
            del self.lanes[lane_id]

        for lane in self.lanes.values():
            lane.entry_lanes = [
                ref
                for ref in lane.entry_lanes
                if ref in self.lanes
                and _distance(lane.polyline[0], self.lanes[ref].polyline[-1]) < _POINT_CLOSE_THRESHOLD
            ]
            lane.exit_lanes = [
                ref
                for ref in lane.exit_lanes
                if ref in self.lanes
                and _distance(lane.polyline[-1], self.lanes[ref].polyline[0]) < _POINT_CLOSE_THRESHOLD
            ]

        for lane in self.lanes.values():
            lane.left_neighbors = self._clean_neighbors(lane, lane.left_neighbors)
            lane.right_neighbors = self._clean_neighbors(lane, lane.right_neighbors)

        for lane in self.lanes.values():
            lane.left_neighbors = [ref for ref in lane.left_neighbors if lane.id in self.lanes[ref].right_neighbors]
            lane.right_neighbors = [ref for ref in lane.right_neighbors if lane.id in self.lanes[ref].left_neighbors]

        for lane in self.lanes.values():
            for left in lane.entry_lanes:
                for right in lane.entry_lanes:
                    if left != right and left in self.lanes and right in self.lanes:
                        self.lanes[left].merge_lanes.add(right)
            for left in lane.exit_lanes:
                for right in lane.exit_lanes:
                    if left != right and left in self.lanes and right in self.lanes:
                        self.lanes[left].diverge_lanes.add(right)

        for lane in self.lanes.values():
            lane.diverge_lanes = {ref for ref in lane.diverge_lanes if lane.id in self.lanes[ref].diverge_lanes}
            lane.merge_lanes = {ref for ref in lane.merge_lanes if lane.id in self.lanes[ref].merge_lanes}

    def _clean_neighbors(self, lane: _LaneRecord, neighbors: list[int]) -> list[int]:
        cleaned: list[int] = []
        for neighbor_id in neighbors:
            neighbor = self.lanes.get(neighbor_id)
            if neighbor is None:
                continue
            neighbor_type = _neighbor_type(lane.polyline, neighbor.polyline)
            if neighbor_type in {"real", "bifurcated-parallel", "merged-parallel"}:
                cleaned.append(neighbor_id)
            if neighbor_type in {"bifurcated", "bifurcated-parallel"}:
                lane.diverge_lanes.add(neighbor_id)
            if neighbor_type in {"merged", "merged-parallel"}:
                lane.merge_lanes.add(neighbor_id)
        return cleaned

    def _find_signalized_intersections(self) -> list[list[int]]:
        if not self.lanes:
            return []

        def is_connection_group(group: list[int]) -> bool:
            return len(group) > 1 and any(
                self.lanes[lane_id].diverge_lanes or self.lanes[lane_id].merge_lanes for lane_id in group
            )

        union_find = _UnionFind(self.lanes)
        for lane in self.lanes.values():
            for neighbor_id in lane.left_neighbors + lane.right_neighbors:
                if neighbor_id not in self.lanes:
                    continue
                if _real_neighbor_type(lane.polyline, self.lanes[neighbor_id].polyline) == "complete":
                    union_find.union(lane.id, neighbor_id)
            for ref in lane.diverge_lanes | lane.merge_lanes:
                if ref in self.lanes:
                    union_find.union(lane.id, ref)

        connection_groups = [group for group in union_find.groups() if is_connection_group(group)]
        internal_lanes = {lane_id for group in connection_groups for lane_id in group}

        for lane_id, lane in self.lanes.items():
            if lane_id in internal_lanes or not lane.entry_lanes or not lane.exit_lanes:
                continue
            if lane.entry_lanes[0] in self.lanes and lane.exit_lanes[0] in self.lanes:
                if union_find.find(lane.entry_lanes[0]) == union_find.find(lane.exit_lanes[0]):
                    union_find.union(lane_id, lane.entry_lanes[0])
                    union_find.union(lane_id, lane.exit_lanes[0])

        signalized = []
        for group in union_find.groups():
            if not is_connection_group(group):
                continue
            if len(group) < 4:
                continue
            if not any(any(state != _TLS.ABSENT for state in self.lanes[lane_id].record_tls) for lane_id in group):
                continue
            invalid_entry = any(
                len(self.lanes[lane_id].entry_lanes) > 1
                and not any(entry_lane in group for entry_lane in self.lanes[lane_id].entry_lanes)
                for lane_id in group
            )
            if not invalid_entry:
                signalized.append(group)

        return signalized

    def _form_lane_center_matrix(self) -> tuple[np.ndarray, dict[int, int]]:
        lane_ids = list(self.lanes)
        max_points = max(len(self.lanes[lane_id].polyline) for lane_id in lane_ids)
        matrix = np.full((len(lane_ids), max_points, 3), np.inf, dtype=np.float64)
        row_to_lane_id = {}
        for row, lane_id in enumerate(lane_ids):
            polyline = self.lanes[lane_id].polyline
            matrix[row, : len(polyline)] = polyline
            row_to_lane_id[row] = lane_id
        return matrix, row_to_lane_id

    def _form_intersection(
        self,
        intersection_ids: list[int],
        veh_assignment: dict[int, list[dict[int, _VehicleState]]],
    ) -> list[list[_ApproachingLane]]:
        internal_ids = set(intersection_ids)
        incoming_ids = sorted(
            {
                entry_lane
                for lane_id in internal_ids
                for entry_lane in self.lanes[lane_id].entry_lanes
                if entry_lane not in internal_ids
            }
        )

        approaching_lanes = []
        for lane_id in incoming_ids:
            lane = self.lanes.get(lane_id)
            if lane is None:
                continue
            approaching = _ApproachingLane(
                id=lane_id,
                shape=lane.polyline,
                record_vehs=veh_assignment.get(lane_id, [{} for _ in range(self.length)]),
            )
            for next_lane_id in lane.exit_lanes:
                if next_lane_id not in internal_ids or next_lane_id not in self.lanes:
                    continue
                approaching.injunction_lanes.append(
                    _InJunctionLane(
                        id=next_lane_id,
                        shape=self.lanes[next_lane_id].polyline,
                        record_tls=list(self.lanes[next_lane_id].record_tls),
                        record_vehs=veh_assignment.get(next_lane_id, [{} for _ in range(self.length)]),
                    )
                )
            if approaching.injunction_lanes:
                approaching_lanes.append(approaching)

        return _group_lanes_into_ways(approaching_lanes)

    def _write_generated_states(
        self,
        intersection: list[list[_ApproachingLane]],
        tls_sequence: list[list[dict[tuple[_Direction, ...], _TLS]]],
        traffic_lights: dict[int, schema.TrafficLightTrack],
    ) -> int:
        updated_lanes: set[int] = set()
        for timestep, tls_state in enumerate(tls_sequence):
            for way_idx, approach in enumerate(intersection):
                if not approach:
                    continue
                for lane in approach:
                    directions = {conn.direction for conn in lane.injunction_lanes}
                    phase = next(
                        (candidate for candidate in tls_state[way_idx] if directions.issubset(set(candidate))),
                        None,
                    )
                    if phase is None:
                        phase = next(
                            candidate
                            for candidate in tls_state[way_idx]
                            if any(direction in candidate for direction in directions)
                        )
                    state = tls_state[way_idx][phase]
                    for conn in lane.injunction_lanes:
                        conn.new_tls[timestep] = state
                        track = traffic_lights.get(conn.id)
                        if track is None:
                            if self._is_tail_lane(conn.id):
                                continue
                        if track is None or len(track.states) != self.length:
                            track = schema.TrafficLightTrack(
                                position=conn.shape[0].astype(np.float64, copy=True),
                                states=[puffer_types.TLState.UNKNOWN] * self.length,
                                controlled_lane=conn.id,
                            )
                            traffic_lights[conn.id] = track
                        track.position = conn.shape[0].astype(np.float64, copy=True)
                        track.states[timestep] = _to_puffer_tls(state)
                        updated_lanes.add(conn.id)
        return len(updated_lanes)

    def _is_tail_lane(self, lane_id: int) -> bool:
        lane = self.lanes.get(lane_id)
        if lane is None or _polyline_length(lane.polyline) >= _TAIL_LANE_LENGTH_THRESHOLD:
            return False
        if len(lane.entry_lanes) != 1:
            return False

        entry_lane = self.lanes.get(lane.entry_lanes[0])
        if entry_lane is None or len(entry_lane.exit_lanes) != 1:
            return False
        if _polyline_length(entry_lane.polyline) <= _TAIL_ENTRY_LENGTH_THRESHOLD:
            return False

        lane_vector = lane.polyline[-1, :2] - lane.polyline[0, :2]
        entry_vector = entry_lane.polyline[-1, :2] - entry_lane.polyline[0, :2]
        if np.linalg.norm(lane_vector) < 1e-6 or np.linalg.norm(entry_vector) < 1e-6:
            return False
        return (
            _angle_of_headings(np.arctan2(lane_vector[1], lane_vector[0]), np.arctan2(entry_vector[1], entry_vector[0]))
            < _TAIL_ALIGNMENT_THRESHOLD
        )


class _TLSGenerator:
    def __init__(self, horizon: int, delta_t: int = 10, smoothing_width: int = 30, yellow_duration: int = 20) -> None:
        self.horizon = horizon
        self.v_green = 3.0
        self.v_red = 1.0
        self.a_green = 0.5
        self.a_red = -1.0
        self.delta_t = delta_t
        self.theta = 0.8
        self.w_big = 100.0
        self.w_small = 0.1
        self.smoothing_width = smoothing_width
        self.yellow_duration = yellow_duration
        self.container_template: list[dict[tuple[_Direction, ...], _TLS | None]] = []

    def gen_period(
        self,
        intersection: list[list[_ApproachingLane]],
        start_step: int = 0,
        end_step: int | None = None,
    ) -> list[list[dict[tuple[_Direction, ...], _TLS]]]:
        if len(intersection) not in (2, 3, 4) or self.horizon == 0:
            return []

        end_step = self.horizon if end_step is None else min(end_step, self.horizon)
        if end_step <= start_step:
            return []

        generated_steps = list(range(max(start_step, self.delta_t), min(end_step, self.horizon - self.delta_t)))
        tl_state_buff: list[list[dict[tuple[_Direction, ...], _TLS]] | None] = [None for _ in range(self.horizon)]

        if not generated_steps:
            step = min(max(start_step, 0), self.horizon - 1)
            state = self.gen_one_moment(intersection, step)
            return [_copy_state(state) for _ in range(self.horizon)]

        prev_state = None
        for step in generated_steps:
            current = self.gen_one_moment(intersection, step, prev_state)
            tl_state_buff[step] = current
            prev_state = current

        first_step = generated_steps[0]
        last_step = generated_steps[-1]
        for step in range(start_step, first_step):
            tl_state_buff[step] = _copy_state(tl_state_buff[first_step])
        for step in range(first_step, last_step + 1):
            if tl_state_buff[step] is None:
                tl_state_buff[step] = _copy_state(tl_state_buff[step - 1])
        for step in range(last_step + 1, end_step):
            tl_state_buff[step] = _copy_state(tl_state_buff[last_step])
        for step in range(start_step):
            tl_state_buff[step] = _copy_state(tl_state_buff[first_step])
        for step in range(end_step, self.horizon):
            tl_state_buff[step] = _copy_state(tl_state_buff[last_step])

        final = [state for state in tl_state_buff if state is not None]
        final[start_step:end_step] = self._smooth_sequence(final[start_step:end_step])
        final[start_step:end_step] = self._add_yellow_light(final[start_step:end_step])
        return final

    def gen_one_moment(
        self,
        intersection: list[list[_ApproachingLane]],
        curr_step: int,
        prev_state: list[dict[tuple[_Direction, ...], _TLS]] | None = None,
    ) -> list[dict[tuple[_Direction, ...], _TLS]]:
        self.container_template = self._gen_state_container(intersection)
        raw_state = self._derive_raw_state(intersection, curr_step)
        estimated_state, confidence = self._derive_estimated_state(intersection, curr_step)
        imputed_state, weight = self._derive_imputed_state(raw_state, estimated_state, confidence)
        if len(intersection) == 2:
            return imputed_state

        feasible_states = self._get_feasible_states()
        candidate_states = self._score_candidate_states(feasible_states, imputed_state, weight)
        if prev_state in candidate_states:
            return _copy_state(prev_state)
        return self._fill_right_turn_signal(_copy_state(candidate_states[0]))

    def _gen_state_container(
        self,
        intersection: list[list[_ApproachingLane]],
    ) -> list[dict[tuple[_Direction, ...], _TLS | None]]:
        state_container = [None for _ in intersection]
        for index, approach in enumerate(intersection):
            movements = {conn.direction for lane in approach for conn in lane.injunction_lanes}
            union_find = _UnionFind(range(3))
            for lane in approach:
                for conn_i in lane.injunction_lanes:
                    for conn_j in lane.injunction_lanes:
                        union_find.union(conn_i.direction.value, conn_j.direction.value)
            phases = [
                tuple(_Direction(direction) for direction in group)
                for group in union_find.groups()
                if all(_Direction(direction) in movements for direction in group)
            ]
            state_container[index] = dict.fromkeys(phases)
        return state_container

    def _derive_raw_state(
        self,
        intersection: list[list[_ApproachingLane]],
        curr_step: int,
    ) -> list[dict[tuple[_Direction, ...], _TLS]]:
        raw_state = _copy_state(self.container_template)
        for index, approach in enumerate(intersection):
            for lane in approach:
                for conn in lane.injunction_lanes:
                    phase = next(candidate for candidate in raw_state[index] if conn.direction in candidate)
                    for step in range(curr_step, max(0, curr_step - 10) - 1, -1):
                        state = conn.record_tls[step]
                        if state in {_TLS.ABSENT, _TLS.UNKNOWN} or raw_state[index][phase] is not None:
                            continue
                        raw_state[index][phase] = _TLS.GREEN if state == _TLS.YELLOW else _TLS(state)
        return raw_state

    def _derive_estimated_state(
        self,
        intersection: list[list[_ApproachingLane]],
        curr_step: int,
    ) -> tuple[list[dict[tuple[_Direction, ...], _TLS]], list[dict[tuple[_Direction, ...], float]]]:
        estimated_state = _copy_state(self.container_template)
        confidence = _copy_state(self.container_template)

        for index, approach in enumerate(intersection):
            for phase in estimated_state[index]:
                if phase == (_Direction.R,):
                    continue
                mean_acc, mean_spd, sum_f, sum_g, must_green = self._get_traj_metrics_at_phase(
                    approach,
                    phase,
                    curr_step,
                )
                if must_green:
                    estimated_state[index][phase] = _TLS.GREEN
                    confidence[index][phase] = self.w_big
                    continue
                if sum_g >= self.theta:
                    if mean_spd >= self.v_green:
                        estimated_state[index][phase] = _TLS.GREEN
                        confidence[index][phase] = sum_g
                    elif mean_spd <= self.v_red:
                        estimated_state[index][phase] = _TLS.RED
                        confidence[index][phase] = sum_g
                if sum_f >= self.theta:
                    if mean_acc >= self.a_green:
                        estimated_state[index][phase] = _TLS.GREEN
                        confidence[index][phase] = sum_f
                    elif mean_acc <= self.a_red:
                        estimated_state[index][phase] = _TLS.RED
                        confidence[index][phase] = sum_f

        return estimated_state, confidence

    def _derive_imputed_state(
        self,
        raw_state: list[dict[tuple[_Direction, ...], _TLS | None]],
        estimated_state: list[dict[tuple[_Direction, ...], _TLS | None]],
        confidence: list[dict[tuple[_Direction, ...], float]],
    ) -> tuple[list[dict[tuple[_Direction, ...], _TLS]], list[dict[tuple[_Direction, ...], float]]]:
        imputed_state = _copy_state(self.container_template)
        weight = _copy_state(self.container_template)
        for index in range(len(raw_state)):
            for phase in raw_state[index]:
                raw_none = raw_state[index][phase] is None
                estimated_none = estimated_state[index][phase] is None
                if raw_none and estimated_none:
                    weight[index][phase] = 0.0
                elif raw_none:
                    imputed_state[index][phase] = estimated_state[index][phase]
                    weight[index][phase] = confidence[index][phase]
                elif estimated_none:
                    imputed_state[index][phase] = raw_state[index][phase]
                    weight[index][phase] = self.w_small
                elif raw_state[index][phase] == estimated_state[index][phase]:
                    imputed_state[index][phase] = estimated_state[index][phase]
                    weight[index][phase] = self.w_big
                elif confidence[index][phase] >= self.theta:
                    imputed_state[index][phase] = estimated_state[index][phase]
                    weight[index][phase] = confidence[index][phase]
                else:
                    imputed_state[index][phase] = raw_state[index][phase]
                    weight[index][phase] = 0.0
        return imputed_state, weight

    def _get_feasible_states(self) -> list[list[dict[tuple[_Direction, ...], _TLS]]]:
        candidate_states = []
        if len(self.container_template) == 4:
            for green_way in range(4):
                candidate = _copy_state(self.container_template)
                for index in range(4):
                    for phase in candidate[index]:
                        candidate[index][phase] = _TLS.GREEN if index == green_way else _TLS.RED
                candidate_states.append(candidate)

            for green_group in ([0, 2], [1, 3]):
                candidate = _copy_state(self.container_template)
                for index in range(4):
                    for phase in candidate[index]:
                        candidate[index][phase] = _TLS.GREEN if index in green_group else _TLS.RED
                candidate_states.append(candidate)

                split_left = not any(
                    any(_Direction.L in phase and _Direction.S in phase for phase in self.container_template[index])
                    for index in green_group
                )
                if not split_left:
                    continue

                candidate = _copy_state(self.container_template)
                for index in range(4):
                    for phase in candidate[index]:
                        if index not in green_group:
                            candidate[index][phase] = _TLS.RED
                        elif _Direction.L in phase:
                            candidate[index][phase] = _TLS.GREEN
                        else:
                            candidate[index][phase] = _TLS.RED
                candidate_states.append(candidate)

                candidate = _copy_state(self.container_template)
                for index in range(4):
                    for phase in candidate[index]:
                        if index not in green_group or _Direction.L in phase:
                            candidate[index][phase] = _TLS.RED
                        else:
                            candidate[index][phase] = _TLS.GREEN
                candidate_states.append(candidate)

        elif len(self.container_template) == 3:
            for green_way in range(3):
                candidate = _copy_state(self.container_template)
                for index in range(3):
                    for phase in candidate[index]:
                        candidate[index][phase] = _TLS.GREEN if index == green_way else _TLS.RED
                candidate_states.append(candidate)

            split_left = not any(
                any(_Direction.S in phase and _Direction.L in phase for phase in self.container_template[index])
                for index in (0, 1)
            )
            if split_left:
                candidate = _copy_state(self.container_template)
                for phase in candidate[2]:
                    candidate[2][phase] = _TLS.RED
                for index in (0, 1):
                    for phase in candidate[index]:
                        candidate[index][phase] = _TLS.RED if _Direction.L in phase else _TLS.GREEN
                candidate_states.append(candidate)

                candidate = _copy_state(self.container_template)
                for phase in candidate[2]:
                    candidate[2][phase] = _TLS.RED
                for index in (0, 1):
                    for phase in candidate[index]:
                        candidate[index][phase] = _TLS.GREEN if _Direction.L in phase else _TLS.RED
                candidate_states.append(candidate)

            candidate = _copy_state(self.container_template)
            for index in range(3):
                for phase in candidate[index]:
                    candidate[index][phase] = _TLS.RED if index == 2 else _TLS.GREEN
            candidate_states.append(candidate)

        return candidate_states

    def _score_candidate_states(
        self,
        feasible_states: list[list[dict[tuple[_Direction, ...], _TLS]]],
        imputed_state: list[dict[tuple[_Direction, ...], _TLS | None]],
        weight: list[dict[tuple[_Direction, ...], float]],
    ) -> list[list[dict[tuple[_Direction, ...], _TLS]]]:
        def match_score(candidate) -> float:
            return sum(
                weight[index][phase]
                for index in range(len(imputed_state))
                for phase in imputed_state[index]
                if imputed_state[index][phase] is not None and imputed_state[index][phase] == candidate[index][phase]
            )

        def conflict_score(candidate) -> float:
            return sum(
                weight[index][phase]
                for index in range(len(imputed_state))
                for phase in imputed_state[index]
                if imputed_state[index][phase] is not None and imputed_state[index][phase] != candidate[index][phase]
            )

        scores = [
            (index, match_score(candidate), conflict_score(candidate))
            for index, candidate in enumerate(feasible_states)
        ]
        best_match = max(scores, key=lambda item: item[1])[1]
        scores = [score for score in scores if score[1] == best_match]
        lowest_conflict = min(scores, key=lambda item: item[2])[2]
        return [feasible_states[index] for index, _, conflict in scores if conflict == lowest_conflict]

    def _fill_right_turn_signal(
        self,
        state: list[dict[tuple[_Direction, ...], _TLS]],
    ) -> list[dict[tuple[_Direction, ...], _TLS]]:
        for lane_state in state:
            if (_Direction.R,) not in lane_state:
                continue
            straight_phase = next((phase for phase in lane_state if _Direction.S in phase), None)
            left_phase = next((phase for phase in lane_state if _Direction.L in phase), None)
            if straight_phase is not None:
                lane_state[(_Direction.R,)] = lane_state[straight_phase]
            elif left_phase is not None:
                lane_state[(_Direction.R,)] = lane_state[left_phase]
        return state

    def _get_traj_metrics_at_phase(
        self,
        approach: list[_ApproachingLane],
        phase: tuple[_Direction, ...],
        curr_step: int,
    ) -> tuple[float, float, float, float, bool]:
        selected_lanes = [lane for lane in approach if any(conn.direction in phase for conn in lane.injunction_lanes)]
        trajectories: dict[int, list[tuple[int, float, float]]] = {}

        def append_record(veh_id: int, pos_idx: int, speed: float, acceleration: float) -> None:
            trajectories.setdefault(veh_id, []).append((pos_idx, speed, acceleration))

        start = max(0, curr_step - self.delta_t)
        end = min(self.horizon, curr_step + self.delta_t + 1)
        for timestep in range(start, end):
            for lane in selected_lanes:
                for veh_id, record in lane.record_vehs[timestep].items():
                    pos_idx = len(lane.shape) - record.lane_pos_idx - 1
                    append_record(veh_id, pos_idx, record.speed, record.acceleration)

                has_right_turn = any(conn.direction == _Direction.R for conn in lane.injunction_lanes)
                for conn in lane.injunction_lanes:
                    for veh_id, record in conn.record_vehs[timestep].items():
                        pos_idx = -record.lane_pos_idx
                        append_record(veh_id, pos_idx, record.speed, record.acceleration)
                        if not has_right_turn and abs(timestep - curr_step) <= 2:
                            if 0 <= record.lane_pos_idx < 10 and record.speed > 0:
                                return 0.0, 0.0, 0.0, 0.0, True

        if not trajectories:
            return 0.0, 0.0, 0.0, 0.0, False

        per_vehicle = {}
        for veh_id, records in trajectories.items():
            pos_idx, speeds, accelerations = zip(*records)
            f_values = [self._f(distance, acc) for distance, acc in zip(pos_idx, accelerations, strict=False)]
            g_values = [self._g(distance, speed) for distance, speed in zip(pos_idx, speeds, strict=False)]
            per_vehicle[veh_id] = (
                np.average(accelerations, weights=f_values) if np.sum(f_values) else 0.0,
                np.average(speeds, weights=g_values) if np.sum(g_values) else 0.0,
                float(np.max(f_values)) if f_values else 0.0,
                float(np.max(g_values)) if g_values else 0.0,
            )

        accelerations, speeds, f_values, g_values = zip(*per_vehicle.values(), strict=False)
        filtered_f = [value for value in f_values if value]
        filtered_acc = [accelerations[idx] for idx, value in enumerate(f_values) if value]
        filtered_g = [value for value in g_values if value]
        filtered_speed = [speeds[idx] for idx, value in enumerate(g_values) if value]

        sum_f = np.log1p(np.sum(filtered_f))
        mean_acc = np.average(filtered_acc, weights=filtered_f) if filtered_f else 0.0
        sum_g = np.log1p(np.sum(filtered_g))
        mean_spd = np.average(filtered_speed, weights=filtered_g) if filtered_g else 0.0
        return float(mean_acc), float(mean_spd), float(sum_f), float(sum_g), False

    @staticmethod
    def _f(index: int, acceleration: float) -> float:
        distance = index * 0.5
        if distance < -8 or (acceleration < 0 and distance < 0):
            return 0.0
        if distance <= 15:
            return 1.0
        if distance >= 30:
            return 0.0
        return ((distance - 30) ** 2) / (15 * 15)

    @staticmethod
    def _g(index: int, speed: float) -> float:
        distance = index * 0.5
        if distance < -12:
            return 0.0

        if speed <= 12:
            distance_limit = ((15 - 6) / (6 * 6)) * (speed - 6) ** 2 + 6
        else:
            distance_limit = min(speed - 12 + 15, 30)

        if distance > 2 * distance_limit:
            return 0.0
        if distance <= distance_limit:
            return 1.0
        return ((distance - 2 * distance_limit) ** 2) / (distance_limit * distance_limit)

    def _smooth_sequence(
        self,
        tl_state_buff: list[list[dict[tuple[_Direction, ...], _TLS]]],
    ) -> list[list[dict[tuple[_Direction, ...], _TLS]]]:
        intervals: set[tuple[int, int]] = set()
        for way_idx in range(len(self.container_template)):
            for phase in self.container_template[way_idx]:
                intervals.update(self._find_short_intervals(tl_state_buff, way_idx, phase))

        for start, end in sorted(intervals):
            for step in range(start, end + 1):
                tl_state_buff[step] = _copy_state(tl_state_buff[start - 1])

        return tl_state_buff

    def _find_short_intervals(
        self,
        tl_state_buff: list[list[dict[tuple[_Direction, ...], _TLS]]],
        way_idx: int,
        phase: tuple[_Direction, ...],
    ) -> list[tuple[int, int]]:
        intervals = []
        index = 0
        while index < len(tl_state_buff):
            current = tl_state_buff[index][way_idx][phase]
            if current in {_TLS.GREEN, _TLS.RED}:
                other = _TLS.RED if current == _TLS.GREEN else _TLS.GREEN
                next_index = index + 1
                while next_index < len(tl_state_buff) and tl_state_buff[next_index][way_idx][phase] == other:
                    next_index += 1
                span = next_index - index - 1
                if next_index < len(tl_state_buff) and 0 < span < self.smoothing_width:
                    if tl_state_buff[next_index][way_idx][phase] == current:
                        intervals.append((index + 1, next_index - 1))
                index = next_index
            else:
                index += 1
        return intervals

    def _add_yellow_light(
        self,
        tl_state_buff: list[list[dict[tuple[_Direction, ...], _TLS]]],
    ) -> list[list[dict[tuple[_Direction, ...], _TLS]]]:
        for way_idx in range(len(self.container_template)):
            for phase in self.container_template[way_idx]:
                red_indices = [
                    step
                    for step in range(1, len(tl_state_buff))
                    if tl_state_buff[step][way_idx][phase] == _TLS.RED
                    and tl_state_buff[step - 1][way_idx][phase] == _TLS.GREEN
                ]
                for step in red_indices:
                    for yellow_step in range(step - self.yellow_duration, step):
                        if 0 <= yellow_step < len(tl_state_buff):
                            tl_state_buff[yellow_step][way_idx][phase] = _TLS.YELLOW
        return tl_state_buff


def _assign_vehicle_states_to_lanes(
    tracks,
    lane_center_matrix: np.ndarray,
    row_to_lane_id: dict[int, int],
    dt: float,
    horizon: int,
) -> dict[int, list[dict[int, _VehicleState]]]:
    assignments_by_row = [[{} for _ in range(horizon)] for _ in range(len(row_to_lane_id))]

    for track_id, track in tracks.items():
        if track.type != puffer_types.AgentType.VEHICLE:
            continue
        positions = np.asarray(track.position, dtype=np.float64)
        headings = np.asarray(track.heading, dtype=np.float64)
        velocities = np.asarray(track.velocity, dtype=np.float64)
        valid = np.asarray(track.valid, dtype=bool)

        for timestep in range(horizon):
            if not valid[timestep]:
                continue
            position = positions[timestep]
            distances = np.linalg.norm(lane_center_matrix - position, axis=2)
            min_distances = np.min(distances, axis=1)
            min_columns = np.argmin(distances, axis=1)
            candidate_rows = [
                row
                for row in range(len(row_to_lane_id))
                if _vehicle_matches_lane(
                    lane_center_matrix[row],
                    int(min_columns[row]),
                    float(min_distances[row]),
                    float(headings[timestep]),
                )
            ]
            if not candidate_rows:
                continue

            best_row = min(candidate_rows, key=lambda row: distances[row, min_columns[row]])
            best_col = int(min_columns[best_row])
            speed = float(np.linalg.norm(velocities[timestep, :2]))
            acceleration = 0.0
            for prev_step in range(timestep - 1, max(-1, timestep - 6), -1):
                if prev_step < 0 or not valid[prev_step]:
                    continue
                prev_speed = float(np.linalg.norm(velocities[prev_step, :2]))
                delta_t = (timestep - prev_step) * dt
                if delta_t > 0:
                    acceleration = (speed - prev_speed) / delta_t
                break
            if abs(acceleration) > _ACCELERATION_MAXLIMIT:
                acceleration = 0.0
            assignments_by_row[best_row][timestep][int(track_id)] = _VehicleState(best_col, speed, acceleration)

    return {row_to_lane_id[row]: assignments_by_row[row] for row in row_to_lane_id}


def _vehicle_matches_lane(
    lane_points: np.ndarray,
    min_column: int,
    min_distance: float,
    heading: float,
) -> bool:
    if min_distance >= _DISTANCE_CRITERIA:
        return False
    if min_column == 0:
        start = lane_points[min_column]
        end = lane_points[min_column + 1]
    else:
        start = lane_points[min_column - 1]
        end = lane_points[min_column]
    if not np.isfinite(start).all() or not np.isfinite(end).all():
        return False
    lane_heading = _vector_heading(end[:2] - start[:2])
    return _angle_of_headings(heading, lane_heading) < _ANGLE_CRITERIA


def _group_lanes_into_ways(approaching_lanes: list[_ApproachingLane]) -> list[list[_ApproachingLane]]:
    if not approaching_lanes:
        return []

    def lane_vector(lane: _ApproachingLane) -> np.ndarray:
        start = lane.shape[-min(50, len(lane.shape))]
        end = lane.shape[-1]
        return end[:2] - start[:2]

    vectors = [lane_vector(lane) for lane in approaching_lanes]
    groups = _group_vectors_by_angles(vectors)
    ways = [[approaching_lanes[index] for index in group] for group in groups]

    if len(ways) == 3:
        angle01 = _angle_of_two_vectors(lane_vector(ways[1][0]), lane_vector(ways[0][0]))
        angle02 = _angle_of_two_vectors(lane_vector(ways[2][0]), lane_vector(ways[0][0]))
        angle12 = _angle_of_two_vectors(lane_vector(ways[2][0]), lane_vector(ways[1][0]))
        if angle02 > angle01 and angle02 > angle12:
            ways = [ways[0], ways[2], ways[1]]
        elif angle12 > angle01 and angle12 > angle02:
            ways = [ways[1], ways[2], ways[0]]
    elif len(ways) == 4:
        ways.sort(key=lambda way: _vector_heading(lane_vector(way[0])))

    return ways


def _group_vectors_by_angles(vectors: list[np.ndarray], angle_threshold: float = np.pi / 6) -> list[list[int]]:
    union_find = _UnionFind(range(len(vectors)))
    for left in range(len(vectors)):
        for right in range(len(vectors)):
            if _angle_of_two_vectors(vectors[left], vectors[right]) < angle_threshold:
                union_find.union(left, right)
    return union_find.groups()


def _classify_direction(points_xy: np.ndarray) -> _Direction:
    vectors = np.diff(points_xy, axis=0)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angle_diffs = (np.diff(angles) + np.pi) % (2 * np.pi) - np.pi
    total_turn_angle = float(np.sum(angle_diffs))
    if abs(total_turn_angle) < np.pi / 6:
        return _Direction.S
    return _Direction.L if total_turn_angle > 0 else _Direction.R


def _neighbor_type(polyline1: np.ndarray, polyline2: np.ndarray) -> str:
    line1 = np.array([polyline1[0, :2], polyline1[-1, :2]], dtype=np.float64)
    line2 = np.array([polyline2[0, :2], polyline2[-1, :2]], dtype=np.float64)
    parallel = _two_lines_parallel(line1, line2)
    start_distance = _distance(polyline1[0], polyline2[0])
    end_distance = _distance(polyline1[-1], polyline2[-1])

    def distance_level(value: float) -> str:
        if value < 1:
            return "low"
        if value < 5:
            return "mid"
        return "high"

    levels = [distance_level(start_distance), distance_level(end_distance)]
    if parallel:
        if "low" in levels:
            return "bifurcated-parallel" if levels[0] == "low" else "merged-parallel"
        return "other"
    if levels[0] in {"low", "mid"}:
        return "bifurcated"
    if levels[1] in {"low", "mid"}:
        return "merged"
    return "other"


def _real_neighbor_type(
    polyline1: np.ndarray,
    polyline2: np.ndarray,
    point_close_threshold: float = _POINT_CLOSE_THRESHOLD,
    length_difference_threshold: float = 3.0,
) -> str:
    start_close = _distance(polyline1[0], polyline2[0]) < point_close_threshold
    end_close = _distance(polyline1[-1], polyline2[-1]) < point_close_threshold
    length1 = _polyline_length(polyline1)
    length2 = _polyline_length(polyline2)
    polyline1_longer = length2 + length_difference_threshold < length1
    if start_close and end_close:
        return "complete"
    if start_close and not end_close and polyline1_longer:
        return "side-start"
    if not start_close and end_close and polyline1_longer:
        return "side-end"
    return "other"


def _distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(point1, dtype=np.float64) - np.asarray(point2, dtype=np.float64)))


def _polyline_length(polyline: np.ndarray) -> float:
    if len(polyline) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(polyline, axis=0), axis=1)))


def _two_lines_parallel(line1: np.ndarray, line2: np.ndarray) -> bool:
    vec1 = line1[1] - line1[0]
    vec2 = line2[1] - line2[0]
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return False
    cosine = float(np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cosine)))
    return angle < _LINE_PARALLEL_THRESHOLD


def _angle_of_two_vectors(left: np.ndarray, right: np.ndarray) -> float:
    norm_left = np.linalg.norm(left)
    norm_right = np.linalg.norm(right)
    if norm_left == 0 or norm_right == 0:
        return np.pi
    cosine = float(np.clip(np.dot(left, right) / (norm_left * norm_right), -1.0, 1.0))
    return float(np.arccos(cosine))


def _vector_heading(vector: np.ndarray) -> float:
    return float(np.arctan2(vector[1], vector[0]))


def _angle_of_headings(left: float, right: float) -> float:
    diff = abs(left - right) % (2 * np.pi)
    return float(min(diff, 2 * np.pi - diff))


def _as_xyz_array(points) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or len(array) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if array.shape[1] == 2:
        return np.column_stack([array, np.zeros(len(array), dtype=np.float64)])
    return array[:, :3]


def _from_puffer_tls(state) -> _TLS:
    value = int(state)
    if value == int(puffer_types.TLState.GREEN):
        return _TLS.GREEN
    if value == int(puffer_types.TLState.YELLOW):
        return _TLS.YELLOW
    if value == int(puffer_types.TLState.RED):
        return _TLS.RED
    return _TLS.UNKNOWN


def _to_puffer_tls(state: _TLS):
    if state == _TLS.GREEN:
        return puffer_types.TLState.GREEN
    if state == _TLS.YELLOW:
        return puffer_types.TLState.YELLOW
    if state == _TLS.RED:
        return puffer_types.TLState.RED
    return puffer_types.TLState.UNKNOWN
