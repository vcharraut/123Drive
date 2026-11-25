"""Traffic-light states imputed and corrected from vehicle trajectories, after Yan et al., arXiv 2506.07150.

1. Signalized intersections (paper section 3.1). Junction lanes group by py123d intersection; maps without
   intersections group lanes that share an entry or exit lane or cross at the same level. A group is signalized when
   one of its connectors has a logged light or follows a light stop zone. Its signalized connectors are those, or,
   when no approach has a light stop zone, every connector whose approach has no stop or yield sign.
2. Movements. Approach lanes, the lanes entering the connectors, group by heading into approaches. A connector's
   turns are those of its paths through the junction, and a connector joined by several lanes belongs to the approach
   of the lane it continues. Connectors leaving one approach lane share its stop line and form one movement; a
   dedicated free turn (right, left in Singapore) follows the straight movement of its approach.
3. Estimation (section 3.2.1). Vehicles from 8 m past to 60 m before the stop line give each movement a state from
   their acceleration and speed, weighted by the confidence functions f and g of Appendix A over +-1 s. Vehicles
   taking a free turn are left out. As in the authors' code, a vehicle moving just past the stop line means green;
   here free turners count too, as they cross the stop line of a lane they share with the straight movement.
4. Imputation and correction (section 3.2.2). Logged and estimated states merge by Table 2, and each frame takes the
   ring-and-barrier configuration that matches them best (Eqs. 9-11), keeping the previous one on ties. A lone
   approach or a lone pair of opposite approaches has no conflict, and its imputed states hold over unknown frames.
5. Temporal filtering (section 3.2.3). A state shorter than 3 s between two opposite states takes the state before or
   after it, whichever the evidence matches better, shortest first; the last 2 s of each green before a red turn yellow.
   A green also turns yellow from a logged yellow that no logged green follows.

Free turns never show RED, as Macadam counts a turn on red as an infraction: a red between two greens shows GREEN,
and a red before the first or after the last green is UNKNOWN.
Intersections without any logged or estimated state keep their logged lights.
"""

import collections
import dataclasses

import numpy as np
import shapely
from scipy.ndimage import maximum_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from bin_factory import puffer_types, schema
from bin_factory.conversion.traffic_controls import light_stop_line
from bin_factory.log_context import log


# Parameters of the paper (Table 5)
WINDOW_S = 1.0  # past and future horizon of the vehicle evidence
A_GREEN = 0.5  # m/s^2, weighted mean acceleration above which a movement is green
A_RED = -2.0  # m/s^2, below which it is red
V_GREEN = 3.5  # m/s, weighted mean speed above which a movement is green
V_RED = 0.5  # m/s, below which it is red
# Movement confidence from which vehicles correct a logged state. Table 5 gives 1.0, which lets one standing vehicle,
# such as a left turner waiting for a gap, overrule a logged green; the authors' code needs log(1 + c) >= 0.8.
THETA = float(np.expm1(0.8))
W_BIG = 100.0  # weight of a logged state the vehicles confirm
W_SMALL = 0.1  # weight of a logged state without vehicle evidence
MIN_STATE_S = 3.0  # t_min: a shorter state between two opposite ones is smoothed out
YELLOW_S = 2.0  # t_yellow
PAST_LINE_M = 8.0  # vehicles up to this far past the stop line count (Appendix A)

# Adaptation to py123d maps and logs
LOOKBACK_S = 1.0  # a logged state holds this long over unknown frames, as in the authors' code
REACH_M = 60.0  # upstream reach along single-successor lanes: twice the largest g0
MATCH_LATERAL_M = 2.0  # vehicle centre to lane centreline
MATCH_HEADING_RAD = np.radians(30.0)
SAME_LEVEL_M = 2.5  # height gap of a vehicle on a lane, or of two crossing lanes
TURN_RAD = np.radians(30.0)  # total turning angle below which a connector goes straight
SAME_APPROACH_RAD = np.radians(30.0)
OPPOSITE_RAD = np.radians(120.0)
END_HEADING_M = 20.0  # approach heading over the last metres of its lanes
FREE_TURN_DEPTH_M = 4.0  # depth into a connector past which its siblings no longer overlap it
EXIT_M = 20.0  # a connector's path through the junction goes on this far past it to tell its turn
CROSSED_M = 5.0  # a vehicle this far past the stop line, still moving, has just crossed it (authors' code)
CROSSING_S = 0.2  # spread of that green evidence (authors' code: 2 steps)

GO, STOP = 1, -1
_GREEN, _YELLOW, _RED, _UNKNOWN = (int(puffer_types.TLState[name]) for name in ("GREEN", "YELLOW", "RED", "UNKNOWN"))


@dataclasses.dataclass
class _Movement:
    approach: int
    turns: frozenset[str]  # "L" across traffic, "S" straight, "R" to the free side
    connectors: list[int]
    free: list[int]  # connectors turning to the free side
    end_distance: dict[int, float]  # lane -> distance from its end to the stop line, negative past the line


def infer_traffic_lights(scenario: schema.PufferScenario, extras: schema.ExtractionExtras) -> None:
    length, dt = scenario.metadata.scenario_length, float(scenario.metadata.dt)
    all_lanes = {lane_id: lane for lane_id, lane in scenario.polylines.items() if lane.is_lane}
    lanes = {
        lane_id: lane
        for lane_id, lane in all_lanes.items()
        if lane.type != puffer_types.LaneType.BIKE_LANE and lane.polyline is not None and len(lane.polyline) >= 2
    }
    tracks = {track.controlled_lane: track for track in extras.traffic_lights.values()}
    intersections = _signalized_intersections(lanes, set(tracks), extras.stop_zones) if length >= 2 else []
    if not intersections:
        return

    left_hand = scenario.metadata.location.startswith("sg-")
    light_zones = [zone for zone in extras.stop_zones if zone.type == puffer_types.TCType.TRAFFIC_LIGHT]
    structures = [
        _movements(connectors, group, lanes, all_lanes, light_zones, left_hand) for connectors, group in intersections
    ]
    corridor = sorted({lane_id for movements, _ in structures for m in movements for lane_id in m.end_distance})
    slot = {lane_id: index for index, lane_id in enumerate(corridor)}
    position, heading, speed, accel, valid = _kinematics(scenario.agents, length, dt)
    lane_of, arc = _match(position, heading, valid, [_points(lanes[lane_id]) for lane_id in corridor])
    lane_length = np.array([_arc(_points(lanes[lane_id]))[-1] for lane_id in corridor] + [np.nan])

    window, lookback, spread = round(WINDOW_S / dt), round(LOOKBACK_S / dt), round(CROSSING_S / dt)
    for movements, pairs in structures:
        deep = np.isin(lane_of, [slot[c] for m in movements for c in m.connectors]) & (arc >= FREE_TURN_DEPTH_M)
        free = np.isin(lane_of, [slot[c] for m in movements for c in m.free])
        turners = (deep & free).sum(1) > (deep & ~free).sum(1)
        entered = deep.sum(1) > 0  # seen inside the junction, past where sibling connectors overlap
        estimated = np.zeros((len(movements), length), dtype=int)
        confidence = np.zeros((len(movements), length))
        for index, movement in enumerate(movements):
            if movement.turns == {"R"}:
                continue  # its vehicles take a free turn
            lookup = np.full(len(corridor) + 1, np.nan)  # the last slot stands for no lane
            lookup[[slot[lane_id] for lane_id in movement.end_distance]] = list(movement.end_distance.values())
            distance = lookup[lane_of] + lane_length[lane_of] - arc
            rows = np.isfinite(distance).any(1)
            estimated[index], confidence[index] = _estimate(
                distance[rows], accel[rows], speed[rows], ~turners[rows], entered[rows], window, spread
            )
        logged, logged_yellow = (
            np.array(rows)
            for rows in zip(
                *(_logged([tracks[c] for c in m.connectors if c in tracks], length, lookback) for m in movements),
                strict=True,
            )
        )
        imputed, weight = _impute(logged, estimated, confidence)
        if not imputed.any():
            continue
        follow = _free_followers(movements)
        approaches = len({m.approach for m in movements})
        if approaches > 2 or (approaches == 2 and not pairs):
            states = _project(imputed, weight, _feasible(movements, pairs, follow))
        else:
            states = _hold(imputed)
            states[list(follow)] = states[list(follow.values())]
        states = _yellow(_smooth(states, imputed, weight, round(MIN_STATE_S / dt)), round(YELLOW_S / dt))
        _write(movements, _keep_logged_yellow(states, logged_yellow), lanes, tracks, extras)
        log.debug(
            "traffic lights: %d approaches, %d movements, %d of %d connectors logged",
            len({m.approach for m in movements}),
            len(movements),
            sum(c in tracks for m in movements for c in m.connectors),
            sum(len(m.connectors) for m in movements),
        )


def _signalized_intersections(
    lanes: dict[int, schema.RoadPolyline], logged: set[int], stop_zones: list[schema.StopZone]
) -> list[tuple[list[int], set[int]]]:
    """Return the signalized connectors and the lanes of each signalized intersection (paper section 3.1)."""
    light = puffer_types.TCType.TRAFFIC_LIGHT
    light_lanes = {lane_id for zone in stop_zones if zone.type == light for lane_id in zone.controlled_lanes}
    sign_lanes = {lane_id for zone in stop_zones if zone.type != light for lane_id in zone.controlled_lanes}
    zone_exits = {exit_id for lane_id in light_lanes if lane_id in lanes for exit_id in lanes[lane_id].exit_lanes}
    signal = (logged | zone_exits) & lanes.keys()
    intersections = []
    for group in _junction_groups(lanes, signal):
        connectors = []
        for lane_id in sorted(group):
            entries = [entry for entry in lanes[lane_id].entry_lanes if entry in lanes]
            if entries and not group.intersection(entries):
                connectors.append(lane_id)
        if not signal.intersection(connectors):
            continue
        trust_zones = any(light_lanes.intersection(lanes[lane_id].entry_lanes) for lane_id in connectors)
        signalized = [
            lane_id
            for lane_id in connectors
            if lane_id in signal or not (trust_zones or sign_lanes.intersection(lanes[lane_id].entry_lanes))
        ]
        intersections.append((signalized, group))
    return intersections


def _junction_groups(lanes: dict[int, schema.RoadPolyline], signal: set[int]) -> list[set[int]]:
    """Group junction lanes by py123d intersection, or else link lanes sharing an entry or exit lane or crossing."""
    by_intersection = collections.defaultdict(set)
    for lane_id, lane in lanes.items():
        if lane.intersection_id >= 0:
            by_intersection[lane.intersection_id].add(lane_id)
    if by_intersection:
        return list(by_intersection.values())

    ids = sorted(lanes)
    index = {lane_id: i for i, lane_id in enumerate(ids)}
    sharing = collections.defaultdict(list)
    for lane_id in ids:
        for entry in lanes[lane_id].entry_lanes:
            sharing["entry", entry].append(index[lane_id])
        for exit_id in lanes[lane_id].exit_lanes:
            sharing["exit", exit_id].append(index[lane_id])
    pairs = [(members[0], other) for members in sharing.values() for other in members[1:]]
    # Lanes inside a junction diverge from or merge with others, or carry a light; they may cross each other
    junction = sorted(
        {i for members in sharing.values() if len(members) > 1 for i in members} | {index[s] for s in signal}
    )
    lines = np.array([shapely.LineString(_points(lanes[ids[i]])[:, :2]) for i in junction], dtype=object)
    for a, b in zip(*shapely.STRtree(lines).query(lines, predicate="crosses"), strict=True):
        first, second = lanes[ids[junction[a]]], lanes[ids[junction[b]]]
        if a >= b or ids[junction[b]] in first.exit_lanes or ids[junction[a]] in second.exit_lanes:
            continue
        point = shapely.get_coordinates(shapely.intersection(lines[a], lines[b]))[0]
        if abs(_height_at(_points(first), point) - _height_at(_points(second), point)) < SAME_LEVEL_M:
            pairs.append((junction[a], junction[b]))
    labels = _components(len(ids), pairs)
    groups = collections.defaultdict(set)
    for lane_id in ids:
        groups[labels[index[lane_id]]].add(lane_id)
    return list(groups.values())


def _movements(
    connectors: list[int],
    group: set[int],
    lanes: dict[int, schema.RoadPolyline],
    all_lanes: dict[int, schema.RoadPolyline],
    light_zones: list[schema.StopZone],
    left_hand: bool,
) -> tuple[list[_Movement], list[tuple[int, int]]]:
    """Return an intersection's movements and its pairs of opposite approaches, approaches in heading order."""
    start_heading = {c: _start_heading(_points(lanes[c])) for c in connectors}
    entry = {
        c: min(
            (e for e in lanes[c].entry_lanes if e in lanes),
            key=lambda e: -np.cos(_end_heading(_points(lanes[e])) - start_heading[c]),
        )
        for c in connectors
    }
    approach_lanes = sorted(set(entry.values()))
    heading = np.array([_end_heading(_points(lanes[lane_id])) for lane_id in approach_lanes])
    close = np.abs(_wrap(heading[:, None] - heading[None])) < SAME_APPROACH_RAD
    labels = _components(len(approach_lanes), list(zip(*np.nonzero(close), strict=True)))
    means = {int(label): _circular_mean(heading[labels == label]) for label in np.unique(labels)}
    order = sorted(means, key=means.__getitem__)
    approach_of = {lane_id: order.index(int(labels[i])) for i, lane_id in enumerate(approach_lanes)}

    # A turn inside the junction stands; a path straight through it takes the turn of the road just past it
    turns = {
        c: {
            inside if (inside := _turn_class(path, left_hand)) != "S" else _turn_class(extended, left_hand)
            for path, extended in _junction_paths(c, group, lanes)
        }
        for c in connectors
    }
    stop = {}
    for c in connectors:
        line_heading, line = light_stop_line(lanes[c], all_lanes, light_zones)
        offset = line.mean(0)[:2] - _points(lanes[c])[0, :2]
        stop[c] = float(offset @ np.array([np.cos(line_heading), np.sin(line_heading)]))
    movements = []
    for approach in range(len(order)):
        phases: list[set[str]] = []  # turn classes linked by a shared approach lane
        for lane_id in [lane_id for lane_id in approach_lanes if approach_of[lane_id] == approach]:
            lane_turns = set().union(*(turns[c] for c in connectors if entry[c] == lane_id))
            phases = [p for p in phases if not p & lane_turns] + [
                lane_turns.union(*(p for p in phases if p & lane_turns))
            ]
        for phase in phases:
            members = [c for c in connectors if approach_of[entry[c]] == approach and turns[c] <= phase]
            end_distance = {c: stop[c] - _arc(_points(lanes[c]))[-1] for c in members}
            for lane_id in {entry[c] for c in members}:
                end_distance.update(_upstream(lane_id, min(stop[c] for c in members if entry[c] == lane_id), lanes))
            free = [c for c in members if turns[c] == {"R"}]
            movements.append(_Movement(approach, frozenset(phase), members, free, end_distance))

    approach_heading = np.array([means[label] for label in order])
    gap = np.abs(_wrap(approach_heading[:, None] - approach_heading[None]))
    partner = gap.argmax(1)
    pairs = [(i, int(j)) for i, j in enumerate(partner) if i < j and partner[j] == i and gap[i, j] > OPPOSITE_RAD]
    return movements, pairs


def _junction_paths(
    start: int, group: set[int], lanes: dict[int, schema.RoadPolyline]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the polylines of the paths through the junction from a connector, following successors in ``group``.

    Each path comes with its extension EXIT_M past the junction along single successors: some slip lanes turn only
    after it.
    """
    paths = []
    stack = [(start, _points(lanes[start])[:, :2], {start})]
    while stack:
        lane_id, path, seen = stack.pop()
        following = [e for e in lanes[lane_id].exit_lanes if e in group and e in lanes and e not in seen]
        stack += [(e, np.vstack([path, _points(lanes[e])[1:, :2]]), seen | {e}) for e in following]
        if following:
            continue
        beyond = [path[-1:]]
        while _arc(np.vstack(beyond))[-1] < EXIT_M:
            exits = [e for e in lanes[lane_id].exit_lanes if e in lanes]
            if len(exits) != 1 or exits[0] in seen:
                break
            lane_id = exits[0]
            seen = seen | {lane_id}
            beyond.append(_points(lanes[lane_id])[1:, :2])
        beyond = np.vstack(beyond)
        arc = _arc(beyond)
        paths.append((path, np.vstack([path, beyond[1:][arc[1:] <= EXIT_M]])))
    return paths


def _upstream(lane_id: int, end_distance: float, lanes: dict[int, schema.RoadPolyline]) -> dict[int, float]:
    """Return the approach lane and its single-successor predecessors within REACH_M, with their end distances."""
    found = {lane_id: end_distance}
    queue = collections.deque([lane_id])
    while queue:
        current = queue.popleft()
        distance = found[current] + _arc(_points(lanes[current]))[-1]
        if distance >= REACH_M:
            continue
        for previous in lanes[current].entry_lanes:
            single = previous in lanes and [e for e in lanes[previous].exit_lanes if e in lanes] == [current]
            if single and previous not in found:
                found[previous] = distance
                queue.append(previous)
    return found


def _kinematics(
    agents: dict[int, schema.Track], length: int, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return vehicle positions, headings, speeds, accelerations and validity, shaped (vehicles, frames)."""
    vehicles = [track for track in agents.values() if track.type == puffer_types.AgentType.VEHICLE]
    if not vehicles:
        empty = np.zeros((0, length))
        return np.zeros((0, length, 3)), empty, empty, empty, empty.astype(bool)
    position = np.stack([np.asarray(track.position, dtype=np.float64)[:length, :3] for track in vehicles])
    heading = np.stack([np.asarray(track.heading, dtype=np.float64)[:length] for track in vehicles])
    valid = np.stack([np.asarray(track.valid)[:length].astype(bool) for track in vehicles])
    velocity = np.stack([np.asarray(track.velocity, dtype=np.float64)[:length, :2] for track in vehicles])
    speed = np.linalg.norm(velocity, axis=-1)
    # Central difference over 0.5 s, zero where a neighbour is invalid or the value is implausible
    half = max(1, round(0.25 / dt))
    frames = np.arange(length)
    low, high = np.maximum(frames - half, 0), np.minimum(frames + half, length - 1)
    accel = (speed[:, high] - speed[:, low]) / np.maximum((high - low) * dt, 1e-9)
    accel = np.where(valid[:, low] & valid[:, high] & (np.abs(accel) <= 10.0), accel, 0.0)
    return position, heading, speed, accel, valid


def _match(
    position: np.ndarray, heading: np.ndarray, valid: np.ndarray, polylines: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Return per vehicle and frame the index of the closest aligned polyline (-1 if none) and the arc length on it."""
    best = np.full(valid.shape, np.inf)
    lane_of = np.full(valid.shape, -1)
    arc = np.zeros(valid.shape)
    for index, polyline in enumerate(polylines):
        low, high = polyline[:, :2].min(0) - MATCH_LATERAL_M, polyline[:, :2].max(0) + MATCH_LATERAL_M
        near = valid & np.all((position[..., :2] >= low) & (position[..., :2] <= high), axis=-1)
        if not near.any():
            continue
        points = position[near]
        distance, s, tangent, height = _locate(points[:, :2], polyline)
        better = (
            (distance <= MATCH_LATERAL_M)
            & (np.abs(_wrap(heading[near] - tangent)) <= MATCH_HEADING_RAD)
            & (np.abs(points[:, 2] - height) <= SAME_LEVEL_M)
            & (distance < best[near])
        )
        rows, cols = (axis[better] for axis in np.nonzero(near))
        best[rows, cols], lane_of[rows, cols], arc[rows, cols] = distance[better], index, s[better]
    return lane_of, arc


def _estimate(
    distance: np.ndarray,
    accel: np.ndarray,
    speed: np.ndarray,
    counted: np.ndarray,
    entered: np.ndarray,
    window: int,
    spread: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a movement's estimated state (GO, STOP or 0) and confidence per frame, from Fig. 5 and Table 1.

    ``distance`` is each vehicle's distance to the stop line per frame, NaN when the vehicle is not on the movement;
    ``counted`` vehicles give the evidence of Table 1. As in the authors' code, a vehicle that ``entered`` the junction
    and moves just past the stop line makes the movement GO with W_BIG within ``spread`` frames: queues still stand as
    a green starts. Free turners count here too, since they cross the stop line they share with the straight movement;
    a lone turn on red makes a short green that smoothing removes.
    """
    on = np.isfinite(distance)
    crossed = entered[:, None] & on & (distance > -CROSSED_M) & (distance <= 0) & (speed > V_RED)
    crossed = maximum_filter1d(crossed.any(0).astype(np.int8), size=2 * spread + 1) > 0
    on &= counted[:, None]
    distance = np.where(on, distance, np.inf)
    mean_accel, f_max = _windowed(_f(distance, accel) * on, accel, window)
    mean_speed, g_max = _windowed(_g(distance, speed) * on, speed, window)
    confidence_f, confidence_g = f_max.sum(0), g_max.sum(0)
    a = (f_max * mean_accel).sum(0) / np.maximum(confidence_f, 1e-12)  # Eq. 5
    v = (g_max * mean_speed).sum(0) / np.maximum(confidence_g, 1e-12)  # Eq. 7
    areas = [  # areas 1 to 4 of Fig. 5; area 5 has no estimate
        (confidence_f > 0) & (a > A_GREEN),
        (confidence_f > 0) & (a < A_RED),
        (confidence_g > 0) & (v > V_GREEN),
        (confidence_g > 0) & (v < V_RED),
    ]
    state = np.where(crossed, GO, np.select(areas, [GO, STOP, GO, STOP], 0))
    confidence = np.select(areas, [confidence_f, confidence_f, confidence_g, confidence_g], 0.0)
    return state, np.where(crossed, W_BIG, confidence)


def _f(d: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Acceleration confidence f(d, a) of Appendix A (Eq. 21), as drawn in Fig. 14a."""
    fade = np.where(d > 15.0, (d - 30.0) ** 2 / 15.0**2, 1.0)
    return np.where((d < -PAST_LINE_M) | (d > 30.0) | ((d < 0) & (a < 0)), 0.0, fade)


def _g(d: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Speed confidence g(d, v) of Appendix A (Eqs. 22-23).

    g0 follows Fig. 14b, 15 m at standstill and at 12 m/s, where Eq. 23 prints 3(v - 6)^2 / 4 for (v - 6)^2 / 4.
    """
    g0 = np.where(v <= 12.0, (v - 6.0) ** 2 / 4.0 + 6.0, np.minimum(5.0 * (v - 12.0) + 15.0, 30.0))
    fade = np.where(d > g0, (d - 2.0 * g0) ** 2 / g0**2, 1.0)
    return np.where((d < -PAST_LINE_M) | (d > 2.0 * g0), 0.0, fade)


def _windowed(weight: np.ndarray, value: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return per vehicle and frame the weighted mean of ``value`` over +-``window`` frames and the largest weight."""
    if not weight.size:
        return weight, weight
    cumulative = np.cumsum(np.pad(np.stack([weight, weight * value]), ((0, 0), (0, 0), (1, 0))), axis=-1)
    frames = np.arange(weight.shape[1])
    high, low = np.minimum(frames + window + 1, weight.shape[1]), np.maximum(frames - window, 0)
    total, weighted = np.take(cumulative, high, axis=-1) - np.take(cumulative, low, axis=-1)
    largest = maximum_filter1d(weight, size=2 * window + 1, axis=1, mode="constant")
    return weighted / np.maximum(total, 1e-12), largest  # Eqs. 6 and 8


def _logged(tracks: list[schema.TrafficLightTrack], length: int, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a movement's logged state per frame, the majority over its connectors with YELLOW counting as GO, and
    which of yellow (+1) or green (-1) its connectors show more."""
    frames = np.arange(length)
    total, yellow = np.zeros(length, dtype=int), np.zeros(length, dtype=int)
    for track in tracks:
        states = np.array([int(state) for state in track.states[:length]])
        known = np.isin(states, (_GREEN, _YELLOW, _RED))
        last = np.maximum.accumulate(np.where(known, frames, -1))
        held = np.where((last >= 0) & (frames - last <= lookback), states[np.maximum(last, 0)], _UNKNOWN)
        total += np.select([(held == _GREEN) | (held == _YELLOW), held == _RED], [GO, STOP], 0)
        yellow += (held == _YELLOW).astype(int) - (held == _GREEN)
    return np.sign(total), np.sign(yellow)


def _impute(logged: np.ndarray, estimated: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return imputed states and their weights, from Table 2."""
    correct = (logged != 0) & (estimated != 0) & (logged != estimated) & (confidence >= THETA)
    imputed = np.where((logged == 0) | correct, estimated, logged)
    weight = np.select(
        [logged == 0, estimated == 0, logged == estimated, correct], [confidence, W_SMALL, W_BIG, confidence], 0.0
    )
    return imputed, weight


def _free_followers(movements: list[_Movement]) -> dict[int, int]:
    """Map each free-turn-only movement to the movement of its approach it follows: straight, else across."""
    follow = {}
    for index, movement in enumerate(movements):
        own = [i for i, m in enumerate(movements) if m.approach == movement.approach and i != index]
        leader = next((i for i in own if "S" in movements[i].turns), next(iter(own), None))
        if movement.turns == {"R"} and leader is not None:
            follow[index] = leader
    return follow


def _feasible(movements: list[_Movement], pairs: list[tuple[int, int]], follow: dict[int, int]) -> np.ndarray:
    """Return the ring-and-barrier configurations, rows of GO and STOP per movement, generalizing Table 3.

    Green goes to each pair of opposite approaches: all their movements, and when no lane turns across traffic and
    goes straight, also their other movements only or their across turns only; then to each approach alone. Ties
    take the first, so unobserved movements follow the most common operation: opposite approaches together.
    """
    groups = [[i for i, m in enumerate(movements) if m.approach == a] for a in sorted({m.approach for m in movements})]
    greens = []
    for a, b in pairs:
        both = groups[a] + groups[b]
        greens.append(both)
        if not any({"L", "S"} <= movements[i].turns for i in both):
            greens.append([i for i in both if "L" not in movements[i].turns])
            greens.append([i for i in both if "L" in movements[i].turns])
    greens += groups
    configurations: list[np.ndarray] = []
    for green in greens:
        row = np.full(len(movements), STOP)
        row[green] = GO
        row[list(follow)] = row[list(follow.values())]
        if (row == GO).any() and not any((row == other).all() for other in configurations):
            configurations.append(row)
    return np.array(configurations)


def _project(imputed: np.ndarray, weight: np.ndarray, configurations: np.ndarray) -> np.ndarray:
    """Return per frame the configuration with the highest match score (Eqs. 9-11), keeping the previous on ties.

    With GO and STOP the only known states, the conflict score of Eq. 9 is the total weight minus the match score, so
    the lowest conflict among the best matches never breaks a tie. Frames before any evidence take the first choice.
    """
    agree = (configurations[:, :, None] == imputed[None]) & (imputed != 0)[None]
    match = (agree * weight[None]).sum(1)
    first = int(np.argmax(weight.sum(0) > 0))
    choice = np.zeros(imputed.shape[1], dtype=int)
    previous = -1
    for t in range(first, imputed.shape[1]):
        tied = np.flatnonzero(match[:, t] >= match[:, t].max() - 1e-9)
        previous = previous if previous in tied else int(tied[0])
        choice[t] = previous
    choice[:first] = choice[first]
    return configurations[choice].T.copy()


def _hold(imputed: np.ndarray) -> np.ndarray:
    """Return the imputed states with each unknown frame taking the last known state, or the first one before it."""
    frames = np.arange(imputed.shape[1])
    last = np.maximum.accumulate(np.where(imputed != 0, frames, -1), axis=1)
    last = np.where(last < 0, np.argmax(imputed != 0, axis=1)[:, None], last)
    return np.take_along_axis(imputed, last, axis=1)


def _smooth(states: np.ndarray, imputed: np.ndarray, weight: np.ndarray, min_frames: int) -> np.ndarray:
    """Replace each run shorter than ``min_frames`` between two opposite states (Eqs. 12-13).

    The whole intersection takes its state before or after the run, both feasible: the one the imputed states over the
    run match best, the state before on ties as in Eq. 13 (the text allows either). Runs go shortest first and are
    found again after each replacement. Each replacement removes the two switches around the run and adds none on
    the other movements, so the loop ends.
    """
    states = states.copy()
    while True:
        intervals = []
        for row in states:
            bounds = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1, len(row)]
            for k in range(1, len(bounds) - 2):
                start, end = bounds[k], bounds[k + 1]
                if row[start] != 0 and end - start < min_frames and row[start - 1] == row[end] == -row[start]:
                    intervals.append((end - start, start, end))
        if not intervals:
            return states
        _, start, end = min(intervals)
        before, after = states[:, start - 1 : start], states[:, end : end + 1]
        evidence = (imputed[:, start:end] != 0) * weight[:, start:end]
        better = (evidence * (imputed[:, start:end] == after)).sum() > (
            evidence * (imputed[:, start:end] == before)
        ).sum()
        states[:, start:end] = after if better else before


def _yellow(states: np.ndarray, frames: int) -> np.ndarray:
    """Return TLState values, with the last ``frames`` frames of each GO run before a STOP YELLOW (Eq. 14)."""
    result = np.select([states == GO, states == STOP], [_GREEN, _RED], _UNKNOWN)
    for row, t in zip(*np.nonzero((states[:, :-1] == GO) & (states[:, 1:] == STOP)), strict=True):
        start = t
        while start > 0 and t - start + 1 < frames and states[row, start - 1] == GO:
            start -= 1
        result[row, start : t + 1] = _YELLOW
    return result


def _keep_logged_yellow(states: np.ndarray, logged_yellow: np.ndarray) -> np.ndarray:
    """Turn each green YELLOW from the first frame logged yellow after its last frame logged green to its end.

    The paper adds a fixed yellow; logged yellows are often longer, or still on when the log ends. A green that
    follows a red keeps its first frame, as no light turns from red to yellow.
    """
    states = states.copy()
    for row, shown in zip(states, logged_yellow, strict=True):
        go = np.isin(row, (_GREEN, _YELLOW)).astype(int)
        bounds = np.flatnonzero(np.diff(np.r_[0, go, 0]))
        for start, end in zip(bounds[::2], bounds[1::2], strict=True):
            last_green = start + max(np.flatnonzero(shown[start:end] < 0), default=-1)
            first_yellow = np.flatnonzero(shown[last_green + 1 : end] > 0)
            after_red = start > 0 and row[start - 1] == _RED
            if len(first_yellow):
                row[max(last_green + 1 + first_yellow[0], start + after_red) : end] = _YELLOW
    return states


def _write(
    movements: list[_Movement],
    states: np.ndarray,
    lanes: dict[int, schema.RoadPolyline],
    tracks: dict[int, schema.TrafficLightTrack],
    extras: schema.ExtractionExtras,
) -> None:
    for movement, row in zip(movements, states, strict=True):
        # A free turn stays GREEN from its first go frame to its last green one: no unknown frames split its states
        free = np.where(row == _RED, _UNKNOWN, row)
        green = np.flatnonzero(row == _GREEN)
        if len(green):
            free[np.argmax(np.isin(row, (_GREEN, _YELLOW))) : green[-1] + 1] = _GREEN
        for c in movement.connectors:
            values = free if c in movement.free else row
            light = [puffer_types.TLState(int(value)) for value in values]
            if c in tracks:
                tracks[c].states = light
            elif values.any():
                extras.traffic_lights[c] = schema.TrafficLightTrack(
                    position=_points(lanes[c])[0].copy(), states=light, controlled_lane=c
                )


def _components(n: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    rows, cols = np.asarray(pairs, dtype=int).reshape(-1, 2).T
    return connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)), directed=False)[1]


def _wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _circular_mean(angles: np.ndarray) -> float:
    return float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))


def _arc(polyline: np.ndarray) -> np.ndarray:
    """Return the 2D arc length at each point."""
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(polyline[:, :2], axis=0), axis=1))]


def _end_heading(polyline: np.ndarray) -> float:
    """Return the direction of the chord over the last END_HEADING_M of a polyline."""
    arc = _arc(polyline)
    start = [np.interp(arc[-1] - END_HEADING_M, arc, polyline[:, k]) for k in range(2)]
    dx, dy = polyline[-1, :2] - start
    return float(np.arctan2(dy, dx))


def _start_heading(polyline: np.ndarray) -> float:
    """Return the direction of the first step of a polyline that has a length."""
    steps = np.diff(polyline[:, :2], axis=0)
    dx, dy = steps[np.argmax(np.linalg.norm(steps, axis=1) > 1e-6)]
    return float(np.arctan2(dy, dx))


def _turn_class(polyline: np.ndarray, left_hand: bool) -> str:
    """Return "S" below TURN_RAD of total turning, else "L" across traffic or "R" to the free side."""
    steps = np.diff(polyline[:, :2], axis=0)
    steps = steps[np.linalg.norm(steps, axis=1) > 1e-6]
    angle = float(_wrap(np.diff(np.arctan2(steps[:, 1], steps[:, 0]))).sum())
    if abs(angle) < TURN_RAD:
        return "S"
    return "L" if (angle > 0) != left_hand else "R"


def _locate(points: np.ndarray, polyline: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return each point's distance to the polyline, and the arc length, direction and height where it projects."""
    start, step = polyline[:-1], np.diff(polyline, axis=0)
    squared = (step[:, :2] ** 2).sum(1)
    t = np.clip(((points[:, None] - start[:, :2]) * step[:, :2]).sum(-1) / np.maximum(squared, 1e-12), 0.0, 1.0)
    gap = np.linalg.norm(points[:, None] - (start[:, :2] + t[..., None] * step[:, :2]), axis=-1)
    gap[:, squared <= 1e-12] = np.inf
    segment = gap.argmin(1)
    rows = np.arange(len(points))
    t = t[rows, segment]
    arc = _arc(polyline)[segment] + t * np.sqrt(squared[segment])
    direction = np.arctan2(step[segment, 1], step[segment, 0])
    return gap[rows, segment], arc, direction, start[segment, 2] + t * step[segment, 2]


def _height_at(polyline: np.ndarray, point: np.ndarray) -> float:
    return float(_locate(np.asarray(point, dtype=np.float64)[None, :2], polyline)[3][0])


def _points(lane: schema.RoadPolyline) -> np.ndarray:
    return np.asarray(lane.polyline, dtype=np.float64)
