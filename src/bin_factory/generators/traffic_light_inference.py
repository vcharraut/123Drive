"""Traffic-light state completion from detections, intersection geometry and vehicle behaviour.

Signalized connectors carry a light track or follow a lane in a traffic-light stop zone.
Connectors in one intersection share a movement group when their approach and turn match or
when they leave the same lane, unless their detections disagree.

Crossing paths from perpendicular approaches conflict. A left turn and oncoming traffic may
share a permissive green. Groups linked by conflicts, a shared approach, or opposite crossing
paths run a joint hidden Markov model. It forbids conflicting greens, favours matching states
on the same approach, and penalizes simultaneous greens on opposite crossing paths.
Lanes detected only YELLOW keep their detections and contribute no evidence.

Detections are evidence, not fixed states: nuPlan labels flicker (sub-second REDs inside a
GREEN) and switch late. Vehicles crossing a stop line support GO, as do vehicles pulling away
from a stop toward it, while vehicles waiting there support STOP. The stop line is the one
the traffic control gets: at the light's stop zone when there is one, else at the connector start.

Each group follows the most likely joint path in which a state holds at least 3 s, so states
are continuous: a RED under 10 s between two GOs is filled, and GREEN followed by RED gets a
yellow tail. Forward-backward gives the confidence of that path. Frames before the first and
after the last confident frame stay UNKNOWN. Free turns never show RED, since Macadam counts
a turn on red as a red-light infraction: their RED frames are UNKNOWN.
"""

from __future__ import annotations

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from bin_factory import puffer_types, schema
from bin_factory.conversion.traffic_controls import stop_line_at_zone


_SWITCH_S = 30.0  # mean time between two switches of one group
_DETECTION_LLR = 1.0  # log-likelihood ratio GO/STOP of one detection frame; vehicles outweigh label noise
_ENTRY_LLR = 12.0  # vehicle crossing the stop line
_FREE_ENTRY_LLR = 1.0  # vehicle crossing on a free turn where turn on red is legal
_PULL_AWAY_LLR = 3.0  # per frame a vehicle moves from a stop to its crossing
_PULL_AWAY_S = 4.0  # longest pull-away credited before a crossing
_WAIT_LLR = -0.2  # per frame a vehicle stands at the stop line
_MAX_GROUP_LLR = 16.0
_COUPLING = 0.05  # per-frame log bonus for each pair of same-approach groups showing the same state
_OPPOSE = 0.2  # per-frame log penalty for each pair of crossing groups from opposite approaches both GO
_PERPENDICULAR_COS = 0.7  # crossing approaches more perpendicular than this conflict, the others oppose
_CONFIDENCE = 0.9
_YELLOW_S = 3.0
_MIN_PHASE_S = 3.0  # a joint state holds at least this long
_MIN_RED_S = 10.0  # a shorter RED between two GOs is filled
_ENTRY_S = 0.5  # GO evidence spread before the crossing frame
_DEPART_S = 2.0  # ignore the end of a stop, when the light is green but the vehicle has not moved yet
_PATH_S = 3.0  # vehicle path length used to tell connectors sharing a start apart
_MAX_JOINT_GROUPS = 14
_MAX_JOINT_STATES = 1024

_GREEN, _YELLOW, _RED = (int(puffer_types.TLState[name]) for name in ("GREEN", "YELLOW", "RED"))


def infer_traffic_lights(scenario: schema.PufferScenario, extras: schema.ExtractionExtras) -> None:
    length, dt = scenario.metadata.scenario_length, float(scenario.metadata.dt)
    polys = {
        lid: np.asarray(e.polyline, dtype=np.float64)[:, :2]
        for lid, e in scenario.polylines.items()
        if e.is_lane and e.polyline is not None and len(e.polyline) > 1 and np.ptp(e.polyline[:, :2], 0).any()
    }
    tracks = {int(tl.controlled_lane): tl for tl in extras.traffic_lights.values()}
    zone_exits = {
        exit_id
        for zone in extras.stop_zones
        if zone.type == puffer_types.TCType.TRAFFIC_LIGHT
        for lid in zone.controlled_lanes
        for exit_id in scenario.polylines[lid].exit_lanes
    }
    ids = sorted((set(tracks) | zone_exits) & set(polys))
    if length <= 0 or not ids:
        return

    lines = [polys[lid] for lid in ids]
    observed = np.array(
        [[int(s) for s in tracks[lid].states[:length]] if lid in tracks else [0] * length for lid in ids]
    )
    obs = np.select([observed == _RED, (observed == _GREEN) | (observed == _YELLOW)], [-1, 1], 0)
    # WOMD maps flashing caution to YELLOW. Keep these detections without GO/STOP inference.
    caution = (observed == _YELLOW).any(1) & ~((observed == _GREEN) | (observed == _RED)).any(1)
    obs[caution] = 0
    start = np.array([line[0] for line in lines])
    start_dir = np.array([_first_dir(line) for line in lines])
    end_dir = np.array([-_first_dir(line[::-1]) for line in lines])
    turn = np.arctan2(_cross(start_dir, end_dir), (start_dir * end_dir).sum(1))
    turn_class = np.where(np.abs(turn) < np.radians(30), 0, np.sign(turn)).astype(int)
    # Free turns allow right turns on red, or left turns in left-hand traffic.
    free = turn_class == (1 if scenario.metadata.location.startswith("sg-") else -1)

    entries = [set(scenario.polylines[lid].entry_lanes) for lid in ids]
    shared = np.array([[bool(a & b) for b in entries] for a in entries])
    group, conflict, couple, oppose = _movement_groups(lines, start_dir, turn_class, free, obs, shared)
    line_start, line_dir = _stop_lines(ids, scenario.polylines, extras.stop_zones, start, start_dir)
    llr = obs * _DETECTION_LLR
    llr += _vehicle_evidence(scenario.agents, lines, line_start, line_dir, free, length, dt)
    group_llr = np.zeros((group.max() + 1, length))
    np.add.at(group_llr, group, llr)
    group_llr = np.clip(group_llr, -_MAX_GROUP_LLR, _MAX_GROUP_LLR)
    go, path = _group_posterior(group_llr, conflict, couple, oppose, dt / _SWITCH_S, round(_MIN_PHASE_S / dt))

    go, path = go[group], _fill_short_reds(path, round(_MIN_RED_S / dt))[group]
    sure = np.where(path == 1, go, 1 - go) > _CONFIDENCE
    frame = np.arange(length)
    known = (frame >= sure.argmax(1)[:, None]) & (frame <= length - 1 - sure[:, ::-1].argmax(1)[:, None])
    states = np.where(known & sure.any(1)[:, None], np.where(path == 1, _GREEN, _RED), 0)
    states = _yellow_tails(states, round(_YELLOW_S / dt))
    states = np.where(free[:, None] & (states == _RED), 0, states)
    states = np.where(caution[:, None], observed, states)
    _write_inferred_states(ids, states, tracks, scenario.polylines, extras.traffic_lights)


def _write_inferred_states(
    ids: list[int],
    states: np.ndarray,
    tracks: dict[int, schema.TrafficLightTrack],
    map_elements: dict[int, schema.RoadPolyline],
    traffic_lights: dict[int, schema.TrafficLightTrack],
) -> None:
    for lid, row in zip(ids, states, strict=True):
        if lid in tracks:
            tracks[lid].states = [puffer_types.TLState(int(s)) for s in row]
        elif row.any():
            traffic_lights[lid] = schema.TrafficLightTrack(
                position=np.asarray(map_elements[lid].polyline, dtype=np.float64)[0].copy(),
                states=[puffer_types.TLState(int(s)) for s in row],
                controlled_lane=lid,
            )


def _first_dir(line: np.ndarray) -> np.ndarray:
    steps = np.diff(line, axis=0)
    step = steps[np.linalg.norm(steps, axis=1) > 0][0]
    return step / np.linalg.norm(step)


def _fill_short_reds(path: np.ndarray, frames: int) -> np.ndarray:
    """Turn each RED run shorter than `frames` between two GO runs of a row into GO."""
    path = path.copy()
    change = np.diff(np.pad(path, ((0, 0), (1, 1))), axis=1)
    rows, starts = np.nonzero(change == -1)
    next_rows, ends = np.nonzero(change == 1)
    short = (rows[:-1] == next_rows[1:]) & (ends[1:] - starts[:-1] < frames)
    for row, start, end in zip(rows[:-1][short], starts[:-1][short], ends[1:][short], strict=True):
        path[row, start:end] = 1
    return path


def _stop_lines(
    ids: list[int],
    polylines: dict[int, schema.RoadPolyline],
    stop_zones: list[schema.StopZone],
    start: np.ndarray,
    start_dir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the midpoint and direction of each connector's stop line: at its stop zone, else at its start."""
    lanes_by_id = {lid: lane for lid, lane in polylines.items() if lane.is_lane}
    zones = [zone for zone in stop_zones if zone.type == puffer_types.TCType.TRAFFIC_LIGHT]
    found = [stop_line_at_zone(lanes_by_id[lid], lanes_by_id, zones) for lid in ids]
    mid = [point if at is None else at[1][:, :2].mean(0) for point, at in zip(start, found, strict=True)]
    direction = [
        along if at is None else (np.cos(at[0]), np.sin(at[0])) for along, at in zip(start_dir, found, strict=True)
    ]
    return np.array(mid), np.array(direction)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _components(n: int, pairs: tuple[np.ndarray, ...]) -> np.ndarray:
    adjacency = coo_matrix((np.ones(len(pairs[0])), (pairs[0], pairs[1])), shape=(n, n))
    return connected_components(adjacency, directed=False)[1]


def _movement_groups(
    lines: list[np.ndarray],
    start_dir: np.ndarray,
    turn_class: np.ndarray,
    free: np.ndarray,
    obs: np.ndarray,
    shared: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a group label per connector and boolean matrices for conflicts, couples, and opposites.

    `shared` marks connector pairs leaving the same lane: one lane end has one signal, so they share a group.
    Conflicts cross from perpendicular approaches. Couples share an approach. Opposites cross from opposite approaches.
    Exclude free turns from conflicts and opposites.
    """
    n = len(lines)
    geoms = [shapely.LineString(line) for line in lines]
    tree = shapely.STRtree(geoms)
    intersection = _components(n, tree.query(geoms, predicate="dwithin", distance=1.0))
    aligned = start_dir @ start_dir.T
    approach = _components(n, np.nonzero((intersection[:, None] == intersection) & (aligned > np.cos(np.radians(30)))))
    disagree = (np.abs(obs) @ np.abs(obs).T - obs @ obs.T) / 2 > 0.05 * np.maximum(np.abs(obs) @ np.abs(obs).T, 1)
    same = ((approach[:, None] == approach) & (turn_class[:, None] == turn_class) | shared) & ~disagree
    group = _components(n, np.nonzero(same))

    i, j = tree.query(geoms, predicate="crosses")
    keep = ~free[i] & ~free[j] & (approach[i] != approach[j])
    perpendicular = np.abs((start_dir[i] * start_dir[j]).sum(1)) < _PERPENDICULAR_COS
    conflict = np.zeros((group.max() + 1,) * 2, dtype=bool)
    conflict[group[i[keep & perpendicular]], group[j[keep & perpendicular]]] = True
    oppose = np.zeros_like(conflict)
    oppose[group[i[keep & ~perpendicular]], group[j[keep & ~perpendicular]]] = True
    couple = np.zeros_like(conflict)
    couple[group[:, None], group] = approach[:, None] == approach
    np.fill_diagonal(couple, False)
    conflict |= conflict.T
    return group, conflict, couple, (oppose | oppose.T) & ~conflict


def _vehicle_evidence(
    agents: dict[int, schema.Track],
    lines: list[np.ndarray],
    start: np.ndarray,
    start_dir: np.ndarray,
    free: np.ndarray,
    length: int,
    dt: float,
) -> np.ndarray:
    """Compute GO evidence from vehicle entries and pull-aways, and STOP evidence from stop-line waits."""
    llr = np.zeros((len(lines), length))
    entry, depart, path = round(_ENTRY_S / dt), round(_DEPART_S / dt), round(_PATH_S / dt)
    pull_away = round(_PULL_AWAY_S / dt)
    line_lengths = np.array([np.linalg.norm(np.diff(line, axis=0), axis=1).sum() for line in lines])
    for track in agents.values():
        if track.type != puffer_types.AgentType.VEHICLE:
            continue
        pos = np.asarray(track.position, dtype=np.float64)[:length, :2]
        valid = np.asarray(track.valid, dtype=bool)[:length]
        speed = np.linalg.norm(np.asarray(track.velocity, dtype=np.float64)[:length, :2], axis=1)
        heading = np.asarray(track.heading, dtype=np.float64)[:length]
        rel = pos[:, None] - start
        along = (rel * start_dir).sum(-1)
        lateral = _cross(start_dir, rel)
        on = valid[:, None] & (np.abs(lateral) < 2.0)
        on &= np.stack([np.cos(heading), np.sin(heading)], -1) @ start_dir.T > np.cos(np.radians(45))
        crossing = np.zeros_like(on)
        crossing[1:] = on[1:] & on[:-1] & (along[:-1] < 0) & (along[1:] >= 0) & (speed[1:, None] > 1.0)
        waiting = on & (speed[:, None] < 0.5) & (along > -6.0) & (along < 0)
        waiting[:-depart] &= waiting[depart:]
        waiting[-depart:] = False

        frames, crossed = np.nonzero(crossing)
        if not len(crossed) and not waiting.any():
            continue
        fit = np.array(
            [
                _path_fit(pos[t : t + path][valid[t : t + path]], lines[c], line_lengths[c])
                for t, c in zip(frames, crossed, strict=True)
            ]
        )
        best = np.array([fit[np.abs(frames - t) <= entry].min() for t in frames])
        taken = (fit < 1.5) & (fit <= best + 0.5)
        for t, c in zip(frames[taken], crossed[taken], strict=True):
            llr[c, max(0, t - entry) : t + 1] += _FREE_ENTRY_LLR if free[c] else _ENTRY_LLR
            stops = np.flatnonzero((speed[:t] < 0.5) & valid[:t])
            if not free[c] and len(stops) and stops[-1] >= t - pull_away:
                llr[c, stops[-1] + 1 : max(0, t - entry)] += _PULL_AWAY_LLR
        targets = np.zeros(len(lines), dtype=bool)
        targets[crossed[taken]] = True
        targets = targets if waiting[:, targets].any() else ~free
        llr += _WAIT_LLR * (waiting & targets & ~free).T
    return llr


def _path_fit(path: np.ndarray, line: np.ndarray, line_length: float) -> float:
    """Return mean distance from the vehicle path, cut to the connector length, to the connector."""
    traveled = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    points = path[traveled <= max(line_length, 3.0)]
    seg_start, seg = line[:-1], np.diff(line, axis=0)
    t = np.clip(((points[:, None] - seg_start) * seg).sum(-1) / np.maximum((seg**2).sum(-1), 1e-9), 0.0, 1.0)
    return float(np.linalg.norm(points[:, None] - (seg_start + t[..., None] * seg), axis=-1).min(1).mean())


def _group_posterior(
    llr: np.ndarray, conflict: np.ndarray, couple: np.ndarray, oppose: np.ndarray, switch: float, dwell: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the GO probability and the most likely GO path per group and frame.

    Decode groups linked by conflicts, couples, or opposites together. If there are too many joint states,
    use conflict links only, then individual groups.
    """
    posterior = np.full_like(llr, np.nan)
    path = np.zeros(llr.shape, dtype=int)
    for links in (conflict | couple | oppose, conflict, np.zeros_like(conflict)):
        component = _components(len(llr), np.nonzero(links))
        for label in np.unique(component[np.isnan(posterior[:, 0])]):
            members = np.flatnonzero(component == label)
            states = _feasible(conflict[np.ix_(members, members)]) if len(members) <= _MAX_JOINT_GROUPS else None
            if states is not None and len(states) <= _MAX_JOINT_STATES:
                pairs = np.ix_(members, members)
                posterior[members], path[members] = _decode(
                    llr[members], states, couple[pairs], oppose[pairs], switch, dwell
                )
    return posterior, path


def _feasible(conflict: np.ndarray) -> np.ndarray:
    states = (np.arange(2 ** len(conflict))[:, None] >> np.arange(len(conflict))) & 1
    return states[np.einsum("sg,gh,sh->s", states, conflict.astype(int), states) == 0]


def _decode(
    llr: np.ndarray, states: np.ndarray, couple: np.ndarray, oppose: np.ndarray, switch: float, dwell: int
) -> tuple[np.ndarray, np.ndarray]:
    flips = (states[:, None] != states[None]).sum(-1)
    trans = switch**flips * (1 - switch) ** (states.shape[1] - flips)
    trans /= trans.sum(1, keepdims=True)
    agree = ((states[:, :, None] == states[:, None]) & couple).sum((1, 2)) / 2
    both_go = ((states[:, :, None] & states[:, None]) * oppose).sum((1, 2)) / 2
    log_emit = states @ llr + _COUPLING * agree[:, None] - _OPPOSE * both_go[:, None]
    emit = np.exp(log_emit - log_emit.max(0))
    alpha, beta = np.empty_like(emit), np.ones_like(emit)
    alpha[:, 0] = emit[:, 0] / emit[:, 0].sum()
    for t in range(1, emit.shape[1]):
        a = (alpha[:, t - 1] @ trans) * emit[:, t]
        alpha[:, t] = a / a.sum()
    for t in range(emit.shape[1] - 2, -1, -1):
        b = trans @ (emit[:, t + 1] * beta[:, t + 1])
        beta[:, t] = b / b.sum()
    joint = alpha * beta
    return states.T @ (joint / joint.sum(0)), states[_min_dwell_path(log_emit, np.log(trans), dwell)].T


def _min_dwell_path(log_emit: np.ndarray, log_trans: np.ndarray, dwell: int) -> np.ndarray:
    """Return the most likely state per frame where every state but the first and last holds at least `dwell` frames."""
    length = log_emit.shape[1]
    log_emit = np.pad(log_emit, ((0, 0), (0, dwell - 1)))  # free frames after the end let the last state be short
    move = log_trans.copy()
    np.fill_diagonal(move, -np.inf)
    cum = np.cumsum(np.pad(log_emit, ((0, 0), (1, 0))), axis=1)
    best = np.empty(log_emit.shape)
    origin = np.full(log_emit.shape, -1)
    best[:, 0] = log_emit[:, 0]
    for t in range(1, log_emit.shape[1]):
        best[:, t] = best[:, t - 1] + log_emit[:, t]
        if t >= dwell:
            # enter a state at t - dwell + 1 and hold it through t
            enter = best[:, t - dwell, None] + move
            score = enter.max(0) + cum[:, t + 1] - cum[:, t - dwell + 1]
            better = score > best[:, t]
            best[better, t] = score[better]
            origin[better, t] = enter.argmax(0)[better]
    path = np.empty(log_emit.shape[1], dtype=int)
    state, t = int(best[:, -1].argmax()), log_emit.shape[1] - 1
    while t >= 0:
        held = dwell if origin[state, t] >= 0 else 1
        path[t - held + 1 : t + 1] = state
        state, t = (int(origin[state, t]) if held == dwell else state), t - held
    return path[:length]


def _yellow_tails(states: np.ndarray, frames: int) -> np.ndarray:
    """Turn the last `frames` GREEN frames before each GREEN->RED switch into YELLOW."""
    states = states.copy()
    for row, t in zip(*np.nonzero((states[:, :-1] == _GREEN) & (states[:, 1:] == _RED)), strict=True):
        tail = np.arange(max(0, t + 1 - frames), t + 1)
        green_run = tail[np.cumprod((states[row, tail] == _GREEN)[::-1])[::-1].astype(bool)]
        states[row, green_run] = _YELLOW
    return states
