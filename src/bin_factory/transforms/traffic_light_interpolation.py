"""Traffic-light state completion from detections, intersection geometry and vehicle behaviour.

Signalized connectors are the lanes carrying a TL track plus the successors of lanes in TL stop zones. Connectors of
one intersection with the same approach and turn share a light (a movement group), unless their detections disagree.
Groups linked by conflicts (crossing paths from different approaches) or by a shared approach run one hidden Markov
model over joint states (every group GO or STOP), restricted to states where no two conflicting groups are both GO
and softly favouring same-approach groups that agree. Per-frame evidence:
- detections: strong, and detected frames are copied to the output, except the last seconds of a RED run before a
  gap (detectors switch to green late), where the inference may overrule them,
- vehicles crossing the stop line onto a connector: GO,
- vehicles stopped right at the stop line: STOP.
Forward-backward yields a GO posterior per group and frame. Confident frames become GREEN or RED, inferred GREEN
ending in RED gets a yellow tail, everything else stays UNKNOWN.
"""

from __future__ import annotations

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from bin_factory import puffer_types, schema


_SWITCH_S = 30.0  # mean time between two switches of one group
_DETECTION_LLR = 4.0  # log-likelihood ratio GO/STOP of one detection frame
_ENTRY_LLR = 12.0  # vehicle crossing the stop line
_FREE_ENTRY_LLR = 1.0  # same on a free turn (right turn on red is legal)
_WAIT_LLR = -0.2  # per frame a vehicle stands at the stop line
_MAX_GROUP_LLR = 16.0
_COUPLING = 0.05  # per-frame log bonus for each pair of same-approach groups showing the same state
_CONFIDENCE = 0.9
_YELLOW_S = 3.0
_LAG_S = 2.0  # RED detections lag the switch to green: their last seconds before a gap can be overruled
_LAG_LLR = 1.0  # detection evidence in those seconds
_ENTRY_S = 0.5  # GO evidence spread before the crossing frame
_DEPART_S = 2.0  # last part of a stop dropped: the light is already green while the vehicle starts
_PATH_S = 3.0  # vehicle path length used to tell connectors sharing a start apart
_MAX_JOINT_GROUPS = 14
_MAX_JOINT_STATES = 1024

_GREEN, _YELLOW, _RED = (int(puffer_types.TLState[name]) for name in ("GREEN", "YELLOW", "RED"))


def interpolate_traffic_lights(scenario: schema.PufferScenario, extras: schema.ExtractionExtras) -> None:
    length, dt = scenario.metadata.scenario_length, float(scenario.metadata.dt)
    polys = {
        lid: np.asarray(e.polyline, dtype=np.float64)[:, :2]
        for lid, e in scenario.map.items()
        if e.is_lane and e.polyline is not None and len(e.polyline) > 1 and np.ptp(e.polyline[:, :2], 0).any()
    }
    tracks = {int(tl.controlled_lane): tl for tl in extras.traffic_lights.values()}
    zone_exits = {
        exit_id
        for zone in extras.stop_zones
        if zone.type == puffer_types.TCType.TRAFFIC_LIGHT
        for lid in zone.controlled_lanes
        for exit_id in scenario.map[lid].exit_lanes
    }
    ids = sorted((set(tracks) | zone_exits) & set(polys))
    if length <= 0 or not ids:
        return

    lines = [polys[lid] for lid in ids]
    observed = np.array(
        [[int(s) for s in tracks[lid].states[:length]] if lid in tracks else [0] * length for lid in ids]
    )
    obs = np.select([observed == _RED, (observed == _GREEN) | (observed == _YELLOW)], [-1, 1], 0)
    start = np.array([line[0] for line in lines])
    start_dir = np.array([_first_dir(line) for line in lines])
    end_dir = np.array([-_first_dir(line[::-1]) for line in lines])
    turn = np.arctan2(_cross(start_dir, end_dir), (start_dir * end_dir).sum(1))
    turn_class = np.where(np.abs(turn) < np.radians(30), 0, np.sign(turn)).astype(int)
    # free turn: right turn on red allowed, left turn in left-hand traffic
    free = turn_class == (1 if scenario.metadata.location.startswith("sg-") else -1)

    group, conflict, couple = _movement_groups(lines, start_dir, turn_class, free, obs)
    lag = _lag_frames(obs, round(_LAG_S / dt))
    llr = obs * np.where(lag, _LAG_LLR, _DETECTION_LLR)
    llr += _vehicle_evidence(scenario.agents, lines, start, start_dir, free, length, dt)
    group_llr = np.zeros((group.max() + 1, length))
    np.add.at(group_llr, group, llr)
    go = _group_posterior(np.clip(group_llr, -_MAX_GROUP_LLR, _MAX_GROUP_LLR), conflict, couple, dt / _SWITCH_S)

    go = go[group]
    inferred = np.select([go > _CONFIDENCE, go < 1 - _CONFIDENCE], [_GREEN, _RED], 0)
    kept = (observed != 0) & ~(lag & (inferred != 0))
    states = np.where(kept, observed, _yellow_tails(inferred, kept, round(_YELLOW_S / dt)))
    for lid, row in zip(ids, states, strict=True):
        if lid in tracks:
            tracks[lid].states = [puffer_types.TLState(int(s)) for s in row]
        elif row.any():
            extras.traffic_lights[lid] = schema.TrafficLightTrack(
                position=np.asarray(scenario.map[lid].polyline, dtype=np.float64)[0].copy(),
                states=[puffer_types.TLState(int(s)) for s in row],
                controlled_lane=lid,
            )


def _first_dir(line: np.ndarray) -> np.ndarray:
    steps = np.diff(line, axis=0)
    step = steps[np.linalg.norm(steps, axis=1) > 0][0]
    return step / np.linalg.norm(step)


def _lag_frames(obs: np.ndarray, frames: int) -> np.ndarray:
    """The last `frames` frames of each detected RED run followed by a gap."""
    lag = np.zeros(obs.shape, dtype=bool)
    lag[:, :-1] = (obs[:, :-1] == -1) & (obs[:, 1:] == 0)
    for _ in range(frames - 1):
        lag[:, :-1] |= lag[:, 1:] & (obs[:, :-1] == obs[:, 1:])
    return lag


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _components(n: int, pairs: np.ndarray) -> np.ndarray:
    adjacency = coo_matrix((np.ones(len(pairs[0])), (pairs[0], pairs[1])), shape=(n, n))
    return connected_components(adjacency, directed=False)[1]


def _movement_groups(
    lines: list[np.ndarray], start_dir: np.ndarray, turn_class: np.ndarray, free: np.ndarray, obs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group label per connector, and boolean group matrices: conflicts (crossing paths from different approaches,
    no free turn) and couples (same approach)."""
    n = len(lines)
    geoms = [shapely.LineString(line) for line in lines]
    tree = shapely.STRtree(geoms)
    intersection = _components(n, tree.query(geoms, predicate="dwithin", distance=1.0))
    aligned = start_dir @ start_dir.T
    approach = _components(n, np.nonzero((intersection[:, None] == intersection) & (aligned > np.cos(np.radians(30)))))
    disagree = (np.abs(obs) @ np.abs(obs).T - obs @ obs.T) / 2 > 0.05 * np.maximum(np.abs(obs) @ np.abs(obs).T, 1)
    same = (approach[:, None] == approach) & (turn_class[:, None] == turn_class) & ~disagree
    group = _components(n, np.nonzero(same))

    i, j = tree.query(geoms, predicate="crosses")
    keep = ~free[i] & ~free[j] & (approach[i] != approach[j])
    conflict = np.zeros((group.max() + 1,) * 2, dtype=bool)
    conflict[group[i[keep]], group[j[keep]]] = True
    couple = np.zeros_like(conflict)
    couple[group[:, None], group] = approach[:, None] == approach
    np.fill_diagonal(couple, False)
    return group, conflict | conflict.T, couple


def _vehicle_evidence(
    agents: dict[int, schema.Track],
    lines: list[np.ndarray],
    start: np.ndarray,
    start_dir: np.ndarray,
    free: np.ndarray,
    length: int,
    dt: float,
) -> np.ndarray:
    """Per connector and frame: GO evidence where vehicles enter it, STOP evidence where they wait at its stop line."""
    llr = np.zeros((len(lines), length))
    entry, depart, path = round(_ENTRY_S / dt), round(_DEPART_S / dt), round(_PATH_S / dt)
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
        targets = np.zeros(len(lines), dtype=bool)
        targets[crossed[taken]] = True
        targets = targets if waiting[:, targets].any() else ~free
        llr += _WAIT_LLR * (waiting & targets & ~free).T
    return llr


def _path_fit(path: np.ndarray, line: np.ndarray, line_length: float) -> float:
    """Mean distance from the vehicle path (cut to the connector length) to the connector."""
    traveled = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    points = path[traveled <= max(line_length, 3.0)]
    seg_start, seg = line[:-1], np.diff(line, axis=0)
    t = np.clip(((points[:, None] - seg_start) * seg).sum(-1) / np.maximum((seg**2).sum(-1), 1e-9), 0.0, 1.0)
    return float(np.linalg.norm(points[:, None] - (seg_start + t[..., None] * seg), axis=-1).min(1).mean())


def _group_posterior(llr: np.ndarray, conflict: np.ndarray, couple: np.ndarray, switch: float) -> np.ndarray:
    """GO posterior per group and frame. Groups linked by conflicts or couples are decoded jointly; components with
    too many joint states fall back to conflict links only, then to single groups."""
    posterior = np.full_like(llr, np.nan)
    for links in (conflict | couple, conflict, np.zeros_like(conflict)):
        component = _components(len(llr), np.nonzero(links))
        for label in np.unique(component[np.isnan(posterior[:, 0])]):
            members = np.flatnonzero(component == label)
            states = _feasible(conflict[np.ix_(members, members)]) if len(members) <= _MAX_JOINT_GROUPS else None
            if states is not None and len(states) <= _MAX_JOINT_STATES:
                posterior[members] = _forward_backward(llr[members], states, couple[np.ix_(members, members)], switch)
    return posterior


def _feasible(conflict: np.ndarray) -> np.ndarray:
    states = (np.arange(2 ** len(conflict))[:, None] >> np.arange(len(conflict))) & 1
    return states[np.einsum("sg,gh,sh->s", states, conflict.astype(int), states) == 0]


def _forward_backward(llr: np.ndarray, states: np.ndarray, couple: np.ndarray, switch: float) -> np.ndarray:
    flips = (states[:, None] != states[None]).sum(-1)
    trans = switch**flips * (1 - switch) ** (states.shape[1] - flips)
    trans /= trans.sum(1, keepdims=True)
    agree = ((states[:, :, None] == states[:, None]) & couple).sum((1, 2)) / 2
    log_emit = states @ llr + _COUPLING * agree[:, None]
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
    return states.T @ (joint / joint.sum(0))


def _yellow_tails(states: np.ndarray, observed: np.ndarray, frames: int) -> np.ndarray:
    """Turn the last `frames` inferred GREEN frames before each GREEN->RED switch into YELLOW."""
    states = states.copy()
    for row, t in zip(*np.nonzero((states[:, :-1] == _GREEN) & (states[:, 1:] == _RED)), strict=True):
        tail = np.arange(max(0, t + 1 - frames), t + 1)
        green_run = tail[np.cumprod(((states[row, tail] == _GREEN) & ~observed[row, tail])[::-1])[::-1].astype(bool)]
        states[row, green_run] = _YELLOW
    return states
