"""Zero out log-only agents that overlap with active agents.

Runs after ``process_agent_routes``: agents with a non-empty route are treated as
"active" (the simulator's controllable set), the rest as log-only. For each active
agent, checks per-timestep bbox overlap against every log-only agent; first
intersection zeros the log agent's ``valid`` array.
"""

import logging

import numpy as np
from shapely import geometry as shapely_geom

from .routes import _compute_control_state


logger = logging.getLogger(__name__)


def invalid_agent_overlap(scenario) -> None:
    agent_ids = list(scenario.agents.keys())
    tracks = [scenario.agents[aid] for aid in agent_ids]
    if len(tracks) < 2:
        return

    active_mask = np.array([bool(t.route) for t in tracks], dtype=bool)
    if not active_mask.any() or active_mask.all():
        return

    active_idxs = np.flatnonzero(active_mask)
    log_idxs = np.flatnonzero(~active_mask)

    corners = _compute_all_corners(tracks)  # (N, T, 4, 2)
    valid = np.stack([np.asarray(t.valid, dtype=bool) for t in tracks], axis=0)  # (N, T)
    aabb_min = corners.min(axis=2)  # (N, T, 2)
    aabb_max = corners.max(axis=2)  # (N, T, 2)

    num_steps = valid.shape[1]
    flagged = np.zeros(len(tracks), dtype=bool)

    for t in range(num_steps):
        active_live = active_idxs[valid[active_idxs, t]]
        log_live = log_idxs[valid[log_idxs, t] & ~flagged[log_idxs]]
        if active_live.size == 0 or log_live.size == 0:
            continue

        a_min = aabb_min[active_live, t]
        a_max = aabb_max[active_live, t]
        l_min = aabb_min[log_live, t]
        l_max = aabb_max[log_live, t]
        overlap = (
            (a_max[:, None, 0] >= l_min[None, :, 0])
            & (a_min[:, None, 0] <= l_max[None, :, 0])
            & (a_max[:, None, 1] >= l_min[None, :, 1])
            & (a_min[:, None, 1] <= l_max[None, :, 1])
        )
        ai, li = np.where(overlap)
        if ai.size == 0:
            continue

        polys = {}
        for pa, pl in zip(ai, li):
            a_idx, l_idx = active_live[pa], log_live[pl]
            if flagged[l_idx]:
                continue
            if a_idx not in polys:
                polys[a_idx] = shapely_geom.Polygon(corners[a_idx, t])
            if l_idx not in polys:
                polys[l_idx] = shapely_geom.Polygon(corners[l_idx, t])
            if polys[a_idx].intersects(polys[l_idx]):
                flagged[l_idx] = True

    flagged_ids = [agent_ids[i] for i in np.flatnonzero(flagged)]
    for aid in flagged_ids:
        logger.debug("agent=%d: zeroing log agent due to overlap with active agent", aid)
        track = scenario.agents[aid]
        track.valid[:] = 0
        track.route = []
        track.route_gt_len = 0
        track.control_state = _compute_control_state(track)

    if flagged_ids:
        logger.debug("zeroed %d overlapping log agents", len(flagged_ids))


def _compute_all_corners(tracks):
    positions = np.stack([t.position[:, :2] for t in tracks], axis=0)  # (N, T, 2)
    headings = np.stack([np.asarray(t.heading) for t in tracks], axis=0)  # (N, T)
    lengths = np.stack([np.asarray(t.length) for t in tracks], axis=0)  # (N, T)
    widths = np.stack([np.asarray(t.width) for t in tracks], axis=0)  # (N, T)

    half_l = lengths / 2.0
    half_w = widths / 2.0
    # local corners: FR, FL, RL, RR, shape (N, T, 4, 2)
    local = np.stack(
        [
            np.stack([half_l, -half_w], axis=-1),
            np.stack([half_l, half_w], axis=-1),
            np.stack([-half_l, half_w], axis=-1),
            np.stack([-half_l, -half_w], axis=-1),
        ],
        axis=-2,
    )
    cos_h = np.cos(headings)[..., None]  # (N, T, 1)
    sin_h = np.sin(headings)[..., None]
    x = local[..., 0]
    y = local[..., 1]
    rot_x = cos_h * x - sin_h * y
    rot_y = sin_h * x + cos_h * y
    rotated = np.stack([rot_x, rot_y], axis=-1)  # (N, T, 4, 2)
    return rotated + positions[:, :, None, :]
