# Binary output review

Open work on the converted binaries, and the changes of 2026-10-20 and 2026-10-21. Earlier review results are in the
git history of this file (`1ed26ef`).

## Left to do

### py123d release

| # | Action | Where | Notes |
|---|---|---|---|
| 28 | Publish py123d and commit the pin | py123d `feat/opendrive-v2` at `fa58134`; 123Drive `pyproject.toml` | The local `[tool.uv.sources]` pin to `../py123d` is uncommitted and the branch is not pushed. CI installs `py123d==0.7.0`, which lacks the `SHOULDER` and `STATIC_OBJECT` layers |
| 29 | Commit the CARLA speed signs and scripts, then regenerate the CARLA bins | py123d `carla_maps/speed_signs.json`, `scripts/opendrive_parity/` | Town11 (1213 rows) and Town12 (2580 rows) speed signs and `export_speed_signs.py`, `check_speed_signs.py`, `check_lane_gaps.py` are uncommitted. The bin maps do not carry the new limits yet |

### CARLA fidelity

| # | Action | Where | Notes |
|---|---|---|---|
| 18 | Speed signs for Town13 | py123d `carla_maps/speed_signs.json` | Loading Town13 exceeds the 62 GB RAM and the server is OOM-killed, also with a 300 m `tile_stream_distance`. Its `.xodr` has no speed signals, so the signs need CARLA: add swap or use a larger machine |

### Blocked

| # | Action | Where | Notes |
|---|---|---|---|
| 7 | TL phases are `-1` for nuPlan and WOD | waits on source data | Their stop zones and detections carry no cycle; CARLA has 3039 / 4951 controls with phases |

## Changes

### 2026-10-21

- Lane-change routes and binary v3 (`f00a4fc`). Routes follow successor links or change lanes where both lanes match
  the same logged point; each route lane carries its `route_entry_s` and `route_exit_s` span in `cum_length`. Same
  routed vehicles as v2; 1844 nuPlan and 332 WOD routes change lanes. Before `--fix`, ego routes more than 5 m off
  their log drop from 9 to 3 in nuPlan and stay at 5 in WOD; other routes drop from 428 to 229 and from 156 to 110.
  PufferDinde `vcha/new_maps` reads v3 (`abf7060e8`) and can add random lane changes to generated routes
  (`route_lane_change_prob`, default 0, `8e7d8d2bd`). Its `--fix` results match v2 except 1 nuPlan agent, whose
  shorter v3 route ends before its off-road step.
- Format checks on v3 bins at `1ed26ef` (1247 nuPlan, 494 WOD, 12 CARLA; same accepted and rejected scenes as v2):
  every file decodes to its last byte, sections are aligned, values are finite, polygons are closed, `cum_length`
  is monotonic, IDs are unique, all references resolve, and ego is ID 0 and valid on every frame. Route spans have
  one entry per lane, lie inside the lane, and cover the whole lane across successor links.
- Lane changes leave a nonzero span on both lanes (former task 27). The DP no longer changes lanes from a lane's
  start or into its end, which removes the drop that joined a lane to the successor of its neighbour without the
  junction check. Before, 796 nuPlan and 68 WOD steps (45 on ego routes) changed lanes on a lane after a split or
  before a merge, and 2 spans had zero length; now none do. 915 nuPlan (50 ego) and 69 WOD (1 ego) routes change,
  and the counts above include the fix. The PufferDinde `--fix` audit was not rerun on these routes.

### 2026-10-20

- Non-ego vehicles whose log moves faster than 50 m/s between consecutive valid frames get no route
  (`MAX_JUMP_SPEED`, `6821f82`); 8 routes are cleared, and no routed vehicle jumps above 50 m/s.
- Same-type road lines within 0.1 m are one line (`LINE_DUPLICATE_DISTANCE`, `56ea740`), removing 20008 nuPlan road
  lines.
- CI and documentation require Python 3.12 (`71db23b`).
- Won't fix: WOD validity gaps of 1 or 2 frames stay in the log (interpolation would change the mask WOSAC scores
  against), and map leftovers stay as in the source (743 type-0 WOD lanes, 516 near-identical WOD lane pairs, and 36
  CARLA lanes under 10 cm all come from the source data).
- Speed signs for Town11 and Town12 exported from CARLA; on random routes, every observed limit change matches an
  exported row (see task 29).
- nuPlan and WOD reconverted at `88429dc` and audited with PufferDinde `vcha/new_maps` at `dd714dce9`. One `--fix`
  pass clears 1183 nuPlan and 286 WOD routes, leaving 16623 and 5327 frame-0 controllable vehicles; the last pass
  shows only mid-log ego infractions (collision / off-road / red light / stop sign: 16 / 21 / 38 / 36 in nuPlan,
  3 / 3 / 8 / 7 in WOD).
