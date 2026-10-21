# Binary output review

Open work on the converted binaries, and the changes of 2026-10-20 and 2026-10-21. Earlier review results are in the
git history of this file (`1ed26ef`).

## Left to do

### Conversion

| # | Action | Where | Notes |
|---|---|---|---|
| 31 | Road lines that become duplicates after simplification | `extractor.py` `_drop_duplicate_lines`, `geometry.py` `process_polylines` | 935 nuPlan-mini pairs (362 identical) among 775k lines, and 139 in the 113 held-out nuPlan-test scenes; none in WOD or CARLA. The dedup leaves none, but it runs before simplification: lines with the same ends and a kink of at least 0.1 m become straight and match afterwards. Dedup the simplified geometry |
| 32 | The `nuplan` preset does not infer traffic lights | `presets.toml` | `nuplan-mini` sets `infer_traffic_lights = true`; full nuPlan conversions keep raw detections only. Confirm whether this is intended |

### CARLA fidelity

| # | Action | Where | Notes |
|---|---|---|---|
| 18 | Speed signs for Town13 | py123d `carla_maps/speed_signs.json` | Loading Town13 exceeds the 62 GB RAM and the server is OOM-killed, also with a 300 m `tile_stream_distance`. Its `.xodr` has no speed signals, so the signs need CARLA: add swap or use a larger machine. Until then Town13 keeps its road limits (up to 120 km/h) |
## Changes

### 2026-10-21

- Final review at `2e43b89`, on the local datasets and on held-out data from
  `gs://valeo-cp2879-driving-policy/datasets/public`: 12 random nuPlan test logs (113 scenes, converted with
  `--preset nuplan --infer_traffic_lights`) and WOD validation shard 00000 (286 scenes). All scenes converted
  (local: 1247 nuPlan, 494 WOD, 12 CARLA, 20 rejections). Every format, reference, route-span, junction-ban,
  zero-span, and jump-rule check passes on all five sets. After `--fix`, nuPlan and WOD bins are byte-identical to
  the audited `b58a1f4` bins. The replay audit needs one pass everywhere and leaves only ego rows, none at step 0;
  held-out data clears 92 of 847 nuPlan and 159 of 3240 WOD frame-0 controllable vehicles (6.6% and 5.1% locally).
  Unknown speed limits are Singapore only (plus 1.8 km in Boston), as decided. Findings: tasks 31 and 32.
  Traffic-light inference was not re-scored; its code did not change.
- Won't fix: traffic-light phases stay `-1` for nuPlan and WOD (former task 7). PufferDinde uses `junction_id` and
  `phase_idx` only to generate light cycles for map-only data; logged scenes replay their per-frame states. CARLA
  has phases on 3039 of 4951 controls.
- CARLA speed limits come from signs (former task 29). py123d `ec19306` adds the Town11 (1213 rows) and Town12
  (2580 rows) speed signs and the CARLA scripts `export_speed_signs.py`, `check_speed_signs.py`, and
  `check_lane_gaps.py`. The Town11 and Town12 maps in `/data/py123d/data/maps/opendrive` were rebuilt at `ec19306`;
  the other towns keep their 2026-10-16 maps. The `opendrive` preset now uses `speed_limit_source = "signs"`, so lanes
  take the last sign's limit and 30 km/h before the first sign, as CARLA vehicles do. Road limits went up to 105 km/h;
  sign limits are 30 to 90 km/h (120 km/h on Town12 highways), and Town10HD, which has no signs, is 30 km/h
  throughout. Town13 is unchanged (task 18). The sign rows split lane sections: Town11 grows from 2357 to 4536
  lanes (54 to 112 MB) and Town12 from 7810 to 10400 (270 to 451 MB), mostly from the lane distance matrix.
- Town12 stop signs go from 628 to 627 (former task 30), still controlling all 1398 lanes. Two `.xodr` stop zones on
  overlapping lanes at (-1460, -4065) become one zone covering both lanes, 0.5 m upstream; 2 other stop lines move
  0.15 m and no traffic light or yield sign moves. The speed-sign section splits cause it: a Town12 map built at
  py123d `fa58134`, without the new sign rows, keeps 628.
- Lane-change routes and binary v3 (`f00a4fc`). Routes follow successor links or change lanes where both lanes match
  the same logged point; each route lane carries its `route_entry_s` and `route_exit_s` span in `cum_length`. Same
  routed vehicles as v2; 1844 nuPlan and 332 WOD routes change lanes. Before `--fix`, ego routes more than 5 m off
  their log drop from 9 to 3 in nuPlan and stay at 5 in WOD; other routes drop from 428 to 229 and from 156 to 110.
  PufferDinde `vcha/new_maps` reads v3 (`abf7060e8`) and can add random lane changes to generated routes
  (`route_lane_change_prob`, default 0, `8e7d8d2bd`).
- Format checks on v3 bins at `1ed26ef` (1247 nuPlan, 494 WOD, 12 CARLA; same accepted and rejected scenes as v2):
  every file decodes to its last byte, sections are aligned, values are finite, polygons are closed, `cum_length`
  is monotonic, IDs are unique, all references resolve, and ego is ID 0 and valid on every frame. Route spans have
  one entry per lane, lie inside the lane, and cover the whole lane across successor links.
- Lane changes leave a nonzero span on both lanes (former task 27). The DP no longer changes lanes from a lane's
  start or into its end, which removes the drop that joined a lane to the successor of its neighbour without the
  junction check. Before, 796 nuPlan and 68 WOD steps (45 on ego routes) changed lanes on a lane after a split or
  before a merge, and 2 spans had zero length; now none do. 915 nuPlan (50 ego) and 69 WOD (1 ego) routes change,
  and the counts above include the fix.
- Replay audit of the `b58a1f4` bins with PufferDinde `vcha/new_maps` at `8e7d8d2bd`: one `--fix` pass clears 1183
  nuPlan and 286 WOD routes and leaves 16623 and 5327 frame-0 controllable vehicles, as with v2. First-pass non-ego
  offenders (collision / off-road / red light / stop sign) are 119 / 520 / 402 / 185 in nuPlan (521 off-road with v2)
  and 26 / 114 / 101 / 54 in WOD. The last pass shows only the same mid-log ego infractions as v2, none at step 0.

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
