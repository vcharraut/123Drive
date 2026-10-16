# Binary output data review

Review of `convert` output, done by decoding every produced `.bin` and checking it against the raw py123d data.

- **Date:** 2026-10-08, status updated 2026-10-10, everything re-measured 2026-10-13
- **Branch / commit:** `dev_v0.3.3` @ `4a2771d` + uncommitted `--edge_max_segment_length` default 10 → 1
- **py123d:** 0.7.0 (installed and pinned in `pyproject.toml`; `CLAUDE.md` still says 0.5.1). Since 2026-10-13 123Drive runs on the local checkout `../py123d` (`feat/opendrive-v2`, uncommitted changes) through `[tool.uv.sources]`
- **Scope:** every scene available locally: 1753 converted, 20 failures, 0 validation errors.

| Dataset | Scenes | Command |
|---|---|---|
| nuPlan-mini, all splits (train/val/test) | 1247 / 1265 | `--preset nuplan-mini` |
| WOD-motion train | 494 / 496 | `--preset wod-motion` |
| CARLA Town01–07, 10HD, 11, 12, 13, 15 (map only) | 12 / 12 | `--preset opendrive` |

nuPlan scenes per city: Las Vegas 995 (+2 failed), Boston 86 (+9), Pittsburgh 82 (+5), Singapore 84 (+2).
Decoded totals: 218k agents, 180k objects, 1.93M map elements (572k lanes), 57k traffic controls.

All 20 failures are the ego gate in `routes.py` (see "Checked in PufferDinde").

Previous review (70 scenes, 2026-10-06) is superseded; its items are re-measured below.

## Re-measured (2026-10-13)

Every check below was rerun on the 1753 binaries; numbers in this document are the re-measured ones unless a date says otherwise.

- The fixes of 2026-10-10 hold, and the PufferDinde audit and the TL scores come out identical.
- One new bug: nuPlan stop-sign headings are wrong on curved lanes (finding 9, task 2).
- Not reproduced: the 17 WOD vehicles driving with heading opposite to their velocity (finding 7).
- The check scripts of the first review were not kept, so the checks were written again. Small differences from the 2026-10-08 numbers can come from the method.

## Since this review (2026-10-10)

Fixed on `dev_v0.3.3`, re-measured on the same 1765 scenes:

| Commit | Change | Effect |
|---|---|---|
| `401fe84` | Speed-limit fill runs over the whole map, not the scene crop | Boston lanes with a limit 6% → 99.4% |
| `70030a4` | `invalid_agent_overlap` removed; overlap handled by `control_state` flags; short tracks judged by speed | No agent loses data (0 emptied, all 2099 WOD `tracks_to_predict` intact); 0 controllable vehicles overlap anything at frame 0 or a log/static agent later; frozen vehicles with p90 speed > 2 m/s 1975 → 48 (nuPlan) |
| `97c3e1f` | Road lines < 10 cm and duplicates (Hausdorff < 5 cm, same type) dropped; lane end hooks trimmed | nuPlan lines 1.20M → 0.80M; CARLA lines 35.5k → 18.0k, lane hooks 570 → 0, > 60° successor jumps 326 → 1 |
| `2020845` | `metadata.dt` is the nominal `--dt` | 0.1 exactly in every scene |

Binaries no longer drop or invalidate log data. `control_state` is the per-agent flag PufferDinde reads at init (0 control candidate, 1 replay from log, 2 frozen in self-play); see the "PufferDinde agent controller review" doc for how each mode uses it.

## Checked in PufferDinde (2026-10-10)

Every binary was loaded in PufferDinde (`vcha/new_maps`) and each controllable vehicle was replayed from its own log with `scripts/replay_audit.py`. An infraction there comes from the data or the simulator, not from a policy.

| Controllable vehicles (self-play) | nuPlan before | nuPlan after | WOD before | WOD after |
|---|---|---|---|---|
| Scenes | 1257 | 1247 | 496 | 494 |
| Controllable | 17830 | 16686 | 5609 | 5334 |
| Collision | 273 | 12 (ego) | 39 | 3 (ego) |
| Off-road | 498 | 21 (ego) | 105 | 3 (ego) |
| Red light | 457 | 40 (ego) | 115 | 8 (ego) |
| Stop sign | 209 | 32 (ego) | 61 | 7 (ego) |
| Removed at reset (at route end) | 34 | 0 | 2 | 0 |

The "after" columns use `--edge_max_segment_length 1.0` and the layout without the stored goal. After the changes no non-ego vehicle has an infraction, and no ego has one at the start frame. What is left is the ego's own log later in the scene, kept on purpose.

What changed:

- **123Drive, `routes.py`:**
  - a non-ego vehicle with less than 15 m of route ahead at the check frame is replayed (`control_state` 1, route kept);
  - conversion fails when the ego starts off-road, overlaps another agent at the start frame, or has less than 15 m of route ahead: 10 more nuPlan scenes and 2 WOD scenes.
- **PufferDinde, `scripts/replay_audit.py --fix`:** run on the converted directory, it rewrites `control_state` to 1 for every non-ego vehicle that collides, leaves the road, runs a red light or runs a stop sign in its own log. A frozen agent standing in a logged path is replayed instead of the vehicle that hits it. nuPlan: 1110 agents (6.0% of controllable vehicles); WOD: 268 (4.8%).
- **PufferDinde simulator:** frozen agents leave the scene when their log track ends. Before, they stayed as ghosts and caused 198 of the 273 nuPlan collisions.

A dataset is now produced by `convert` followed by `replay_audit.py --fix`; the second step has to be rerun when PufferDinde's infraction checks change.

Rerun on 2026-10-13 with PufferDinde `vcha/new_maps` @ `e2dcac8c1`: same "after" columns. `--fix` needs 4 rewriting passes on nuPlan (1094 + 13 + 2 + 1 agents) and 1 on WOD (268). The audit only sees infractions the simulator raises, so it cannot see a control that never fires (finding 9).

Rerun on 2026-10-14 with PufferDinde `vcha/new_maps` @ `43cdca48f`, on bins converted with the stop-zone heading fix (task 2): the last pass still has ego rows only, none at the start frame. nuPlan: 1116 agents handed to log replay (1100 + 13 + 2 + 1 over 4 passes), 16680 controllable left; ego infractions 12 collision, 21 off-road, 40 red light, 36 stop sign. WOD: unchanged (268 handed over, 5334 controllable). The first nuPlan pass finds 218 stop-sign infractions instead of 207: stop signs that pointed the wrong way now apply to their lanes. Rerun after the traffic-light fix (task 1): logged vehicles running a red light drop from 457 to 438 in the first nuPlan pass and from 115 to 108 in WOD, so fewer vehicles are handed to log replay (nuPlan 1099, 16697 controllable; WOD 261, 5341 controllable). The last pass still has ego rows only (nuPlan red light 38, was 40), none at the start frame.

---

## Findings (most severe first; finding 9 was added on 2026-10-13)

### 1. `invalid_agent_overlap` deletes WOD prediction targets

- 6 WOD `tracks_to_predict` (5 pedestrians, 1 cyclist, 6 scenes) come out with no valid frame. Before the transform they had 70–91 valid frames.
- Cause: `invalid_agent_overlap` (`src/bin_factory/transforms/invalid_agents.py:80`) zeroes every non-routed agent whose box touches a routed one at any frame. Pedestrians and cyclists never get routes, so they are always on the "log" side.
- Scenes: `269b95fa04b70912`, `394a75089673ce23`, `4d7c19aa675fd360`, `bc3c2aa6f879c1d2`, `bd0ddf7aee72b014`, `bea9ff2756e5e3b8`.
- Overall the transform empties 57 WOD agents (32 pedestrians) and 820 nuPlan agents (248 pedestrians, 88 cyclists).
- **Suggested fix:** never zero ids listed in `metadata.tracks_to_predict` / `objects_of_interest`. Also consider leaving pedestrians alone.
- **Status:** fixed in `70030a4` (transform removed). Re-measured: 0 of 2092 WOD `tracks_to_predict` are empty, and no agent is without a valid frame.

### 2. Duplicate tracks: two controllable vehicles in the same spot (nuPlan)

- 79 pairs of controllable vehicles, in 76 nuPlan scenes (6%), overlap by more than 50% of the smaller box.
  - Median centre distance 0.44 m, median overlap 18 frames (1.8 s), same size and speed.
  - So these are two track tokens for one car, e.g. scene `049f3a2a…`, agents 60 and 78, both from frame 0.
- In simulation both spawn and collide on the first step. `invalid_agent_overlap` only compares controllable with log agents, so it doesn't catch them.
- 1535 more pairs where at least one side is non-controllable. WOD has 0 controllable pairs.
- **Suggested fix:** before routing, merge or drop the shorter track of any vehicle pair overlapping more than 50% for at least 1 s.
- **Status:** fixed in `70030a4` without dropping data: such vehicles get `control_state` 1 (replayed), not 0. Re-measured: 2 controllable pairs are left in nuPlan, overlapping for 1 and 3 frames in the middle of the scene; after `replay_audit.py --fix` no non-ego collision remains.

### 3. TL interpolation: conflict model forces RED on moving traffic

Scored with `scripts/tl_score.py` on all scenes with lights (1234 nuPlan, 347 WOD):

| Metric | nuPlan raw | nuPlan interp | WOD raw | WOD interp |
|---|---|---|---|---|
| Coverage of signalized frames | 19.6% | 45.5% | 59.6% | 93.4% |
| Fidelity to raw detections | 100% | 99.89% | 100% | 99.97% |
| Red runs / 100 entries | 2.13 | 1.75 | 1.19 | **2.37** |
| Short phases / lane-min | 0.03 | 0.06 | 0.02 | **0.15** |
| G→R or R→Y / lane-min | 0.08 | 0.12 | 0.02 | **0.15** |
| Crossing pairs both green / pair-min | 1.79 | 1.93 | 0.56 | 0.56 |
| Lane hold-out accuracy | – | 96.9% | – | 97.5% |

The WOD regressions come from two things:

- **Permissive left turns are treated as hard conflicts.**
  - `_movement_groups` (`traffic_light_interpolation.py:125`) marks every crossing pair from different approaches as a conflict. That includes a left turn and the oncoming straight, which share a green in the US.
  - Of the conflict pairs, 25% (nuPlan) and 19% (WOD) are opposing-direction pairs.
  - Example WOD `347766e7d811c4`: left turners from the east approach and westbound straight traffic both enter. The HMM can't make both GO, so westbound straight gets RED from frame 49 to the end while 3 vehicles drive through it.
  - `tl_score` itself only counts perpendicular pairs as conflicts, so the scorer and the model disagree.
- **Flashing or isolated YELLOW detections get RED filled in around them.**
  - WOMD `FLASHING_CAUTION` maps to YELLOW and shows up as yellow / unknown alternating.
  - The HMM fills the gaps with RED, giving `YYYY RRRRRRR YYYY…` (WOD `58a876325e1b26f9`, `e4bed279d8a9b843`). A single 1 s yellow becomes `RRR…YYYYYYYYYY…RRR` (`a4efb612a711c1f3`).
  - Those scenes alone account for most of the illegal R→Y transitions.
- **Suggested fix:**
  - Count only perpendicular approaches (|cos| < 0.7 between start directions) as conflicts, or make opposing-left conflicts soft.
  - Leave lanes whose detections are all YELLOW or alternate YELLOW/UNKNOWN uninterpolated.
- **Status:** fixed 2026-10-14 (uncommitted), both suggested fixes, with opposite-approach conflicts made soft (-0.2 per frame both GO) rather than dropped (dropping them raised conflicting greens on nuPlan from 1.93 to 2.19 per pair-minute). Scored with `scripts/tl_score.py`:

  | Metric | WOD before | WOD after | nuPlan before | nuPlan after |
  |---|---|---|---|---|
  | Coverage of signalized frames | 93.4% | 92.8% | 45.5% | 45.9% |
  | Red runs / 100 entries | 2.37 | 1.32 | 1.75 | 1.55 |
  | Short phases / lane-min | 0.15 | 0.09 | 0.06 | 0.06 |
  | G->R or R->Y / lane-min | 0.15 | 0.06 | 0.12 | 0.11 |
  | Crossing pairs both green / pair-min | 0.56 | 0.56 | 1.93 | 1.92 |
  | Lane hold-out accuracy | 97.5% | 98.1% | 96.9% | 97.2% |
  | Gap hold-out accuracy | 98.0% | 98.6% | 96.1% | 96.3% |

  Raw WOD detections have 1.19 red runs per 100 entries but know the state at only 74% of entries; the 3 red runs added over raw are vehicles entering 0.5–1 s before the detector shows RED (late yellow).

TL controls that stay UNKNOWN for the whole scene: Las Vegas 14% (4148/29443), Boston 46%, Pittsburgh 47%, Singapore 52%, WOD 4% (138/3898).

### 4. Speed limits: missing, crop-dependent, or wrong

| Source | Lanes with a limit in the binaries |
|---|---|
| Singapore | 0% of 33068 lane instances (raw map has none) |
| Boston | 99.4% (25110 / 25253); 6% before `401fe84` |
| Las Vegas, Pittsburgh, WOD (car lanes), CARLA | 100% |
| WOD bike lanes / type-0 lanes | 38 / 5241 and 2 / 735 |

- **The Boston fill depended on the crop (fixed).**
  - `_fill_missing_speed_limits` (`extractor.py:547`) ran a BFS only over the lanes inside the 250 m crop.
  - On the full Boston map the same BFS fills 2984 / 3019 lanes from the 227 seeded ones. In a scene crop the seeds are usually outside, so nothing gets filled.
  - The same lane therefore gets a limit in map-only output but not in scene output.
  - **Fix:** fill on the full map (or cache per map) before cropping. Singapore still needs a per-location default (50 km/h).
- **Las Vegas 5 and 10 mph limits look wrong.**
  - They cover 20% of LV lane instances.
  - Vehicles on 5 mph lanes drive at a median 1.26× the limit, and 36% of samples are above 1.5×. On 10 mph lanes: median 1.0×, 18% above 1.5×. Every limit from 15 mph up has a median of 0.36–0.85× and at most 1.2% above 1.5×.
  - These are source values. Consider a floor (e.g. 15 mph) for routed lanes.
  - WOD's 5 mph lanes (582 lane instances) show the same pattern on 273 samples: median 1.15×, 40% above 1.5×.
- **CARLA Town13 limits are 1.6× too high.**
  - The xodr itself says `unit="mph"` with the same numbers Town12 uses in `km/h` (30/40/50/120), so 170 lanes come out at 120 mph = 53.6 m/s.
  - It is a CARLA source bug, but we pass it through. Town03's 50–55 mph urban limits also come straight from its xodr.
- **Status:** crop dependence fixed in `401fe84`; Singapore, Las Vegas and Town13 open (task 3).

### 5. CARLA Town12/13 lane centerlines start with a hook

- Raw py123d centerlines of 224 (Town12) and 111 (Town13) junction lanes start with a ~0.25 m segment at 90°, then 0.125 m back at ~178°, then the real lane.
- After simplification about 450 lanes keep a sub-0.5 m spur at the start and the end. Effects:
  - `heading[0]` is ~90° off.
  - 390 lanes get curvature above 0.1 1/m from the spur alone.
  - 326 exit links show a 85–90° heading jump.
  - TL/stop controls look 90° off their controlled lane.
- **Suggested fix:** drop leading/trailing segments under 0.5 m that turn more than 30°. Add a `PY123D-REPORT[opendrive]`.
- **Status:** fixed in `97c3e1f`.

### 6. Degenerate and duplicated road lines

- **Duplicates:** raw py123d road lines are about 40% duplicates, in either direction.
  - Pittsburgh: 3780 → 2282 unique. CARLA Town03: 2591 → 1586.
  - In nuPlan scene `00605dc7…`, 489 of 1146 `UNKNOWN` (type 10) lines are exact copies of another line, plus reversed copies.
  - They eat into the road-observation budget and the file size.
- **Tiny lines (CARLA):** 12.7k of 35.5k road lines (36%) are shorter than 10 cm.
  - Town12: 4782 of 10380; Town13: 4961 of 11077. They are already like this in py123d.
  - CARLA also has 36 lanes shorter than 10 cm (Town15, Town13).
- WOD has 49 duplicated lanes (same geometry and topology, different ids).
- **Suggested fix:** in `sanitize`, drop lines under ~0.5 m and dedupe lines by direction-agnostic rounded geometry.
- **Status:** fixed in `97c3e1f` at extraction (10 cm, Hausdorff dedupe). Still open:
  - 9024 of 795k nuPlan lines (1.1%) are repeated after simplification (task 11).
  - WOD has 44 lanes with the same geometry and topology as another lane (50 with the same geometry).
  - CARLA keeps 36 lanes under 10 cm: Town15 20, Town13 13, Town07 3 (task 17).

### 7. Heading flips still pass through

| Data | Vehicles with flips (>120° in one frame) | Flip frames | Controllable |
|---|---|---|---|
| nuPlan (1247) | 3951 | 16200 | 82 |
| WOD (494) | 244 | 631 | 18 |

- Pedestrians: 287 flip in nuPlan, 410 in WOD.
- The yaw-rate channel is gone, so the ±15.7 rad/s spikes of the last review no longer reach the binary. The raw headings are still unfiltered.
- Most nuPlan flips happen at standstill: only 114 flip frames (79 vehicles) are above 1 m/s. WOD: 374 frames (187 vehicles).
- Vehicle frames above 1 m/s with heading more than 90° off the velocity: 767 in nuPlan (293 vehicles), 2597 in WOD (947 vehicles). This is what the fix of task 5 would touch.
- The 17 WOD vehicles moving above 3 m/s with heading opposite their velocity for 5+ frames (2026-10-08) are not reproduced: 0 at 90°, 120° or 150°, from the velocity channel or from displacement.
- **Status:** open (task 5).

### 8. Smaller items

- **Removed agents keep stale data.** Fixed: `70030a4` no longer empties agents and `4a2771d` removed the stored goal.
- **Traffic-control groups section is all `-1`** in every dataset. py123d 0.5.1 stop zones have no `intersection_id` / `signal_group_id` / `signal_sequence`, and lights from detections never get them. The trailing section carries no information yet (task 7).
- **`metadata.dt` was the measured mean step.** Fixed in `2020845`: 0.1 in every scene. `metadata.location` is still not serialized (task 6).
- **WOD validity holes:** 29% of WOD vehicles (8264 / 28811) have gaps in `valid`; 53% of the 13770 gaps are single frames and 75% are 1–2 frames. 850 controllable vehicles have one. They come from the source. Consider interpolating 1–2 frame holes (task 10).
- **WOD neighbours:** 16% of neighbour references are one-way (8018 / 50376). 5620 point at a lane whose opposite-side neighbour is a different lane (likely because py123d keeps one per side); 2398 point at a lane with no neighbour on that side. Sides and directions are correct: 12 wrong-side and 14 opposite-direction references out of 50k (checked at the lane midpoint).
- **WOD broken exit links:**
  - 324 exit links in 155 scenes have a gap above 3 m (max 206 m). 237 of them also turn by more than 60°.
  - No GT route uses one, but 12 route extensions do, 3 of them ego routes. They also shorten lane-graph distances (task 8).
- **WOD stop signs listing far lanes:** 17 controlled-lane references lie more than 5 m from their stop line. In scene `9fee7172effd…`, a stop sign controls 3 lanes 97 m away (source data). Since `controlled_lanes[0]` sets the heading, a far lane can also skew it (task 9).
- **nuPlan TL stop lines from stop polygons:**
  - 10% have their midpoint more than 2 m from the first controlled lane (max 20 m).
  - 6% of the lane references lie more than 5 m outside the line's half-width. The polygon is narrower than the lanes it lists (task 9).
- **Raw track noise:**
  - WOD: 19 vehicles have single-frame jumps above 50 m/s (max 203 m/s), 6 of them controllable. nuPlan: 3 vehicles (max 55 m/s), 1 controllable.
  - 163 of 3954 WOD pedestrians (4%) exceed 8 m/s.
  - Frozen vehicles (`control_state` 2) with a p90 speed above 2 m/s: 51 in nuPlan, 112 in WOD. The speed is mostly noise: only 7 and 23 of them have a net displacement above 1.5 m, 1 and 4 above 3 m.
  - 3 frozen WOD vehicles span 6.2–8.5 m (`a4fe6991e8f98201` agent 25, `ddaad9bc12a65c49` agents 32 and 34). Each is a 3–7 frame track that ends with one 6–8 m jump.
- **WOD ego velocity** points along the heading: about 4–5% of moving frames are more than 5° off the motion direction (p99 9–12°). Other agents: 0.
- **20 scenes fail the ego gate** (no route, under 15 m of route ahead, off-road start, or overlap at the start frame):
  - nuPlan 18: Boston log `2021.08.09.17.55.59_veh-28_00021_00307` 9 (ego parked about 10 m from any lane), Pittsburgh 5 (logs `2021.08.17.18.54.02_veh-45_00665_01065` and `2021.08.17.17.17.01_veh-45_02314_02798`), Singapore 2, Las Vegas 2. One Las Vegas scene is the only start-frame overlap.
  - WOD 2: `360f14d603c9a914`, `48bde9cca82316c7`.
- **Lanes that stay `UNKNOWN` (type 0).**
  - nuPlan: 243 lanes isolated by the crop keep type 0 because type inference only sees lanes inside the crop.
  - Pittsburgh also has 4 real street↔bike connectors that stay type 0 (188 instances over the scenes).
  - WOD: 735 type-0 lanes (21.9 km), all without entry or exit links and 733 without a speed limit. Not checked against the source yet (task 17).
  - Low impact: routes only use street and freeway lanes.
- **CARLA Town11 has one traffic light with a zero-length stop line** (control 116, lane 1255). The lane boundaries probably coincide at the lane start: `_lane_width_from_boundaries` only falls back to 3.5 m when the width is 0 or less (task 17).
- **Ego routes miss lane changes.**
  - 9 nuPlan and 5 WOD ego GT routes deviate more than 5 m from the ego path (max 12.9 m), mostly at a lane change near the end (GT prefix only 1–3 lanes).
  - This is the lane-change route limitation, already designed in a separate worktree.

### 9. nuPlan stop-sign heading is taken at the wrong end of the lane

- For a stop zone, `process_traffic_controls` (`traffic_controls.py:84`) sets the heading with `_compute_heading_from_incoming_lanes(controlled_lanes[0])`. That is the travel direction at the **start** of the lane.
- nuPlan stop polygons reference the approach lane and sit at its **end**: of the 8576 nuPlan stop signs whose first lane passes within 2 m of the line, 6134 are in the last 10% of that lane and 374 in the first 10%. WOD and CARLA lines sit at the lane start, where the same code is right.
- So when the approach lane curves, the heading is off. Of the 9962 nuPlan stop signs with a controlled lane within 3 m of the line:

  | Heading vs lane direction at the stop line | Stop signs |
  |---|---|
  | more than 20° off | 1502 (15%) |
  | more than 45° off | 766 (7.7%) |
  | more than 90° off | 150 (1.5%) |

  WOD: 2 signs above 45°. CARLA: 0.
- The stop line is built perpendicular to this heading through the polygon centre, so it is rotated by the same angle.
- Effect in PufferDinde: `agent_is_near_and_aligned_with_stop_line` only applies a stop sign to agents within π/4 of its heading. The 766 signs should therefore never fire for the lane they control. This is read from the simulator code, not measured in a rollout, and the replay audit cannot see it (no infraction is raised).
- Examples (heading / lane direction at the line):
  - `020c36cf-fe2b-50a8-8038-1b7ea3733238`, control 3, lane 104 (118.7 m): −91° / +89°.
  - `0939c150-dcd3-5b98-9f7c-8db9bb8d041a`, control 11, lane 284 (30.1 m): −131° / +105°.
  - `0a9cb34a-c5d4-5640-b4ea-1b7ffecd6165`, control 25, lane 32 (25.1 m): 179° / 55°.
- The same code path sets the heading of TL controls built from stop polygons, but only 26 of 34304 nuPlan TL controls are more than 45° off.
- **Suggested fix:** for stop zones, take the heading from the direction of `controlled_lanes[0]` at its point nearest to the polygon centre. Keep the incoming-lane heading for lights built from detections, which sit at the lane start.
- **Status:** fixed 2026-10-14 (uncommitted): a stop zone's heading is the direction of the lane passing nearest its centre (controlled lanes and the lanes entering them). nuPlan stop signs more than 45° off: 776 → 0, more than 20°: 1540 → 5; polygon traffic lights 26 → 0; WOD stop signs 5 → 0. Replay audit rerun with `--fix`: last pass ego rows only (nuPlan 1116 agents handed to replay, 16680 controllable; WOD unchanged).

---

## Open tasks

Most important first. PufferDinde-side issues (WOSAC freezing non-target cars, IDM on parked cars, `control_agents`) are in the controller review doc, not here.

| # | Task | Where | Notes |
|---|---|---|---|
| 1 | ~~TL interpolation: opposing conflicts and flashing yellow~~ done 2026-10-14 | `traffic_light_interpolation.py` | Finding 3 |
| 2 | ~~nuPlan stop-sign heading~~ done 2026-10-14 | `traffic_controls.py` `_heading_near` | Finding 9 |
| 3 | Speed-limit defaults: Singapore (none in source), Las Vegas 5/10 mph floor, CARLA Town13 mph/km/h | `extractor.py` or presets | Finding 4; needs a decision per item. CARLA towns: `--speed_limit_source signs` (2026-10-14) gives the limits CARLA applies from its speed signs for Town01–07, 10HD and 15; Town11, 12, 13 have no sign export and keep the OpenDRIVE values |
| 4 | Controllable vehicles with a short logged future: 14% (nuPlan) and 13% (WOD) end within 2 s, 16% and 17% within 2–5 s; 37% and 52% last to the final frame | `routes.py` | Partly done: vehicles with < 15 m of route ahead at the start are replayed. A short logged track with a long route ahead is still controllable |
| 5 | Heading flips: snap vehicle heading to the velocity direction when > 90° off and speed > 1 m/s | `extractor.py` | Finding 7; touches 767 nuPlan and 2597 WOD frames. Flips at standstill (most of nuPlan's 16200) need another rule |
| 6 | ~~Serialize `metadata.location`~~ done 2026-10-14 (uncommitted) | `serialize.py`, viewer, `mapforge`, PufferDinde | `char[32] location` after `dataset` in the metadata (empty for WOD, which has none in py123d). The layout changes: older bins no longer load. PufferDinde reads it in the worktree `../PufferDinde-bin-location` (detached at `vcha/new_maps`), with its 8 shipped CARLA bins migrated; its 2 shipped nuPlan bins were already in the v0.3.2 layout |
| 7 | Traffic-control groups section is all `-1` | waits on a py123d release | The CARLA map files already store intersection, signal group and sequence per stop zone (py123d `feat/opendrive-v2`); the released 0.7.0 does not read them. With `pyproject.toml` pointed at the local checkout (2026-10-13) the CARLA groups are filled. nuPlan and WOD stay `-1` |
| 8 | Stop route extension at WOD exit links with a gap > 3 m | `routes.py` `_extend_route_to_dead_end` | 12 extensions (3 ego) cross one of the 324 broken links |
| 9 | Stop lines: WOD stop signs listing lanes far away (17 references); nuPlan TL lines narrower than their lanes (10% with the midpoint > 2 m from the first lane) | `traffic_controls.py` | Finding 8; task 2 is done, so this can start |
| 10 | WOD validity holes of 1–2 frames (75% of 13770 gaps; 850 controllable vehicles have a gap) | `extractor.py` | Would add interpolated frames; decide whether that fits "log data unchanged" |
| 11 | Residual repeated road lines (9024, 1.1% of nuPlan) | `extractor.py` | Raw copies differ by > 5 cm, identical after simplification |
| 12 | CARLA: 541 successor links with a > 1 cm gap (Town12 286, Town13 239) | check raw OpenDRIVE | Already in the source before `97c3e1f`; 2 are above 1 m (Town03, Town15) |
| 13 | Pre-existing test failures (15, also on `ba3270a`; not rerun on 2026-10-13) | `tests/bin_factory` | `test_map_types_in_valid_ranges` caps types at 31 (carpark is 33); binary walkers in `test_serialize.py` / `test_real_data_e2e.py` don't read the groups section |
| 14 | Stale doc: `docs/routes.rst` describes a 0.3 m off-road threshold for stationary vehicles that the code doesn't have | `docs/routes.rst` | |
| 15 | ~~20 scenes dropped by the ego gate (18 nuPlan, 2 WOD)~~ decided 2026-10-14: keep rejecting them | `routes.py` | No change |
| 16 | Lane-change routes | separate worktree | GT routes don't follow lane changes; 9 nuPlan and 5 WOD ego routes deviate > 5 m |
| 17 | Map leftovers: 36 CARLA lanes under 10 cm, one zero-length TL stop line (Town11), 735 isolated type-0 WOD lanes, 44 duplicated WOD lanes | `extractor.py`, `traffic_controls.py` | Findings 6 and 8; low impact, check the WOD ones against the source first |
| 18 | CARLA speed signs for the large maps (Town11, 12, 13): the CARLA export found almost no trigger hits there (1 of 1173 signs in Town11), so `--speed_limit_source signs` keeps the OpenDRIVE values | py123d `carla_maps/speed_signs.json` | Likely cause: tiles load only around a hero vehicle. Retry with the test car spawned as hero, else derive the lanes from the trigger boxes |
| 19 | Commit the CARLA work and drop the temporary py123d pin | py123d `feat/opendrive-v2`, 123Drive | py123d (uncommitted): traffic-light stop lines merged per signal group and placed 0–0.5 m before the junction entry; lanes cut at speed signs and `Lane.sign_speed_limit_mps`; CARLA stop and yield signs from an export (`carla_maps/speed_signs.json`, `stop_signs.json`). 123Drive (uncommitted): `--speed_limit_source`, stop-zone heading (`_heading_near`), docs. Then release py123d and remove `[tool.uv.sources]` |
| 20 | Regenerate the nuPlan and WOD datasets | `convert` + PufferDinde `replay_audit.py --fix` | The heading fix changes 1239 of 1247 nuPlan and 381 of 494 WOD bins (stop-zone headings and lines only). Checked on copies on 2026-10-14 (see "Checked in PufferDinde"); the real datasets are not regenerated |
| 21 | PufferDrive CARLA bridge offsets (`TOWN_OFFSETS`) | PufferDrive `pufferlib/ocean/cosim/carla_bridge.py` | The new CARLA bins sit 4–50 m from the old origin, so the constants and the 2 m calibration window no longer fit. The offset is exactly minus the bin's origin (the mean of the lane centreline points): checked in CARLA, lanes then match within 7 cm (p95) and heights exactly. Store the origin in the bin (the bridge's `town_offset` already reads a `centroid`) or leave map-only conversions uncentred |
| 22 | PufferDrive 3.0 compatibility | PufferDrive | The 3.0 loader expects the v0.3.2 layout (no neighbours, curvature or groups section) and cannot read current bins; `binaries/carla_lhs` is a mirror of an older export, not of `binaries/carla` |
| 23 | Road edges and road lines not checked against CARLA | `extractor.py`, py123d surfaces | 3.0 and current edges sit 0.3–0.64 m apart where both exist; which matches CARLA's curbs is unknown |
| 24 | Keep the CARLA export and check scripts | a repo of your choice | Speed signs, stop/yield signs, red-light lines, simulated drives, offsets: written during the 2026-10-13/14 review, only in the session scratchpad |

---

## Checked and fine

- **Format:** every file decodes to the last byte with the layout in `docs/binary-format.rst` (no yaw rate; groups section present). IDs are contiguous after reindex.
- **Structure:** no NaN/Inf. Polygons are closed. Element heading matches its segments. Lane `cum_length` is monotonic and ends at `length`.
- **References:** all entry/exit/neighbour/route/TC lane references point to lanes. Entry and exit links are symmetric except 6 WOD links.
- **Velocities** (2 s window, frames with speed above 1 m/s), median velocity / displacement:

  | Data | Vehicles | Pedestrians | Cyclists | Ego |
  |---|---|---|---|---|
  | nuPlan | 1.001 | 0.987 | 0.998 | 1.000 |
  | WOD | 0.999 | 1.000 | 0.999 | 1.000 |

  Per-frame nuPlan jitter: p99 17° between velocity and motion direction for non-ego.
- **Ego:** id 0, VEHICLE, CONTROLLABLE, valid on every frame, in all 1741 log scenes.
- **Routes:**
  - Only vehicles have routes. Every step follows an exit link, with no repeats and only street/freeway lanes.
  - Every route's `route_gt_len` is in [1, len]. GT start and end are reachable in the lane graph.
  - Every controllable vehicle has a route, and only vehicles are controllable. 868 nuPlan and 54 WOD vehicles handed to log replay by `convert` keep theirs.
- **Controllable vehicles:** 17754 in nuPlan and 5601 in WOD before `replay_audit.py --fix`. All are valid at frame 0; none overlaps another agent at frame 0 or a replayed or frozen agent later, except the ego later in its log (9 nuPlan and 2 WOD scenes).
- **Parked detection:** no static vehicle spans more than 4.9 m in nuPlan. In WOD none spans more than 5 m apart from 3 short noisy tracks (finding 8).
- **Moving vehicles without a route:** 99% (nuPlan) and 96% (WOD) are invalid at t=0, as `route_check_timestep=0` requires.
- **Lane graph:** ids are exactly the street/freeway lanes, zero diagonal, nothing negative. Each exit edge is at most the source lane's length. 0 issues.
- **Exit-link continuity:** nuPlan exit gaps are 0 m everywhere. CARLA has 2 links above 1 m.
- **Neighbours:** densified to 0.5 m; nuPlan and CARLA have 0 wrong-side and 0 opposite-direction pairs.
- **Road edges:** road on the left of the edge for 92% (WOD), 97% (nuPlan) and 99% (CARLA) of samples near a lane (nearest-lane heuristic).
- **Traffic controls:**
  - TL states have length T and values 0–4. Signs have no states.
  - No lane is controlled twice. Stop lines are perpendicular to the heading; the heading itself is wrong for 7.7% of nuPlan stop signs (finding 9).
  - WOD TL stop lines sit at the lane start (0 m).
- **Stop-sign behaviour:** of the vehicles crossing a stop line, 23% (nuPlan, 474 crossings) and 59% (WOD, 105 crossings) drop under 0.5 m/s in the 5 s before, and 68% and 86% under 2 m/s. On 2026-10-08 this read 25–44% and 54–79% for both datasets. Rolling stops, not misplaced lines; redo the nuPlan number after task 2.
- **nuPlan `TURN_STOP` zones:** mapped to STOP_SIGN, but in py123d they never reference lanes, so all are dropped. No stray stop signs.
- **Timing:** nuPlan 201 frames, WOD 91 frames, CARLA map-only `T=0`.
- **Speed vs limit** (where limits exist and look sane, 15+ mph): median 0.36–0.85×, at most 1.2% of samples above 1.5× (ignoring a 7-sample Pittsburgh bucket).

---

## How to reproduce

1. Convert everything into a scratch directory:

   ```bash
   uv run convert --preset nuplan-mini --py123d_path /data/py123d/data --output /tmp/review/nuplan --workers 12
   uv run convert --preset wod-motion  --py123d_path /data/py123d/data --output /tmp/review/wod --workers 10
   uv run convert --preset opendrive   --py123d_path /data/py123d/data --output /tmp/review/carla --workers 4
   ```

2. Decode with a reader that follows `docs/binary-format.rst`. Agents and objects have 9 float channels plus `valid`. Lanes carry topology, speed limit, length, `cum_length` and curvature. The file ends with the groups section.
3. Checks that surfaced the findings:
   - **Prediction targets:** count `tracks_to_predict` ids with no valid frame; rerun the pipeline with `invalid_agent_overlap` off to compare.
   - **Duplicate tracks:** box overlap above 50% of the smaller box between any two valid vehicles, per frame, split by `control_state`.
   - **Traffic lights:**
     - Run `scripts/tl_score.py` on all scenes:

       ```bash
       uv run python scripts/tl_score.py --preset wod-motion \
           --impl none bin_factory.transforms.traffic_light_interpolation --cache /tmp/tl_cache --workers 12 --json tl.json
       ```

     - Then classify the HMM's conflict pairs by start-direction dot product (opposing below −0.7).
   - **Speed limits:** convert the full map with `loader.extract_scenario(ArrowMapAPI(...))` and compare known-limit counts with scene crops; compare speed with the limit of the nearest GT route lane, bucketed by limit.
   - **Geometry:**
     - Endpoint spurs: first or last segment under 0.5 m turning more than 30°.
     - Duplicates: direction-agnostic rounded xy key.
     - Tiny lines: total length under 0.1 m.
   - **Heading flips / velocity:** heading change above 120° between consecutive valid frames; velocity vs displacement over a 2 s window.
   - **Stop-sign heading:** angle between the control's heading and the direction of each controlled lane at its point nearest to the stop-line midpoint, for lanes within 3 m of it.
4. Acceptance check, from a PufferDinde checkout of `vcha/new_maps` and on a copy of the binaries:

   ```bash
   python scripts/replay_audit.py /tmp/review/nuplan --fix
   ```

   The last pass must print only ego rows, none at step 0.
