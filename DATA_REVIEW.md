# Binary output review

This report compares converted binaries with the source py123d data.

## Current format

Since `f00a4fc`, 123Drive writes version 3 binaries (`123DRIVE` header with an 8-section directory: `AGENTS`,
`ROADS`, `TRAFFIC`, `OBJECTS`, `LANEGRPH`, `METADATA`, `TLPHASE1`, `SOURCE01`). Sections and indexed records are
8-byte aligned. Track records pack only the active `[first, first + n)` slice, compress constant dimensions and zero
`z` via a `flags` bitmask, and store `valid` as `uint8`. Agent records end with the route and, per route lane, the
`route_entry_s` and `route_exit_s` span in that lane's `cum_length`. `SOURCE01` records `origin_xyz`, `map_has_z`,
`timestamps_are_relative`, `timestamps_us`, source metadata strings, and per-agent/object `source_id` strings.

A nonempty `route` determines control eligibility. Binaries store no `control_state` or `route_gt_len`, and element
IDs keep their extracted values (ego is always ID 0). Ineligible agents keep their logged trajectories and validity
masks with empty routes. PufferDinde chooses background behavior.

## Measurement setup

- nuPlan and WOD were converted and audited on 2026-10-20 as v2 bins at `88429dc`, with PufferDinde `vcha/new_maps`
  at `dd714dce9`. On 2026-10-21 both were reconverted with the lane-change routes now in `f00a4fc` and audited with
  the local PufferDinde v3 reader (`abf7060e8`, `8e7d8d2bd`, not pushed). v3 keeps the same routed vehicles; the
  `--fix` results match v2 except 1 nuPlan agent, whose shorter v3 route ends before its off-road step. The counts
  below are the v2 ones. CARLA bins and the checks without a date are from the 2026-10-19 run at `2e6cc63`;
  map-only bins carry no routes, so v3 changes only their version.
- The local setup uses py123d 0.8.0 from `../py123d`, branch `feat/opendrive-v2` at `fa58134`,
  through `[tool.uv.sources]`. The pin is uncommitted and the py123d branch is not pushed. The committed
  `py123d==0.7.0` lacks the `SHOULDER` and `STATIC_OBJECT` layers this checkout uses. CI needs a
  published version with those layers. Publishing py123d is outside this review.
- The converted v2 datasets occupy 2111.3 MB for `nuplan-mini` (1247 bins), 211.8 MB for `wod-motion`
  (494 bins), and 726.2 MB for `opendrive` (12 bins).
- The run covered every locally available scene. It converted 1753 scenes, rejected 20 through
  the ego checks, and reported no validation errors.

| Dataset | Scenes | Command |
|---|---|---|
| nuPlan-mini, all splits | 1247 / 1265 | `--preset nuplan-mini` |
| WOD-motion train | 494 / 496 | `--preset wod-motion` |
| CARLA Town01 through Town07, 10HD, 11, 12, 13, 15, map only | 12 / 12 | `--preset opendrive` |

## Conversion decisions

- Binaries never drop or invalidate log data. Ineligible agents have empty routes; PufferDinde chooses background replay, freezing, or exclusion.
- The converter rejects a scene if ego has no route, has less than 15 m of route ahead, starts off-road, or overlaps an agent at the start frame.
- Non-ego vehicles search frames `0..10` (`MAX_INIT_TIMESTEP = 10`) for their first valid, on-road frame with at least 15 m of route ahead.
- A dataset needs conversion followed by a route-clearing `--fix` pass from PufferDinde `scripts/replay_audit.py`. The last audit pass must print only ego rows, with none at the start frame. Mid-log ego infractions remain in the data.
- Routes follow successor links, or change to a neighbouring lane where both lanes match the same logged point, never on junction lanes (inside an intersection, after a split, or before a merge). A lane left where it was entered, or changed into at its end, drops out of the route. 1893 nuPlan and 343 WOD routes change lanes. Before `--fix`, 3 nuPlan and 5 WOD ego routes and 219 nuPlan and 108 WOD other routes deviate more than 5 m from their log (9, 5, 428, and 156 without lane changes). PufferDinde can add random lane changes to generated routes (`route_lane_change_prob`, default 0).
- Opposite-approach TL conflicts are soft, and lanes whose detections are only YELLOW/UNKNOWN stay uninterpolated.
- Speed limits of logged data stay as in the source: no default for Singapore (none in the map), no floor for the Las Vegas/WOD 5 and 10 mph lanes, no interpolation.
- The only consumer to stay compatible with is PufferDinde `vcha/new_maps`. PufferDrive 3.0 does not need to read current bins.
- Controllable vehicles with a short logged future stay controllable (14.2% of nuPlan and 13.3% of WOD frame-0 controllable vehicles end within 2 s). Vehicles with < 15 m of route ahead have their routes cleared.
- Heading flips are source noise and stay in the log, as log replay needs it unchanged.
- Same-type road lines whose ends and Hausdorff distance are within 0.1 m, the simplification tolerance, are one line (`LINE_DUPLICATE_DISTANCE`). On 40 nuPlan scenes this keeps 22224 of 22903 lines and leaves 4 identical repeats; 40 WOD scenes and 4 CARLA towns are unchanged.
- Map leftovers stay as in the source (won't fix). The 743 type-0 WOD lanes (155 scenes) are type 0 in the raw WOMD protos with no links, neighbours, or boundaries; routes and the lane graph skip them. The 516 near-identical WOD lane pairs (845 lanes, 181 scenes, median 1.5 m) are in WOMD too (40 exact, the rest 1 to 5 cm apart) and share direction and links; 96 of 6145 routes (6 ego) end on one, and dropping a twin would not extend any. The 36 CARLA lanes under 10 cm (Town07 3, Town13 13, Town15 20) are 1 cm `.xodr` lane sections that carry lane graph links.
- WOD validity gaps of 1 or 2 frames stay in the log (won't fix). They are 75.5% (14151 / 18738) of internal gaps, and 640 frame-0 controllable vehicles (819 total routed) have one after `--fix`. Interpolation would change the validity mask that WOSAC scores against; PufferDinde `drive.h` marks a replayed agent `AGENT_SIM_STATUS_ABSENT` on those frames.
- Non-ego vehicles whose log moves faster than 50 m/s between consecutive valid frames get no route (`MAX_JUMP_SPEED` in `routes.py`). Their log stays unchanged.
- WOD entry and exit links whose endpoints are more than 3 m apart are dropped: 324 of each in 155 scenes.

## Dataset review

- PufferDinde `map_data.h` maps any agent with `route_length > 0` to `CONTROL_STATE_CONTROLLABLE` and all other
  agents to `CONTROL_STATE_REPLAY`. Because non-routed agents replay from the first pass, a single route-clearing
  `--fix` pass resolves every non-ego infraction (see [Replay audit](#replay-audit)).
- PufferDinde resets at `init_step = 0` and only activates agents with `valid[0] == 1`. Consequently, the 1392 nuPlan
  and 532 WOD late-init vehicles (`first` in `1..10`) are not active during `replay_audit.py` and keep their routes.
  Totals of routed vehicles are 19198 nuPlan and 6145 WOD before `--fix`, 18015 and 5859 after.
- Among frame-0 controllable vehicles (16623 nuPlan, 5327 WOD), 1 nuPlan vehicle (at standstill) and no WOD vehicle
  has a heading flip after `--fix`, and no routed vehicle jumps above 50 m/s. Among unaudited late-init routed
  vehicles, 12 nuPlan and 8 WOD vehicles have heading flips.
- `tl_score.py` reproduced the traffic-light table below. Fidelity to raw detections is 99.88% in nuPlan and 99.97%
  in WOD, below the script's 100% target.

## Next actions

### CARLA fidelity

| # | Action | Where | Notes |
|---|---|---|---|
| 18 | Speed signs for Town13 | py123d `carla_maps/speed_signs.json` | Town11 and Town12 are exported (see CARLA maps). Loading Town13 exceeds the 62 GB RAM and the server is OOM-killed, also with a 300 m `tile_stream_distance`. Its `.xodr` has no speed signals, so the signs need CARLA: add swap or use a larger machine |

### Blocked

| # | Action | Where | Notes |
|---|---|---|---|
| 7 | TL phases are `-1` for nuPlan and WOD | waits on source data | Their stop zones and detections carry no cycle; CARLA has 3039 / 4951 controls with phases |

## Open findings

### Traffic lights

The reviewed `tl_score.py` run covered all scenes with lights:

| Metric | nuPlan | WOD |
|---|---|---|
| Coverage of signalized frames | 45.9% | 92.8% |
| Red runs / 100 entries | 1.55 | 1.32 |
| Lane hold-out accuracy | 97.2% | 98.1% |

Controls that stay UNKNOWN for the whole scene: Las Vegas 14%, Boston 46%, Pittsburgh 47%, Singapore 52%, WOD 4%.

### Recorded track noise

The source noise remains in the trajectories:

- A heading changed by more than 120° in one frame for 3951 nuPlan vehicles across 16200 frames and 244 WOD vehicles across 631 frames. After `--fix`, one frame-0 controllable nuPlan vehicle still had a flip at standstill (plus 12 nuPlan and 8 WOD late-init routed vehicles).
- WOD had 19 vehicles with single-frame jumps above 50 m/s, with a maximum of 203 m/s, and nuPlan had 3, with a maximum of 55 m/s. None is routed.
- Of 3954 WOD pedestrians, 163 exceeded 8 m/s, or about 4%.
- Among low-displacement background vehicles, 51 nuPlan and 112 WOD tracks had a 90th-percentile speed above 2 m/s. Only 7 nuPlan and 23 WOD tracks had net movement above 1.5 m.
- WOD ego velocity differed from the motion direction by more than 5° in 4% to 5% of moving frames. The 99th-percentile difference was 9° to 12°.
- nuPlan type-0 lanes: 243 isolated by the crop, plus 4 Pittsburgh street↔bike connectors. Routes only use street and freeway lanes.

### CARLA maps

On 2026-10-19, CARLA 0.9.16 checked the installed towns (Town01 to Town07, Town10HD) headless with
downward `world.cast_ray` probes every 4 m across each edge and line. Link checks loaded every `.xodr.gz`
into a client-side `carla.Map`. Map-only bins map to CARLA as `x = carla_x`, `y = -carla_y`. PufferDrive 3.0
`TOWN_OFFSETS` are 0.3 to 1.2 m off, so its bins were calibrated against the lanes first.

- Road edges match CARLA's kerbs. Where the kerb has no gutter (Town01, 02), 123Drive edges sit
  on the kerb face (median +0.02 m). Where CARLA has a 0.45 m gutter `Shoulder` lane, labelled sidewalk at road
  height (Town03, 05, 06, 07, 10HD), they follow the road and gutter boundary, 0.47 m inside the kerb, with
  either `--shoulders` value. Wide paved shoulders stay 1.5 to 2.5 m inside the road through `shoulders = "offroad"`.
  PufferDrive 3.0 edges sit 0.18 to 0.28 m beyond the kerb face on the sidewalk and exclude parking lanes.
- Town04 guardrails stand 0.73 m inside both edge sets. The 3782 Town04 `BARRIER` objects (about 3 m by 0.4 m)
  represent them; coverage of each rail was not checked.
- Road lines lie on the painted markings: median 0.01 m, 90th percentile 0.06 to 0.13 m in Town04, 05, 07,
  and 10HD. Town01, 02, 03, and 06 markings cannot be hit by rays.
- Successor gaps above 1 cm: 441 of 540 are in the source. Town12 and Town13 roads with
  2.75 m lanes join 2.5 m junction connectors, which offsets their centres by 0.125 m; CARLA measures the same
  gap. The other 99 (Town12 49, Town13 35, Town15 9, Town06 5, Town07 1; up to 1.25 m in Town15) are links py123d
  adds: a lane opening or closing at zero width takes over its driving neighbour's links and runs at full width
  (`collection.py`, `lane_helper.py` on `feat/opendrive-v2`). These lanes overlap the neighbour near the split, and
  their centerlines deviate from CARLA's lane centre by up to half the lane width. All gaps are below
  `MAX_LANE_GAP`.
- Speed signs for Town11 and Town12 are in py123d `speed_signs.json`, uncommitted.
  Large maps put actors farther than `actor_active_distance` (2000 m) from a hero to sleep, and sleeping
  triggers never fire. With one hero kept alive and the distance raised to
  1e6 m, triggers fire without loaded tiles. A car placed every 2 m along each lane near a sign gives the first
  point where it takes the limit; for 30 km/h signs it first takes another limit from a nearby sign. Town11 has
  1213 rows from 1173 signs and Town12 has 2580 rows from 1947 signs. Driving 1 m steps along random routes, every
  observed limit change (512 in Town11, 35 in Town12) matches an exported row; 5 occur 0.1 to 1.5 m before the
  road boundary where the row starts. On Town04 the new export keeps every exported lane and limit, adds 4 lanes,
  and shifts some positions by 2 to 4 m. The `.xodr` files hold no speed signals, and the bin maps have not been
  regenerated with the new limits.
- CARLA scripts live in py123d `scripts/opendrive_parity/`, uncommitted: `export_speed_signs.py`
  writes `speed_signs.json` (it reproduces the Town04 rows exactly), `check_speed_signs.py` drives the random
  routes, and `check_lane_gaps.py` classifies successor gaps. On Town04 the route check leaves 2 of 21 changes
  unexplained; both follow a row of the same lane and limit by 3.8 to 4.4 m, a lag of unknown cause. Not kept: the
  edge and line ray probe (it reads 123Drive and PufferDrive 3.0 bins) and the CARLA gap and centerline checks,
  whose inputs came from scripts that were lost.

### Rejected scenes

- nuPlan 18: Boston log `2021.08.09.17.55.59_veh-28_00021_00307` 9 (ego parked ~10 m from any lane), Pittsburgh 5 (`2021.08.17.18.54.02_veh-45_00665_01065`, `2021.08.17.17.17.01_veh-45_02314_02798`), Singapore 2, Las Vegas 2 (one start-frame overlap).
- WOD 2: `360f14d603c9a914`, `48bde9cca82316c7`.

## Replay audit

The 2026-10-20 audit used PufferDinde `vcha/new_maps` at `dd714dce9` on the `88429dc` v2 binaries,
with the route-clearing `--fix` step from `4b3900d1c`:

| | nuPlan | WOD |
|---|---|---|
| Frame-0 controllable before `--fix` | 17806 | 5613 |
| First-pass non-ego offenders: collision / off-road / red light / stop sign | 119 / 521 / 402 / 185 | 26 / 114 / 101 / 54 |
| Routes cleared (`--fix` passes) | 1183 in 657 scenes (1 pass) | 286 in 185 scenes (1 pass) |
| Frame-0 controllable after `--fix` | 16623 | 5327 |
| Last pass, ego only: collision / off-road / red light / stop sign | 16 / 21 / 38 / 36 | 3 / 3 / 8 / 7 |

The audit found no ego infraction at the start frame. The audit reports only infractions the simulator detects; a
traffic control that never triggers will not appear as a violation.

## Replay modes

Measured on 2026-10-20 on the `--fix` bins. PufferDinde decides each mode's behaviour at runtime; the bins carry no
mode, "still", or reference-infraction flags.

- Control ego only, others on log replay (`control_sdc_only`): no ego infraction at step 0. Later ego infractions
  (collision / off-road / red light / stop sign) are 23 / 21 / 38 / 36 in nuPlan and 5 / 3 / 8 / 7 in WOD; collisions
  exceed the default mode's 16 / 3 because only this mode replays agents that appear after frame 0. Scoring can skip
  an infraction the logged ego also commits (DriveZero follows NAVSIM's human-reference filtering);
  `replay_audit.py --mode control_sdc_only --out` lists them per scene.
- Control routed vehicles, others frozen at their frame-0 pose: not reachable from the bins in `vcha/new_maps`, where
  an empty route always maps to replay. A box-overlap estimate (a frozen agent leaves at its log end, as in
  PufferDinde) finds that 1127 nuPlan (6.8%, 527 scenes, 142 ego) and 172 WOD (3.2%, 115 scenes, 21 ego) frame-0
  controllable vehicles drive their own log into a frozen agent, at a median 5.4 s / 3.9 s, with the agent staying
  7.3 s / 4.6 s more. All but 7 of those agents move in their log (net displacement > 1.5 m). Freezing only
  agents still in their log and dropping moving ones leaves 7 nuPlan conflicts and none in WOD. What happens to
  vehicles whose log hits a frozen agent is PufferDinde's choice.
- Control routed vehicles, no one else: the default audit already covers it, since only off-road, red light, stop
  sign, and collisions between controlled vehicles remain.

## Checks on the output

These checks ran on v2 bins and were not repeated on v3.

- Every file decoded to its last byte with `tests/bin_factory/binary_reader.py`, with 8-byte aligned sections and records, valid `SOURCE01` origins and timestamps, and no NaN or infinity in actor or map geometry. Polygons (`type` 30..33) were closed. Lane `cum_length` increased monotonically from 0.
- Element IDs preserved source extraction IDs (non-contiguous for roads in all scenes and for agents/objects in nuPlan); ego was always ID 0. All lane, route, traffic-control, `objects_of_interest`, and `tracks_to_predict` references resolved. Entry and exit links were symmetric except for 6 WOD entry links.
- The median ratio of velocity to displacement-derived speed over 2 s was 0.975 to 1.004 across all agent classes in nuPlan and WOD.
- Ego had ID 0, type VEHICLE, and a nonempty route. Every logged ego frame was valid.
- Only vehicles had routes. Without lane changes, each route step followed an exit link through street or freeway lanes.
- After `--fix`, none of the frame-0 controllable vehicles overlapped another agent at frame 0 or a replayed agent later. Later ego collisions remained in 16 nuPlan scenes and 3 WOD scenes. All WOD `tracks_to_predict` had valid frames.
- Static vehicles spanned at most 4.9 m in nuPlan and 5 m in WOD, apart from 3 short noisy WOD tracks.
- Lane-graph IDs matched the street and freeway lanes in road-section order. The matrix had a zero diagonal and no negative distances.
- nuPlan and CARLA had no wrong-side or opposite-direction neighbour pairs. WOD had 12 wrong-side and 14 opposite-direction pairs among 50,000 pairs.
- Near-lane samples had road on the left of the edge in 92% of WOD, 97% of nuPlan, and 99% of CARLA cases.
- Traffic-light sequences had length `T` and states 0 through 4. No lane was controlled twice. Stop-sign crossings reflected rolling stops rather than misplaced lines; speeds were below 0.5 m/s in 28% of nuPlan crossings and 62% of WOD crossings.
- `metadata.dt` was 0.1 s. nuPlan had 201 frames, WOD had 91, and CARLA map-only scenes had `T=0` and `dt=0.0`.

## Rerun the review

1. Convert into a scratch directory:

   ```bash
   uv run convert --preset nuplan-mini --py123d_path /data/py123d/data --output /tmp/review/nuplan --workers 12
   uv run convert --preset wod-motion  --py123d_path /data/py123d/data --output /tmp/review/wod --workers 10
   uv run convert --preset opendrive   --py123d_path /data/py123d/data --output /tmp/review/carla --workers 4
   ```

2. Decode with `tests/bin_factory/binary_reader.py` (`parse_binary`), which validates the v3 layout against `src/bin_factory/serialize.py`.
3. Traffic lights, for `nuplan-mini` and `wod-motion`: `uv run python scripts/tl_score.py --preset wod-motion --impl none bin_factory.generators.traffic_light_inference --cache /tmp/tl_cache --workers 12 --json tl.json`.
4. Build PufferDinde `vcha/new_maps` in a scratch copy (`git archive vcha/new_maps | tar -x -C <copy>`, then `NO_TRAIN=1 python setup.py build_ext --inplace --force`) and run `PYTHONPATH=<copy> python scripts/replay_audit.py /tmp/review/nuplan --fix` from it with PufferDinde's venv. Without `PYTHONPATH`, the venv's editable install loads the checkout's `binding.so`, which crashes on bins of another version when the checkout is on another branch and leaves the worker pool hanging. The last pass must print only ego rows, with none at step 0.
