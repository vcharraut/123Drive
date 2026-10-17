# Binary output review

This report compares converted binaries with the source py123d data.

## Current format

As of 2026-10-18, a nonempty route determines control eligibility. Binaries no longer store
`control_state` or `route_gt_len`. Ineligible agents keep their logged trajectories and validity
masks but have empty routes. PufferDinde chooses background behavior.

The measured datasets below use the old format. They have not been regenerated. The PufferDinde
`vcha/new_maps` reader and audit tool need matching updates before they can read or rewrite new bins.
In particular, `--fix` must clear routes instead of changing a control-state integer.

## Measurement setup

- The first review ran on 2026-10-16 at `13f5e80`. Commits `e0e1433` through `92945fb` produced
  byte-identical nuPlan and WOD bins. CARLA retained 4695 on-road static objects out of 11599 and
  gained 48 shoulder caps.
- The local setup uses py123d 0.8.0 from `../py123d`, branch `feat/opendrive-v2` at `fa58134`,
  through `[tool.uv.sources]`. These changes are uncommitted and py123d is not pushed. The committed
  `py123d==0.7.0` lacks the `SHOULDER` and `STATIC_OBJECT` layers this checkout uses. CI needs a
  published version with those layers. Publishing py123d is outside this review.
- The 2026-10-17 datasets at `92945fb`, after `--fix`, contain 1247 bins in `output/nuplan-mini`
  and 494 in `output/wod-motion`.
- The run covered every locally available scene. It converted 1753 scenes, rejected 20 through
  the ego checks, and reported no validation errors.

| Dataset | Scenes | Command |
|---|---|---|
| nuPlan-mini, all splits | 1247 / 1265 | `--preset nuplan-mini` |
| WOD-motion train | 494 / 496 | `--preset wod-motion` |
| CARLA Town01 through Town07, 10HD, 11, 12, 13, 15, map only | 12 / 12 | `--preset opendrive` |

## Conversion decisions

- Binaries never drop or invalidate log data. Ineligible agents have empty routes; PufferDinde chooses background replay, freezing, or exclusion.
- The converter rejects a scene if ego has no route, has less than 15 m of route ahead, starts off-road, or overlaps an agent at the start frame. The measured run rejected 20 scenes.
- A dataset needs conversion followed by the updated PufferDinde `scripts/replay_audit.py --fix`. The last audit pass must print only ego rows, with none at the start frame. Mid-log ego infractions remain in the data.
- Opposite-approach TL conflicts are soft, and lanes whose detections are only YELLOW/UNKNOWN stay uninterpolated.
- Speed limits of logged data stay as in the source: no default for Singapore (none in the map), no floor for the Las Vegas/WOD 5 and 10 mph lanes, no interpolation.
- The only consumer to stay compatible with is PufferDinde `vcha/new_maps`. PufferDrive 3.0 does not need to read current bins.
- Controllable vehicles with a short logged future stay controllable (13.6% of nuPlan and 12.6% of WOD end within 2 s). Vehicles with < 15 m of route ahead have their routes cleared.
- Heading flips are source noise and stay in the log, as log replay needs it unchanged.

## Old-format dataset review

These checks ran on 2026-10-17 against the final `output/` bins after `--fix`. The comparison used
the pre-fix `13f5e80` bins from 2026-10-16.

- Each of the 1741 decoded bins matched its 2026-10-16 counterpart except for `control_state`. Map, trajectory, and traffic-control measurements therefore stayed the same.
- `--fix` changed only `control_state`, never the ego's:

  | Change | nuPlan | WOD |
  |---|---|---|
  | Vehicle state 0 to 1, control candidate to replay | 1057 | 260 |
  | Vehicle state 2 to 1, frozen in a logged path to replay | 35 | 1 |
  | Pedestrian state 2 to 1 | 7 | 0 |
  | Total | 1099 | 261 |

- After `--fix`, 16697 nuPlan and 5341 WOD vehicles remained controllable. All had routes and valid frame-0 data. The audit reported no overlaps for those vehicles. Another 1925 nuPlan and 314 WOD vehicles replayed while retaining routes under the old format.
- `--fix` reduced controllable heading-flip counts from 82 to 1 in nuPlan and from 18 to 0 in WOD. The remaining nuPlan flip occurred at standstill. Duplicate controllable pairs fell from 2 to 0, and nuPlan controllable jumps above 50 m/s fell from 1 to 0.
- Five WOD vehicles remained controllable despite a single-frame jump of 5 to 7.5 m, between 1 and 11 frames before their track ends. Task 26 covers these cases. The jump changes the last valid position, which PufferDinde uses as the goal.
- `tl_score.py` not rerun: the TL code and its input are unchanged since `13f5e80`.

## Next actions

### P1 training data

| # | Action | Where | Size |
|---|---|---|---|
| 8 | Stop route extension at WOD exit links with a gap > 3 m | `routes.py` `_extend_route_to_dead_end` | 324 broken links in 155 scenes; 12 route extensions use one, 3 of them ego. They also shorten lane-graph distances |
| 26 | Clear routes of controllable vehicles whose log jumps above 50 m/s | `routes.py` | 5 WOD vehicles jump between 1 and 11 frames before the track end. Goals shift by 5 to 7.5 m. Scene and agent pairs are `4e55008decc258f7` 32, `c54c78b994225840` 98, `c7bb5289bdcf2e84` 2, `ddaad9bc12a65c49` 17, `e56d0ad1cdcd09ee` 39 |
| 9 | Stop lines: WOD stop signs listing far lanes; nuPlan TL lines narrower than their lanes | `traffic_controls.py` | 17 WOD references > 5 m away (one sign controls lanes 97 m off); 10% of nuPlan TL lines have the midpoint > 2 m from the first lane |

### P2 CARLA fidelity

| # | Action | Where | Notes |
|---|---|---|---|
| 21 | PufferDrive CARLA bridge `TOWN_OFFSETS` | PufferDrive `pufferlib/ocean/cosim/carla_bridge.py` | New bins sit 4 to 50 m from the old origin. The offset is the negative mean of lane centerline points. After correction, the 95th-percentile lane error is 7 cm. Store the origin in the bin or leave map-only conversions uncentred |
| 18 | Speed signs for Town11, 12, 13 | py123d `carla_maps/speed_signs.json` | The export found 1 trigger hit of 1173 signs in Town11. Likely cause: tiles load only around a hero vehicle. Retry with the car spawned as hero, else derive lanes from trigger boxes |
| 23 | Check road edges and lines against CARLA | `extractor.py`, py123d geometry | PufferDrive 3.0 and current edges sit 0.3 to 0.64 m apart. It is not yet clear which matches CARLA's kerbs |
| 12 | 541 successor links with a > 1 cm gap (Town12 286, Town13 239; 2 above 1 m in Town03/15) | raw OpenDRIVE | In the source |

### P3 lower priority or blocked

| # | Action | Where | Notes |
|---|---|---|---|
| 13 | `test_map_types_in_valid_ranges` caps types at 31 (carpark is 33); rerun the suite | `tests/bin_factory/test_real_data_e2e.py` | Suite not rerun since 2026-10-16 |
| 24 | Keep the bin decoder and the CARLA export/check scripts somewhere durable | outside git (`scripts/` stays untracked) | Today they live in session scratchpads and get rewritten each review |
| 10 | WOD validity gaps of 1 or 2 frames | `extractor.py` | 75% of 13770 gaps; 802 controllable vehicles after `--fix`. Interpolation would add frames and needs agreement to change the rule that log data stays unchanged |
| 11 | Residual repeated road lines | `extractor.py` | 9024 nuPlan lines (1.1%), identical only after simplification |
| 17 | Map leftovers | `extractor.py` | 36 CARLA lanes < 10 cm, 735 isolated type-0 WOD lanes, 44 duplicated WOD lanes; check the WOD ones against the source first |
| 16 | Lane-change routes | separate worktree | 9 nuPlan and 5 WOD ego GT routes deviate > 5 m |
| 7 | TL phases are `-1` for nuPlan and WOD | waits on source data | Their stop zones and detections carry no cycle; CARLA has 3039/3039 |

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

These measurements use the old-format data. The source noise remains in the trajectories.

- A heading changed by more than 120° in one frame for 3951 nuPlan vehicles across 16200 frames and 244 WOD vehicles across 631 frames. After `--fix`, one controllable nuPlan vehicle still had a flip at standstill.
- WOD had 19 vehicles with single-frame jumps above 50 m/s, with a maximum of 203 m/s. Five remained controllable after `--fix`; see task 26. nuPlan had 3 such vehicles, with a maximum of 55 m/s, and none remained controllable.
- Of 3954 WOD pedestrians, 163 exceeded 8 m/s, or about 4%.
- Among frozen vehicles, 51 nuPlan and 112 WOD tracks had a 90th-percentile speed above 2 m/s. Only 7 nuPlan and 23 WOD tracks had net movement above 1.5 m.
- WOD ego velocity differed from the motion direction by more than 5° in 4% to 5% of moving frames. The 99th-percentile difference was 9° to 12°.
- nuPlan type-0 lanes: 243 isolated by the crop, plus 4 Pittsburgh street↔bike connectors. Routes only use street and freeway lanes.

### Rejected scenes

- nuPlan 18: Boston log `2021.08.09.17.55.59_veh-28_00021_00307` 9 (ego parked ~10 m from any lane), Pittsburgh 5 (`2021.08.17.18.54.02_veh-45_00665_01065`, `2021.08.17.17.17.01_veh-45_02314_02798`), Singapore 2, Las Vegas 2 (one start-frame overlap).
- WOD 2: `360f14d603c9a914`, `48bde9cca82316c7`.

## Old-format replay audit

The 2026-10-17 audit used PufferDinde `vcha/new_maps` at `961b350bb` and the `92945fb` datasets in
`output/`. Its results matched the 2026-10-16 run on copies of the `13f5e80` bins.

| | nuPlan | WOD |
|---|---|---|
| Controllable before `--fix` | 17754 | 5601 |
| First-pass non-ego offenders: collision / off-road / red light / stop sign | 49 / 473 / 400 / 184 | 17 / 97 / 100 / 54 |
| Handed to log replay (passes) | 1099 (1083 + 13 + 2 + 1) | 261 (1 pass) |
| Controllable after | 16697 | 5341 |
| Last pass, ego only: collision / off-road / red light / stop sign | 12 / 21 / 38 / 36 | 3 / 3 / 8 / 7 |

The audit found no ego infraction at the start frame. It can report only infractions the simulator
detects. A traffic control that never triggers will not appear as a violation.

## Checks on the old-format output

- Every file decoded to its last byte using the format at `92945fb`. Agent, map, control, and object IDs were contiguous from 0. Actor and map-geometry arrays contained no NaN or infinity. Polygons were closed. Lane `cum_length` increased monotonically and ended at `length`.
- All lane references resolved. Entry and exit links were symmetric except for 6 WOD links.
- The median ratio of velocity to displacement-derived speed over 2 s was 0.987 to 1.001 for every class in nuPlan and WOD.
- Ego had ID 0, type VEHICLE, and controllable status. Every logged ego frame was valid.
- Only vehicles had routes. Each route step followed an exit link through street or freeway lanes, and every controllable vehicle had a route.
- After `--fix`, 16697 nuPlan and 5341 WOD vehicles remained controllable. None overlapped another agent at frame 0 or a replayed or frozen agent later. Later ego overlaps remained in 7 nuPlan scenes and 1 WOD scene. All WOD `tracks_to_predict` had valid frames.
- Static vehicles spanned at most 4.9 m in nuPlan and 5 m in WOD, apart from 3 short noisy WOD tracks.
- Lane-graph IDs matched the street and freeway lanes. The matrix had a zero diagonal and no negative distances.
- nuPlan and CARLA had no wrong-side or opposite-direction neighbour pairs. WOD had 12 wrong-side and 14 opposite-direction pairs among 50,000 pairs.
- Near-lane samples had road on the left of the edge in 92% of WOD, 97% of nuPlan, and 99% of CARLA cases.
- Traffic-light sequences had length T and states 0 through 4. No lane was controlled twice, and no control faced more than 45° away from its lane. WOD stop lines sat at lane starts. Stop-sign crossings reflected rolling stops rather than misplaced lines. Speeds were below 0.5 m/s in 28% of nuPlan crossings and 62% of WOD crossings.
- `metadata.dt` was 0.1 s. nuPlan had 201 frames, WOD had 91, and CARLA map-only scenes had `T=0`.

## Rerun the review

These commands produce the current format. Update the decoder and PufferDinde audit before
rerunning them. Do not use the old control-state writer on new bins. The measurements above
remain historical until the new output has been audited.

1. Convert into a scratch directory:

   ```bash
   uv run convert --preset nuplan-mini --py123d_path /data/py123d/data --output /tmp/review/nuplan --workers 12
   uv run convert --preset wod-motion  --py123d_path /data/py123d/data --output /tmp/review/wod --workers 10
   uv run convert --preset opendrive   --py123d_path /data/py123d/data --output /tmp/review/carla --workers 4
   ```

2. Decode with a reader that follows `src/bin_factory/serialize.py`. Task 24 tracks the missing committed decoder.
3. Traffic lights: `uv run python scripts/tl_score.py --preset wod-motion --impl none bin_factory.transforms.traffic_light_interpolation --cache /tmp/tl_cache --workers 12 --json tl.json`
4. After updating PufferDinde's reader and audit, run `python scripts/replay_audit.py /tmp/review/nuplan --fix` from `vcha/new_maps`. It rewrites the scratch bins in place. The last pass must print only ego rows, with none at step 0.
