# PufferDrive Binary Format (.bin)

Binary files are written by `scenario_to_binary()` in `src/bin_factory/serialize.py`. All multi-byte values are little-endian. Floats are 32-bit IEEE 754 unless noted. Enum integer values (`type`, `control_state`, `states`, …) are defined in `src/bin_factory/puffer_types.py` — that file is the single source of truth; this doc only names the enum.

## Header

| Offset | Type | Field |
|--------|------|-------|
| 0 | int32 | `n_agents` |
| 4 | int32 | `n_road_elements` |
| 8 | int32 | `n_traffic_controls` |
| 12 | int32 | `n_objects` |

## Agents (× n_agents)

Each agent is laid out sequentially:

| Type | Field | Notes |
|------|-------|-------|
| int32 | `id` | Track ID as stored in the scenario |
| int32 | `type` | `AgentType` enum |
| int32 | `T` | Trajectory length (number of timesteps) |
| float32 × T | `x` | Column-major: all x values first |
| float32 × T | `y` | |
| float32 × T | `z` | |
| float32 × T | `heading` | Radians |
| float32 × T | `vx` | Velocity x component |
| float32 × T | `vy` | Velocity y component |
| float32 × T | `length` | Bounding box length per timestep |
| float32 × T | `width` | Bounding box width per timestep |
| float32 × T | `height` | Bounding box height per timestep |
| int32 × T | `valid` | 1 if observation exists at this timestep |
| int32 | `n_route_lanes` | Number of lane IDs in route |
| int32 × n_route_lanes | `route_lane_ids` | Ordered lane ID sequence |
| int32 | `route_gt_len` | Number of leading route lanes supported by GT before extension |
| float32 | `goal_x` | Last valid x position |
| float32 | `goal_y` | Last valid y position |
| float32 | `goal_z` | Last valid z position |
| int32 | `control_state` | `ControlState` enum |

## Road Map Elements (× n_road_elements)

| Type | Field | Notes |
|------|-------|-------|
| int32 | `id` | Element ID |
| int32 | `type` | See type ranges below |
| int32 | `N` | Number of polyline/polygon points |
| float32 × N | `x` | Column-major |
| float32 × N | `y` | |
| float32 × N | `z` | |
| float32 × N | `heading` | Per-point heading (radians) |

**Lane-only fields** (type 0–9):

| Type | Field | Notes |
|------|-------|-------|
| int32 | `n_entry_lanes` | |
| int32 × n_entry_lanes | `entry_lane_ids` | |
| int32 | `n_exit_lanes` | |
| int32 × n_exit_lanes | `exit_lane_ids` | |
| float32 | `speed_limit` | m/s, -1.0 if unknown |
| float32 | `length` | Total polyline arc length (meters) |
| float32 × N | `cum_length` | Per-point cumulative arc length, `cum[0]=0`, `cum[-1]=length` |

### Road element type ranges

The `type` int selects a category by range; exact values live in `puffer_types.py` (`is_road_lane` / `is_road_line` / `is_road_edge` test these ranges).

| Range | Category | Enum |
|-------|----------|------|
| 0–9 | Lanes | `LaneType` |
| 10–19 | Road lines | `RoadLineType` |
| 20–29 | Road edges | `RoadEdgeType` |
| 30+ | Areas | `MiscRoadType` |

## Traffic Control Elements (× n_traffic_controls)

| Type | Field | Notes |
|------|-------|-------|
| int32 | `id` | Element ID |
| int32 | `type` | `TCType` enum |
| float32 × 6 | `stop_line` | Two 3D points |
| float32 | `heading` | |
| int32 | `n_states` | |
| int32 × n_states | `states` | `TLState` enum (per-frame) |
| int32 | `n_controlled_lanes` | |
| int32 × n_controlled_lanes | `controlled_lane_ids` | |

## Objects (× n_objects)

Same dynamic state layout as agents, without route/goal:

| Type | Field | Notes |
|------|-------|-------|
| int32 | `id` | Object ID |
| int32 | `type` | `ObjectType` enum |
| int32 | `T` | Trajectory length |
| float32 × T | `x` | Column-major |
| float32 × T | `y` | |
| float32 × T | `z` | |
| float32 × T | `heading` | Radians |
| float32 × T | `vx` | Velocity x component |
| float32 × T | `vy` | Velocity y component |
| float32 × T | `length` | Bounding box length per timestep |
| float32 × T | `width` | Bounding box width per timestep |
| float32 × T | `height` | Bounding box height per timestep |
| int32 × T | `valid` | 1 if observation exists at this timestep |

## Lane Graph Distances

All-pairs shortest lane-to-lane distances precomputed via Dijkstra on FREEWAY and SURFACE_STREET lanes. Edge weight = source lane arc length. Unreachable pairs stored as IEEE 754 `inf`.

| Type | Field | Notes |
|------|-------|-------|
| int32 | `n_lanes_graph` | Number of lane nodes (0 = no graph data) |
| int32 × n | `lane_ids` | Lane IDs in matrix row/col order |
| float32 × n² | `distances` | Row-major shortest path matrix |

## Metadata (tail of file)

| Type | Field | Notes |
|------|-------|-------|
| char[128] | `id` | UTF-8, null-padded |
| char[32] | `dataset` | UTF-8, null-padded |
| int32 | `scenario_length` | Number of timesteps |
| float32 | `dt` | Seconds between timesteps |
| int32 | `n_objects_of_interest` | |
| int32 × n | `objects_of_interest_ids` | |
| int32 | `n_tracks_to_predict` | |
| int32 × n | `tracks_to_predict_ids` | |

## Traffic Light Phases (optional tagged section)

Written after the metadata by every current serializer; readers treat its absence as "no phase info".

| Type | Field | Notes |
|---|---|---|
| char[8] | `tag` | `TLPHASE1` |
| int32 × 2 × n_traffic_controls | `junction_id`, `phase_idx` | In traffic-control order. `-1, -1` when the element is not part of a signalized junction cycle. Per junction, `phase_idx` is compact `0..N-1`; lights with the same phase are green together, phases cycle in index order. |
