# PufferDrive Binary Format (.bin)

Binary files are written by `scenario_to_binary()` in `src/bin_factory/serialize.py`. All multi-byte values are little-endian. Floats are 32-bit IEEE 754 unless noted.

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
| int32 | `type` | 1=VEHICLE, 2=PEDESTRIAN, 3=CYCLIST, 4=OTHER |
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
| float32 | `goal_x` | Last valid x position |
| float32 | `goal_y` | Last valid y position |
| float32 | `goal_z` | Last valid z position |
| int32 | `mark_as_expert` | 0 if route exists, 1 otherwise |

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

| Type | Field |
|------|-------|
| int32 | `n_entry_lanes` |
| int32 × n_entry_lanes | `entry_lane_ids` |
| int32 | `n_exit_lanes` |
| int32 × n_exit_lanes | `exit_lane_ids` |
| float32 | `speed_limit` | m/s, -1.0 if unknown |

### Road element type ranges

| Range | Category | Values |
|-------|----------|--------|
| 0–9 | Lanes | 0=UNKNOWN, 1=FREEWAY, 2=SURFACE_STREET, 3=BIKE_LANE, 4=BUS_LANE |
| 10–19 | Road lines | 10=UNKNOWN, 11=BROKEN_WHITE, 12=SOLID_WHITE, 13=DOUBLE_SOLID_WHITE, 14=BROKEN_YELLOW, 15=BROKEN_DOUBLE_YELLOW, 16=SOLID_YELLOW, 17=DOUBLE_SOLID_YELLOW, 18=PASSING_DOUBLE_YELLOW |
| 20–29 | Road edges | 20=UNKNOWN, 21=BOUNDARY, 22=MEDIAN |
| 30+ | Areas | 30=UNKNOWN, 31=CROSSWALK, 32=SPEED_BUMP |

## Traffic Control Elements (× n_traffic_controls)

| Type | Field | Notes |
|------|-------|-------|
| int32 | `id` | Element ID |
| int32 | `type` | 1=TRAFFIC_LIGHT, 2=STOP_SIGN, 3=YIELD_SIGN |
| float32 × 6 | `stop_line` | Two 3D points |
| float32 | `heading` | |
| int32 | `n_states` | |
| int32 × n_states | `states` | 0=UNKNOWN, 1=GREEN, 2=YELLOW, 3=RED, 4=OFF |
| int32 | `n_controlled_lanes` | |
| int32 × n_controlled_lanes | `controlled_lane_ids` | |

## Objects (× n_objects)

Same dynamic state layout as agents, without route/goal/mark_as_expert:

| Type | Field | Notes |
|------|-------|-------|
| int32 | `id` | Object ID |
| int32 | `type` | Object type enum |
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
| float32 × n | `lane_lengths` | Arc length of each lane's polyline |
| float32 × n² | `distances` | Row-major shortest path matrix |

## Metadata (tail of file)

| Type | Field | Notes |
|------|-------|-------|
| char[128] | `id` | UTF-8, null-padded |
| int32 | `map_id` | |
| char[32] | `dataset` | UTF-8, null-padded |
| int32 | `scenario_length` | Number of timesteps |
| float32 | `dt` | Seconds between timesteps |
| int32 | `n_objects_of_interest` | |
| int32 × n | `objects_of_interest_ids` | |
| int32 | `n_tracks_to_predict` | |
| int32 × n | `tracks_to_predict_ids` | |
