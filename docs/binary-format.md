# PufferDrive Binary Format (.bin)

Binary files are written by `puffer_dict_to_binary()` in `src/bin_factory/serialize.py`. All multi-byte values are little-endian. Floats are 32-bit IEEE 754 unless noted.

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
| int32 | `id` | Sequential index (0-based) |
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
| float32 | `goal_x` | Last valid position x |
| float32 | `goal_y` | Last valid position y |
| float32 | `goal_z` | Last valid position z |
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

**Lane-only fields** (type 1–9):

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
| 1–9 | Lanes | 1=FREEWAY, 2=SURFACE_STREET, 3=BIKE_LANE |
| 10–19 | Road lines | 10=UNKNOWN, 11=BROKEN_WHITE, 12=SOLID_WHITE, 13=DOUBLE_SOLID_WHITE, 14=BROKEN_YELLOW, 15=BROKEN_DOUBLE_YELLOW, 16=SOLID_YELLOW, 17=DOUBLE_SOLID_YELLOW, 18=PASSING_DOUBLE_YELLOW |
| 20–29 | Road edges | 20=UNKNOWN, 21=BOUNDARY, 22=MEDIAN, 23=SIDEWALK |
| 31+ | Areas | 31=CROSSWALK, 32=SPEED_BUMP, 33=DRIVEWAY |

## Traffic Control Elements (× n_traffic_controls)

| Type | Field | Notes |
|------|-------|-------|
| int32 | `id` | Element ID |
| int32 | `type` | 1=TRAFFIC_LIGHT, 2=STOP_SIGN, 3=YIELD_SIGN |
| float32 | `x` | Position |
| float32 | `y` | |
| float32 | `z` | |
| int32 | `n_states` | |
| int32 × n_states | `states` | Per-timestep light state (see below) |
| int32 | `n_controlled_lanes` | |
| int32 × n_controlled_lanes | `controlled_lane_ids` | |

### Traffic light states

| Value | Meaning |
|-------|---------|
| 0 | UNKNOWN / UNOBSERVED |
| 1 | ARROW_RED |
| 2 | ARROW_YELLOW |
| 3 | ARROW_GREEN |
| 4 | RED |
| 5 | YELLOW |
| 6 | GREEN |
| 7 | FLASHING_RED |
| 8 | FLASHING_YELLOW |

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
| char[128] | `scenario_id` | UTF-8, null-padded |
| int32 | `map_id` | |
| char[64] | `dataset_name` | UTF-8, null-padded |
| int32 | `scenario_length` | Number of timesteps |
| int32 | `sdc_index` | Index into agents array for ego vehicle |
| int32 | `n_objects_of_interest` | |
| int32 × n | `objects_of_interest_ids` | |
| int32 | `n_tracks_to_predict` | |
| int32 × n | `tracks_to_predict_ids` | |
