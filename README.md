# 123Drive

Converts [py123d](https://github.com/autonomousvision/py123d) data to [PufferDrive](https://github.com/Emerge-Lab/PufferDrive) binary format.

## Pipeline

Three tools, run in order:

```
Raw dataset  →  [py123d-docker]  →  py123d Arrow files  →  [bin_factory]  →  .bin  →  [viz]
```

1. **`py123d-docker`** — Docker wrapper that runs py123d's extraction pipeline, producing Arrow files per dataset/split.
2. **`bin_factory`** — Reads Arrow files, applies transforms (interpolation, traffic lights, validation), outputs `.bin` scenarios.
3. **`viz`** — FastAPI server for inspecting `.bin` files in a browser.

## Usage

```bash
# Basic conversion
uv run convert --dataset_path /path/to/data --output_dir ./output

# With traffic lights and strict validation
uv run convert --dataset_path /path/to/data --output_dir ./output --traffic_lights --validate_level 2

# Parallel processing with dataset filtering
uv run convert --dataset_path /path/to/data --output_dir ./output --num_workers 8 --datasets nuplan --max_scenarios 100
```

## Options

**Core:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset_path` | required | Path to py123d dataset (logs/ and maps/) |
| `--output_dir` | `./output` | Directory to save binary files |
| `--num_workers` | `1` | Number of parallel workers |

**Dataset filtering:**

| Flag | Default | Description |
|------|---------|-------------|
| `--max_scenarios N` | all | Limit number of scenarios |
| `--datasets` | all | Dataset names to include (e.g. `nuplan wod-motion`) |
| `--split_types` | all | Split types (e.g. `train val test`) |
| `--split_names` | all | Split names (e.g. `nuplan-mini_val`) |
| `--log_names` | all | Specific log names |
| `--duration_s` | `0` | Scenario duration in seconds (0 = full) |
| `--history_s` | `0` | History duration in seconds |
| `--map_only` | off | Load map-only scenarios (no logs) |

**Transforms & validation:**

| Flag | Default | Description |
|------|---------|-------------|
| `--traffic_lights` | off | Generate synthetic traffic light sequences |
| `--validate_level` | `1` | 0 = none, 1 = soft (structure), 2+ = strict (physics) |
| `--max_segment_length` | `2.0` | Max segment length for polyline interpolation (m) |
| `--area_threshold` | `0.1` | Visvalingam-Whyatt simplification threshold (0 = disabled) |
| `--dist_threshold` | `10.0` | Distance threshold for road graph |
| `--min_route_valid_points` | `0` | Min valid trajectory points for route computation (0 = no filter) |
| `--route_check_timestep` | `0` | Timestep at which agent must be valid for route computation |

## Visualize output

```bash
uv pip install -e ".[viz]"
viz --dir ./output --port 8080
```

Open http://localhost:8080 to inspect generated `.bin` scenarios.

## Data

Features available in py123d and their extraction status. [X] = extracted, [ ] = not extracted.

### Agents

**Ego vehicle** (from `EgoStateSE3`):

| Feature | Extracted | Notes |
|---------|-----------|-------|
| Position (x, y, z) | [X] | Center SE3 pose, centered around map centroid |
| Heading (yaw) | [X] | From `center_se3.pose_se2.yaw` |
| Velocity (vx, vy) | [X] | 2D only, from `dynamic_state_se3.velocity_3d` |
| Length, width, height | [X] | From `vehicle_parameters` |
| Valid mask | [X] | Always true for ego |
| Velocity (vz) | [ ] | Z-component of 3D velocity |
| Acceleration (ax, ay, az) | [ ] | Full 3D acceleration vector |
| Angular velocity (roll, pitch, yaw rates) | [ ] | 3D angular velocity |
| Tire steering angle | [ ] | |
| Wheel base | [ ] | Axle geometry from `vehicle_parameters` |
| Rear axle to center offsets | [ ] | Longitudinal + vertical |

**Other agents** (from `BoxDetectionSE3`):

| Feature | Extracted | Notes |
|---------|-----------|-------|
| Position (x, y, z) | [X] | From `bounding_box_se3.center_se3` |
| Heading (yaw) | [X] | |
| Velocity (vx, vy) | [X] | 2D only, from `velocity_2d` |
| Length, width, height | [X] | From bounding box dimensions |
| Valid mask | [X] | True when detection exists at frame |
| Track token | [X] | Used internally for identity tracking |
| Agent type | [X] | Mapped to vehicle/person/bicycle/other |
| Velocity (vz) | [ ] | Z-component from `velocity_3d` |

**Agent type mapping** (py123d `DefaultBoxDetectionLabel` to puffer int):

| py123d label | Puffer type | Extracted |
|--------------|-------------|-----------|
| EGO | 1 (vehicle) | [X] |
| VEHICLE | 1 (vehicle) | [X] |
| PERSON | 2 (pedestrian) | [X] |
| BICYCLE | 3 (cyclist) | [X] |
| TRAIN | 4 (other) | [X] |
| ANIMAL | 4 (other) | [X] |
| TRAFFIC_SIGN | 4 (other) | [X] |
| TRAFFIC_CONE | 4 (other) | [X] |
| TRAFFIC_LIGHT | 4 (other) | [X] |
| BARRIER | 4 (other) | [X] |
| GENERIC_OBJECT | 4 (other) | [X] |

**Puffer-computed fields** (not from py123d):

| Feature | Notes |
|---------|-------|
| Routes | Lane ID sequences computed from GT trajectory + map |
| Goal (x, y, z) | Last valid position |
| mark_as_expert | 1 if no route found, 0 otherwise |

### Maps

**Lane** (`MapLayer.LANE`):

| Feature | Extracted | Notes |
|---------|-----------|-------|
| Centerline polyline (3D) | [X] | |
| Speed limit | [X] | Stored as m/s |
| Predecessor lanes | [X] | Connectivity |
| Successor lanes | [X] | Connectivity |
| Left/right lane IDs | [X] | Stored in intermediate, not output to binary yet |
| Lane type | [ ] | Always hardcoded to SURFACE_STREET |
| Left boundary polyline | [ ] | |
| Right boundary polyline | [ ] | |
| Lane group ID | [ ] | |
| Outline polygon | [ ] | |

**Road line** (`MapLayer.ROAD_LINE`):

| Feature | Extracted | Notes |
|---------|-----------|-------|
| Polyline (3D) | [X] | |
| Road line type | [X] | 14 types mapped (dashed/solid white/yellow, etc.) |

**Road edge** (`MapLayer.ROAD_EDGE`):

| Feature | Extracted | Notes |
|---------|-----------|-------|
| Polyline (3D) | [X] | |
| Road edge type | [X] | BOUNDARY, MEDIAN |

**Crosswalk** (`MapLayer.CROSSWALK`):

| Feature | Extracted | Notes |
|---------|-----------|-------|
| Outline polygon (3D) | [X] | |

**Not extracted map layers:**

| Layer | Extracted | Notes |
|-------|-----------|-------|
| LANE_GROUP | [ ] | Lane group boundaries, connectivity, intersection membership |
| INTERSECTION | [ ] | Intersection outlines and lane group associations |
| WALKWAY | [ ] | Pedestrian walkway surfaces |
| CARPARK | [ ] | Parking area surfaces |
| GENERIC_DRIVABLE | [ ] | Generic drivable area surfaces |
| STOP_ZONE | [X] | Stop zone polygon + controlled lane IDs |

### Traffic Lights

From `TrafficLightDetection`:

| Feature | Extracted | Notes |
|---------|-----------|-------|
| Lane ID (controlled lane) | [X] | |
| State per timestep | [X] | GREEN, YELLOW, RED mapped to ints |
| Position | [X] | Computed from controlled lane's first centerline point |
| OFF state | [ ] | Mapped to 0 (unknown) |
| UNKNOWN state | [ ] | Mapped to 0 (unknown) |

**Puffer traffic control type mapping** (defined but only TRAFFIC_LIGHT populated):

| Type | ID | Populated |
|------|----|-----------|
| TRAFFIC_LIGHT | 1 | [X] |
| STOP_SIGN | 2 | [ ] |
| YIELD_SIGN | 3 | [ ] |
| SPEED_LIMIT_SIGN | 4 | [ ] |
| TRAFFIC_CONE | 5 | [ ] |
| TRAFFIC_BARRIER | 6 | [ ] |
| GUARDRAIL | 7 | [ ] |

### Others

**Not extracted from py123d box detections as map-like objects:**

| Object type | Notes |
|-------------|-------|
| Traffic cones | Detected as agents (type 4), not as static map objects |
| Barriers | Detected as agents (type 4), not as static map objects |
| Traffic signs | Detected as agents (type 4), not as static map objects |
| Stop signs | Type 2 in traffic_controls, not populated from source |
| Speed bumps | Type 32 in road_map_elements, not populated from source |
| Driveways | Type 33 in road_map_elements, not populated from source |


## Acknowledgements

- [py123d](https://github.com/autonomousvision/py123d)
- [Improving Traffic Signal Data Quality for the Waymo Open Motion Dataset](https://arxiv.org/abs/2506.07150) for the traffic lights improvement on WOMD
