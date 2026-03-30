# Data Surface

All fields available in [py123d](https://github.com/autonomousvision/py123d) and whether they are included in the PufferDrive binary.

## Ego State

| Field | py123d | PufferDrive |
|-------|:------:|:-----------:|
| Position (x, y, z) | x | x |
| Heading (yaw) | x | x |
| Velocity (vx, vy) | x | x |
| Bounding box (length, width, height) | x | x |
| Validity per timestep | x | x |
| Acceleration (ax, ay, az) | x | - |
| Angular velocity (roll, pitch, yaw rates) | x | - |
| Tire steering angle | x | - |
| Full SE3 pose (quaternion) | x | - |
| Vehicle metadata (wheelbase, name, calibrations) | x | - |
| IMU / rear axle poses | x | - |
| Route (ordered lane IDs) | - | x |
| Goal position | - | x |

## Agent Detections

| Field | py123d | PufferDrive |
|-------|:------:|:-----------:|
| Position (x, y, z) | x | x |
| Heading (yaw) | x | x |
| Velocity (vx, vy) | x | x |
| Bounding box (length, width, height) | x | x |
| Validity per timestep | x | x |
| Agent type (vehicle, pedestrian, cyclist, ...) | x | x |
| Track token | x | - |
| Acceleration (ax, ay, az) | x | - |
| Full SE3 pose (quaternion) | x | - |
| Lidar point count per detection | x | - |
| Route (ordered lane IDs) | - | x |
| Goal position | - | x |

## Map - Lanes

| Field | py123d | PufferDrive |
|-------|:------:|:-----------:|
| Centerline polyline | x | x |
| Lane type (freeway, surface street, bike, bus) | x | x |
| Speed limit (m/s) | x | x |
| Predecessor / successor lane IDs | x | x |
| Left / right boundary polylines | x | - |
| Left / right neighbor lane IDs | x | - |
| Lane group ID | x | - |
| Lane outline polygon | x | - |

## Map - Road Lines

| Field | py123d | PufferDrive |
|-------|:------:|:-----------:|
| Polyline | x | x |
| Line type (solid/dashed, white/yellow, single/double) | x | x |

## Map - Road Edges

| Field | py123d | PufferDrive |
|-------|:------:|:-----------:|
| Polyline | x | x |
| Edge type (boundary, median) | x | x |

## Map - Areas

| Field | py123d | PufferDrive |
|-------|:------:|:-----------:|
| Crosswalk polygon | x | x |
| Walkway polygon | x | - |
| Carpark polygon | x | - |
| Generic drivable area polygon | x | - |

## Map - Topology

| Field | py123d | PufferDrive |
|-------|:------:|:-----------:|
| Lane graph (all-pairs shortest distances) | - | x |
| Lane groups | x | - |
| Intersections (type, constituent lane groups) | x | - |

## Traffic Controls

| Field | py123d | PufferDrive |
|-------|:------:|:-----------:|
| Traffic light state per timestep (green/yellow/red/off) | x | x |
| Controlled lane IDs | x | x |
| Stop line geometry | x | x |
| Stop sign / yield sign zones | x | x |
| Speed-limit signs | x | - |
| Cones, barriers | x | - |

## Metadata

| Field | py123d | PufferDrive |
|-------|:------:|:-----------:|
| Scenario ID | x | x |
| Dataset name | x | x |
| Location | x | x |
| Scenario length (timesteps) | x | x |
| Timestep duration (dt) | x | x |
| Objects of interest IDs | - | x |
| Tracks to predict IDs | - | x |
| Split name / type | x | - |
| Log name | x | - |
| Map metadata (has_z, is_per_log) | x | - |

## Sensors

| Field | py123d | PufferDrive |
|-------|:------:|:-----------:|
| Lidar point clouds | x | - |
| Lidar features (intensity, ring, range, timestamps) | x | - |
| Camera images | x | - |
| Camera intrinsics / extrinsics | x | - |
| Sensor-to-IMU calibrations | x | - |
