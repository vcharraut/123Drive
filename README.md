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
# Install
uv sync --extra all

# Basic conversion
uv run convert --dataset_path /path/to/data --output_dir ./output

# With traffic lights and physical validation as warnings
uv run convert --dataset_path /path/to/data --output_dir ./output --traffic_lights --validate_level 2

# Parallel processing with dataset filtering
uv run convert --dataset_path /path/to/data --output_dir ./output --num_workers 8 --datasets nuplan --max_scenarios 100

# Visualize output
uv run viz --dir ./output
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
| `--validate_level` | `1` | 0 = off, 1 = mandatory schema, 2 = physics warnings, 3 = reject hard physics, 4 = reject all physics |
| `--max_segment_length` | `2.0` | Max segment length for polyline interpolation (m) |
| `--area_threshold` | `0.1` | Visvalingam-Whyatt simplification threshold (0 = disabled) |
| `--dist_threshold` | `10.0` | Distance threshold for road graph |
| `--min_route_valid_points` | `0` | Min valid trajectory points for route computation (0 = no filter) |
| `--route_check_timestep` | `0` | Timestep at which agent must be valid for route computation |

### Validation modes

`bin_factory` always validates the Puffer dict after conversion. The selected mode controls which checks run and which issues reject a scenario.

| Level | Name | Behavior |
|------|------|----------|
| `0` | Off | Skip all validation |
| `1` | Mandatory | Reject broken structure: missing keys, wrong container types, wrong array shapes, inconsistent trajectory lengths, invalid ids, missing metadata |
| `2` | Warn physics | Run mandatory checks, then run physical/coherence checks as warnings only |
| `3` | Reject hard physics | Run mandatory checks, reject hard physical violations, keep softer anomalies as warnings |
| `4` | Reject all physics | Run mandatory checks, reject every physical/coherence violation |

High-level mandatory checks cover dataset organization issues such as malformed top-level fields, missing required keys, invalid lane or traffic-light references, and inconsistent per-agent tensor lengths.

Physical/coherence checks cover suspicious motion or map behavior such as teleportation, impossible dimensions, excessive speed, high acceleration, heading/velocity mismatch, short or sharply bent road polylines, and suspicious traffic-light transitions.

Hard physical violations currently include issues that make the scenario unsafe to trust downstream, such as teleportation, non-positive dimensions, and extreme speed violations. Softer anomalies remain warnings at level `3`.

## Acknowledgements

- [py123d](https://github.com/autonomousvision/py123d)
- [Improving Traffic Signal Data Quality for the Waymo Open Motion Dataset](https://arxiv.org/abs/2506.07150) for the traffic lights improvement on WOMD
