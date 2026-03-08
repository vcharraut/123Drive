# 123Drive

Convert `py123d` datasets to PufferDrive `.bin` files.

Pipeline:

```text
raw dataset -> py123d-docker -> py123d Arrow -> convert -> .bin -> viz
```

This release is intentionally small:
- extract with `py123d-docker`
- convert with `convert`
- inspect in the browser with `viz`
- export PNG or MP4 only when needed with `viz-png`, `viz-video`, `viz-batch`

## Install

`123Drive` is currently `uv`-first. Recommended install:

```bash
uv sync --extra all
```

## Quickstart

```bash
# install
uv sync --extra all

# 1) Extract raw dataset to py123d Arrow
uv run py123d-docker --dataset nuplan-mini --dataset_path /data/nuplan --output /data/py123d

# 2) Convert Arrow to PufferDrive .bin
uv run convert --py123d_path /data/py123d --output_dir ./output

# 3) Inspect in the browser
uv run viz --dir ./output
```

Open `http://localhost:8080`.

## Convert

Basic use:

```bash
uv run convert --py123d_path /path/to/py123d --output_dir ./output
```

Common examples:

```bash
# Parallel conversion
uv run convert --py123d_path /path/to/py123d --output_dir ./output --num_workers 8

# Filter datasets / splits / logs
uv run convert --py123d_path /path/to/py123d --output_dir ./output \
  --datasets nuplan --split_types val --max_scenarios 100

# Route filtering knobs
uv run convert --py123d_path /path/to/py123d --output_dir ./output \
  --min_route_valid_points 10 --route_check_timestep 5

# Stricter validation
uv run convert --py123d_path /path/to/py123d --output_dir ./output --validate_level 3
```

Core flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--py123d_path` | default to PY123D_DATA_ROOT env varialbe, can be override | Path to py123d dataset with `logs/` and `maps/` |
| `--output_dir` | `./output` | Directory for `.bin` files |
| `--num_workers` | `1` | Parallel workers |
| `--validate_level` | `1` | Validation strictness |

Filtering flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--max_scenarios` | all | Limit number of scenarios |
| `--datasets` | all | Dataset names to include |
| `--split_types` | all | Split types to include |
| `--split_names` | all | Split names to include |
| `--log_names` | all | Specific log names to include |
| `--duration_s` | `0` | Scenario duration in seconds, `0` = full |
| `--history_s` | `0` | History duration in seconds |
| `--map_only` | off | Load map-only scenarios |

Geometry + route flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--max_segment_length` | `2.0` | Max segment length for polyline interpolation |
| `--area_threshold` | `0.1` | Polyline simplification threshold, `0` = off |
| `--dist_threshold` | `10.0` | Distance threshold for road graph processing |
| `--min_route_valid_points` | `0` | Min valid trajectory points for route computation |
| `--route_check_timestep` | `0` | Timestep that must be valid for route computation |

Validation levels:

| Level | Behavior |
|------|----------|
| `0` | Skip validation |
| `1` | Mandatory schema / integrity checks only |
| `2` | Mandatory errors + physical checks as warnings |
| `3` | Reject hard physical issues, keep soft ones as warnings |
| `4` | Reject all physical / coherence issues |

## Web Viz

Default viewer:

```bash
uv run viz --dir ./output --port 8080
```

Features:
- browse `.bin` scenarios from a directory
- inspect map, agents, routes, and traffic controls
- playback, selection, follow-ego, and layer toggles

## Matplotlib Viz

Secondary export/debug tools:

```bash
# Single PNG
uv run viz-png ./output/map_000.bin ./frame.png

# Single MP4
uv run viz-video ./output/map_000.bin ./video.mp4

# Batch export
uv run viz-batch ./output ./exports --format both
```

