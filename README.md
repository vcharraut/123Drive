# 123Drive

Convert `123D` arrow datasets to PufferDrive `.bin` files.

```text
raw dataset -> [123D] -> .arrow -> [123Drive] -> .bin
```

## Install

`123Drive` is `uv`-first. Recommended install:

```bash
uv sync --extra all
```

## Quickstart

```bash
# Install
uv sync --extra all

# Convert Arrow to PufferDrive .bin
uv run convert --py123d_path /data/123d --output ./output

# Inspect in the browser
uv run web --dir ./output
```

Open `http://localhost:8080`.

## CLIs

- `convert`: 123D Arrow -> PufferDrive `.bin`
- `build`: build Docker images for 123D/123Drive pipelines
- `web`: browser viewer for `.bin`
- `viz`: matplotlib mp4

## Convert

Basic use:

```bash
uv run convert --py123d_path /path/to/123d --output ./output
```

Examples:

```bash
# Parallel conversion
uv run convert --py123d_path /path/to/123d --output ./output --workers 8

# Filter datasets / splits / logs
uv run convert --py123d_path /path/to/123d --output ./output \
  --datasets nuplan --split_types val --num_scenes 100

# Route filtering knobs
uv run convert --py123d_path /path/to/123d --output ./output \
  --min_route_valid_points 10 --route_check_timestep 5

# Map-only conversion
uv run convert --py123d_path /path/to/123d --output ./output --map_only
```

Core flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--py123d_path` | `PY123D_DATA_ROOT` or required | Path to 123D dataset with `logs/` and `maps/` |
| `--output` | `./output` | Directory for `.bin` files |
| `--workers` | `1` | Parallel workers |
| `--validate_level` | `1` | Validation strictness |
| `--fail_fast` | off | Stop on first error |

Filtering flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--num_scenes` | all | Limit number of scenarios |
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
| `--reindex_id` | off | Reindex all element IDs to contiguous `range(0, n)` |

Validation levels:

| Level | Behavior |
|------|----------|
| `0` | Skip validation |
| `1` | Schema checks: required keys, container types, array shapes, and length consistency |
| `2` | Semantic checks: schema plus topology refs, finite values, valid traffic-light states, and ego-only temporal sanity |

## Viz

```bash
uv run web --dir ./output --port 8080
```

- browse `.bin` scenarios from a directory
- inspect map, agents, route, and traffic controls
- playback, follow-ego, selection, and layer toggles

## Docker Images

```bash
# List available datasets
uv run build list

# Build py123d dataset image
uv run build py123d --dataset nuplan-mini

# Build 123Drive converter image
uv run build 123drive

# Build with custom refs
uv run build py123d --dataset nuplan --py123d_ref my-branch --no_cache
uv run build 123drive --drive123_ref v0.2.0
```

Images are portable — run them however you want (`docker run`, Kubernetes, etc.).

## Docs

- Binary format: `docs/binary-format.md`
- Supported data surface: `docs/data.md`
- Route search notes: `docs/route-algorithm.md`
