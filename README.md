# 123Drive

Convert `py123d` datasets to PufferDrive `.bin` files.

```text
raw dataset -> [py123d-docker] -> py123d Arrow -> [convert] -> .bin -> [viz]
```

## Install

`123Drive` is `uv`-first. Recommended install:

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
uv run convert --py123d_path /data/py123d --output ./output

# 3) Inspect in the browser
uv run viz --dir ./output
```

Open `http://localhost:8080`.

## CLIs

- `convert`: py123d Arrow -> PufferDrive `.bin`
- `py123d-docker`: raw dataset -> py123d Arrow through Docker
- `viz`: browser viewer for `.bin`
- `viz-png`, `viz-video`, `viz-batch`: optional matplotlib exports

## Convert

Basic use:

```bash
uv run convert --py123d_path /path/to/py123d --output ./output
```

Examples:

```bash
# Parallel conversion
uv run convert --py123d_path /path/to/py123d --output ./output --workers 8

# Filter datasets / splits / logs
uv run convert --py123d_path /path/to/py123d --output ./output \
  --datasets nuplan --split_types val --num_scenes 100

# Route filtering knobs
uv run convert --py123d_path /path/to/py123d --output ./output \
  --min_route_valid_points 10 --route_check_timestep 5

# Map-only conversion
uv run convert --py123d_path /path/to/py123d --output ./output --map_only
```

Core flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--py123d_path` | `PY123D_DATA_ROOT` or required | Path to py123d dataset with `logs/` and `maps/` |
| `--output` | `./output` | Directory for `.bin` files |
| `--workers` | `1` | Parallel workers |
| `--validate_level` | `1` | Validation strictness |

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

Validation levels:

| Level | Behavior |
|------|----------|
| `0` | Skip validation |
| `1` | Schema checks: required keys, container types, array shapes, and length consistency |
| `2` | Semantic checks: schema plus topology refs, finite values, valid traffic-light states, and ego-only temporal sanity |

## Viz

```bash
uv run viz --dir ./output --port 8080
```

- browse `.bin` scenarios from a directory
- inspect map, agents, route, and traffic controls
- playback, follow-ego, selection, and layer toggles


## Docker extractor

```bash
uv run py123d-docker --list
uv run py123d-docker --dataset nuplan-mini --dataset_path /data/nuplan --output /data/py123d
```

## Docs

- Binary format: `docs/binary-format.md`
- Supported data surface: `docs/data.md`
- Route search notes: `docs/route-algorithm.md`
