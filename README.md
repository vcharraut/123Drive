# 123Drive

Convert `123D` arrow datasets to PufferDrive `.bin` files.

```text
raw dataset -> [123D] -> .arrow -> [123Drive] -> .bin
```

## Install

`123Drive` is `uv`-only for now. Use Python `3.10`-`3.13`.

| Workflow | Status |
|------|------|
| `uv sync`, `uv run` | supported |
| Docker build flow | supported |

```bash
uv sync --extra all
```

If you only need conversion:

```bash
uv sync
```

## Quickstart

```bash
# Install
uv sync --extra all

# Convert py123d Arrow output to PufferDrive .bin
uv run convert --py123d_path /data/123d --output ./output

# Inspect in the browser
uv run web --dir ./output

# Or render an mp4
uv run viz ./output/map_000.bin ./output/map_000.mp4
```

Open `http://localhost:8080`.

`viz` uses `ffmpeg` through Matplotlib's `FFMpegWriter`, so install `ffmpeg` on your system first.

## CLIs

- `convert`: py123d output root (`logs/` + `maps/`) -> PufferDrive `.bin`
- `build`: build Docker images for the extraction/conversion pipeline
- `web`: browser viewer for `.bin` files
- `viz`: render `.bin` files to mp4

## Convert

Basic use:

```bash
uv run convert --py123d_path /path/to/123d --output ./output
```

Mental model:

```text
py123d output root -> load scene/map -> extract PufferScenario -> transforms -> serialize -> .bin
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
# Browser viewer
uv sync --extra viz
uv run web --dir ./output --port 8080

# MP4 renderer
uv run viz ./output/map_000.bin ./videos/map_000.mp4
```

- browse `.bin` scenarios from a directory
- inspect map, agents, route, and traffic controls
- playback, follow-ego, selection, and layer toggles
- `web` and `viz` require `uv sync --extra viz` or `uv sync --extra all`
- `viz` requires `ffmpeg` on your system

## Docker Images

```bash
# List available datasets
uv run build list

# Build py123d image
uv run build py123d --dataset nuplan-mini

# Build 123Drive converter image
uv run build 123drive

# Build py123d with a custom ref
uv run build py123d --dataset nuplan --ref my-branch --no_cache
```

To build a specific `123Drive` version, check out that branch, tag, or commit locally first, then run `uv run build 123drive`.

Images are portable — run them however you want (`docker run`, Kubernetes, etc.).

- `py123d-<dataset>` is an opinionated BEV-oriented extractor with raw sensors disabled
- `123drive:latest` is a thin uv-backed runtime image built from the current checkout and forwards args directly to `convert`
- `build` requires Docker
- Dockerfiles require BuildKit because they use `RUN --mount=type=cache`

## Docs

- Binary format: `docs/binary-format.md`
- Supported data surface: `docs/data.md`
- Route search notes: `docs/route-algorithm.md`
