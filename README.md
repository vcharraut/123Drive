# 123Drive

Convert `123D` arrow datasets to PufferDrive `.bin` files.

```text
raw dataset -> [123D] -> .arrow -> [123Drive] -> .bin
```

## Install

`123Drive` is `uv`-only. Use Python `3.11`-`3.13` from a local git checkout.

Pick the smallest extra that matches your workflow:

- `uv sync --extra convert`: dataset conversion
- `uv sync --extra viz`: browser viewer
- `uv sync --extra docker`: Docker image builder
- `uv sync --extra all`: everything

`uv sync` without extras installs only the minimal base package.

## First 5 Minutes

Convert only:

```bash
uv sync --extra convert
uv run convert --py123d_path /data/123d --output ./output
```

Inspect existing `.bin` output in the browser:

```bash
uv sync --extra viz
uv run web --dir ./output
```

Build Docker images:

```bash
uv sync --extra docker
uv run build list
```

Open `http://localhost:8080`.

## CLIs

- `convert`: py123d output root (`logs/` + `maps/`) -> PufferDrive `.bin`
- `build`: build Docker images for the extraction/conversion pipeline
- `web`: browser viewer for `.bin` files

## Convert

Basic use:

```bash
uv run convert --py123d_path /path/to/123d --output ./output
```

Output files are named from dataset + scenario identity (for example `nuplan__<scenario>.bin`). Existing files are not overwritten unless you pass `--overwrite`.

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
| `--overwrite` | off | Replace existing output files |
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
| `--map_only` | off | Load map-only scenarios |

Geometry + route flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--max_segment_length` | `10.0` | Max segment length for polyline interpolation |
| `--area_threshold` | `0.1` | Polyline simplification threshold, `0` = off |
| `--min_route_valid_points` | `0` | Min valid trajectory points for route computation |
| `--route_check_timestep` | `0` | Timestep that must be valid for route computation |
| `--reindex_id` | off | Reindex all element IDs to contiguous `range(0, n)` |

Validation levels:

| Level | Behavior |
|------|----------|
| `0` | Skip validation |
| `1` | Schema checks: required keys, container types, array shapes, and length consistency |
| `2` | Semantic checks: schema plus topology refs, finite values, valid traffic-light states, and ego-only temporal sanity |

## Web Viewer

```bash
uv sync --extra viz
uv run web --dir ./output --port 8080
```

- browse `.bin` scenarios from a directory
- inspect map, agents, route, and traffic controls
- playback, follow-ego, selection, and layer toggles
- path finder uses serialized lane-graph data

## Docker Images

```bash
uv sync --extra docker

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

Images are portable - run them however you want (`docker run`, Kubernetes, etc.).

- `py123d-<dataset>` is an opinionated BEV-oriented extractor with raw sensors disabled
- `123drive:latest` is a thin uv-backed runtime image built from the current checkout and forwards args directly to `convert`
- `build` requires Docker
- Dockerfiles require BuildKit because they use `RUN --mount=type=cache`

## Docs

- Binary format: `docs/binary-format.md`
- Supported data surface: `docs/data.md`
- Route search notes: `docs/route-algorithm.md`
