# 123Drive

Convert [py123d](https://github.com/autonomousvision/py123d) arrow datasets to [PufferDrive](https://github.com/Emerge-Lab/PufferDrive) `.bin` files.

```text
raw dataset -> [123D] -> .arrow -> [123Drive] -> .bin
```

## Install

`123Drive` is `uv`-only. Use Python `3.11`-`3.13` from a local git checkout.

Pick the extra that matches your workflow:

- `uv sync`: dataset conversion (base dependencies)
- `uv sync --extra viz`: browser viewer
- `uv sync --extra all`: everything

`uv sync` without extras installs only the minimal base package.

## First 5 Minutes

Convert only:

```bash
uv sync
uv run convert --py123d_path /data/123d --output ./output
```

Inspect existing `.bin` output in the browser:

```bash
uv sync --extra viz
uv run web --dir ./output
```

Open `http://localhost:8080`.

## CLIs

- `convert`: py123d output root (`logs/` + `maps/`) -> PufferDrive `.bin`
- `augment-static-maps`: affine variants of static map `.bin` files
- `build`: build Docker images for the extraction/conversion pipeline
- `web`: browser viewer for `.bin` files

## Convert

Basic use:

```bash
uv run convert --py123d_path /path/to/123d --output ./output
```

Output files are named from dataset + scenario identity (for example `nuplan__<scenario>.bin`).
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

### Presets

Presets bundle the right defaults to reproduce a dataset with one command. Each
pins a dataset family; pick a split on top with `--split_names`. Explicit CLI
flags always override preset values. Defined in `src/bin_factory/presets.toml`.

```bash
uv run convert --preset nuplan --py123d_path /path/to/123d --output ./output

# narrow to a split / override anything inline
uv run convert --preset nuplan --split_names nuplan-mini_val \
  --py123d_path /path/to/123d --output ./output --num_scenes 100
```

Core flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--preset` | none | Apply a dataset preset (`av2`/`kitti360`/`nuplan`/`nuscenes`/`nureasoning`/`wod-motion`) |
| `--py123d_path` | `PY123D_DATA_ROOT` or required | Path to 123D dataset with `logs/` and `maps/` |
| `--output` | `./output` | Directory for `.bin` files |
| `--workers` | `0` | Parallel workers (`0` = 80% of CPU cores) |
| `--chunk_target_scenes` | `10000` | Scenarios per worker dispatch batch |
| `--validate_level` | `1` | Validation strictness |
| `--log_level` | `INFO` | Root logging level (`DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL`) |

Failures are written to `failures.jsonl` under `--output`.

Filtering flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--num_scenes` | all | Limit number of scenarios |
| `--datasets` | all | Dataset names to include |
| `--split_types` | all | Split types to include |
| `--split_names` | all | Split names to include |
| `--log_names` | all | Specific log names to include |
| `--scene_uuids` | all | Specific scene UUIDs to include (debugging) |
| `--duration_s` | `0` | Scenario duration in seconds, `0` = full |
| `--map_only` | off | Load map-only scenarios |

Geometry + route flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--max_segment_length` | `10.0` | Max segment length for polyline interpolation |
| `--area_threshold` | `0.1` | Polyline simplification threshold, `0` = off |
| `--min_route_valid_points` | `0.0` | Min valid trajectory percentage for route computation (`0`-`100`) |
| `--route_check_timestep` | `0` | Timestep that must be valid for route computation |
| `--no_reindex` | off | Skip reindexing element IDs to contiguous `range(0, n)` |
| `--interpolate_tl` | off | Interpolate traffic light states from vehicle trajectories |
| `--invalid_agent_overlap` | off | Zero out log-only agents whose bbox overlaps an active agent during replay |

Validation levels:

| Level | Behavior |
|------|----------|
| `0` | Skip validation |
| `1` | Schema checks: required keys, container types, array shapes, and length consistency |
| `2` | Semantic checks: schema plus topology refs, finite values, valid traffic-light states, and ego-only temporal sanity |

## Static Map Augmentation

The `augment-static-maps` CLI generates static map variants from `.bin` map files.

```bash
uv run augment-static-maps \
  --config src/bin_factory/configs/static_map_transforms_affine_balanced.yaml \
  --input-dir data/static_maps \
  --output-dir data/static_maps_affine
```

Pass a YAML config explicitly. The config contains only `limit` and `transforms`;
input/output directories are required CLI arguments, and original maps are always copied.

| Preset | Config | Output | Count |
|--------|--------|--------|-------|
| Balanced affine | `src/bin_factory/configs/static_map_transforms_affine_balanced.yaml` | `data/static_maps_affine` | 8 maps x (1 original + 15 transforms) = 128 |
| FlipX only | `src/bin_factory/configs/static_map_transforms_flipx_only.yaml` | `data/static_maps_flipx` | 8 maps x (1 original + 1 flip) = 16 |

Use the flip-only preset explicitly:

```bash
uv run augment-static-maps \
  --config src/bin_factory/configs/static_map_transforms_flipx_only.yaml \
  --input-dir data/static_maps \
  --output-dir data/static_maps_flipx
```

## Web Viewer

```bash
uv sync --extra viz
uv run web --dir ./output --port 8080
```

- browse `.bin` scenarios from a directory
- inspect map, agents, route, and traffic controls
- playback, follow-ego, selection, and layer toggles

## Docker Images

```bash
# Build py123d image
uv run build py123d --dataset nuplan-mini

```

Images are portable - run them however you want (`docker run`, Kubernetes, etc.).

- `py123d-<dataset>` is an opinionated BEV-oriented extractor with raw sensors disabled
- `123drive:latest` is a thin uv-backed runtime image built from the current checkout and forwards args directly to `convert`
- `build` requires Docker
- Dockerfiles require BuildKit because they use `RUN --mount=type=cache`

## Docs

- Binary format: `docs/binary-format.md`
- Route search notes: `docs/route-algorithm.md`
