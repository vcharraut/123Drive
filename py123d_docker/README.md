# py123d-docker

Convert raw AV datasets to py123d Arrow format with Docker.

## Prerequisites

- Docker
- `uv run py123d-docker` from this repo after `uv sync --extra all`

## Usage

```bash
# List available datasets
uv run py123d-docker --list

# Convert one dataset (builds image automatically if needed)
uv run py123d-docker --dataset nuplan-mini --dataset_path /data/nuplan --output /data/py123d

# Select specific splits
uv run py123d-docker --dataset nuplan --dataset_path /data/nuplan --output /data/py123d \
  --splits nuplan_train nuplan_val --workers 16

# Extra Hydra overrides (appended after pre-defined ones)
uv run py123d-docker --dataset wod-motion --dataset_path /data/waymo --output /data/py123d \
  --extra "dataset.dataset_converter_config.include_route=true"

# Dry run
uv run py123d-docker --dataset nuplan-mini --dataset_path /data/nuplan --output /data/py123d --dry_run

# Build image only
uv run py123d-docker --dataset nuplan-mini --build_only

# Override py123d ref (branch, tag, or commit)
uv run py123d-docker --dataset nuplan-mini --build_only --rebuild --py123d_ref my-ref
```

## py123d install ref

- Docker builds install `py123d` from `dev_v0.1.0` by default.
- Override with `--py123d_ref` to use a different branch, tag, or commit.
- Use `--rebuild` when changing `--py123d_ref` or when upstream dependency metadata changes.

## Dataset path

`--dataset_path` points directly to the dataset directory (e.g. `/data/nuplan`, `/data/nuscenes`).
For nuplan, extra paths (`maps/`, `sensor_blobs/`) are resolved relative to the dataset path automatically.

## Defaults (all datasets)

- Cameras and lidars disabled
- Ray by default, process pool when `--workers > 1`
- Container mounts dataset path at `/data` and output at `/output`

## Minimal flow

```bash
uv run py123d-docker --dataset nuplan-mini --dataset_path /data/nuplan --output /data/py123d
uv run convert --py123d_path /data/py123d --output_dir ./output
uv run viz --dir ./output
```

## Adding a dataset

Add an entry to `configs.py::DATASET_CONFIGS`:

```python
"my-dataset": _dataset_config(
    "my_extras",              # pip extra in py123d[extras]
    "my_data_root",           # hydra key for dataset_paths.my_data_root
    ["my-dataset_train", "my-dataset_val"],
    extra_paths={"my_maps_root": "maps"},  # optional, relative to dataset_path
),
```
