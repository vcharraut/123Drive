# py123d-docker

Convert raw AV datasets to py123d Arrow format with Docker.

## Prerequisites

- Docker
- `uv run py123d-docker` from this repo after `uv sync --extra all`

## Usage

```bash
# List available datasets and expected data layout
uv run py123d-docker --list

# Convert one dataset (builds image automatically if needed)
uv run py123d-docker --dataset nuplan-mini --data_root /data --output /data/py123d

# Select specific splits
uv run py123d-docker --dataset nuplan --data_root /data --output /data/py123d \
  --splits nuplan_train nuplan_val --workers 16

# Extra Hydra overrides (appended after pre-defined ones)
uv run py123d-docker --dataset wod-motion --data_root /data --output /data/py123d \
  --extra "dataset.dataset_converter_config.include_route=true"

# Dry run
uv run py123d-docker --dataset nuplan-mini --data_root /data --output /data/py123d --dry_run

# Build image only
uv run py123d-docker --dataset nuplan-mini --build_only

# Override py123d ref (branch, tag, or commit)
uv run py123d-docker --dataset nuplan-mini --build_only --rebuild --py123d_ref my-ref
```

## py123d install ref

- Docker builds install `py123d` from `dev_v0.1.0` by default.
- Override with `--py123d_ref` to use a different branch, tag, or commit.
- Use `--rebuild` when changing `--py123d_ref` or when upstream dependency metadata changes.

## Expected data layout

```
data_root/
├── nuplan/dataset/
├── nuplan/maps/
├── nuplan/sensor_blobs/
├── waymo_open_motion/
├── waymo_open_perception/
├── nuscenes/
├── av2/
├── kitti360/
└── pandaset/
```

## Defaults (all datasets)

- Cameras and lidars disabled
- Ray by default, process pool when `--workers > 1`
- Container mounts raw data at `/data` and output at `/output`

## Minimal flow

```bash
uv run py123d-docker --dataset nuplan-mini --data_root /data --output /data/py123d
uv run convert --dataset_path /data/py123d --output_dir ./output
uv run viz --dir ./output
```

## Adding a dataset

Add an entry to `configs.py::DATASET_CONFIGS`:

```python
"my-dataset": {
    "extras": "my_extras",        # pip extra in py123d[extras]
    "path_keys": {
        "my_data_root": "my_data",  # dataset_paths.my_data_root → /data/my_data
    },
    "sensor_overrides": _SENSOR_DISABLE("my-dataset"),
    "default_splits": ["my-dataset_train", "my-dataset_val"],
},
```
