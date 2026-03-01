# py123d-convert

Converts raw AV datasets (nuPlan, Waymo, nuScenes, etc.) to py123d Arrow format using Docker. One image per dataset — no local dep conflicts.

## Prerequisites

- Docker
- `uv run py123d-convert` (installed via `uv pip install -e ".[all]"`)

## Usage

```bash
# List available datasets and expected data layout
uv run py123d-convert --list

# Convert (builds image automatically if not present)
uv run py123d-convert --dataset nuplan-mini --data_root /data --output /data/py123d

# Select specific splits
uv run py123d-convert --dataset nuplan --splits nuplan_train nuplan_val --workers 16

# Extra Hydra overrides (appended after pre-defined ones)
uv run py123d-convert --dataset wod-motion --data_root /data --output /data/py123d \
    --extra "datasets.wod-motion.dataset_converter_config.include_route=true"

# Dry run — print docker build + run commands without executing
uv run py123d-convert --dataset nuplan-mini --data_root /data --output /data/py123d --dry_run

# Build image only (no conversion)
uv run py123d-convert --dataset nuplan-mini --build_only
```

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

- Sensors disabled (2D only): cameras, lidars, fisheye
- Execution: ray (single worker) or process pool (`--workers N` where N > 1)
- Output always mounted at `/output` inside container

## Adding a dataset

Add an entry to `configs.py::DATASET_CONFIGS`:

```python
"my-dataset": {
    "extras": "my_extras",        # pip extra in py123d[extras]
    "devkit": None,               # or a pip install string
    "path_keys": {
        "my_data_root": "my_data",  # dataset_paths.my_data_root → /data/my_data
    },
    "sensor_overrides": _SENSOR_DISABLE("my-dataset"),
    "default_splits": ["my-dataset_train", "my-dataset_val"],
},
```
