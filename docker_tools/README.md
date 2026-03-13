# docker_tools

Build-only CLI for 123Drive Docker images. Produces reusable images for both pipeline steps — users run them however they want.

## Images

| Image | Dockerfile | Purpose |
|-------|-----------|---------|
| `123d-{dataset}` | `dockerfiles/123d.Dockerfile` | Raw dataset → 123D Arrow |
| `123d-convert:{ref}` | `dockerfiles/123drive.Dockerfile` | 123D Arrow → PufferDrive `.bin` |

## Usage

```bash
# List available datasets
uv run 123d-docker list

# Build 123D extraction image for a dataset
uv run 123d-docker build 123d --dataset nuplan-mini
uv run 123d-docker build 123d --dataset wod-motion --123d_ref my-branch --rebuild

# Build convert image
uv run 123d-docker build convert
uv run 123d-docker build convert --drive123_ref v0.2.0

# Dry run (print docker commands without executing)
uv run 123d-docker build 123d --dataset nuplan-mini --dry_run
uv run 123d-docker build convert --dry_run
```

## Running images

The CLI only builds images. Run them directly:

```bash
# 123D extraction
docker run --rm 123d-nuplan-mini \
  dataset=nuplan-mini dataset_paths.nuplan_data_root=/data

# convert
docker run --rm 123d-convert:main \
  --data_path /input --output /output
```

## Adding a dataset

Add an entry to `configs.py::DATASET_CONFIGS`:

```python
"my-dataset": _dataset_config(
    "my_extras",              # pip extra in py123d[extras]
    "my_data_root",           # hydra key for dataset_paths.my_data_root
    ["my-dataset_train", "my-dataset_val"],
    extra_paths={"my_maps_root": "maps"},
),
```
