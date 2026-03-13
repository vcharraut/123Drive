# docker_tools

Build-only CLI for 123Drive Docker images. Produces reusable images for both pipeline steps — users run them however they want.

## Images

| Image | Dockerfile | Purpose |
|-------|-----------|---------|
| `123d-{dataset}` | `dockerfiles/123d.Dockerfile` | Raw dataset → 123D Arrow |
| `123d-convert:{ref}` | `dockerfiles/123drive.Dockerfile` | 123D Arrow → PufferDrive `.bin` |

## Build

```bash
# List supported py123d datasets
uv run build list

# Build one py123d image per dataset
uv run build py123d --dataset nuplan-mini
uv run build py123d --dataset nuplan --py123d_ref my-branch

# Build 123Drive runtime image
uv run build 123drive
uv run build 123drive --drive123_ref v0.2.0

# Print docker commands without executing
uv run build py123d --dataset wod-motion --dry_run
uv run build 123drive --dry_run
```

## Run

The CLI only builds images. Run them directly with Docker.

### py123d image

The py123d image is a BEV-oriented wrapper. It exposes only a few runtime knobs and bakes in:

- no pinhole cameras
- no lidars
- no box lidar points
- no fisheye cameras

Runtime arguments:

- `--input`: raw dataset root mounted in the container
- `--output`: py123d output root
- `--splits`: optional split override
- `--workers`: conversion workers

Example:

```bash
docker run --rm \
  -v /data/nuplan:/mnt/input \
  -v /tmp/py123d-nuplan-mini:/mnt/output \
  py123d-nuplan-mini \
  --workers 8 \
  --splits nuplan-mini_train nuplan-mini_val
```

For nuPlan datasets, mount the dataset root that contains both `maps/` and `nuplan-v1.1/`.

### 123Drive image

The 123Drive image is runtime only. Pass any `convert` argument supported by `src/bin_factory/main.py`.

Example:

```bash
docker run --rm \
  -v /tmp/py123d-nuplan-mini:/input \
  -v /tmp/pufferdrive:/output \
  123drive:main \
  --py123d_path /input \
  --output /output \
  --workers 8 \
  --validate_level 1
```
