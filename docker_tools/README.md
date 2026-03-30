# docker_tools

Build-only CLI for 123Drive Docker images. Produce reusable images for both pipeline steps, then run them however you want.

Install policy:

- `uv sync` installs everything needed for `convert` and `build`
- the `123drive` image installs the repo with `uv sync --frozen`
- Docker builds require BuildKit because the Dockerfiles use `RUN --mount=type=cache`

## Images

| Image              | Dockerfile                        | Purpose                         |
| ------------------ | --------------------------------- | ------------------------------- |
| `py123d-{dataset}` | `dockerfiles/py123d.Dockerfile`   | Raw dataset → 123D Arrow        |
| `123drive:latest`  | `dockerfiles/123drive.Dockerfile` | 123D Arrow → PufferDrive `.bin` |

## Supported py123d datasets

| Dataset       | Extras   |
| ------------- | -------- |
| `nuplan`      | `nuplan` |
| `nuplan-mini` | `nuplan` |
| `wod-motion`  | `waymo`  |
| `av2-sensor`  | `av2`    |

## Build

```bash
uv sync

# Build one py123d image per dataset
uv run build py123d --dataset nuplan-mini

# Build 123Drive runtime image from the current checkout
uv run build 123drive

# Print docker commands without executing
uv run build py123d --dataset wod-motion --dry_run
uv run build 123drive --dry_run
```

To build a specific `123Drive` version, check out that branch, tag, or commit locally first, then run `uv run build 123drive`.

## Run

The CLI only builds images. Run them with Docker, Kubernetes, Slurm, or any container runtime.

Both images follow the same convention: **read from `/input`, write to `/output`**. Mount your host directories to these paths with `-v <host_path>:<container_path>`.

### py123d image

BEV-only extraction (no cameras, no lidar). Runtime args:

| Arg | Default | Description |
|-----|---------|-------------|
| `--splits` | all | Split override |
| `--worker_type` | `ray` | `ray`, `process_pool`, or `thread_pool` |
| `--workers` | 80% CPUs | Conversion workers |

```bash
docker run --rm \
  -v /data/nuplan:/input \          # host dataset dir  → container /input
  -v /data/py123d_out:/output \     # host output dir   → container /output
  --shm-size=10g \
  py123d-nuplan-mini \
  --splits nuplan-mini_train nuplan-mini_val
```

For nuPlan, mount the root containing both `maps/` and `nuplan-v1.1/`.

### 123Drive image

Converts py123d output to PufferDrive `.bin`. Accepts all `convert` CLI args (see `src/bin_factory/main.py`).

```bash
docker run --rm \
  -v /data/py123d_out:/input \      # py123d output     → container /input
  -v /data/bins:/output \           # binary output     → container /output
  --shm-size=10g \
  123drive \
  --workers 8 \
  --validate_level 1
```
