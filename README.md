# 123Drive

Convert [123D](https://github.com/autonomousvision/py123d) Arrow datasets into
`.bin` scenarios for PufferDinde.

```text
raw dataset -> 123D Arrow -> 123Drive -> .bin
```

## Documentation

[Process, dataset differences, and command options](docs/index.md).

## Quick start

```bash
uv sync
uv run convert --py123d_path /data/123d --output ./output
```

Use a dataset preset when available:

```bash
uv run convert \
  --preset nuplan \
  --py123d_path /data/123d \
  --output ./output
```

Inspect generated binaries:

```bash
uv sync --extra viz
uv run web --dir ./output
```

Python 3.11 to 3.13. MIT licensed.

## Agent control and format changes

An agent with a nonempty route is eligible for control. An empty route makes it
a background entity. PufferDinde chooses the controller and background behavior.

The binary no longer stores `control_state` or `route_gt_len`. Reconvert existing
scenes and update the PufferDinde reader before using new output. See the
[binary writer](src/bin_factory/serialize.py).
