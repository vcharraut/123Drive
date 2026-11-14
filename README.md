# 123Drive

Convert [123D](https://github.com/autonomousvision/py123d) Arrow datasets into
`.bin` scenarios for Macadam.

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

Python 3.12 to 3.13. MIT licensed.

## Agent control and format changes

An agent with a nonempty route is eligible for control. An empty route makes it
a background entity. Macadam chooses the controller and background behavior.

Binary format v0.4 has a versioned header and indexed sections. Road polylines (lanes, lines,
edges) and road surfaces (crosswalks, walkways, shoulders, and other outlines) have separate
sections and keep py123d's map layers and types. Routes may change lanes,
and each route lane carries its `route_entry_s` and `route_exit_s` span. It does not store
`control_state` or `route_gt_len`. Reconvert existing scenes and update the
Macadam reader before using new output. See the
[binary writer](src/bin_factory/serialize.py).
