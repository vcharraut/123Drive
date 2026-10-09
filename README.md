# 123Drive

Convert [123D](https://github.com/autonomousvision/py123d) Arrow datasets into
[PufferDrive](https://github.com/Emerge-Lab/PufferDrive) `.bin` scenarios.

```text
raw dataset -> 123D Arrow -> 123Drive -> PufferDrive binary
```

## Documentation

Full documentation: **https://vcharraut.github.io/123Drive/**

It covers installation, datasets, conversion, presets, the processing pipeline, validation,
route and traffic-light algorithms, the binary format, Mapforge, the viewer, and Docker.

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

Python 3.11–3.13. MIT licensed.
