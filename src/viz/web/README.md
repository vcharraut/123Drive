# Web Viz

Minimal browser viewer for PufferDrive `.bin` scenarios.

Run from the source checkout:

```bash
uv sync --extra all
uv run viz --dir /path/to/bin/files --port 8080
```

Open `http://localhost:8080`.

## What it does

- lists `.bin` scenarios from a directory
- renders map, agents, routes, and traffic controls
- supports playback, follow-ego, selection, and layer toggles

## Keyboard shortcuts

- `Space` play / pause
- `Left` / `Right` previous / next timestep
- `+` / `-` zoom in / out
- `F` fit view
