# Puffer Web Visualization

Interactive browser viewer for PufferDrive `.bin` scenarios.

## Install

```bash
uv pip install -e ".[viz]"
```

## Run

```bash
viz-server --dir /path/to/bin/files --port 8080
```

Open http://localhost:8080.

## What it does

- Lists all `.bin` scenarios from the provided directory
- Renders map, agents, trajectories, and traffic lights
- Supports 2D/3D view toggle, follow-ego, playback controls, and layer toggles
- Supports element selection + details panel and ID search

## Keyboard shortcuts

- `Space`: play / pause
- `Left` / `Right`: previous / next timestep
- `+` / `-`: zoom in / out
- `F`: fit view

## Implementation layout

```
puffer_viz/
├── server.py         # FastAPI API + static hosting
├── binary_loader.py  # .bin parser
└── web/
    ├── utils.py
    └── static/
        ├── index.html
        ├── helpers.js
        ├── info_panel.js
        ├── app.js
        └── style.css
```
