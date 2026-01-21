# Puffer Web Visualization

Interactive scenario explorer for PufferDrive binary files.

## Install

```bash
uv pip install -e ".[viz]"
```

## Run

```bash
# Load scenario directly
python puffer_viz/web/app.py path/to/scenario.bin

# Or start empty and upload via UI
python puffer_viz/web/app.py

# Options
python puffer_viz/web/app.py --port 8080 --debug scenario.bin
```

Opens at http://localhost:8050

## Features

**Map Elements**
- Lanes (gray), road lines (white/yellow, solid/dashed), road edges (black)
- Crosswalks, stop signs, speed bumps
- Layer toggles in left panel

**Agents**
- Colored rectangles with heading arrows
- Trajectory history + future (dotted)
- Routes as dashed lines
- Red border = track to predict, blue = object of interest

**Traffic Lights**
- Colored circles (red/yellow/green)
- State timeline in info panel

**Interactivity**
- Scroll to zoom, drag to pan
- Click element → detailed info panel
- Search by ID (agent/road/traffic light)
- Playback: play/pause, speed control (0.5x-4x), step forward/back
- Follow ego button centers view on SDC

**Info Panel (right side)**
- Scenario overview: ID, dataset, counts
- Selected element details:
  - Agent: position, velocity, heading, dimensions, route, full trajectory table
  - Road: type, speed limit, entry/exit lanes, polyline coordinates
  - Traffic light: state timeline, controlled lanes

## Keyboard Shortcuts

None yet - use UI controls.

## File Structure

```
puffer_viz/web/
├── app.py           # Entry point, CLI
├── layout.py        # UI components
├── callbacks.py     # Interactivity
├── render.py        # Plotly figure generation
├── info_panels.py   # Element info display
└── utils.py         # Type mappings, colors
```
