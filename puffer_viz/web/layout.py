"""UI layout components for Puffer web visualization."""

from dash import dcc, html
import dash_bootstrap_components as dbc


def create_layout():
    """Create main app layout."""
    return dbc.Container([
        # Stores for state
        dcc.Store(id="scenario-data", data=None),
        dcc.Store(id="selected-element", data=None),
        dcc.Store(id="layer-state", data={
            "lanes": True,
            "road_lines": True,
            "road_edges": True,
            "crosswalks": True,
            "agents": True,
            "routes": True,
            "trajectories": True,
            "traffic_lights": True,
            "agent_ids": True,
        }),
        dcc.Store(id="highlight-lanes", data=[]),
        dcc.Interval(id="playback-interval", interval=100, disabled=True),

        # Header
        _create_header(),

        # Main content
        dbc.Row([
            # Left sidebar - Layers panel
            dbc.Col(_create_layers_panel(), width=2, className="pe-0"),

            # Center - Visualization
            dbc.Col(_create_viz_panel(), width=7, className="px-1"),

            # Right sidebar - Info panel
            dbc.Col(_create_info_panel(), width=3, className="ps-0"),
        ], className="g-0 flex-grow-1", style={"height": "calc(100vh - 120px)"}),

        # Footer - Timeline controls
        _create_timeline_controls(),

    ], fluid=True, className="vh-100 d-flex flex-column p-2")


def _create_header():
    """Create header bar."""
    return dbc.Row([
        dbc.Col([
            html.H5([
                html.Span("Puffer Viz", className="fw-bold"),
                html.Span(" | ", className="text-muted mx-2"),
                html.Span(id="scenario-id-display", className="text-muted"),
                html.Span(" | ", className="text-muted mx-2"),
                html.Span(id="dataset-display", className="text-muted"),
            ], className="mb-0"),
        ], width=8),
        dbc.Col([
            html.Div([
                html.Span("Timestep: ", className="small text-muted"),
                html.Span(id="timestep-display", className="fw-bold"),
                html.Span(" / ", className="small text-muted"),
                html.Span(id="max-timestep-display", className="small"),
            ], className="text-end"),
        ], width=4),
    ], className="mb-2 align-items-center")


def _create_layers_panel():
    """Create layers toggle panel."""
    return dbc.Card([
        dbc.CardHeader("Layers", className="py-2"),
        dbc.CardBody([
            _layer_toggle("lanes", "Lanes", True),
            _layer_toggle("road_lines", "Road Lines", True),
            _layer_toggle("road_edges", "Road Edges", True),
            _layer_toggle("crosswalks", "Features", True),
            html.Hr(className="my-2"),
            _layer_toggle("agents", "Agents", True),
            _layer_toggle("routes", "Routes", True),
            _layer_toggle("trajectories", "Trajectories", True),
            _layer_toggle("agent_ids", "Agent IDs", True),
            html.Hr(className="my-2"),
            _layer_toggle("traffic_lights", "Traffic Lights", True),
            html.Hr(className="my-2"),
            html.Div([
                dbc.Button("Fit All", id="btn-fit-all", size="sm", color="secondary", className="w-100 mb-1"),
                dbc.Button("Follow Ego", id="btn-follow-ego", size="sm", color="secondary", className="w-100 mb-1"),
            ]),
            html.Hr(className="my-2"),
            html.Div([
                html.Label("Zoom Radius (m)", className="small mb-1"),
                dbc.Select(
                    id="zoom-radius-select",
                    options=[
                        {"label": "50m", "value": "50"},
                        {"label": "100m", "value": "100"},
                        {"label": "200m", "value": "200"},
                        {"label": "500m", "value": "500"},
                    ],
                    value="100",
                    size="sm",
                ),
            ]),
        ], className="py-2"),
    ], className="h-100")


def _layer_toggle(layer_id, label, default=True):
    """Create a layer toggle checkbox."""
    return dbc.Checkbox(
        id=f"layer-{layer_id}",
        label=label,
        value=default,
        className="small mb-1",
    )


def _create_viz_panel():
    """Create main visualization panel."""
    return dbc.Card([
        dbc.CardBody([
            dcc.Graph(
                id="main-graph",
                config={
                    "scrollZoom": True,
                    "displayModeBar": True,
                    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                    "displaylogo": False,
                },
                style={"height": "100%"},
            ),
        ], className="p-1 h-100"),
    ], className="h-100")


def _create_info_panel():
    """Create info panel (scenario + selected element)."""
    return dbc.Card([
        dbc.CardBody([
            # Scenario info
            html.Div(id="scenario-info-panel"),

            # Search
            dbc.InputGroup([
                dbc.Input(id="search-input", placeholder="Search ID...", size="sm"),
                dbc.Select(
                    id="search-type",
                    options=[
                        {"label": "Agent", "value": "agent"},
                        {"label": "Road", "value": "road"},
                        {"label": "TL", "value": "traffic_light"},
                    ],
                    value="agent",
                    size="sm",
                    style={"maxWidth": "80px"},
                ),
                dbc.Button("Go", id="btn-search", size="sm", color="primary"),
            ], className="mb-3"),

            # Selected element info
            html.Div(id="element-info-panel"),

        ], className="py-2", style={"overflowY": "auto", "height": "100%"}),
    ], className="h-100")


def _create_timeline_controls():
    """Create timeline/playback controls."""
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                # Playback buttons
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button("⏮", id="btn-start", size="sm", color="secondary"),
                        dbc.Button("◀", id="btn-prev", size="sm", color="secondary"),
                        dbc.Button("▶", id="btn-play", size="sm", color="primary"),
                        dbc.Button("▶▶", id="btn-next", size="sm", color="secondary"),
                        dbc.Button("⏭", id="btn-end", size="sm", color="secondary"),
                    ], className="me-3"),
                ], width="auto"),

                # Slider
                dbc.Col([
                    dcc.Slider(
                        id="timestep-slider",
                        min=0,
                        max=90,
                        step=1,
                        value=0,
                        marks=None,
                        tooltip={"placement": "top", "always_visible": False},
                        className="flex-grow-1",
                    ),
                ]),

                # Speed selector
                dbc.Col([
                    dbc.Select(
                        id="playback-speed",
                        options=[
                            {"label": "0.5x", "value": "200"},
                            {"label": "1x", "value": "100"},
                            {"label": "2x", "value": "50"},
                            {"label": "4x", "value": "25"},
                        ],
                        value="100",
                        size="sm",
                        style={"width": "80px"},
                    ),
                ], width="auto"),

                # File upload
                dbc.Col([
                    dcc.Upload(
                        id="upload-file",
                        children=dbc.Button("Load File", size="sm", color="secondary"),
                    ),
                ], width="auto"),
            ], className="align-items-center"),
        ], className="py-2"),
    ], className="mt-2")


def create_filter_modal():
    """Create filter modal (Phase 2)."""
    return dbc.Modal([
        dbc.ModalHeader("Filter Elements"),
        dbc.ModalBody([
            html.H6("Agent Filters"),
            dbc.Checklist(
                id="filter-agent-types",
                options=[
                    {"label": "Vehicles", "value": "vehicle"},
                    {"label": "Pedestrians", "value": "pedestrian"},
                    {"label": "Cyclists", "value": "cyclist"},
                ],
                value=["vehicle", "pedestrian", "cyclist"],
            ),
            dbc.Checkbox(id="filter-valid-only", label="Valid at current timestep only", value=False),
            dbc.Checkbox(id="filter-ttp-only", label="Tracks to predict only", value=False),
            dbc.Checkbox(id="filter-ooi-only", label="Objects of interest only", value=False),
            dbc.Checkbox(id="filter-with-routes", label="Agents with routes only", value=False),
        ]),
        dbc.ModalFooter([
            dbc.Button("Apply", id="btn-apply-filters", color="primary"),
            dbc.Button("Close", id="btn-close-filters", color="secondary"),
        ]),
    ], id="filter-modal", is_open=False)
