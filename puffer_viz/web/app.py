#!/usr/bin/env python
"""Main Dash app entry point for Puffer visualization."""

import argparse
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc


# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from puffer_viz.binary_loader import load_puffer_binary
from puffer_viz.web.callbacks import _serialize_scenario, register_callbacks
from puffer_viz.web.layout import create_layout


def create_app(scenario=None):
    """Create and configure the Dash app."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )

    app.layout = create_layout()
    register_callbacks(app)

    # If scenario provided, inject it
    if scenario is not None:
        _inject_initial_scenario(app, scenario)

    return app


def _inject_initial_scenario(app, scenario):
    """Inject initial scenario data into the app."""

    # Serialize for storage
    serialized = _serialize_scenario(scenario)
    metadata = scenario.get("metadata", {})
    length = metadata.get("scenario_length", 91)
    scenario_id = scenario.get("scenario_id", "unknown")
    dataset = metadata.get("dataset_name", "unknown")

    # Override layout to include initial data
    original_layout = app.layout

    def serve_layout():
        layout = original_layout
        # Update stores with initial data
        for component in _find_components(layout, "scenario-data"):
            component.data = serialized
        for component in _find_components(layout, "timestep-slider"):
            component.max = length - 1
            component.value = 0
        for component in _find_components(layout, "scenario-id-display"):
            component.children = scenario_id
        for component in _find_components(layout, "dataset-display"):
            component.children = dataset
        for component in _find_components(layout, "max-timestep-display"):
            component.children = str(length - 1)
        return layout

    app.layout = serve_layout


def _find_components(layout, component_id):
    """Recursively find components by ID."""
    if hasattr(layout, "id") and layout.id == component_id:
        yield layout
    if hasattr(layout, "children"):
        children = layout.children
        if isinstance(children, list):
            for child in children:
                yield from _find_components(child, component_id)
        elif children is not None:
            yield from _find_components(children, component_id)


def main():
    parser = argparse.ArgumentParser(description="Puffer Scenario Visualization")
    parser.add_argument("file", nargs="?", help="Binary scenario file to load")
    parser.add_argument("--host", default="127.0.0.1", help="Host to run on")
    parser.add_argument("--port", type=int, default=8050, help="Port to run on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    # Load initial scenario if provided
    scenario = None
    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"Error: File not found: {path}")
            sys.exit(1)
        print(f"Loading scenario from {path}...")
        scenario = load_puffer_binary(path)
        print(f"Loaded scenario: {scenario.get('scenario_id', 'unknown')}")

    app = create_app(scenario)

    print(f"\nStarting Puffer Viz at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
