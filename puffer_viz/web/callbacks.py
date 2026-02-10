"""Dash callbacks for Puffer web visualization."""

import base64
from pathlib import Path

import numpy as np
from dash import Input, Output, State, ctx, no_update

from .info_panels import create_element_info, create_scenario_info
from .render import create_figure


def register_callbacks(app):
    """Register all callbacks with the app."""

    @app.callback(
        [
            Output("scenario-data", "data"),
            Output("timestep-slider", "max"),
            Output("timestep-slider", "value"),
            Output("scenario-id-display", "children"),
            Output("dataset-display", "children"),
            Output("max-timestep-display", "children"),
        ],
        [Input("upload-file", "contents")],
        [State("upload-file", "filename")],
        prevent_initial_call=True,
    )
    def load_uploaded_file(contents, filename):
        """Load scenario from uploaded file."""
        if contents is None:
            return no_update, no_update, no_update, no_update, no_update, no_update

        # Decode file
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        # Save to temp and load
        import tempfile

        from puffer_viz.binary_loader import load_puffer_binary

        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(decoded)
            temp_path = f.name

        scenario = load_puffer_binary(temp_path)

        # Convert numpy arrays to lists for JSON serialization
        scenario = _serialize_scenario(scenario)

        metadata = scenario.get("metadata", {})
        length = metadata.get("scenario_length", 91)
        scenario_id = scenario.get("scenario_id", "unknown")
        dataset = metadata.get("dataset_name", "unknown")

        return scenario, length - 1, 0, scenario_id, dataset, str(length - 1)

    @app.callback(
        Output("main-graph", "figure"),
        [
            Input("scenario-data", "data"),
            Input("timestep-slider", "value"),
            Input("layer-lanes", "value"),
            Input("layer-road_lines", "value"),
            Input("layer-road_edges", "value"),
            Input("layer-crosswalks", "value"),
            Input("layer-agents", "value"),
            Input("layer-routes", "value"),
            Input("layer-trajectories", "value"),
            Input("layer-traffic_lights", "value"),
            Input("layer-agent_ids", "value"),
            Input("selected-element", "data"),
            Input("highlight-lanes", "data"),
        ],
    )
    def update_figure(
        scenario,
        timestep,
        lanes,
        road_lines,
        road_edges,
        crosswalks,
        agents,
        routes,
        trajectories,
        traffic_lights,
        agent_ids,
        selected_element,
        highlight_lanes,
    ):
        """Update the main visualization figure."""
        if scenario is None:
            return _empty_figure()

        # Deserialize numpy arrays
        scenario = _deserialize_scenario(scenario)

        layers = {
            "lanes": lanes,
            "road_lines": road_lines,
            "road_edges": road_edges,
            "crosswalks": crosswalks,
            "agents": agents,
            "routes": routes,
            "trajectories": trajectories,
            "traffic_lights": traffic_lights,
            "agent_ids": agent_ids,
        }

        return create_figure(
            scenario,
            timestep=timestep,
            layers=layers,
            selected_element=selected_element,
            highlight_lanes=highlight_lanes or [],
        )

    @app.callback(
        [Output("selected-element", "data"), Output("highlight-lanes", "data")],
        [Input("main-graph", "clickData"), Input("btn-search", "n_clicks"), Input("btn-clear-selection", "n_clicks")],
        [State("search-input", "value"), State("search-type", "value"), State("scenario-data", "data")],
    )
    def handle_selection(click_data, search_clicks, clear_clicks, search_value, search_type, scenario):
        """Handle click on element, search, or clear."""
        triggered = ctx.triggered_id

        if triggered == "btn-clear-selection":
            return None, []

        if triggered == "btn-search" and search_value:
            # Search by ID
            try:
                search_id = int(search_value)
                selected = {"type": search_type, "id": search_id}
                highlight = _get_highlight_lanes(scenario, selected)
                return selected, highlight
            except ValueError:
                return no_update, no_update

        if triggered == "main-graph" and click_data:
            # Click on element
            point = click_data.get("points", [{}])[0]
            customdata = point.get("customdata")

            if customdata and isinstance(customdata, dict):
                selected = customdata
                highlight = _get_highlight_lanes(scenario, selected)
                return selected, highlight

        return no_update, no_update

    @app.callback(
        Output("scenario-info-panel", "children"),
        [Input("scenario-data", "data")],
    )
    def update_scenario_info(scenario):
        """Update scenario info panel."""
        if scenario is None:
            return create_scenario_info(None)
        scenario = _deserialize_scenario(scenario)
        return create_scenario_info(scenario)

    @app.callback(
        Output("element-info-panel", "children"),
        [Input("selected-element", "data"), Input("timestep-slider", "value")],
        [State("scenario-data", "data")],
    )
    def update_element_info(selected_element, timestep, scenario):
        """Update selected element info panel."""
        if scenario is None:
            return create_element_info(None, None, 0)
        scenario = _deserialize_scenario(scenario)
        return create_element_info(scenario, selected_element, timestep)

    @app.callback(
        Output("timestep-display", "children"),
        [Input("timestep-slider", "value")],
    )
    def update_timestep_display(timestep):
        return str(timestep)

    # Playback controls
    @app.callback(
        [
            Output("timestep-slider", "value", allow_duplicate=True),
            Output("playback-interval", "disabled"),
            Output("btn-play", "children"),
        ],
        [
            Input("btn-play", "n_clicks"),
            Input("btn-start", "n_clicks"),
            Input("btn-end", "n_clicks"),
            Input("btn-prev", "n_clicks"),
            Input("btn-next", "n_clicks"),
            Input("playback-interval", "n_intervals"),
        ],
        [
            State("timestep-slider", "value"),
            State("timestep-slider", "max"),
            State("playback-interval", "disabled"),
            State("playback-speed", "value"),
        ],
        prevent_initial_call=True,
    )
    def playback_controls(
        play_clicks,
        start_clicks,
        end_clicks,
        prev_clicks,
        next_clicks,
        intervals,
        current_ts,
        max_ts,
        is_paused,
        speed,
    ):
        triggered = ctx.triggered_id

        if triggered == "btn-play":
            # Toggle play/pause
            return no_update, not is_paused, "⏸" if is_paused else "▶"

        if triggered == "btn-start":
            return 0, True, "▶"

        if triggered == "btn-end":
            return max_ts, True, "▶"

        if triggered == "btn-prev":
            return max(0, current_ts - 1), True, "▶"

        if triggered == "btn-next":
            return min(max_ts, current_ts + 1), True, "▶"

        if triggered == "playback-interval":
            # Auto-advance during playback
            if current_ts >= max_ts:
                return 0, True, "▶"  # Stop at end
            return current_ts + 1, False, "⏸"

        return no_update, no_update, no_update

    @app.callback(
        Output("playback-interval", "interval"),
        [Input("playback-speed", "value")],
    )
    def update_playback_speed(speed):
        return int(speed)

    # View controls
    @app.callback(
        Output("main-graph", "figure", allow_duplicate=True),
        [Input("btn-fit-all", "n_clicks")],
        [
            State("scenario-data", "data"),
            State("timestep-slider", "value"),
            State("layer-lanes", "value"),
            State("layer-road_lines", "value"),
            State("layer-road_edges", "value"),
            State("layer-crosswalks", "value"),
            State("layer-agents", "value"),
            State("layer-routes", "value"),
            State("layer-trajectories", "value"),
            State("layer-traffic_lights", "value"),
            State("layer-agent_ids", "value"),
            State("selected-element", "data"),
            State("highlight-lanes", "data"),
        ],
        prevent_initial_call=True,
    )
    def fit_all(
        n_clicks,
        scenario,
        timestep,
        lanes,
        road_lines,
        road_edges,
        crosswalks,
        agents,
        routes,
        trajectories,
        traffic_lights,
        agent_ids,
        selected_element,
        highlight_lanes,
    ):
        if scenario is None:
            return no_update

        scenario = _deserialize_scenario(scenario)

        layers = {
            "lanes": lanes,
            "road_lines": road_lines,
            "road_edges": road_edges,
            "crosswalks": crosswalks,
            "agents": agents,
            "routes": routes,
            "trajectories": trajectories,
            "traffic_lights": traffic_lights,
            "agent_ids": agent_ids,
        }

        fig = create_figure(scenario, timestep, layers, selected_element, highlight_lanes)
        fig.update_layout(xaxis={"autorange": True}, yaxis={"autorange": True})
        return fig

    @app.callback(
        Output("main-graph", "figure", allow_duplicate=True),
        [Input("btn-follow-ego", "n_clicks")],
        [
            State("scenario-data", "data"),
            State("timestep-slider", "value"),
            State("zoom-radius-select", "value"),
            State("layer-lanes", "value"),
            State("layer-road_lines", "value"),
            State("layer-road_edges", "value"),
            State("layer-crosswalks", "value"),
            State("layer-agents", "value"),
            State("layer-routes", "value"),
            State("layer-trajectories", "value"),
            State("layer-traffic_lights", "value"),
            State("layer-agent_ids", "value"),
            State("selected-element", "data"),
            State("highlight-lanes", "data"),
        ],
        prevent_initial_call=True,
    )
    def follow_ego(
        n_clicks,
        scenario,
        timestep,
        zoom_radius,
        lanes,
        road_lines,
        road_edges,
        crosswalks,
        agents,
        routes,
        trajectories,
        traffic_lights,
        agent_ids,
        selected_element,
        highlight_lanes,
    ):
        if scenario is None:
            return no_update

        scenario = _deserialize_scenario(scenario)
        metadata = scenario.get("metadata", {})
        sdc_index = metadata.get("sdc_index", -1)
        dynamic_agents = scenario.get("agents", [])

        # Find ego position
        ego_x, ego_y = None, None
        if 0 <= sdc_index < len(dynamic_agents):
            ego_agent = dynamic_agents[sdc_index]
            states = ego_agent.get("states", {})
            xyz = states.get("xyz", np.array([]))
            valid = states.get("valid", np.array([]))
            if timestep < len(xyz) and valid[timestep]:
                ego_x, ego_y = xyz[timestep, 0], xyz[timestep, 1]

        if ego_x is None:
            return no_update

        layers = {
            "lanes": lanes,
            "road_lines": road_lines,
            "road_edges": road_edges,
            "crosswalks": crosswalks,
            "agents": agents,
            "routes": routes,
            "trajectories": trajectories,
            "traffic_lights": traffic_lights,
            "agent_ids": agent_ids,
        }

        fig = create_figure(scenario, timestep, layers, selected_element, highlight_lanes)

        radius = float(zoom_radius)
        fig.update_layout(
            xaxis={"range": [ego_x - radius, ego_x + radius]},
            yaxis={"range": [ego_y - radius, ego_y + radius]},
        )
        return fig


def _empty_figure():
    """Create empty figure placeholder."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(
        showlegend=False,
        xaxis={"visible": False},
        yaxis={"visible": False},
        plot_bgcolor="white",
        annotations=[
            {
                "text": "Load a scenario file to begin",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 16, "color": "gray"},
            },
        ],
    )
    return fig


def _serialize_scenario(scenario):
    """Convert numpy arrays to lists for JSON storage."""

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    return convert(scenario)


def _deserialize_scenario(scenario):
    """Convert lists back to numpy arrays."""
    if scenario is None:
        return None

    # Convert agent states
    for agent in scenario.get("agents", []):
        states = agent.get("states", {})
        for key in ["xyz", "heading", "velocity", "length", "width", "height", "valid"]:
            if key in states and isinstance(states[key], list):
                states[key] = np.array(states[key])

    # Convert road element xyz
    for road in scenario.get("road_map_elements", []):
        if "xyz" in road and isinstance(road["xyz"], list):
            road["xyz"] = np.array(road["xyz"])

    # Convert traffic element xyz
    for traffic in scenario.get("traffic_control_elements", []):
        if "xyz" in traffic and isinstance(traffic["xyz"], list):
            traffic["xyz"] = np.array(traffic["xyz"])

    return scenario


def _get_highlight_lanes(scenario, selected):
    """Get lanes to highlight based on selection."""
    if scenario is None or selected is None:
        return []

    scenario = _deserialize_scenario(scenario) if isinstance(scenario, dict) else scenario

    elem_type = selected.get("type")
    elem_id = selected.get("id")

    if elem_type == "agent":
        # Highlight route lanes
        for agent in scenario.get("agents", []):
            if agent["id"] == elem_id:
                routes = agent.get("routes", [])
                if routes:
                    return routes[0]  # Return first route lane IDs
        return []

    if elem_type == "road":
        # Highlight connected lanes
        for road in scenario.get("road_map_elements", []):
            if road["id"] == elem_id:
                entry = road.get("entry_lanes", [])
                exit_ = road.get("exit_lanes", [])
                return entry + exit_
        return []

    if elem_type == "traffic_light":
        # Highlight controlled lanes
        for traffic in scenario.get("traffic_control_elements", []):
            if traffic["id"] == elem_id:
                return traffic.get("controlled_lanes", [])
        return []

    return []
