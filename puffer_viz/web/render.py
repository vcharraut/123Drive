"""Plotly figure generation for Puffer scenarios."""

import numpy as np
import plotly.graph_objects as go

from .utils import (
    build_lane_map,
    compute_route_polyline,
    get_agent_color,
    get_agent_type_name,
    get_heading_arrow,
    get_road_styling,
    get_road_type_name,
    get_traffic_state_color,
    get_traffic_state_name,
    get_vehicle_corners,
    is_lane,
    is_road_edge,
    is_road_line,
)


def create_figure(
    scenario,
    timestep=0,
    layers=None,
    selected_element=None,
    highlight_lanes=None,
):
    """Create Plotly figure for scenario at given timestep."""
    if layers is None:
        layers = {
            "lanes": True,
            "road_lines": True,
            "road_edges": True,
            "crosswalks": True,
            "agents": True,
            "routes": True,
            "trajectories": True,
            "traffic_lights": True,
            "agent_ids": True,
        }

    highlight_lanes = highlight_lanes or []
    metadata = scenario.get("metadata", {})

    fig = go.Figure()

    # Render layers in order (bottom to top)
    if layers.get("lanes"):
        _add_lanes(fig, scenario, highlight_lanes)
    if layers.get("road_lines"):
        _add_road_lines(fig, scenario)
    if layers.get("road_edges"):
        _add_road_edges(fig, scenario)
    if layers.get("crosswalks"):
        _add_crosswalks(fig, scenario)
    if layers.get("routes"):
        _add_routes(fig, scenario, metadata)
    if layers.get("trajectories"):
        _add_trajectories(fig, scenario, timestep, metadata)
    if layers.get("agents"):
        _add_agents(fig, scenario, timestep, metadata, layers.get("agent_ids", True))
    if layers.get("traffic_lights"):
        _add_traffic_lights(fig, scenario, timestep)

    # Highlight selected element
    if selected_element:
        _highlight_selection(fig, scenario, selected_element, timestep)

    # Configure layout
    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        dragmode="pan",
        uirevision="constant",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        plot_bgcolor="white",
    )

    return fig


def _add_lanes(fig, scenario, highlight_lanes):
    road_elements = scenario.get("road_map_elements", [])
    for elem in road_elements:
        if not is_lane(elem.get("type", 0)):
            continue
        xyz = elem.get("xyz", np.array([]))
        if len(xyz) < 2:
            continue

        elem_id = elem["id"]
        color, _, width = get_road_styling(elem["type"])
        if elem_id in highlight_lanes:
            color = "#FF6600"
            width = 4

        fig.add_trace(go.Scatter(
            x=xyz[:, 0], y=xyz[:, 1],
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="text",
            hovertext=f"Lane {elem_id}<br>Type: {get_road_type_name(elem['type'])}",
            customdata=[{"type": "road", "id": elem_id}] * len(xyz),
            name=f"lane_{elem_id}",
        ))


def _add_road_lines(fig, scenario):
    road_elements = scenario.get("road_map_elements", [])
    for elem in road_elements:
        if not is_road_line(elem.get("type", 0)):
            continue
        xyz = elem.get("xyz", np.array([]))
        if len(xyz) < 2:
            continue

        color, dash, width = get_road_styling(elem["type"])
        fig.add_trace(go.Scatter(
            x=xyz[:, 0], y=xyz[:, 1],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hoverinfo="text",
            hovertext=f"Road Line {elem['id']}<br>Type: {get_road_type_name(elem['type'])}",
            customdata=[{"type": "road", "id": elem["id"]}] * len(xyz),
            name=f"line_{elem['id']}",
        ))


def _add_road_edges(fig, scenario):
    road_elements = scenario.get("road_map_elements", [])
    for elem in road_elements:
        if not is_road_edge(elem.get("type", 0)):
            continue
        xyz = elem.get("xyz", np.array([]))
        if len(xyz) < 2:
            continue

        color, _, width = get_road_styling(elem["type"])
        fig.add_trace(go.Scatter(
            x=xyz[:, 0], y=xyz[:, 1],
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="text",
            hovertext=f"Road Edge {elem['id']}<br>Type: {get_road_type_name(elem['type'])}",
            customdata=[{"type": "road", "id": elem["id"]}] * len(xyz),
            name=f"edge_{elem['id']}",
        ))


def _add_crosswalks(fig, scenario):
    road_elements = scenario.get("road_map_elements", [])
    for elem in road_elements:
        elem_type = elem.get("type", 0)
        if elem_type not in [31, 32, 33]:  # crosswalk, speed_bump, stop_sign
            continue
        xyz = elem.get("xyz", np.array([]))
        if len(xyz) < 2:
            continue

        color, _, width = get_road_styling(elem_type)
        type_name = get_road_type_name(elem_type)

        if elem_type == 33:  # stop sign - marker
            fig.add_trace(go.Scatter(
                x=[xyz[0, 0]], y=[xyz[0, 1]],
                mode="markers",
                marker=dict(color=color, size=10, symbol="square"),
                hoverinfo="text",
                hovertext=f"Stop Sign {elem['id']}",
                customdata=[{"type": "road", "id": elem["id"]}],
                name=f"stop_{elem['id']}",
            ))
        else:  # crosswalk, speed bump - lines
            fig.add_trace(go.Scatter(
                x=xyz[:, 0], y=xyz[:, 1],
                mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="text",
                hovertext=f"{type_name} {elem['id']}",
                customdata=[{"type": "road", "id": elem["id"]}] * len(xyz),
                name=f"feature_{elem['id']}",
            ))


def _add_routes(fig, scenario, metadata):
    agents = scenario.get("dynamic_agents", [])
    road_elements = scenario.get("road_map_elements", [])
    sdc_index = metadata.get("sdc_index", -1)

    lane_map = build_lane_map(road_elements)

    for i, agent in enumerate(agents):
        routes = agent.get("routes", [])
        if not routes:
            continue

        agent_id = agent["id"]
        is_ego = i == sdc_index
        color = get_agent_color(agent_id, is_ego)

        # Use first route
        route = routes[0]
        start_pos = agent["states"]["xyz"][0] if len(agent["states"]["xyz"]) > 0 else None
        route_pts = compute_route_polyline(route, lane_map, start_pos)

        if route_pts is None or len(route_pts) < 2:
            continue

        fig.add_trace(go.Scatter(
            x=route_pts[:, 0], y=route_pts[:, 1],
            mode="lines",
            line=dict(color=color, width=2, dash="dash"),
            opacity=0.5,
            hoverinfo="text",
            hovertext=f"Route for Agent {agent_id}",
            name=f"route_{agent_id}",
        ))


def _add_trajectories(fig, scenario, timestep, metadata):
    agents = scenario.get("dynamic_agents", [])
    sdc_index = metadata.get("sdc_index", -1)

    for i, agent in enumerate(agents):
        states = agent.get("states", {})
        xyz = states.get("xyz", np.array([]))
        valid = states.get("valid", np.array([]))

        if len(xyz) == 0:
            continue

        agent_id = agent["id"]
        is_ego = i == sdc_index
        color = get_agent_color(agent_id, is_ego)

        # History trajectory (up to current timestep)
        history_mask = np.zeros(len(valid), dtype=bool)
        history_mask[:min(timestep + 1, len(valid))] = valid[:min(timestep + 1, len(valid))] > 0
        if np.sum(history_mask) > 1:
            hist_xyz = xyz[history_mask]
            fig.add_trace(go.Scatter(
                x=hist_xyz[:, 0], y=hist_xyz[:, 1],
                mode="lines",
                line=dict(color=color, width=1.5),
                opacity=0.4,
                hoverinfo="skip",
                name=f"hist_{agent_id}",
            ))

        # Future trajectory (from current timestep)
        future_mask = np.zeros(len(valid), dtype=bool)
        future_mask[timestep:] = valid[timestep:] > 0
        if np.sum(future_mask) > 1:
            fut_xyz = xyz[future_mask]
            fig.add_trace(go.Scatter(
                x=fut_xyz[:, 0], y=fut_xyz[:, 1],
                mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                opacity=0.3,
                hoverinfo="skip",
                name=f"fut_{agent_id}",
            ))


def _add_agents(fig, scenario, timestep, metadata, show_ids):
    agents = scenario.get("dynamic_agents", [])
    sdc_index = metadata.get("sdc_index", -1)
    tracks_to_predict = metadata.get("tracks_to_predict", [])
    objects_of_interest = metadata.get("objects_of_interests", [])

    for i, agent in enumerate(agents):
        states = agent.get("states", {})
        xyz = states.get("xyz", np.array([]))
        heading = states.get("heading", np.array([]))
        valid = states.get("valid", np.array([]))
        length = states.get("length", np.array([]))
        width = states.get("width", np.array([]))
        velocity = states.get("velocity", np.array([]))

        if len(xyz) == 0 or timestep >= len(xyz) or not valid[timestep]:
            continue

        agent_id = agent["id"]
        agent_type = agent.get("type", 0)
        is_ego = i == sdc_index
        color = get_agent_color(agent_id, is_ego)

        x, y = xyz[timestep, 0], xyz[timestep, 1]
        h = heading[timestep] if timestep < len(heading) else 0
        l = length[timestep] if timestep < len(length) else 4.5
        w = width[timestep] if timestep < len(width) else 2.0
        vx = velocity[timestep, 0] if timestep < len(velocity) else 0
        vy = velocity[timestep, 1] if timestep < len(velocity) else 0

        vel_mag = np.sqrt(vx**2 + vy**2)

        # Vehicle body polygon
        xs, ys = get_vehicle_corners(x, y, h, l, w)

        is_ttp = agent_id in tracks_to_predict
        is_ooi = agent_id in objects_of_interest
        edge_color = "red" if is_ttp else ("blue" if is_ooi else "black")
        edge_width = 2 if (is_ttp or is_ooi) else 1

        hover_text = (
            f"Agent {agent_id}<br>"
            f"Type: {get_agent_type_name(agent_type)}<br>"
            f"Pos: ({x:.1f}, {y:.1f})<br>"
            f"Vel: {vel_mag:.1f} m/s<br>"
            f"{'EGO' if is_ego else ''}"
        )

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            fill="toself",
            fillcolor=color,
            line=dict(color=edge_color, width=edge_width),
            opacity=0.8,
            hoverinfo="text",
            hovertext=hover_text,
            customdata=[{"type": "agent", "id": agent_id}] * len(xs),
            name=f"agent_{agent_id}",
        ))

        # Heading arrow
        ax, ay = get_heading_arrow(x, y, h, l)
        fig.add_trace(go.Scatter(
            x=[x, ax], y=[y, ay],
            mode="lines",
            line=dict(color="black", width=2),
            hoverinfo="skip",
            name=f"heading_{agent_id}",
        ))

        # Agent ID label
        if show_ids:
            fig.add_trace(go.Scatter(
                x=[x], y=[y + w * 0.8],
                mode="text",
                text=[str(agent_id)],
                textposition="top center",
                textfont=dict(size=10, color="black"),
                hoverinfo="skip",
                name=f"label_{agent_id}",
            ))


def _add_traffic_lights(fig, scenario, timestep):
    traffic_elements = scenario.get("traffic_control_elements", [])

    for elem in traffic_elements:
        xyz = elem.get("xyz", np.array([]))
        if len(xyz) < 3:
            continue

        x, y = xyz[0], xyz[1]
        states = elem.get("states", [])
        state = states[timestep] if timestep < len(states) else 0
        color = get_traffic_state_color(state)

        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers",
            marker=dict(color=color, size=12, line=dict(color="black", width=1)),
            hoverinfo="text",
            hovertext=f"Traffic Light {elem['id']}<br>State: {get_traffic_state_name(state)}",
            customdata=[{"type": "traffic_light", "id": elem["id"]}],
            name=f"tl_{elem['id']}",
        ))


def _highlight_selection(fig, scenario, selected_element, timestep):
    """Add highlight for selected element."""
    elem_type = selected_element.get("type")
    elem_id = selected_element.get("id")

    if elem_type == "agent":
        agents = scenario.get("dynamic_agents", [])
        for agent in agents:
            if agent["id"] == elem_id:
                states = agent.get("states", {})
                xyz = states.get("xyz", np.array([]))
                if timestep < len(xyz):
                    x, y = xyz[timestep, 0], xyz[timestep, 1]
                    fig.add_trace(go.Scatter(
                        x=[x], y=[y],
                        mode="markers",
                        marker=dict(
                            color="rgba(255,0,0,0.3)",
                            size=30,
                            symbol="circle",
                        ),
                        hoverinfo="skip",
                        name="selection_highlight",
                    ))
                break

    elif elem_type == "road":
        road_elements = scenario.get("road_map_elements", [])
        for elem in road_elements:
            if elem["id"] == elem_id:
                xyz = elem.get("xyz", np.array([]))
                if len(xyz) > 0:
                    fig.add_trace(go.Scatter(
                        x=xyz[:, 0], y=xyz[:, 1],
                        mode="lines",
                        line=dict(color="red", width=5),
                        opacity=0.5,
                        hoverinfo="skip",
                        name="selection_highlight",
                    ))
                break

    elif elem_type == "traffic_light":
        traffic_elements = scenario.get("traffic_control_elements", [])
        for elem in traffic_elements:
            if elem["id"] == elem_id:
                xyz = elem.get("xyz", np.array([]))
                if len(xyz) >= 2:
                    fig.add_trace(go.Scatter(
                        x=[xyz[0]], y=[xyz[1]],
                        mode="markers",
                        marker=dict(
                            color="rgba(255,0,0,0.3)",
                            size=25,
                            symbol="circle",
                        ),
                        hoverinfo="skip",
                        name="selection_highlight",
                    ))
                break
