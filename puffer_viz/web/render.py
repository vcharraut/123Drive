"""Plotly figure generation for Puffer scenarios - optimized for performance."""

import numpy as np
import plotly.graph_objects as go

from .utils import (
    ROAD_COLORS,
    build_lane_map,
    compute_route_polyline,
    get_agent_color,
    get_agent_type_name,
    get_heading_arrow,
    get_road_type_name,
    get_traffic_state_color,
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

    # Render layers in order (bottom to top) - batched for performance
    if layers.get("lanes"):
        _add_lanes_batched(fig, scenario, highlight_lanes)
    if layers.get("road_lines"):
        _add_road_lines_batched(fig, scenario)
    if layers.get("road_edges"):
        _add_road_edges_batched(fig, scenario)
    if layers.get("crosswalks"):
        _add_crosswalks_batched(fig, scenario)
    if layers.get("routes"):
        _add_routes_batched(fig, scenario, metadata)
    if layers.get("trajectories"):
        _add_trajectories_batched(fig, scenario, timestep, metadata)
    if layers.get("agents"):
        _add_agents_batched(fig, scenario, timestep, metadata, layers.get("agent_ids", True))
    if layers.get("traffic_lights"):
        _add_traffic_lights_batched(fig, scenario, timestep)

    # Add click targets for road elements (must be on top of lines for click detection)
    if layers.get("lanes") or layers.get("road_edges") or layers.get("road_lines"):
        _add_click_targets(fig, scenario)

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


def _batch_polylines(elements, filter_fn):
    """Batch multiple polylines into single arrays with None separators."""
    xs, ys, ids, hover_texts = [], [], [], []

    for elem in elements:
        if not filter_fn(elem.get("type", 0)):
            continue
        xyz = elem.get("xyz", np.array([]))
        if len(xyz) < 2:
            continue

        xs.extend(xyz[:, 0].tolist())
        xs.append(None)
        ys.extend(xyz[:, 1].tolist())
        ys.append(None)

        # Store ID for each point (for click detection)
        elem_id = elem["id"]
        ids.extend([{"type": "road", "id": elem_id}] * len(xyz))
        ids.append(None)

        hover_texts.extend([f"ID: {elem_id}<br>Type: {get_road_type_name(elem['type'])}"] * len(xyz))
        hover_texts.append(None)

    return xs, ys, ids, hover_texts


def _add_lanes_batched(fig, scenario, highlight_lanes):
    road_elements = scenario.get("road_map_elements", [])

    # Regular lanes
    xs, ys, ids, hovers = _batch_polylines(road_elements, is_lane)
    if xs:
        fig.add_trace(
            go.Scattergl(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=ROAD_COLORS["lane"], width=1.5),
                hoverinfo="text",
                hovertext=hovers,
                customdata=ids,
                name="lanes",
            ),
        )

    # Highlighted lanes (separate trace)
    if highlight_lanes:
        hxs, hys = [], []
        for elem in road_elements:
            if elem["id"] in highlight_lanes and is_lane(elem.get("type", 0)):
                xyz = elem.get("xyz", np.array([]))
                if len(xyz) >= 2:
                    hxs.extend(xyz[:, 0].tolist())
                    hxs.append(None)
                    hys.extend(xyz[:, 1].tolist())
                    hys.append(None)
        if hxs:
            fig.add_trace(
                go.Scattergl(
                    x=hxs,
                    y=hys,
                    mode="lines",
                    line=dict(color="#FF6600", width=4),
                    hoverinfo="skip",
                    name="lanes_highlight",
                ),
            )


def _add_road_lines_batched(fig, scenario):
    road_elements = scenario.get("road_map_elements", [])

    # Group by style (white solid, white broken, yellow solid, yellow broken)
    groups = {
        "white_solid": ([], []),
        "white_broken": ([], []),
        "yellow_solid": ([], []),
        "yellow_broken": ([], []),
    }
    all_ids, all_hovers = [], []

    for elem in road_elements:
        road_type = elem.get("type", 0)
        if not is_road_line(road_type):
            continue
        xyz = elem.get("xyz", np.array([]))
        if len(xyz) < 2:
            continue

        # Determine group
        is_yellow = road_type in [14, 15, 16, 17, 18]
        is_broken = road_type in [11, 14]

        if is_yellow:
            key = "yellow_broken" if is_broken else "yellow_solid"
        else:
            key = "white_broken" if is_broken else "white_solid"

        groups[key][0].extend(xyz[:, 0].tolist())
        groups[key][0].append(None)
        groups[key][1].extend(xyz[:, 1].tolist())
        groups[key][1].append(None)

    # Add traces for each group
    styles = {
        "white_solid": (ROAD_COLORS["road_line_white"], "solid"),
        "white_broken": (ROAD_COLORS["road_line_white"], "dot"),
        "yellow_solid": (ROAD_COLORS["road_line_yellow"], "solid"),
        "yellow_broken": (ROAD_COLORS["road_line_yellow"], "dot"),
    }

    for key, (xs, ys) in groups.items():
        if xs:
            color, dash = styles[key]
            fig.add_trace(
                go.Scattergl(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=color, width=1.5, dash=dash),
                    hoverinfo="skip",
                    name=f"road_lines_{key}",
                ),
            )


def _add_road_edges_batched(fig, scenario):
    road_elements = scenario.get("road_map_elements", [])
    xs, ys, ids, hovers = _batch_polylines(road_elements, is_road_edge)

    if xs:
        fig.add_trace(
            go.Scattergl(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=ROAD_COLORS["road_edge"], width=2.5),
                hoverinfo="text",
                hovertext=hovers,
                customdata=ids,
                name="road_edges",
            ),
        )


def _add_crosswalks_batched(fig, scenario):
    road_elements = scenario.get("road_map_elements", [])

    # Crosswalks (31)
    cw_xs, cw_ys = [], []
    # Speed bumps (32)
    sb_xs, sb_ys = [], []
    # Stop signs (33) - points
    ss_xs, ss_ys = [], []

    for elem in road_elements:
        elem_type = elem.get("type", 0)
        xyz = elem.get("xyz", np.array([]))
        if len(xyz) < 1:
            continue

        if elem_type == 31:
            cw_xs.extend(xyz[:, 0].tolist())
            cw_xs.append(None)
            cw_ys.extend(xyz[:, 1].tolist())
            cw_ys.append(None)
        elif elem_type == 32:
            sb_xs.extend(xyz[:, 0].tolist())
            sb_xs.append(None)
            sb_ys.extend(xyz[:, 1].tolist())
            sb_ys.append(None)
        elif elem_type == 33:
            ss_xs.append(xyz[0, 0])
            ss_ys.append(xyz[0, 1])

    if cw_xs:
        fig.add_trace(
            go.Scattergl(
                x=cw_xs,
                y=cw_ys,
                mode="lines",
                line=dict(color=ROAD_COLORS["crosswalk"], width=3),
                hoverinfo="skip",
                name="crosswalks",
            ),
        )

    if sb_xs:
        fig.add_trace(
            go.Scattergl(
                x=sb_xs,
                y=sb_ys,
                mode="lines",
                line=dict(color=ROAD_COLORS["speed_bump"], width=3),
                hoverinfo="skip",
                name="speed_bumps",
            ),
        )

    if ss_xs:
        fig.add_trace(
            go.Scattergl(
                x=ss_xs,
                y=ss_ys,
                mode="markers",
                marker=dict(color=ROAD_COLORS["stop_sign"], size=8, symbol="square"),
                hoverinfo="skip",
                name="stop_signs",
            ),
        )


def _add_routes_batched(fig, scenario, metadata):
    agents = scenario.get("dynamic_agents", [])
    road_elements = scenario.get("road_map_elements", [])
    sdc_index = metadata.get("sdc_index", -1)

    if not agents:
        return

    lane_map = build_lane_map(road_elements)

    # Batch all routes
    xs, ys = [], []

    for i, agent in enumerate(agents):
        routes = agent.get("routes", [])
        if not routes:
            continue

        start_pos = agent["states"]["xyz"][0] if len(agent["states"]["xyz"]) > 0 else None
        route_pts = compute_route_polyline(routes[0], lane_map, start_pos)

        if route_pts is None or len(route_pts) < 2:
            continue

        xs.extend(route_pts[:, 0].tolist())
        xs.append(None)
        ys.extend(route_pts[:, 1].tolist())
        ys.append(None)

    if xs:
        fig.add_trace(
            go.Scattergl(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="#666666", width=2, dash="dash"),
                opacity=0.5,
                hoverinfo="skip",
                name="routes",
            ),
        )


def _add_trajectories_batched(fig, scenario, timestep, metadata):
    agents = scenario.get("dynamic_agents", [])

    if not agents:
        return

    # Batch history and future trajectories
    hist_xs, hist_ys = [], []
    fut_xs, fut_ys = [], []

    for agent in agents:
        states = agent.get("states", {})
        xyz = states.get("xyz", np.array([]))
        valid = states.get("valid", np.array([]))

        if len(xyz) == 0:
            continue

        # History
        for t in range(min(timestep + 1, len(valid))):
            if valid[t]:
                hist_xs.append(xyz[t, 0])
                hist_ys.append(xyz[t, 1])
        hist_xs.append(None)
        hist_ys.append(None)

        # Future
        for t in range(timestep, len(valid)):
            if valid[t]:
                fut_xs.append(xyz[t, 0])
                fut_ys.append(xyz[t, 1])
        fut_xs.append(None)
        fut_ys.append(None)

    if hist_xs:
        fig.add_trace(
            go.Scattergl(
                x=hist_xs,
                y=hist_ys,
                mode="lines",
                line=dict(color="#888888", width=1.5),
                opacity=0.4,
                hoverinfo="skip",
                name="trajectories_history",
            ),
        )

    if fut_xs:
        fig.add_trace(
            go.Scattergl(
                x=fut_xs,
                y=fut_ys,
                mode="lines",
                line=dict(color="#888888", width=1, dash="dot"),
                opacity=0.3,
                hoverinfo="skip",
                name="trajectories_future",
            ),
        )


def _add_agents_batched(fig, scenario, timestep, metadata, show_ids):
    agents = scenario.get("dynamic_agents", [])
    sdc_index = metadata.get("sdc_index", -1)
    tracks_to_predict = metadata.get("tracks_to_predict", [])
    objects_of_interest = metadata.get("objects_of_interests", [])

    if not agents:
        return

    # Collect agent data
    body_xs, body_ys, body_colors, body_ids, body_hovers = [], [], [], [], []
    arrow_xs, arrow_ys = [], []
    label_xs, label_ys, label_texts = [], [], []

    # Special markers for TTP/OOI
    ttp_xs, ttp_ys = [], []
    ooi_xs, ooi_ys = [], []

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

        # Vehicle body
        bxs, bys = get_vehicle_corners(x, y, h, l, w)
        body_xs.extend(bxs)
        body_xs.append(None)
        body_ys.extend(bys)
        body_ys.append(None)

        hover = f"Agent {agent_id}<br>Type: {get_agent_type_name(agent_type)}<br>Vel: {vel_mag:.1f} m/s"
        body_hovers.extend([hover] * len(bxs))
        body_hovers.append(None)
        body_ids.extend([{"type": "agent", "id": agent_id}] * len(bxs))
        body_ids.append(None)

        # Heading arrow
        ax, ay = get_heading_arrow(x, y, h, l)
        arrow_xs.extend([x, ax, None])
        arrow_ys.extend([y, ay, None])

        # Label
        if show_ids:
            label_xs.append(x)
            label_ys.append(y + w * 0.8)
            label_texts.append(str(agent_id))

        # TTP/OOI markers
        if agent_id in tracks_to_predict:
            ttp_xs.append(x)
            ttp_ys.append(y)
        if agent_id in objects_of_interest:
            ooi_xs.append(x)
            ooi_ys.append(y)

    # Add traces
    if body_xs:
        fig.add_trace(
            go.Scattergl(
                x=body_xs,
                y=body_ys,
                mode="lines",
                fill="toself",
                fillcolor="#1F77B4",
                line=dict(color="black", width=1),
                opacity=0.8,
                hoverinfo="text",
                hovertext=body_hovers,
                customdata=body_ids,
                name="agents",
            ),
        )

    if arrow_xs:
        fig.add_trace(
            go.Scattergl(
                x=arrow_xs,
                y=arrow_ys,
                mode="lines",
                line=dict(color="black", width=2),
                hoverinfo="skip",
                name="agent_headings",
            ),
        )

    if label_xs:
        fig.add_trace(
            go.Scattergl(
                x=label_xs,
                y=label_ys,
                mode="text",
                text=label_texts,
                textposition="top center",
                textfont=dict(size=10, color="black"),
                hoverinfo="skip",
                name="agent_labels",
            ),
        )

    if ttp_xs:
        fig.add_trace(
            go.Scattergl(
                x=ttp_xs,
                y=ttp_ys,
                mode="markers",
                marker=dict(color="rgba(255,0,0,0.3)", size=20, symbol="circle"),
                hoverinfo="skip",
                name="tracks_to_predict",
            ),
        )

    if ooi_xs:
        fig.add_trace(
            go.Scattergl(
                x=ooi_xs,
                y=ooi_ys,
                mode="markers",
                marker=dict(color="rgba(0,0,255,0.3)", size=20, symbol="circle"),
                hoverinfo="skip",
                name="objects_of_interest",
            ),
        )


def _add_traffic_lights_batched(fig, scenario, timestep):
    traffic_elements = scenario.get("traffic_control_elements", [])

    if not traffic_elements:
        return

    # Group by state color
    by_color = {}

    for elem in traffic_elements:
        xyz = elem.get("xyz", np.array([]))
        if len(xyz) < 3:
            continue

        states = elem.get("states", [])
        state = states[timestep] if timestep < len(states) else 0
        color = get_traffic_state_color(state)

        if color not in by_color:
            by_color[color] = ([], [], [])

        by_color[color][0].append(xyz[0])
        by_color[color][1].append(xyz[1])
        by_color[color][2].append({"type": "traffic_light", "id": elem["id"]})

    for color, (xs, ys, ids) in by_color.items():
        fig.add_trace(
            go.Scattergl(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(color=color, size=12, line=dict(color="black", width=1)),
                hoverinfo="text",
                hovertext=[f"Traffic Light {d['id']}" for d in ids],
                customdata=ids,
                name=f"traffic_lights_{color}",
            ),
        )


def _add_click_targets(fig, scenario):
    """Add invisible markers along road elements for easier click detection."""
    road_elements = scenario.get("road_map_elements", [])

    xs, ys, ids, hovers = [], [], [], []

    for elem in road_elements:
        elem_type = elem.get("type", 0)
        if not (is_lane(elem_type) or is_road_edge(elem_type) or is_road_line(elem_type)):
            continue
        xyz = elem.get("xyz", np.array([]))
        if len(xyz) < 2:
            continue

        # Sample every ~10 points or minimum 3 points per element
        step = max(1, len(xyz) // 10)
        elem_id = elem["id"]
        for i in range(0, len(xyz), step):
            xs.append(xyz[i, 0])
            ys.append(xyz[i, 1])
            ids.append({"type": "road", "id": elem_id})
            hovers.append(f"ID: {elem_id}<br>Type: {get_road_type_name(elem_type)}")

    if xs:
        fig.add_trace(
            go.Scattergl(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(size=12, color="rgba(0,0,0,0)", line=dict(width=0)),
                hoverinfo="text",
                hovertext=hovers,
                customdata=ids,
                name="click_targets",
            ),
        )


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
                    fig.add_trace(
                        go.Scattergl(
                            x=[x],
                            y=[y],
                            mode="markers",
                            marker=dict(color="rgba(255,0,0,0.4)", size=35),
                            hoverinfo="skip",
                            name="selection",
                        ),
                    )
                break

    elif elem_type == "road":
        road_elements = scenario.get("road_map_elements", [])
        for elem in road_elements:
            if elem["id"] == elem_id:
                xyz = elem.get("xyz", np.array([]))
                if len(xyz) > 0:
                    fig.add_trace(
                        go.Scattergl(
                            x=xyz[:, 0],
                            y=xyz[:, 1],
                            mode="lines",
                            line=dict(color="red", width=6),
                            opacity=0.6,
                            hoverinfo="skip",
                            name="selection",
                        ),
                    )
                break

    elif elem_type == "traffic_light":
        traffic_elements = scenario.get("traffic_control_elements", [])
        for elem in traffic_elements:
            if elem["id"] == elem_id:
                xyz = elem.get("xyz", np.array([]))
                if len(xyz) >= 2:
                    fig.add_trace(
                        go.Scattergl(
                            x=[xyz[0]],
                            y=[xyz[1]],
                            mode="markers",
                            marker=dict(color="rgba(255,0,0,0.4)", size=30),
                            hoverinfo="skip",
                            name="selection",
                        ),
                    )
                break
