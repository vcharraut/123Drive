"""Info panel generation for selected elements."""

import dash_bootstrap_components as dbc
import numpy as np
from dash import html

from .utils import (
    format_heading,
    format_velocity,
    get_agent_type_name,
    get_road_type_name,
    get_traffic_state_color,
    get_traffic_state_name,
    is_lane,
)


def create_scenario_info(scenario):
    """Create scenario overview panel."""
    if not scenario:
        return html.Div("No scenario loaded", className="text-muted")

    metadata = scenario.get("metadata", {})
    agents = scenario.get("dynamic_agents", [])
    roads = scenario.get("road_map_elements", [])
    traffic = scenario.get("traffic_control_elements", [])

    sdc_index = metadata.get("sdc_index", -1)
    ooi = metadata.get("objects_of_interests", [])
    ttp = metadata.get("tracks_to_predict", [])

    return dbc.Card(
        [
            dbc.CardHeader("Scenario Info"),
            dbc.CardBody(
                [
                    _info_row("ID", scenario.get("scenario_id", "unknown")),
                    _info_row("Dataset", metadata.get("dataset_name", "unknown")),
                    _info_row("Map ID", metadata.get("map_id", "unknown")),
                    _info_row("Length", f"{metadata.get('scenario_length', 0)} steps"),
                    html.Hr(className="my-2"),
                    _info_row("Agents", len(agents)),
                    _info_row("Road Elements", len(roads)),
                    _info_row("Traffic Lights", len(traffic)),
                    _info_row("SDC Index", sdc_index if sdc_index >= 0 else "None"),
                    html.Hr(className="my-2"),
                    html.Div(
                        [
                            html.Strong("Objects of Interest: ", className="small"),
                            html.Span(", ".join(map(str, ooi)) if ooi else "None", className="small"),
                        ],
                    ),
                    html.Div(
                        [
                            html.Strong("Tracks to Predict: ", className="small"),
                            html.Span(", ".join(map(str, ttp)) if ttp else "None", className="small"),
                        ],
                    ),
                ],
            ),
        ],
        className="mb-3",
    )


def create_element_info(scenario, selected_element, timestep):
    """Create detailed info panel for selected element."""
    if not scenario or not selected_element:
        return html.Div("Click on an element to see details", className="text-muted p-3")

    elem_type = selected_element.get("type")
    elem_id = selected_element.get("id")

    if elem_type == "agent":
        return _agent_info(scenario, elem_id, timestep)
    elif elem_type == "road":
        return _road_info(scenario, elem_id)
    elif elem_type == "traffic_light":
        return _traffic_light_info(scenario, elem_id, timestep)

    return html.Div(f"Unknown element type: {elem_type}")


def _agent_info(scenario, agent_id, timestep):
    """Create detailed agent info panel."""
    agents = scenario.get("dynamic_agents", [])
    metadata = scenario.get("metadata", {})
    sdc_index = metadata.get("sdc_index", -1)
    ttp = metadata.get("tracks_to_predict", [])
    ooi = metadata.get("objects_of_interests", [])

    agent = None
    agent_idx = -1
    for i, a in enumerate(agents):
        if a["id"] == agent_id:
            agent = a
            agent_idx = i
            break

    if not agent:
        return html.Div(f"Agent {agent_id} not found")

    states = agent.get("states", {})
    xyz = states.get("xyz", np.array([]))
    heading = states.get("heading", np.array([]))
    velocity = states.get("velocity", np.array([]))
    length = states.get("length", np.array([]))
    width = states.get("width", np.array([]))
    height = states.get("height", np.array([]))
    valid = states.get("valid", np.array([]))
    routes = agent.get("routes", [])

    is_ego = agent_idx == sdc_index
    is_ttp = agent_id in ttp
    is_ooi = agent_id in ooi

    # Current state
    current_valid = valid[timestep] if timestep < len(valid) else False
    if current_valid and timestep < len(xyz):
        x, y, z = xyz[timestep]
        h = heading[timestep] if timestep < len(heading) else 0
        vx, vy = velocity[timestep] if timestep < len(velocity) else (0, 0)
        l = length[timestep] if timestep < len(length) else 0
        w = width[timestep] if timestep < len(width) else 0
        ht = height[timestep] if timestep < len(height) else 0
    else:
        x = y = z = h = vx = vy = l = w = ht = None

    # Stats
    valid_frames = np.sum(valid > 0)
    valid_indices = np.where(valid > 0)[0]
    first_valid = int(valid_indices[0]) if len(valid_indices) > 0 else None
    last_valid = int(valid_indices[-1]) if len(valid_indices) > 0 else None

    # Build panel
    badges = []
    if is_ego:
        badges.append(dbc.Badge("EGO", color="danger", className="me-1"))
    if is_ttp:
        badges.append(dbc.Badge("Track to Predict", color="warning", className="me-1"))
    if is_ooi:
        badges.append(dbc.Badge("Object of Interest", color="info", className="me-1"))

    route_section = []
    if routes:
        route_ids = routes[0] if routes else []
        route_section = [
            html.Hr(className="my-2"),
            html.Strong("Route (Lane IDs):", className="small d-block"),
            html.Div(
                ", ".join(map(str, route_ids[:20])) + ("..." if len(route_ids) > 20 else ""),
                className="small text-monospace",
                style={"maxHeight": "60px", "overflow": "auto"},
            ),
        ]

    # Trajectory table (collapsed by default)
    traj_rows = []
    for t in range(len(xyz)):
        v = valid[t] if t < len(valid) else 0
        if v:
            tx, ty, tz = xyz[t]
            th = heading[t] if t < len(heading) else 0
            tvx, tvy = velocity[t] if t < len(velocity) else (0, 0)
            traj_rows.append(
                html.Tr(
                    [
                        html.Td(t, className="small"),
                        html.Td(f"{tx:.1f}", className="small"),
                        html.Td(f"{ty:.1f}", className="small"),
                        html.Td(f"{np.degrees(th):.0f}°", className="small"),
                        html.Td(f"{np.sqrt(tvx**2 + tvy**2):.1f}", className="small"),
                    ],
                ),
            )

    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Span(f"Agent {agent_id}", className="fw-bold"),
                    html.Span(badges, className="ms-2"),
                ],
            ),
            dbc.CardBody(
                [
                    _info_row("Type", get_agent_type_name(agent.get("type", 0))),
                    _info_row("Valid at t=" + str(timestep), "Yes" if current_valid else "No"),
                    html.Hr(className="my-2"),
                    html.Strong("Current State:", className="small d-block mb-1"),
                    _info_row("Position", f"({x:.2f}, {y:.2f}, {z:.2f})" if x is not None else "N/A"),
                    _info_row("Heading", format_heading(h) if h is not None else "N/A"),
                    _info_row("Velocity", format_velocity(vx, vy) if vx is not None else "N/A"),
                    _info_row("Dimensions", f"{l:.1f} x {w:.1f} x {ht:.1f} m" if l is not None else "N/A"),
                    html.Hr(className="my-2"),
                    html.Strong("Statistics:", className="small d-block mb-1"),
                    _info_row("Valid Frames", f"{valid_frames} / {len(valid)}"),
                    _info_row("First Valid", first_valid),
                    _info_row("Last Valid", last_valid),
                    *route_section,
                    html.Hr(className="my-2"),
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                [
                                    html.Div(
                                        [
                                            dbc.Table(
                                                [
                                                    html.Thead(
                                                        html.Tr(
                                                            [
                                                                html.Th("t", className="small"),
                                                                html.Th("x", className="small"),
                                                                html.Th("y", className="small"),
                                                                html.Th("hdg", className="small"),
                                                                html.Th("vel", className="small"),
                                                            ],
                                                        ),
                                                    ),
                                                    html.Tbody(traj_rows[:50]),  # Limit rows
                                                ],
                                                size="sm",
                                                striped=True,
                                                bordered=True,
                                            ),
                                            html.Small(
                                                f"Showing {min(50, len(traj_rows))}/{len(traj_rows)} valid frames",
                                            )
                                            if len(traj_rows) > 50
                                            else None,
                                        ],
                                        style={"maxHeight": "200px", "overflow": "auto"},
                                    ),
                                ],
                                title="Trajectory Data",
                                className="small",
                            ),
                        ],
                        start_collapsed=True,
                    ),
                ],
            ),
        ],
    )


def _road_info(scenario, road_id):
    """Create detailed road element info panel."""
    road_elements = scenario.get("road_map_elements", [])

    elem = None
    for r in road_elements:
        if r["id"] == road_id:
            elem = r
            break

    if not elem:
        return html.Div(f"Road element {road_id} not found")

    xyz = elem.get("xyz", np.array([]))
    entry_lanes = elem.get("entry_lanes", [])
    exit_lanes = elem.get("exit_lanes", [])
    speed_limit = elem.get("speed_limit", 0)
    road_type = elem.get("type", 0)

    # Compute bounding box
    if len(xyz) > 0:
        min_x, min_y = xyz[:, 0].min(), xyz[:, 1].min()
        max_x, max_y = xyz[:, 0].max(), xyz[:, 1].max()
        bbox = f"({min_x:.1f}, {min_y:.1f}) to ({max_x:.1f}, {max_y:.1f})"
    else:
        bbox = "N/A"

    # Build panel sections
    connectivity_section = []
    if is_lane(road_type):
        connectivity_section = [
            html.Hr(className="my-2"),
            html.Strong("Connectivity:", className="small d-block mb-1"),
            html.Div(
                [
                    html.Strong("Entry Lanes: ", className="small"),
                    html.Span(", ".join(map(str, entry_lanes)) if entry_lanes else "None", className="small"),
                ],
            ),
            html.Div(
                [
                    html.Strong("Exit Lanes: ", className="small"),
                    html.Span(", ".join(map(str, exit_lanes)) if exit_lanes else "None", className="small"),
                ],
            ),
            _info_row("Speed Limit", f"{speed_limit:.1f} m/s" if speed_limit > 0 else "N/A"),
        ]

    # Polyline table
    poly_rows = []
    for i, pt in enumerate(xyz[:30]):  # Limit to 30 points
        poly_rows.append(
            html.Tr(
                [
                    html.Td(i, className="small"),
                    html.Td(f"{pt[0]:.2f}", className="small"),
                    html.Td(f"{pt[1]:.2f}", className="small"),
                    html.Td(f"{pt[2]:.2f}", className="small"),
                ],
            ),
        )

    return dbc.Card(
        [
            dbc.CardHeader(f"Road Element {road_id}"),
            dbc.CardBody(
                [
                    _info_row("Type", get_road_type_name(road_type)),
                    _info_row("Point Count", len(xyz)),
                    _info_row("Bounding Box", bbox),
                    *connectivity_section,
                    html.Hr(className="my-2"),
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                [
                                    html.Div(
                                        [
                                            dbc.Table(
                                                [
                                                    html.Thead(
                                                        html.Tr(
                                                            [
                                                                html.Th("#", className="small"),
                                                                html.Th("x", className="small"),
                                                                html.Th("y", className="small"),
                                                                html.Th("z", className="small"),
                                                            ],
                                                        ),
                                                    ),
                                                    html.Tbody(poly_rows),
                                                ],
                                                size="sm",
                                                striped=True,
                                                bordered=True,
                                            ),
                                            html.Small(f"Showing {min(30, len(xyz))}/{len(xyz)} points")
                                            if len(xyz) > 30
                                            else None,
                                        ],
                                        style={"maxHeight": "200px", "overflow": "auto"},
                                    ),
                                ],
                                title="Polyline Coordinates",
                                className="small",
                            ),
                        ],
                        start_collapsed=True,
                    ),
                ],
            ),
        ],
    )


def _traffic_light_info(scenario, tl_id, timestep):
    """Create detailed traffic light info panel."""
    traffic_elements = scenario.get("traffic_control_elements", [])

    elem = None
    for t in traffic_elements:
        if t["id"] == tl_id:
            elem = t
            break

    if not elem:
        return html.Div(f"Traffic light {tl_id} not found")

    xyz = elem.get("xyz", np.array([]))
    states = elem.get("states", [])
    controlled_lanes = elem.get("controlled_lanes", [])

    x, y, z = xyz[0], xyz[1], xyz[2] if len(xyz) >= 3 else (0, 0, 0)
    current_state = states[timestep] if timestep < len(states) else 0
    current_color = get_traffic_state_color(current_state)

    # Find state changes
    changes = []
    prev_state = None
    for t, s in enumerate(states):
        if s != prev_state:
            changes.append((t, s))
            prev_state = s

    # State timeline table
    state_rows = []
    for t, s in enumerate(states[:91]):  # Limit
        state_rows.append(
            html.Tr(
                [
                    html.Td(t, className="small"),
                    html.Td(get_traffic_state_name(s), className="small"),
                    html.Td(
                        html.Div(
                            style={
                                "width": "20px",
                                "height": "20px",
                                "backgroundColor": get_traffic_state_color(s),
                                "borderRadius": "50%",
                                "border": "1px solid #333",
                            },
                        ),
                    ),
                ],
                style={"backgroundColor": "#ffe0e0" if t == timestep else "inherit"},
            ),
        )

    return dbc.Card(
        [
            dbc.CardHeader(f"Traffic Light {tl_id}"),
            dbc.CardBody(
                [
                    _info_row("Position", f"({x:.2f}, {y:.2f}, {z:.2f})"),
                    html.Hr(className="my-2"),
                    html.Div(
                        [
                            html.Strong("Current State: ", className="small"),
                            html.Span(get_traffic_state_name(current_state), className="small me-2"),
                            html.Div(
                                style={
                                    "width": "20px",
                                    "height": "20px",
                                    "backgroundColor": current_color,
                                    "borderRadius": "50%",
                                    "border": "1px solid #333",
                                    "display": "inline-block",
                                    "verticalAlign": "middle",
                                },
                            ),
                        ],
                    ),
                    html.Hr(className="my-2"),
                    html.Div(
                        [
                            html.Strong("Controlled Lanes: ", className="small d-block"),
                            html.Span(
                                ", ".join(map(str, controlled_lanes)) if controlled_lanes else "None",
                                className="small",
                            ),
                        ],
                    ),
                    html.Hr(className="my-2"),
                    html.Div(
                        [
                            html.Strong("State Changes:", className="small d-block"),
                            html.Span(
                                " → ".join([f"t{t}:{get_traffic_state_name(s)}" for t, s in changes[:10]]),
                                className="small",
                            ),
                        ],
                    ),
                    html.Hr(className="my-2"),
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                [
                                    html.Div(
                                        [
                                            dbc.Table(
                                                [
                                                    html.Thead(
                                                        html.Tr(
                                                            [
                                                                html.Th("t", className="small"),
                                                                html.Th("State", className="small"),
                                                                html.Th("", className="small"),
                                                            ],
                                                        ),
                                                    ),
                                                    html.Tbody(state_rows),
                                                ],
                                                size="sm",
                                                striped=True,
                                                bordered=True,
                                            ),
                                        ],
                                        style={"maxHeight": "200px", "overflow": "auto"},
                                    ),
                                ],
                                title="State Timeline",
                                className="small",
                            ),
                        ],
                        start_collapsed=True,
                    ),
                ],
            ),
        ],
    )


def _info_row(label, value):
    """Create a label-value row."""
    return html.Div(
        [
            html.Strong(f"{label}: ", className="small"),
            html.Span(str(value), className="small"),
        ],
        className="mb-1",
    )
