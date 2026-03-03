"""
Bird's Eye View (BEV) renderer for Puffer format scenarios.

This module provides visualization functions for Puffer format data
to help debug and understand scenario structure.
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter


# Color scheme for visualization
COLORS = {
    "ego": "#FF0000",  # Red
    "vehicle": "#1F77B4",  # Blue
    "pedestrian": "#2CA02C",  # Green
    "cyclist": "#FF7F0E",  # Orange
    "road_line": "#808080",  # Grey
    "road_edge": "#000000",  # Black
    "lane": "#D3D3D3",  # Light gray
    "lane_unknown": "#00BFFF",  # Deep sky blue
    "road_line_unknown": "#FF00FF",  # Magenta
    "road_edge_unknown": "#00FFFF",  # Cyan
    "crosswalk": "#FFD700",  # Gold
    "speed_bump": "#FF69B4",  # Pink
    "stop_sign": "#FF0000",  # Red
    "traffic_light": "#00FF00",  # Green (default)
}

# Color palette for vehicles (distinct colors for each agent)
VEHICLE_COLORS = [
    "#FF0000",  # Red (ego)
    "#1F77B4",  # Blue
    "#FF7F0E",  # Orange
    "#2CA02C",  # Green
    "#9467BD",  # Purple
    "#8C564B",  # Brown
    "#E377C2",  # Pink
    "#BCBD22",  # Yellow-green
    "#17BECF",  # Cyan
    "#AEC7E8",  # Light blue
    "#FFBB78",  # Light orange
    "#98DF8A",  # Light green
    "#FF9896",  # Light red
    "#C5B0D5",  # Light purple
    "#C49C94",  # Light brown
    "#F7B6D2",  # Light pink
    "#DBDB8D",  # Light yellow-green
    "#9EDAE5",  # Light cyan
]


def get_agent_color(agent_id: int, is_ego: bool) -> str:
    """Get consistent color for an agent based on ID."""
    if is_ego:
        return VEHICLE_COLORS[0]  # Red for ego
    # Use modulo to cycle through colors if more agents than colors
    return VEHICLE_COLORS[(agent_id % (len(VEHICLE_COLORS) - 1)) + 1]


# Puffer agent type mapping
AGENT_TYPE_NAMES = {
    0: "unset",
    1: "vehicle",
    2: "pedestrian",
    3: "cyclist",
    4: "other",
}

# Traffic light state mapping
TRAFFIC_LIGHT_STATES = {
    0: "unknown",
    1: "arrow_stop",
    2: "arrow_caution",
    3: "arrow_go",
    4: "stop",
    5: "caution",
    6: "go",
    7: "flashing_stop",
    8: "flashing_caution",
}

TRAFFIC_LIGHT_COLORS = {
    0: "#808080",  # Unknown - gray
    1: "#FF0000",  # Arrow red
    2: "#FFFF00",  # Arrow yellow
    3: "#00FF00",  # Arrow green
    4: "#FF0000",  # Red
    5: "#FFFF00",  # Yellow
    6: "#00FF00",  # Green
    7: "#FF6600",  # Flashing red
    8: "#FFFF00",  # Flashing yellow
}


def render_scenario_png(
    puffer_scenario: dict,
    output_path: str,
    timestep: int = 0,
    show_routes: bool = True,
    show_future: bool = True,
    figsize: tuple = (24, 24),
    dpi: int = 150,
    zoom_center: tuple | None = None,
    zoom_radius: float | None = None,
    follow_ego: bool = False,
    road_render_mode: str = "plot",
    show_road_headings: bool = True,
) -> None:
    """
    Render a Puffer scenario as a PNG image at a specific timestep.

    Args:
        puffer_scenario: Puffer format scenario dict
        output_path: Output PNG file path
        timestep: Which timestep to visualize (default: 0)
        show_routes: Show agent routes if available (default: True)
        show_future: Show trajecot
        figsize: Figure size in inches (default: (24, 24))
        dpi: Image resolution (default: 300)
        zoom_center: (x, y) center point for zoom, or None for full view
        zoom_radius: Radius in meters for zoom view, or None for full view
        follow_ego: If True, center view on ego vehicle (overrides zoom_center)
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal")
    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)

    scenario_id = puffer_scenario.get("scenario_id", "unknown")
    metadata = puffer_scenario.get("metadata", {})
    dataset_name = metadata.get("dataset_name", "unknown")

    ax.set_title(
        f"Puffer BEV Visualization - {dataset_name}\nScenario: {scenario_id}\nTimestep: {timestep}",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Render road map elements
    _render_road_map(ax, puffer_scenario, render_mode=road_render_mode, show_headings=show_road_headings)

    # 2. Render routes if requested
    if show_routes:
        _render_routes(ax, puffer_scenario)

    # 3. Render traffic lights
    _render_traffic_lights(ax, puffer_scenario, timestep)

    # 4. Render dynamic agents
    _render_agents(ax, puffer_scenario, timestep, show_future)

    # 5. Add legend
    _add_legend(ax, puffer_scenario)

    # 6. Set view bounds (zoom or auto-scale)
    if follow_ego:
        # Center on ego vehicle
        ego_pos = _get_ego_position(puffer_scenario, timestep)
        if ego_pos is not None:
            zoom_center = ego_pos
            if zoom_radius is None:
                zoom_radius = 100.0  # Default zoom radius if following ego

    if zoom_center is not None and zoom_radius is not None:
        # Zoom to specified region
        x_c, y_c = zoom_center
        ax.set_xlim(x_c - zoom_radius, x_c + zoom_radius)
        ax.set_ylim(y_c - zoom_radius, y_c + zoom_radius)
    else:
        # Auto-scale to show full scene
        ax.autoscale()
        ax.margins(0.1)

    # Save figure
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if path has a directory component
        os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved PNG to {output_path}")


def render_scenario_video(
    puffer_scenario: dict,
    output_path: str,
    fps: int = 10,
    show_routes: bool = True,
    show_future: bool = True,
    figsize: tuple = (20, 20),
    dpi: int = 150,
    zoom_center: tuple | None = None,
    zoom_radius: float | None = 200,
    follow_ego: bool = True,
    road_render_mode: str = "plot",
    show_road_headings: bool = False,
) -> None:
    """
    Render a Puffer scenario as an MP4 video animation.

    Args:
        puffer_scenario: Puffer format scenario dict
        output_path: Output MP4 file path
        fps: Frames per second (default: 10)
        show_routes: Show agent routes if available (default: True)
        show_future: Show trajectory future (default: True)
        figsize: Figure size in inches (default: (20, 20))
        dpi: Video resolution (default: 150)
        zoom_center: (x, y) center point for zoom, or None for full view
        zoom_radius: Radius in meters for zoom view, or None for full view
        follow_ego: If True, center view on ego vehicle at each timestep
    """
    metadata = puffer_scenario.get("metadata", {})
    length = metadata.get("scenario_length", 0)

    print(f"Scenario has {length} timesteps")

    # Setup figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal")

    scenario_id = puffer_scenario.get("scenario_id", "unknown")
    dataset_name = metadata.get("dataset_name", "unknown")

    # Setup video writer
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    writer = FFMpegWriter(fps=fps, metadata={"title": f"Puffer Scenario {scenario_id}"})

    with writer.saving(fig, output_path, dpi=dpi):
        for timestep in range(length):
            ax.clear()
            ax.set_aspect("equal")
            ax.set_xlabel("X (meters)", fontsize=10)
            ax.set_ylabel("Y (meters)", fontsize=10)
            ax.set_title(
                f"Puffer BEV - {dataset_name} | Scenario: {scenario_id}\nTimestep: {timestep}/{length - 1}",
                fontsize=12,
                fontweight="bold",
            )

            # Render all elements for this frame
            _render_road_map(ax, puffer_scenario, render_mode=road_render_mode, show_headings=show_road_headings)

            if show_routes:
                _render_routes(ax, puffer_scenario)

            _render_traffic_lights(ax, puffer_scenario, timestep)
            _render_agents(ax, puffer_scenario, timestep, show_future)

            # Set view bounds (zoom or auto-scale)
            current_zoom_center = zoom_center
            if follow_ego:
                # Center on ego vehicle at this timestep
                ego_pos = _get_ego_position(puffer_scenario, timestep)
                if ego_pos is not None:
                    current_zoom_center = ego_pos
                    current_radius = 50.0 if zoom_radius is None else zoom_radius
                else:
                    current_radius = zoom_radius
            else:
                current_radius = zoom_radius

            if current_zoom_center is not None and current_radius is not None:
                # Zoom to specified region
                x_c, y_c = current_zoom_center
                ax.set_xlim(x_c - current_radius, x_c + current_radius)
                ax.set_ylim(y_c - current_radius, y_c + current_radius)
            else:
                # Auto-scale to show full scene
                ax.autoscale()
                ax.margins(0.1)

            # Add frame to video
            writer.grab_frame()

            # Progress indicator
            if (timestep + 1) % max(1, length // 10) == 0:
                print(f"  Progress: {timestep + 1}/{length} frames ({(timestep + 1) / length * 100:.0f}%)")

    plt.close(fig)
    print(f"✓ Saved video to {output_path}")


def _render_road_map(ax: plt.Axes, puffer_scenario: dict, render_mode="plot", show_headings=False) -> None:
    """Render road map elements (lanes, road lines, road edges, crosswalks)."""
    road_elements = puffer_scenario.get("road_map_elements", [])

    # Sort by road element by id
    road_elements = sorted(road_elements, key=lambda e: e.get("id", 0))

    for element in road_elements:
        element_type = element.get("type", 0)
        xyz = element.get("xyz", np.array([]))

        if len(xyz) == 0:
            print(f"Skipping road element {element.get('id', 'unknown')}: empty geometry")
            continue

        x, y = xyz[:, 0], xyz[:, 1]

        # Determine color/style based on element type
        if element_type == 0:  # Unknown lane
            color, alpha, lw, zorder = COLORS["lane_unknown"], 0.95, 1.8, 3
        elif 1 <= element_type <= 9:  # Lanes
            color, alpha, lw, zorder = COLORS["lane"], 0.7, 0.8, 1
        elif element_type == 10:  # Unknown road line
            color, alpha, lw, zorder = COLORS["road_line_unknown"], 0.95, 2.0, 4
        elif 11 <= element_type <= 19:  # Road lines
            color, alpha, lw, zorder = COLORS["road_line"], 0.7, 1.0, 2
        elif element_type == 20:  # Unknown road edge
            color, alpha, lw, zorder = COLORS["road_edge_unknown"], 0.95, 2.8, 4
        elif 21 <= element_type <= 29:  # Road edges
            color, alpha, lw, zorder = COLORS["road_edge"], 1.0, 1.5, 2
        elif element_type == 31:  # Crosswalk
            color, alpha, lw, zorder = COLORS["crosswalk"], 0.6, 2.0, 3
        elif element_type == 32:  # Speed bump
            color, alpha, lw, zorder = COLORS["speed_bump"], 0.7, 2.5, 3
        elif element_type == 33:  # Driveway
            ax.scatter(x, y, color=COLORS["road_line"], s=15, marker="s", zorder=10)
            continue
        else:
            print(f"Skipping road element {element.get('id', 'unknown')}: unknown type {element_type}")
            continue

        # Render polyline
        if render_mode == "scatter":
            ax.scatter(x, y, color=color, alpha=alpha, s=lw * 2, zorder=zorder)
        else:  # plot
            ax.plot(x, y, color=color, alpha=alpha, linewidth=lw, zorder=zorder)

        # Draw heading arrows
        if show_headings and len(xyz) > 1:
            dx = xyz[1:, 0] - xyz[:-1, 0]
            dy = xyz[1:, 1] - xyz[:-1, 1]
            headings = np.arctan2(dy, dx)
            scale = 1.0
            for xi, yi, hi in zip(x[:-1], y[:-1], headings):
                ax.arrow(
                    xi,
                    yi,
                    scale * np.cos(hi),
                    scale * np.sin(hi),
                    head_width=0.3,
                    head_length=0.2,
                    fc=color,
                    ec=color,
                    alpha=alpha * 0.8,
                    zorder=zorder + 1,
                )


def _render_routes(ax: plt.Axes, puffer_scenario: dict) -> None:
    """Render agent routes as dashed lines with agent-specific colors."""
    agents = puffer_scenario.get("agents", [])
    road_elements = puffer_scenario.get("road_map_elements", [])
    metadata = puffer_scenario.get("metadata", {})
    ego_id = metadata.get("ego_id", metadata.get("sdc_index", -1))

    # Build lane ID to polyline mapping
    lane_map = {}
    for element in road_elements:
        element_id = element.get("id")
        element_type = element.get("type", 0)
        if element_type in [1, 2, 3]:  # Only lanes
            lane_map[element_id] = element.get("xyz", np.array([]))

    # Render each agent's route with matching color
    for agent in agents:
        agent_id = agent.get("id", -1)
        routes = agent.get("routes", [])

        if not routes:
            continue

        initial_pos_x = agent["states"]["xyz"][0, 0]
        initial_pos_y = agent["states"]["xyz"][0, 1]

        for route in routes[:1]:
            # Get agent color
            is_ego = agent_id == ego_id
            agent_color = get_agent_color(agent_id, is_ego)

            # Routes is a list of lane IDs
            route_points = []
            for lane_id in route:
                if lane_id in lane_map:
                    lane_xyz = lane_map[lane_id]
                    if len(lane_xyz) > 0:
                        route_points.extend(lane_xyz)

            if len(route_points) > 1:
                route_points = np.array(route_points)

                # Crop route to start near agent's initial position
                dists = np.linalg.norm(route_points[:, :2] - np.array([initial_pos_x, initial_pos_y]), axis=1)
                start_idx = np.argmin(dists)
                route_points = route_points[start_idx:]

                ax.text(
                    route_points[0, 0],
                    route_points[0, 1],
                    f"{agent_id}",
                    fontsize=6,
                    color=agent_color,
                    ha="center",
                    va="bottom",
                    zorder=6,
                )

                ax.plot(
                    route_points[:, 0],
                    route_points[:, 1],
                    color=agent_color,  # Use agent's color
                    linewidth=2.0,
                    alpha=0.6,
                    linestyle="--",
                    zorder=5,
                )


def _render_traffic_lights(ax: plt.Axes, puffer_scenario: dict, timestep: int) -> None:
    """Render traffic lights with their states at the given timestep."""
    traffic_elements = puffer_scenario.get("traffic_control_elements", [])

    for element in traffic_elements:
        xyz = element.get("xyz", np.array([]))
        if len(xyz) < 3:
            continue

        x, y = xyz[0], xyz[1]

        # Get state at timestep
        states = element.get("states", np.array([]))
        state = 0 if len(states) == 0 or timestep >= len(states) else int(states[timestep])

        # Get color based on state
        color = TRAFFIC_LIGHT_COLORS.get(state, "#808080")

        # Draw traffic light as a circle
        ax.add_patch(
            plt.Circle(
                (x, y),
                radius=0.6,
                alpha=0.9,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
                zorder=15,
            ),
        )

        # Draw stop line if geometry is present
        stop_line = element.get("stop_line")
        if stop_line is not None:
            sl = np.asarray(stop_line)
            ax.plot(
                [sl[0, 0], sl[1, 0]],
                [sl[0, 1], sl[1, 1]],
                color="red",
                linewidth=1.5,
                zorder=16,
                solid_capstyle="round",
            )

            # Draw orientation arrow from the traffic light xyz position
            orientation = element.get("orientation")
            if orientation is not None:
                orient = np.asarray(orientation)
                arrow_len = 3.0
                ax.quiver(
                    x, y,
                    orient[0] * arrow_len, orient[1] * arrow_len,
                    angles="xy", scale_units="xy", scale=1,
                    color="red", width=0.008,
                    headwidth=4, headlength=5, headaxislength=4,
                    zorder=17,
                )


def _render_agents(ax: plt.Axes, puffer_scenario: dict, timestep: int, show_future: bool) -> None:
    """Render dynamic agents (vehicles, pedestrians, cyclists) at the given timestep."""
    agents = puffer_scenario.get("agents", [])
    metadata = puffer_scenario.get("metadata", {})

    idx_agents_to_predict = []
    if metadata.get("tracks_to_predict"):
        for t in metadata["tracks_to_predict"]:
            idx_agents_to_predict.append(t)

    ego_id = metadata.get("ego_id", -1)

    for agent in agents:
        agent_id = agent.get("id", -1)
        states = agent.get("states", {})

        # Get states
        xyz = states.get("xyz", np.array([]))
        heading = states.get("heading", np.array([]))
        valid = states.get("valid", np.array([]))
        length = states.get("length", np.array([]))
        width = states.get("width", np.array([]))

        if len(xyz) == 0 or timestep >= len(xyz):
            continue

        if not valid[timestep]:
            continue

        # Get current position and heading
        x, y = xyz[timestep, 0], xyz[timestep, 1]
        heading = heading[timestep]
        length = length[timestep] if timestep < len(length) else 4.5
        width = width[timestep] if timestep < len(width) else 2.0

        # Determine color based on agent ID (consistent with routes)
        is_ego = agent_id == ego_id
        color = get_agent_color(agent_id, is_ego)

        edge_color = "red" if agent_id in idx_agents_to_predict else "black"

        # Draw agent as oriented rectangle
        # Rectangle in local coordinates (centered at origin)
        rect = mpatches.Rectangle(
            (-length / 2, -width / 2),
            length,
            width,
            facecolor=color,
            edgecolor=edge_color,
            linewidth=1.0,
            alpha=0.8,
            zorder=10,
        )

        # Transform to world coordinates: rotate then translate
        t = plt.matplotlib.transforms.Affine2D().rotate(heading).translate(x, y) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

        # Draw heading indicator (arrow)
        arrow_length = length * 0.6
        dx = arrow_length * np.cos(heading)
        dy = arrow_length * np.sin(heading)
        ax.arrow(
            x,
            y,
            dx,
            dy,
            head_width=width * 0.5,
            head_length=width * 0.3,
            fc=color,
            ec="black",
            linewidth=0.5,
            alpha=0.9,
            zorder=11,
        )

        # Show agent ID
        ax.text(
            x,
            y + width,
            f"{agent_id}",
            fontsize=6,
            color="black",
            ha="center",
            va="bottom",
            zorder=12,
        )

        # Show trajectory history
        if show_future:
            valid_traj_indices = np.where(valid)[0]
            if len(valid_traj_indices) > 1:
                traj_xyz = xyz[valid_traj_indices]
                ax.plot(
                    traj_xyz[:, 0],
                    traj_xyz[:, 1],
                    color=color,
                    linewidth=1.5,
                    alpha=0.6,
                    linestyle="-",
                    zorder=9,
                )


def _get_ego_position(puffer_scenario: dict, timestep: int) -> tuple:
    """
    Get ego vehicle position at a specific timestep.

    Args:
        puffer_scenario: Puffer format scenario dict
        timestep: Timestep index

    Returns:
        (x, y) position tuple, or None if ego not found
    """
    metadata = puffer_scenario.get("metadata", {})
    ego_id = metadata.get("sdc_index")
    dynamic_agents = puffer_scenario.get("agents", [])

    for agent in dynamic_agents:
        agent_id = agent.get("id")

        if agent_id == ego_id:
            states = agent.get("states", {})
            xyz = np.array(states.get("xyz", []))
            valid = np.array(states.get("valid", []))

            if timestep < len(xyz) and valid[timestep]:
                return (xyz[timestep][0], xyz[timestep][1])

    return None


def _add_legend(ax: plt.Axes, puffer_scenario: dict) -> None:
    """Add a legend to the plot."""
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["ego"], edgecolor="black", label="Ego Vehicle"),
        mpatches.Patch(facecolor=COLORS["vehicle"], edgecolor="black", label="Vehicle"),
        mpatches.Patch(facecolor=COLORS["pedestrian"], edgecolor="black", label="Pedestrian"),
        mpatches.Patch(facecolor=COLORS["cyclist"], edgecolor="black", label="Cyclist"),
        mpatches.Patch(facecolor=COLORS["lane"], label="Lane"),
        mpatches.Patch(facecolor=COLORS["lane_unknown"], label="Lane (Unknown)"),
        mpatches.Patch(facecolor=COLORS["road_line"], label="Road Line"),
        mpatches.Patch(facecolor=COLORS["road_line_unknown"], label="Road Line (Unknown)"),
        mpatches.Patch(facecolor=COLORS["road_edge"], label="Road Edge"),
        mpatches.Patch(facecolor=COLORS["road_edge_unknown"], label="Road Edge (Unknown)"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )
