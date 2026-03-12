"""Bird's Eye View (BEV) video renderer for Puffer format scenarios."""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.transforms import Affine2D
from tqdm import tqdm

from bin_factory.convert.types import MiscRoadType, TCType, is_road_edge, is_road_lane, is_road_line
from viz.binary_loader import load_puffer_binary
from viz.utils import (
    ROAD_COLORS,
    build_lane_map,
    compute_route_polyline,
    get_heading_arrow,
    get_road_styling,
    get_traffic_state_color,
)


# (alpha, zorder) for specific road types — unknown/special subtypes highlighted
_ROAD_ALPHA_ZORDER = {
    0: (0.95, 3),   # LaneType.UNKNOWN
    10: (0.95, 4),  # RoadLineType.UNKNOWN
    20: (0.95, 4),  # RoadEdgeType.UNKNOWN
    MiscRoadType.CROSSWALK: (0.6, 3),
    MiscRoadType.SPEED_BUMP: (0.7, 3),
}


def _road_alpha_zorder_fallback(element_type):
    if is_road_lane(element_type):
        return (0.7, 1)
    if is_road_line(element_type):
        return (0.7, 2)
    if is_road_edge(element_type):
        return (1.0, 2)
    return (None, None)


AGENT_TYPE_COLORS = {
    1: "#388bfd",
    2: "#3fb950",
    3: "#d29922",
}
EGO_COLOR = "#dc2626"
ROUTELESS_COLOR = "#94a3b8"


def _get_agent_display_color(agent, ego_id):
    if agent.get("id", -1) == ego_id:
        return EGO_COLOR
    if not agent.get("route", []):
        return ROUTELESS_COLOR
    return AGENT_TYPE_COLORS.get(agent.get("type", 4), ROUTELESS_COLOR)


def _build_scene(puffer_scenario, show_routes):
    metadata = puffer_scenario.get("metadata", {})
    agents = puffer_scenario.get("agents", [])
    road_elements = sorted(puffer_scenario.get("road_map_elements", []), key=lambda e: e.get("id", 0))
    sdc_index = metadata.get("sdc_index", -1)
    ego_id = agents[sdc_index].get("id", -1) if isinstance(sdc_index, int) and 0 <= sdc_index < len(agents) else -1
    predict_ids = set(metadata.get("tracks_to_predict", []))

    road_items = []
    for element in road_elements:
        xyz = np.asarray(element.get("xyz", np.array([])))
        if len(xyz) == 0:
            print(f"Skipping road element {element.get('id', 'unknown')}: empty geometry")
            continue

        element_type = element.get("type", 0)
        if element_type == MiscRoadType.DRIVEWAY:
            road_items.append(
                {
                    "kind": "scatter",
                    "xy": xyz[:, :2],
                    "color": ROAD_COLORS["road_line_white"],
                    "alpha": 1.0,
                    "size": 15,
                    "marker": "s",
                    "zorder": 10,
                },
            )
            continue

        color, dash, linewidth = get_road_styling(element_type)
        alpha, zorder = _ROAD_ALPHA_ZORDER.get(element_type, _road_alpha_zorder_fallback(element_type))
        if alpha is None:
            print(f"Skipping road element {element.get('id', 'unknown')}: unknown type {element_type}")
            continue

        linestyle = "-" if dash is None else {"dot": ":", "dash": "--"}.get(dash, "-")
        road_items.append(
            {
                "kind": "plot",
                "xy": xyz[:, :2],
                "color": color,
                "alpha": alpha,
                "linewidth": linewidth,
                "linestyle": linestyle,
                "zorder": zorder,
            },
        )

    lane_map = build_lane_map(road_elements)
    route_items = []
    if show_routes:
        for agent in agents:
            states = agent.get("states", {})
            xyz = np.asarray(states.get("xyz", np.array([])))
            route = agent.get("route", [])
            if len(xyz) == 0 or not route:
                continue

            route_points = compute_route_polyline(route, lane_map, start_pos=xyz[0])
            if route_points is None:
                continue

            route_items.append(
                {
                    "id": agent.get("id", -1),
                    "color": _get_agent_display_color(agent, ego_id),
                    "xy": route_points[:, :2],
                },
            )

    agent_items = []
    ego_xyz = np.array([])
    ego_valid = np.array([])
    for agent in agents:
        states = agent.get("states", {})
        xyz = np.asarray(states.get("xyz", np.array([])))
        valid = np.asarray(states.get("valid", np.array([])), dtype=bool)
        heading = np.asarray(states.get("heading", np.array([])))
        length = np.asarray(states.get("length", np.array([])))
        width = np.asarray(states.get("width", np.array([])))
        valid_future = xyz[valid] if len(xyz) == len(valid) and len(valid) else np.array([])
        agent_id = agent.get("id", -1)

        agent_items.append(
            {
                "id": agent_id,
                "xyz": xyz,
                "valid": valid,
                "heading": heading,
                "length": length,
                "width": width,
                "color": _get_agent_display_color(agent, ego_id),
                "edgecolor": "red" if agent_id in predict_ids else "black",
                "future_xyz": valid_future,
            },
        )

        if agent_id == ego_id:
            ego_xyz, ego_valid = xyz, valid

    traffic_controls = [
        {
            "stop_line": np.asarray(element["stop_line"]),
            "type": element.get("type", TCType.TRAFFIC_LIGHT),
            "states": np.asarray(element.get("states", np.array([]))),
        }
        for element in puffer_scenario.get("traffic_control_elements", [])
        if "stop_line" in element
    ]

    return {
        "length": metadata.get("scenario_length", 0),
        "road_items": road_items,
        "route_items": route_items,
        "agent_items": agent_items,
        "traffic_controls": traffic_controls,
        "ego_xyz": ego_xyz,
        "ego_valid": ego_valid,
        "title": f"Puffer BEV - {metadata.get('dataset_name', 'unknown')} | Scenario: {puffer_scenario.get('scenario_id', 'unknown')}",
        "video_title": f"Puffer Scenario {puffer_scenario.get('scenario_id', 'unknown')}",
    }


def _render_frame(ax, scene, timestep, show_future, zoom_center, zoom_radius, follow_ego, road_render_mode):
    ax.clear()
    ax.set_aspect("equal")
    ax.set_xlabel("X (meters)", fontsize=10)
    ax.set_ylabel("Y (meters)", fontsize=10)
    ax.set_title(f"{scene['title']}\nTimestep: {timestep}/{scene['length'] - 1}", fontsize=12, fontweight="bold")

    for road in scene["road_items"]:
        x, y = road["xy"][:, 0], road["xy"][:, 1]
        if road["kind"] == "scatter" or road_render_mode == "scatter":
            size = road["size"] if "size" in road else road["linewidth"] * 2
            ax.scatter(
                x,
                y,
                color=road["color"],
                alpha=road["alpha"],
                s=size,
                marker=road.get("marker", "o"),
                zorder=road["zorder"],
            )
            continue

        ax.plot(
            x,
            y,
            color=road["color"],
            alpha=road["alpha"],
            linewidth=road["linewidth"],
            linestyle=road["linestyle"],
            zorder=road["zorder"],
        )

    for route in scene["route_items"]:
        x, y = route["xy"][:, 0], route["xy"][:, 1]
        ax.text(x[0], y[0], f"{route['id']}", fontsize=6, color=route["color"], ha="center", va="bottom", zorder=6)
        ax.plot(x, y, color=route["color"], linewidth=2.0, alpha=0.6, linestyle="--", zorder=5)

    for control in scene["traffic_controls"]:
        sl = control["stop_line"]
        if control["type"] == TCType.TRAFFIC_LIGHT:
            state = 0 if timestep >= len(control["states"]) else int(control["states"][timestep])
            ax.plot(
                [sl[0, 0], sl[1, 0]], [sl[0, 1], sl[1, 1]],
                color=get_traffic_state_color(state), linewidth=2.5, solid_capstyle="round",
                alpha=0.9, zorder=15,
            )
        elif control["type"] == TCType.STOP_SIGN:
            ax.plot(
                [sl[0, 0], sl[1, 0]], [sl[0, 1], sl[1, 1]],
                color="#DC2626", linewidth=2.5, solid_capstyle="round",
                alpha=0.9, zorder=15,
            )
        elif control["type"] == TCType.YIELD_SIGN:
            ax.plot(
                [sl[0, 0], sl[1, 0]], [sl[0, 1], sl[1, 1]],
                color="#EAB308", linewidth=2.5, solid_capstyle="round",
                alpha=0.9, zorder=15,
            )

    for agent in scene["agent_items"]:
        if timestep >= len(agent["xyz"]) or timestep >= len(agent["valid"]) or not agent["valid"][timestep]:
            continue

        x, y = agent["xyz"][timestep, :2]
        heading = agent["heading"][timestep] if timestep < len(agent["heading"]) else 0.0
        length = agent["length"][timestep] if timestep < len(agent["length"]) else 4.5
        width = agent["width"][timestep] if timestep < len(agent["width"]) else 2.0

        if show_future and len(agent["future_xyz"]) > 1:
            ax.plot(
                agent["future_xyz"][:, 0],
                agent["future_xyz"][:, 1],
                color=agent["color"],
                linewidth=1.5,
                alpha=0.6,
                linestyle="-",
                zorder=9,
            )

        rect = mpatches.Rectangle(
            (-length / 2, -width / 2),
            length,
            width,
            facecolor=agent["color"],
            edgecolor=agent["edgecolor"],
            linewidth=1.0,
            alpha=0.8,
            zorder=10,
        )
        rect.set_transform(Affine2D().rotate(heading).translate(x, y) + ax.transData)
        ax.add_patch(rect)

        arrow_x, arrow_y = get_heading_arrow(x, y, heading, length)
        ax.arrow(
            x,
            y,
            arrow_x - x,
            arrow_y - y,
            head_width=width * 0.5,
            head_length=width * 0.3,
            fc=agent["color"],
            ec="black",
            linewidth=0.5,
            alpha=0.9,
            zorder=11,
        )
        ax.text(x, y + width, f"{agent['id']}", fontsize=6, color="black", ha="center", va="bottom", zorder=12)

    center, radius = zoom_center, zoom_radius
    if (
        follow_ego
        and timestep < len(scene["ego_xyz"])
        and timestep < len(scene["ego_valid"])
        and scene["ego_valid"][timestep]
    ):
        center = tuple(map(float, scene["ego_xyz"][timestep, :2]))
        radius = 50.0 if zoom_radius is None else zoom_radius

    if center is not None and radius is not None:
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        return

    ax.autoscale()
    ax.margins(0.1)


def render_scenario_video(
    puffer_scenario,
    output_path,
    fps=10,
    show_routes=True,
    show_future=True,
    figsize=(20, 20),
    dpi=150,
    zoom_center=None,
    zoom_radius=None,
    follow_ego=True,
    road_render_mode="plot",
):
    scene = _build_scene(puffer_scenario, show_routes)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, metadata={"title": scene["video_title"]})

    with writer.saving(fig, output_path, dpi=dpi):
        for timestep in tqdm(
            range(scene["length"]), total=scene["length"], desc="Frames", leave=False, dynamic_ncols=True
        ):
            _render_frame(ax, scene, timestep, show_future, zoom_center, zoom_radius, follow_ego, road_render_mode)
            writer.grab_frame()

    plt.close(fig)
    tqdm.write(f"Saved video to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Render Puffer scenarios as MP4 videos")
    parser.add_argument("input", help="Input .bin file or directory of .bin files")
    parser.add_argument("output", help="Output .mp4 path (single) or output directory (batch)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    parser.add_argument("--no-routes", action="store_true", help="Don't show agent routes")
    parser.add_argument("--no-future", action="store_true", help="Don't show trajectory history")
    parser.add_argument("--figsize", type=int, default=20, help="Figure size in inches (default: 20)")
    parser.add_argument("--dpi", type=int, default=150, help="Video resolution DPI (default: 150)")
    parser.add_argument("--zoom-x", type=float, help="X coordinate of zoom center (meters)")
    parser.add_argument("--zoom-y", type=float, help="Y coordinate of zoom center (meters)")
    parser.add_argument("--zoom-radius", type=float, help="Zoom radius in meters")
    parser.add_argument("--follow-ego", action="store_true", help="Center view on ego vehicle (dynamic)")
    parser.add_argument("--max-scenarios", type=int, help="Limit files in batch/directory mode")
    args = parser.parse_args()

    if (args.zoom_x is None) != (args.zoom_y is None):
        print("Error: --zoom-x and --zoom-y must be specified together", file=sys.stderr)
        return 1

    zoom_center = (args.zoom_x, args.zoom_y) if args.zoom_x is not None else None

    def render_file(bin_path, output_path):
        render_scenario_video(
            puffer_scenario=load_puffer_binary(bin_path),
            output_path=output_path,
            fps=args.fps,
            show_routes=not args.no_routes,
            show_future=not args.no_future,
            figsize=(args.figsize, args.figsize),
            dpi=args.dpi,
            zoom_center=zoom_center,
            zoom_radius=args.zoom_radius,
            follow_ego=args.follow_ego,
        )

    input_path = Path(args.input)
    if input_path.is_file():
        jobs = [(input_path, Path(args.output))]
    elif input_path.is_dir():
        bin_files = sorted(input_path.glob("*.bin"))
        if args.max_scenarios:
            bin_files = bin_files[: args.max_scenarios]
        if not bin_files:
            print(f"No .bin files found in {input_path}", file=sys.stderr)
            return 1

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(bin_file, output_dir / f"{bin_file.stem}.mp4") for bin_file in bin_files]
    else:
        print(f"Error: {args.input} is not a file or directory", file=sys.stderr)
        return 1

    for bin_path, output_path in tqdm(jobs, total=len(jobs), desc="Scenarios", dynamic_ncols=True):
        render_file(bin_path, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
