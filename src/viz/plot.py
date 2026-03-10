"""Bird's Eye View (BEV) video renderer for Puffer format scenarios."""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, FancyBboxPatch, RegularPolygon
from matplotlib.transforms import Affine2D

from bin_factory.convert.types import TL_STATE_COLORS, is_road_edge, is_road_lane, is_road_line
from viz.binary_loader import load_puffer_binary
from viz.web.utils import build_lane_map, compute_route_polyline, get_agent_color


COLORS = {
    "ego": "#FF0000",
    "vehicle": "#1F77B4",
    "pedestrian": "#2CA02C",
    "cyclist": "#FF7F0E",
    "road_line": "#808080",
    "road_edge": "#000000",
    "lane": "#D3D3D3",
    "lane_unknown": "#00BFFF",
    "road_line_unknown": "#FF00FF",
    "road_edge_unknown": "#00FFFF",
    "crosswalk": "#FFD700",
    "speed_bump": "#FF69B4",
    "stop_sign": "#FF0000",
    "traffic_light": "#00FF00",
}


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
    metadata = puffer_scenario.get("metadata", {})
    length = metadata.get("scenario_length", 0)

    print(f"Scenario has {length} timesteps")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal")

    scenario_id = puffer_scenario.get("scenario_id", "unknown")
    dataset_name = metadata.get("dataset_name", "unknown")

    output_path = Path(output_path)
    if output_path.parent != Path():
        output_path.parent.mkdir(parents=True, exist_ok=True)
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

            _render_road_map(ax, puffer_scenario, render_mode=road_render_mode)

            if show_routes:
                _render_routes(ax, puffer_scenario)

            _render_traffic_controls(ax, puffer_scenario, timestep)
            _render_agents(ax, puffer_scenario, timestep, show_future)

            current_zoom_center = zoom_center
            if follow_ego:
                ego_pos = _get_ego_position(puffer_scenario, timestep)
                if ego_pos is not None:
                    current_zoom_center = ego_pos
                    current_radius = 50.0 if zoom_radius is None else zoom_radius
                else:
                    current_radius = zoom_radius
            else:
                current_radius = zoom_radius

            if current_zoom_center is not None and current_radius is not None:
                x_c, y_c = current_zoom_center
                ax.set_xlim(x_c - current_radius, x_c + current_radius)
                ax.set_ylim(y_c - current_radius, y_c + current_radius)
            else:
                ax.autoscale()
                ax.margins(0.1)

            writer.grab_frame()

            if (timestep + 1) % max(1, length // 10) == 0:
                print(f"  Progress: {timestep + 1}/{length} frames ({(timestep + 1) / length * 100:.0f}%)")

    plt.close(fig)
    print(f"✓ Saved video to {output_path}")


# --- render helpers ---


def _render_road_map(ax, puffer_scenario, render_mode="plot", show_headings=False):
    road_elements = sorted(
        puffer_scenario.get("road_map_elements", []),
        key=lambda e: e.get("id", 0),
    )

    for element in road_elements:
        element_type = element.get("type", 0)
        xyz = element.get("xyz", np.array([]))

        if len(xyz) == 0:
            print(f"Skipping road element {element.get('id', 'unknown')}: empty geometry")
            continue

        x, y = xyz[:, 0], xyz[:, 1]

        if is_road_lane(element_type):
            if element_type == 0:
                color, alpha, lw, zorder = COLORS["lane_unknown"], 0.95, 1.8, 3
            else:
                color, alpha, lw, zorder = COLORS["lane"], 0.7, 0.8, 1
        elif is_road_line(element_type):
            if element_type == 10:
                color, alpha, lw, zorder = COLORS["road_line_unknown"], 0.95, 2.0, 4
            else:
                color, alpha, lw, zorder = COLORS["road_line"], 0.7, 1.0, 2
        elif is_road_edge(element_type):
            if element_type == 20:
                color, alpha, lw, zorder = COLORS["road_edge_unknown"], 0.95, 2.8, 4
            else:
                color, alpha, lw, zorder = COLORS["road_edge"], 1.0, 1.5, 2
        elif element_type == 31:
            color, alpha, lw, zorder = COLORS["crosswalk"], 0.6, 2.0, 3
        elif element_type == 32:
            color, alpha, lw, zorder = COLORS["speed_bump"], 0.7, 2.5, 3
        elif element_type == 33:
            ax.scatter(x, y, color=COLORS["road_line"], s=15, marker="s", zorder=10)
            continue
        else:
            print(f"Skipping road element {element.get('id', 'unknown')}: unknown type {element_type}")
            continue

        if render_mode == "scatter":
            ax.scatter(x, y, color=color, alpha=alpha, s=lw * 2, zorder=zorder)
        else:
            ax.plot(x, y, color=color, alpha=alpha, linewidth=lw, zorder=zorder)

        if show_headings and len(xyz) > 1:
            dx = xyz[1:, 0] - xyz[:-1, 0]
            dy = xyz[1:, 1] - xyz[:-1, 1]
            headings = np.arctan2(dy, dx)
            scale = 1.0
            for xi, yi, hi in zip(x[:-1], y[:-1], headings, strict=False):
                ax.arrow(
                    xi, yi, scale * np.cos(hi), scale * np.sin(hi),
                    head_width=0.3, head_length=0.2,
                    fc=color, ec=color, alpha=alpha * 0.8, zorder=zorder + 1,
                )


def _render_routes(ax, puffer_scenario):
    agents = puffer_scenario.get("agents", [])
    road_elements = puffer_scenario.get("road_map_elements", [])
    ego_agent = _get_ego_agent(puffer_scenario)
    ego_id = ego_agent.get("id", -1) if ego_agent else -1

    lane_map = build_lane_map(road_elements)

    for agent in agents:
        agent_id = agent.get("id", -1)
        route = agent.get("route", [])
        if not route:
            continue

        start_pos = agent["states"]["xyz"][0]
        is_ego = agent_id == ego_id
        agent_color = get_agent_color(agent_id, is_ego)
        route_points = compute_route_polyline(route, lane_map, start_pos=start_pos)

        if route_points is not None:
            ax.text(
                route_points[0, 0], route_points[0, 1], f"{agent_id}",
                fontsize=6, color=agent_color, ha="center", va="bottom", zorder=6,
            )
            ax.plot(
                route_points[:, 0], route_points[:, 1],
                color=agent_color, linewidth=2.0, alpha=0.6, linestyle="--", zorder=5,
            )


def _render_traffic_controls(ax, puffer_scenario, timestep):
    for element in puffer_scenario.get("traffic_control_elements", []):
        xyz = element.get("xyz", np.array([]))
        if len(xyz) < 3:
            continue

        x, y = xyz[0], xyz[1]
        tc_type = element.get("type", 1)

        if tc_type == 1:
            states = element.get("states", np.array([]))
            state = 0 if len(states) == 0 or timestep >= len(states) else int(states[timestep])
            color = TL_STATE_COLORS.get(state, "#808080")
            ax.add_patch(
                Circle((x, y), radius=0.6, alpha=0.9, facecolor=color, edgecolor="black", linewidth=0.5, zorder=15),
            )
        elif tc_type == 2:
            ax.add_patch(
                FancyBboxPatch(
                    (x - 0.5, y - 0.5), 1.0, 1.0,
                    alpha=0.9, facecolor="#DC2626", edgecolor="black", linewidth=0.5, zorder=15,
                ),
            )
        elif tc_type == 3:
            ax.add_patch(
                RegularPolygon(
                    (x, y), numVertices=3, radius=0.7,
                    alpha=0.9, facecolor="#EAB308", edgecolor="black", linewidth=0.5, zorder=15,
                ),
            )


def _render_agents(ax, puffer_scenario, timestep, show_future):
    agents = puffer_scenario.get("agents", [])
    metadata = puffer_scenario.get("metadata", {})

    idx_agents_to_predict = list(metadata.get("tracks_to_predict", []))

    ego_agent = _get_ego_agent(puffer_scenario)
    ego_id = ego_agent.get("id", -1) if ego_agent else -1

    for agent in agents:
        agent_id = agent.get("id", -1)
        states = agent.get("states", {})

        xyz = states.get("xyz", np.array([]))
        heading = states.get("heading", np.array([]))
        valid = states.get("valid", np.array([]))
        length = states.get("length", np.array([]))
        width = states.get("width", np.array([]))

        if len(xyz) == 0 or timestep >= len(xyz):
            continue
        if not valid[timestep]:
            continue

        x, y = xyz[timestep, 0], xyz[timestep, 1]
        h = heading[timestep]
        l = length[timestep] if timestep < len(length) else 4.5
        w = width[timestep] if timestep < len(width) else 2.0

        is_ego = agent_id == ego_id
        color = get_agent_color(agent_id, is_ego)
        edge_color = "red" if agent_id in idx_agents_to_predict else "black"

        rect = mpatches.Rectangle(
            (-l / 2, -w / 2), l, w,
            facecolor=color, edgecolor=edge_color, linewidth=1.0, alpha=0.8, zorder=10,
        )
        t = Affine2D().rotate(h).translate(x, y) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

        arrow_length = l * 0.6
        dx = arrow_length * np.cos(h)
        dy = arrow_length * np.sin(h)
        ax.arrow(
            x, y, dx, dy,
            head_width=w * 0.5, head_length=w * 0.3,
            fc=color, ec="black", linewidth=0.5, alpha=0.9, zorder=11,
        )

        ax.text(
            x, y + w, f"{agent_id}",
            fontsize=6, color="black", ha="center", va="bottom", zorder=12,
        )

        if show_future:
            valid_traj_indices = np.where(valid)[0]
            if len(valid_traj_indices) > 1:
                traj_xyz = xyz[valid_traj_indices]
                ax.plot(
                    traj_xyz[:, 0], traj_xyz[:, 1],
                    color=color, linewidth=1.5, alpha=0.6, linestyle="-", zorder=9,
                )


def _get_ego_position(puffer_scenario, timestep):
    ego_agent = _get_ego_agent(puffer_scenario)
    if ego_agent is not None:
        states = ego_agent.get("states", {})
        xyz = np.array(states.get("xyz", []))
        valid = np.array(states.get("valid", []))
        if timestep < len(xyz) and valid[timestep]:
            return float(xyz[timestep][0]), float(xyz[timestep][1])
    return None


def _get_ego_agent(puffer_scenario):
    metadata = puffer_scenario.get("metadata", {})
    sdc_index = metadata.get("sdc_index", -1)
    agents = puffer_scenario.get("agents", [])
    if not isinstance(sdc_index, int) or not (0 <= sdc_index < len(agents)):
        return None
    return agents[sdc_index]


# --- CLI ---


def _render_one(bin_path, output_path, args):
    scenario = load_puffer_binary(bin_path)

    zoom_center = None
    if args.zoom_x is not None and args.zoom_y is not None:
        zoom_center = (args.zoom_x, args.zoom_y)
    elif args.zoom_x is not None or args.zoom_y is not None:
        print("Error: --zoom-x and --zoom-y must be specified together", file=sys.stderr)
        return 1

    render_scenario_video(
        puffer_scenario=scenario,
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
    return 0


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
    input_path = Path(args.input)

    if input_path.is_file():
        return _render_one(input_path, args.output, args)

    if input_path.is_dir():
        bin_files = sorted(input_path.glob("*.bin"))
        if args.max_scenarios:
            bin_files = bin_files[: args.max_scenarios]

        if not bin_files:
            print(f"No .bin files found in {input_path}", file=sys.stderr)
            return 1

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, f in enumerate(bin_files, 1):
            print(f"[{i}/{len(bin_files)}] {f.stem}")
            ret = _render_one(f, output_dir / f"{f.stem}.mp4", args)
            if ret != 0:
                return ret

        return 0

    print(f"Error: {args.input} is not a file or directory", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
