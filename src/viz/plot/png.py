import argparse
import sys

from viz.binary_loader import load_puffer_binary
from viz.plot.renderer import render_scenario_png


def main():
    parser = argparse.ArgumentParser(description="Visualize Puffer scenarios as PNG images")

    parser.add_argument("input", help="Input Puffer `.bin` file path")
    parser.add_argument("output", help="Output PNG file path")
    parser.add_argument("--timestep", type=int, default=0, help="Which timestep to visualize (default: 0)")
    parser.add_argument("--no-routes", action="store_true", help="Don't show agent routes")
    parser.add_argument("--no-future", action="store_true", help="Don't show trajectory history")
    parser.add_argument("--figsize", type=int, default=24, help="Figure size in inches (default: 24)")
    parser.add_argument("--dpi", type=int, default=300, help="Image resolution DPI (default: 300)")
    parser.add_argument("--zoom-x", type=float, help="X coordinate of zoom center (meters)")
    parser.add_argument("--zoom-y", type=float, help="Y coordinate of zoom center (meters)")
    parser.add_argument("--zoom-radius", type=float, help="Zoom radius in meters")
    parser.add_argument("--follow-ego", action="store_true", help="Center view on ego vehicle")

    args = parser.parse_args()

    scenario = load_puffer_binary(args.input)

    zoom_center = None
    if args.zoom_x is not None and args.zoom_y is not None:
        zoom_center = (args.zoom_x, args.zoom_y)
    elif args.zoom_x is not None or args.zoom_y is not None:
        print("Error: --zoom-x and --zoom-y must be specified together", file=sys.stderr)
        return 1

    render_scenario_png(
        puffer_scenario=scenario,
        output_path=args.output,
        timestep=args.timestep,
        show_routes=not args.no_routes,
        show_future=not args.no_future,
        figsize=(args.figsize, args.figsize),
        dpi=args.dpi,
        zoom_center=zoom_center,
        zoom_radius=args.zoom_radius,
        follow_ego=args.follow_ego,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
