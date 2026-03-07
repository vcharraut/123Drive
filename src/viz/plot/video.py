import argparse
import sys

from viz.binary_loader import load_puffer_binary
from viz.plot.renderer import render_scenario_video


def main():
    parser = argparse.ArgumentParser(description="Visualize Puffer scenarios as MP4 videos")

    parser.add_argument("input", help="Input Puffer `.bin` file path")
    parser.add_argument("output", help="Output MP4 file path")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    parser.add_argument("--no-routes", action="store_true", help="Don't show agent routes")
    parser.add_argument("--no-future", action="store_true", help="Don't show trajectory history")
    parser.add_argument("--figsize", type=int, default=20, help="Figure size in inches (default: 20)")
    parser.add_argument("--dpi", type=int, default=150, help="Video resolution DPI (default: 150)")
    parser.add_argument("--zoom-x", type=float, help="X coordinate of zoom center (meters)")
    parser.add_argument("--zoom-y", type=float, help="Y coordinate of zoom center (meters)")
    parser.add_argument("--zoom-radius", type=float, help="Zoom radius in meters")
    parser.add_argument("--follow-ego", action="store_true", help="Center view on ego vehicle (dynamic)")

    args = parser.parse_args()

    scenario = load_puffer_binary(args.input)

    zoom_center = None
    if args.zoom_x is not None and args.zoom_y is not None:
        zoom_center = (args.zoom_x, args.zoom_y)
    elif args.zoom_x is not None or args.zoom_y is not None:
        print("Error: --zoom-x and --zoom-y must be specified together", file=sys.stderr)
        return 1

    render_scenario_video(
        puffer_scenario=scenario,
        output_path=args.output,
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


if __name__ == "__main__":
    sys.exit(main())
