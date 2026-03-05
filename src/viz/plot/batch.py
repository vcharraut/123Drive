import argparse
import os
import sys
from pathlib import Path

from src.viz.binary_loader import load_scenario
from src.viz.plot.renderer import render_scenario_png, render_scenario_video


def main():
    parser = argparse.ArgumentParser(description="Batch visualize Puffer scenarios from a directory")

    parser.add_argument("input_dir", help="Input directory with Puffer files (.json or .bin)")
    parser.add_argument("output_dir", help="Output directory for visualizations")
    parser.add_argument(
        "--format",
        choices=["png", "video", "both"],
        default="png",
        help="Output format (default: png)",
    )
    parser.add_argument("--timestep", type=int, default=0, help="Timestep for PNG export (default: 0)")
    parser.add_argument("--fps", type=int, default=10, help="FPS for video export (default: 10)")
    parser.add_argument("--max-scenarios", type=int, help="Maximum number of scenarios to process")
    parser.add_argument("--no-routes", action="store_true", help="Don't show agent routes")
    parser.add_argument("--no-future", action="store_true", help="Don't show trajectory history")
    parser.add_argument("--png-dpi", type=int, default=150, help="PNG DPI (default: 150)")
    parser.add_argument("--video-dpi", type=int, default=100, help="Video DPI (default: 100)")

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    all_files = sorted(input_path.glob("*.bin")) + sorted(input_path.glob("*.json"))
    if args.max_scenarios:
        all_files = all_files[: args.max_scenarios]

    if not all_files:
        print(f"No Puffer files found in {args.input_dir}", file=sys.stderr)
        return 1

    png_dir = os.path.join(args.output_dir, "png") if args.format in ["png", "both"] else None
    video_dir = os.path.join(args.output_dir, "videos") if args.format in ["video", "both"] else None

    os.makedirs(args.output_dir, exist_ok=True)
    if png_dir:
        os.makedirs(png_dir, exist_ok=True)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    errors = 0
    for i, f in enumerate(all_files, 1):
        name = f.stem
        print(f"[{i}/{len(all_files)}] {name}")
        scenario = load_scenario(f)
        length = scenario.get("metadata", {}).get("scenario_length", 0)
        timestep = min(args.timestep, max(0, length - 1))

        if png_dir:
            render_scenario_png(
                puffer_scenario=scenario,
                output_path=os.path.join(png_dir, f"{name}.png"),
                timestep=timestep,
                show_routes=not args.no_routes,
                show_future=not args.no_future,
                figsize=(20, 20),
                dpi=args.png_dpi,
            )
        if video_dir:
            render_scenario_video(
                puffer_scenario=scenario,
                output_path=os.path.join(video_dir, f"{name}.mp4"),
                fps=args.fps,
                show_routes=not args.no_routes,
                show_future=not args.no_future,
                figsize=(16, 16),
                dpi=args.video_dpi,
            )

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
