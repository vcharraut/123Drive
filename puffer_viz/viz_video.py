#!/usr/bin/env python3
"""
CLI script to visualize Puffer scenarios as MP4 videos.

Usage:
    python viz_video.py <puffer_file_path> <output_mp4_path> [--fps 10] [--no-routes] [--no-future]

Example:
    python viz_video.py /path/to/scenario.bin output.mp4 --fps 10
    python viz_video.py /path/to/scenario.json output.mp4 --fps 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from puffer_renderer import render_scenario_video


def load_puffer_scenario(file_path: str) -> dict:
    """
    Load a Puffer scenario from JSON or binary file.

    Args:
        file_path: Path to .json or .bin file

    Returns:
        Puffer scenario dict
    """
    file_path = Path(file_path)

    if file_path.suffix == ".bin":
        # Load from binary
        from binary_loader import load_puffer_binary

        return load_puffer_binary(file_path)
    else:
        # Load from JSON (default)
        with open(file_path) as f:
            scenario = json.load(f)

        # Convert lists back to numpy arrays where needed
        if "agents" in scenario:
            for agent in scenario["agents"]:
                if "states" in agent:
                    states = agent["states"]
                    for key in ["xyz", "heading", "velocity", "length", "width", "height", "valid"]:
                        if key in states and isinstance(states[key], list):
                            states[key] = np.array(states[key])

                if "routes" in agent and isinstance(agent["routes"], list):
                    agent["routes"] = agent["routes"]

        if "road_map_elements" in scenario:
            for element in scenario["road_map_elements"]:
                if "xyz" in element and isinstance(element["xyz"], list):
                    element["xyz"] = np.array(element["xyz"])
                if "dir_xyz" in element and isinstance(element["dir_xyz"], list):
                    element["dir_xyz"] = np.array(element["dir_xyz"])

        if "traffic_control_elements" in scenario:
            for element in scenario["traffic_control_elements"]:
                if "xyz" in element and isinstance(element["xyz"], list):
                    element["xyz"] = np.array(element["xyz"])
                if "states" in element and isinstance(element["states"], list):
                    element["states"] = np.array(element["states"])

        return scenario


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize Puffer scenarios as MP4 videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create video following ego vehicle
  python viz_video.py scenario.json output.mp4 --follow-ego --zoom-radius 40

  # Create video at 20 FPS with high quality
  python viz_video.py scenario.json output.mp4 --fps 20 --dpi 200

  # Zoom to specific location
  python viz_video.py scenario.json output.mp4 --zoom-x 100 --zoom-y 50 --zoom-radius 50

  # Full scene high quality
  python viz_video.py scenario.json output.mp4 --figsize 24 --dpi 150
        """,
    )

    parser.add_argument("input", help="Input Puffer file path (.json or .bin)")
    parser.add_argument("output", help="Output MP4 file path")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    parser.add_argument("--no-routes", action="store_true", help="Don't show agent routes")
    parser.add_argument("--no-future", action="store_true", help="Don't show trajectory history")
    parser.add_argument("--figsize", type=int, default=20, help="Figure size in inches (default: 20)")
    parser.add_argument("--dpi", type=int, default=150, help="Video resolution DPI (default: 150)")
    parser.add_argument("--zoom-x", type=float, help="X coordinate of zoom center (meters)")
    parser.add_argument("--zoom-y", type=float, help="Y coordinate of zoom center (meters)")
    parser.add_argument("--zoom-radius", type=float, help="Zoom radius in meters (e.g., 50)")
    parser.add_argument("--follow-ego", action="store_true", help="Center view on ego vehicle (dynamic)")

    args = parser.parse_args()

    # Load scenario
    print(f"Loading Puffer scenario from {args.input}...")
    try:
        scenario = load_puffer_scenario(args.input)
    except Exception as e:
        print(f"✗ Error loading scenario: {e}")
        return 1

    # Check length
    metadata = scenario.get("metadata", {})
    length = metadata.get("scenario_length", 91)

    # Parse zoom parameters
    zoom_center = None
    if args.zoom_x is not None and args.zoom_y is not None:
        zoom_center = (args.zoom_x, args.zoom_y)
    elif args.zoom_x is not None or args.zoom_y is not None:
        print("✗ Error: Both --zoom-x and --zoom-y must be specified together")
        return 1

    print(f"Scenario has {length} timesteps")
    print(f"Creating video at {args.fps} FPS...")
    if args.follow_ego:
        print("  Mode: Following ego vehicle (dynamic)")
    elif zoom_center:
        print(f"  Mode: Zoomed to ({zoom_center[0]}, {zoom_center[1]}) ± {args.zoom_radius}m")
    else:
        print("  Mode: Full scene view")

    # Render
    try:
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
    except Exception as e:
        print(f"✗ Error rendering: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
