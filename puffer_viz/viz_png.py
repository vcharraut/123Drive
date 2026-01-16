#!/usr/bin/env python3
"""
CLI script to visualize Puffer scenarios as PNG images.

Usage:
    python viz_png.py <puffer_file_path> <output_png_path> [--timestep 0] [--no-routes] [--no-future]

Example:
    python viz_png.py /path/to/scenario.bin output.png --timestep 10
    python viz_png.py /path/to/scenario.json output.png --timestep 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from puffer_renderer import render_scenario_png


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

        print(scenario)

        # Convert lists back to numpy arrays where needed
        if "dynamic_agents" in scenario:
            for agent in scenario["dynamic_agents"]:
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

        if "metadata" in scenario:
            metadata = scenario["metadata"]
            if "timesteps" in metadata and isinstance(metadata["timesteps"], list):
                metadata["timesteps"] = np.array(metadata["timesteps"])

        return scenario


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize Puffer scenarios as PNG images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full scene with high resolution
  python viz_png.py scenario.json output.png --timestep 50

  # Follow ego vehicle with 30m radius
  python viz_png.py scenario.json output.png --timestep 50 --follow-ego --zoom-radius 30

  # Zoom to specific location
  python viz_png.py scenario.json output.png --timestep 50 --zoom-x 100 --zoom-y 50 --zoom-radius 40

  # Ultra high resolution for details
  python viz_png.py scenario.json output.png --dpi 600 --figsize 32
        """,
    )

    parser.add_argument("input", help="Input Puffer file path (.json or .bin)")
    parser.add_argument("output", help="Output PNG file path")
    parser.add_argument("--timestep", type=int, default=0, help="Which timestep to visualize (default: 0)")
    parser.add_argument("--no-routes", action="store_true", help="Don't show agent routes")
    parser.add_argument("--no-future", action="store_true", help="Don't show trajectory history")
    parser.add_argument("--figsize", type=int, default=24, help="Figure size in inches (default: 24)")
    parser.add_argument("--dpi", type=int, default=300, help="Image resolution DPI (default: 300)")
    parser.add_argument("--zoom-x", type=float, help="X coordinate of zoom center (meters)")
    parser.add_argument("--zoom-y", type=float, help="Y coordinate of zoom center (meters)")
    parser.add_argument("--zoom-radius", type=float, help="Zoom radius in meters (e.g., 50)")
    parser.add_argument("--follow-ego", action="store_true", help="Center view on ego vehicle")

    args = parser.parse_args()

    # Load scenario
    print(f"Loading Puffer scenario from {args.input}...")
    try:
        scenario = load_puffer_scenario(args.input)
    except Exception as e:
        print(f"✗ Error loading scenario: {e}")
        return 1

    # Validate timestep
    metadata = scenario.get("metadata", {})
    length = metadata.get("length", 0)
    # if args.timestep >= length:
    #     print(f"✗ Error: Timestep {args.timestep} is out of range (scenario has {length} timesteps)")
    #     return 1

    # Parse zoom parameters
    zoom_center = None
    if args.zoom_x is not None and args.zoom_y is not None:
        zoom_center = (args.zoom_x, args.zoom_y)
    elif args.zoom_x is not None or args.zoom_y is not None:
        print("✗ Error: Both --zoom-x and --zoom-y must be specified together")
        return 1

    # Render
    print(f"Rendering timestep {args.timestep}/{length - 1}...")
    if args.follow_ego:
        print("  Mode: Following ego vehicle")
    elif zoom_center:
        print(f"  Mode: Zoomed to ({zoom_center[0]}, {zoom_center[1]}) ± {args.zoom_radius}m")
    else:
        print("  Mode: Full scene view")

    try:
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
    except Exception as e:
        print(f"✗ Error rendering: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
