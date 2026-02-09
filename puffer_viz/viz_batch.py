#!/usr/bin/env python3
"""
CLI script to batch visualize Puffer scenarios from a directory.

Usage:
    python viz_batch.py <input_dir> <output_dir> [options]

Example:
    python viz_batch.py /data/puffer/scenarios /output/viz --format both --max-scenarios 10
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from puffer_renderer import render_scenario_png, render_scenario_video


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

        return scenario


def find_puffer_files(input_dir: str, max_scenarios: int = None) -> list[str]:
    """Find all Puffer files (.json or .bin) in directory."""
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"✗ Error: Directory {input_dir} does not exist")
        return []

    # Find both JSON and binary files
    json_files = sorted(input_path.glob("*.json"))
    bin_files = sorted(input_path.glob("*.bin"))
    all_files = json_files + bin_files

    if max_scenarios:
        all_files = all_files[:max_scenarios]

    return [str(f) for f in all_files]


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Batch visualize Puffer scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch PNG export at timestep 10
  python viz_batch.py /data/scenarios /output/png --format png

  # Batch video export
  python viz_batch.py /data/scenarios /output/videos --format video

  # Both PNG and video
  python viz_batch.py /data/scenarios /output --format both

  # Limit to first 10 scenarios
  python viz_batch.py /data/scenarios /output --max-scenarios 10
        """,
    )

    parser.add_argument("input_dir", help="Input directory with Puffer files (.json or .bin)")
    parser.add_argument("output_dir", help="Output directory for visualizations")
    parser.add_argument(
        "--format",
        choices=["png", "video", "both"],
        default="png",
        help="Output format (default: png)",
    )
    parser.add_argument("--timestep", type=int, default=0, help="Timestep for PNG export (default: 10)")
    parser.add_argument("--fps", type=int, default=10, help="FPS for video export (default: 10)")
    parser.add_argument("--max-scenarios", type=int, help="Maximum number of scenarios to process")
    parser.add_argument("--no-routes", action="store_true", help="Don't show agent routes")
    parser.add_argument("--no-future", action="store_true", help="Don't show trajectory history")
    parser.add_argument("--png-dpi", type=int, default=150, help="PNG DPI (default: 150)")
    parser.add_argument("--video-dpi", type=int, default=100, help="Video DPI (default: 100)")

    args = parser.parse_args()

    # Find all Puffer files
    print(f"Scanning {args.input_dir} for Puffer scenarios...")
    puffer_files = find_puffer_files(args.input_dir, args.max_scenarios)

    if not puffer_files:
        print("✗ No Puffer files found")
        return 1

    print(f"Found {len(puffer_files)} scenario(s)")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    if args.format in ["png", "both"]:
        png_dir = os.path.join(args.output_dir, "png")
        os.makedirs(png_dir, exist_ok=True)
    if args.format in ["video", "both"]:
        video_dir = os.path.join(args.output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

    # Process each scenario
    success_count = 0
    error_count = 0

    for i, puffer_file in enumerate(puffer_files, 1):
        scenario_name = Path(puffer_file).stem
        print(f"\n[{i}/{len(puffer_files)}] Processing {scenario_name}...")

        try:
            # Load scenario
            scenario = load_puffer_scenario(puffer_file)
            metadata = scenario.get("metadata", {})
            length = metadata.get("scenario_length", 0)

            print(f"  → Scenario has {length} timesteps")

            # Validate timestep
            timestep = args.timestep
            if timestep >= length:
                timestep = min(10, length - 1)
                print(f"  ⚠ Adjusted timestep to {timestep} (scenario has {length} timesteps)")

            # Generate PNG
            if args.format in ["png", "both"]:
                png_path = os.path.join(png_dir, f"{scenario_name}.png")
                print(f"  → Creating PNG at timestep {timestep}...")
                render_scenario_png(
                    puffer_scenario=scenario,
                    output_path=png_path,
                    timestep=timestep,
                    show_routes=not args.no_routes,
                    show_future=not args.no_future,
                    figsize=(20, 20),
                    dpi=args.png_dpi,
                )

            # Generate video
            if args.format in ["video", "both"]:
                video_path = os.path.join(video_dir, f"{scenario_name}.mp4")
                print(f"  → Creating video ({length} frames @ {args.fps} FPS)...")
                render_scenario_video(
                    puffer_scenario=scenario,
                    output_path=video_path,
                    fps=args.fps,
                    show_routes=not args.no_routes,
                    show_future=not args.no_future,
                    figsize=(16, 16),
                    dpi=args.video_dpi,
                )

            success_count += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            error_count += 1
            continue

    # Summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total scenarios: {len(puffer_files)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")

    if args.format in ["png", "both"]:
        print(f"\nPNG files saved to: {png_dir}")
    if args.format in ["video", "both"]:
        print(f"Video files saved to: {video_dir}")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
