"""Write src/bin_factory/data/carla_light_stop_points.json from a running CARLA server.

For every traffic light: one point per lane its trigger volume spans, advanced to the junction entry in 0.5 m steps,
the line RunningRedLightTest scores (leaderboard atomic criteria). Needs CARLA's PythonAPI, not 123Drive's venv:

    <carla python> tools/dump_carla_light_stop_points.py --port 2000 --output src/bin_factory/data/carla_light_stop_points.json
"""

import argparse
import json
import math
import time

import carla


TOWNS = ("Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD")
TRIGGER_SAMPLE_SPAN_FRACTION = 0.9  # RunningRedLightTest: avoids crossing into adjacent lanes
TRIGGER_SAMPLE_STEP_M = 1.0
JUNCTION_ADVANCE_STEP_M = 0.5
MAX_JUNCTION_ADVANCE_STEPS = 400
LOAD_WORLD_SETTLE_S = 5.0


def scored_waypoints(light, carla_map):
    transform = light.get_transform()
    trigger = light.trigger_volume
    lane_waypoints = []
    x = -TRIGGER_SAMPLE_SPAN_FRACTION * trigger.extent.x
    while x < TRIGGER_SAMPLE_SPAN_FRACTION * trigger.extent.x:
        waypoint = carla_map.get_waypoint(transform.transform(trigger.location + carla.Location(x=float(x))))
        if waypoint is not None and (
            not lane_waypoints
            or lane_waypoints[-1].road_id != waypoint.road_id
            or lane_waypoints[-1].lane_id != waypoint.lane_id
        ):
            lane_waypoints.append(waypoint)
        x += TRIGGER_SAMPLE_STEP_M
    scored = []
    for waypoint in lane_waypoints:
        for _ in range(MAX_JUNCTION_ADVANCE_STEPS):
            if waypoint.is_junction:
                break
            ahead = waypoint.next(JUNCTION_ADVANCE_STEP_M)
            if not ahead or ahead[0].is_junction:
                break
            waypoint = ahead[0]
        scored.append(waypoint)
    return scored


def town_points(world):
    carla_map = world.get_map()
    rows = []
    for light in world.get_actors().filter("traffic.traffic_light"):
        for waypoint in scored_waypoints(light, carla_map):
            location = waypoint.transform.location
            rows.append(
                {
                    "light": int(light.id),
                    "x": round(location.x, 4),
                    "y": round(-location.y, 4),
                    "heading": round(-math.radians(waypoint.transform.rotation.yaw), 6),
                    "in_junction": bool(waypoint.is_junction),
                }
            )
    return sorted(rows, key=lambda row: (row["light"], row["x"], row["y"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--towns", nargs="+", default=list(TOWNS))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(300.0)
    world = client.get_world()
    towns = {}
    for town in args.towns:
        if town not in world.get_map().name:
            world = client.load_world(town)
            time.sleep(LOAD_WORLD_SETTLE_S)
        towns[town] = town_points(world)
        print(f"{town}: {len(towns[town])} scored points", flush=True)
    output = {
        "carla_version": client.get_server_version(),
        "frame": "OpenDRIVE: x = CARLA x, y = -CARLA y, heading = -CARLA yaw (rad)",
        "towns": towns,
    }
    with open(args.output, "w") as output_file:
        output_file.write(json.dumps(output, indent=1) + "\n")


if __name__ == "__main__":
    main()
