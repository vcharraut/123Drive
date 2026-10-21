# 123Drive

123Drive converts [123D](https://github.com/autonomousvision/py123d) Arrow
datasets into `.bin` scenarios for PufferDinde. Python 3.12 to 3.13 is required.

## Convert and view

The input directory must contain the extracted `logs/` and `maps/` directories.
Choose a [dataset preset](presets.md) and select splits separately:

```bash
uv sync --extra viz
uv run convert --preset nuplan --split_names nuplan-mini_val \
  --py123d_path /data/123d --output ./output
uv run web --dir ./output
```

Open <http://localhost:8080> to inspect the output. See [Command options](cli-reference.md) for all options.

## Conversion process

The converter selects scenes, resamples trajectories, and loads nearby map geometry.
It centers coordinates, processes geometry and traffic controls, computes vehicle
routes and lane distances, validates the result, and writes one binary per scene.
Map-only conversion writes static maps without agents.

Metadata stores the coordinate origin in float64 `origin_xyz`. It also stores the source
scene UUID, log name, split, and extraction versions for logs and maps.
Add the origin to restore source coordinates. When `map_has_z` is false, elevation stays flat.
`timestamps_us` preserves the exact per-frame source timestamps as int64 microseconds.
`timestamps_are_relative` distinguishes WOD's relative clock from absolute timestamps.

Each track retains its original tracking token or static map-object ID as `source_id`,
independent of local numeric IDs. Map-only binaries have no scene UUID or timestamps.

Binary format v3 uses a versioned header, a section directory, and indexed records.
Validity uses one byte per frame. The writer omits repeated constant dimensions and zero
elevation, and readers restore them. Curvature and all other sections remain available.
v3 adds per-lane route spans to agent records; regenerate v2 bins and update simulator readers to support v3.

The code defines the details: [pipeline], [loader], [schema], and [binary writer].

## Routes and eligible agents

A route is an ordered list of lane IDs. The converter matches the recorded vehicle
trajectory to lanes using position, heading, and elevation. Consecutive lanes follow a
successor link, or a lane change where the recorded vehicle moved to a neighbouring lane
matched at the same point; there are no lane changes on lanes inside intersections, after a
split, or before a merge. It extends the route toward a dead end, choosing the straightest
continuation at branches.

`route_entry_s` and `route_exit_s` give, per route lane, the `cum_length` span driven on it:
the whole lane across successor links, the switch points across a lane change. They carry
positions only, no timing.

A non-ego vehicle keeps its route only if it:

* passes the parked-vehicle filter;
* never moves faster than 50 m/s between consecutive valid frames in the log;
* is valid, on-road, and has at least 15 metres of route ahead at some timestep in `0..10`;
* has no logged overlap at its first eligible frame or later overlap with an agent without a route.

The overlap check repeats when another vehicle loses its route. Non-vehicles have
empty routes. Ego bypasses the parked filter; conversion rejects scenes where ego
has no usable route, starts off-road, or overlaps another agent at frame 0.

A nonempty route means eligible for control. An empty route means background.
PufferDinde chooses the controller and whether to replay, freeze, or omit background
entities. Conversion keeps their recorded trajectories and validity masks.
There is no separate control-state or route-prefix-length field. See [route code].

## Traffic-light inference

By default, missing light observations remain unknown. `--infer_traffic_lights` combines
observed signals, lane topology, vehicle crossings, and stopped vehicles to infer
missing or inconsistent states. It groups intersection movements and uses conflicts
between them to constrain inference. Confident estimates become red or green, with yellow
transitions. Uncertain frames stay unknown.
Yellow-only detections stay unchanged. See [inference code].

- [Dataset differences](presets.md)
- [Command options](cli-reference.md)

[pipeline]: https://github.com/vcharraut/123Drive/blob/main/src/bin_factory/preparation.py
[loader]: https://github.com/vcharraut/123Drive/tree/main/src/bin_factory/loader
[schema]: https://github.com/vcharraut/123Drive/blob/main/src/bin_factory/schema.py
[binary writer]: https://github.com/vcharraut/123Drive/blob/main/src/bin_factory/serialize.py
[route code]: https://github.com/vcharraut/123Drive/blob/main/src/bin_factory/generators/routes.py
[inference code]: https://github.com/vcharraut/123Drive/blob/main/src/bin_factory/generators/traffic_light_inference.py
