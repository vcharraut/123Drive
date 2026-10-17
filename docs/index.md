# 123Drive

123Drive converts [123D](https://github.com/autonomousvision/py123d) Arrow
datasets into `.bin` scenarios for PufferDinde. Python 3.11 to 3.13 is required.

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

The code defines the details: [pipeline], [loader], [schema], and [binary writer].

## Routes and eligible agents

A route is an ordered list of lane IDs. The converter matches the recorded vehicle
trajectory to connected lanes using position, heading, and elevation. It extends
the route toward a dead end, choosing the straightest continuation at branches.

A non-ego vehicle keeps its route only if it:

* passes the parked-vehicle filter;
* is valid and on-road at `--route_check_timestep`;
* has at least 15 metres of route ahead;
* has no logged overlap at that frame or later overlap with an agent without a route.

The overlap check repeats when another vehicle loses its route. Non-vehicles have
empty routes. Ego bypasses the parked filter; conversion rejects scenes where ego
has no usable route, starts off-road, or overlaps another agent at the check frame.

A nonempty route means eligible for control. An empty route means background.
PufferDinde chooses the controller and whether to replay, freeze, or omit background
entities. Conversion keeps their recorded trajectories and validity masks.
There is no separate control-state or route-prefix-length field. See [route code].

## Traffic-light interpolation

By default, missing light observations remain unknown. `--interpolate_tl` combines
observed signals, lane topology, vehicle crossings, and stopped vehicles to infer
missing or inconsistent states. It groups intersection movements and uses conflicts
between them to constrain inference. Confident
estimates become red or green, with yellow transitions; uncertain frames stay unknown.
Yellow-only detections stay unchanged. See [interpolation code].

- [Dataset differences](presets.md)
- [Command options](cli-reference.md)

[pipeline]: https://github.com/vcharraut/123Drive/blob/main/src/bin_factory/preparation.py
[loader]: https://github.com/vcharraut/123Drive/tree/main/src/bin_factory/loader
[schema]: https://github.com/vcharraut/123Drive/blob/main/src/bin_factory/schema.py
[binary writer]: https://github.com/vcharraut/123Drive/blob/main/src/bin_factory/serialize.py
[route code]: https://github.com/vcharraut/123Drive/blob/main/src/bin_factory/transforms/routes.py
[interpolation code]: https://github.com/vcharraut/123Drive/blob/main/src/bin_factory/transforms/traffic_light_interpolation.py
