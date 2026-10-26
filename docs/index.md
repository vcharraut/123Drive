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

## Generate towns

Generate static maps from scratch, without an Arrow dataset:

```bash
uv run mapgen --size 500 --seed 0 --count 5 --output_dir ./output/towns
uv run web --dir ./output/towns
```

The default `mixed` style combines diagonal avenues with skewed streets, variable
block sizes, and an irregular town outline. Seeds vary the outline, rotation,
diagonal connections, T-junctions, one-way streets, and parking placement.
`paris` also uses this blend; `lyon` has fewer diagonal streets. These are synthetic
layouts inspired by the cities, not reconstructions or measured street distributions.

Large crossings and roundabouts share a limit of two per map. At most one crossing
has more than four arms. Other junctions use compact three- or four-way layouts,
with 7 m approach setbacks even along avenues.

Maps include connected right-hand traffic lanes and junction turns, traffic controls,
crosswalks, 2.2 m sidewalks, curb parking, and small carparks with driveway openings.
Tiny one-way streets use 2.5–2.8 m lanes and usually omit curb parking. Two-way
local lanes are 2.75–3.05 m wide; avenues use 3.25 m lanes. These dimensions are
generator choices informed by the [Certu urban-road guide's 3 m lane recommendation](https://www.msr25.doubs.developpement-durable.gouv.fr/IMG/pdf/guide_certu_te-et-amenagements-de-voirie-en-milieu-urbain.pdf).
Street speed limits are 20 km/h on narrow one-way streets, 30 km/h on local two-way
streets, and 50 km/h on avenues. Ordinary junction connectors use 20 km/h.
STOP/YIELD lines sit 0.8 m before junction entries, after the pedestrian crossing.
Crossings leave room for a waiting car; parking starts farther back for visibility.

Unsignalized crossings keep a priority route through the junction. Only minor
approaches receive STOP or YIELD controls; there are no all-way stops or all-way
yields at ordinary intersections. Some quiet crossings stay unsigned for
[priority to the right](https://www.onisr.securite-routiere.gouv.fr/politique-de-securite-routiere/code-de-la-route).
STOP placement is a seeded variation, not a visibility assessment of real buildings.

One or two suitable crossings receive traffic lights. Each approach has a protected
phase: 16 s green, 3 s amber, then 2 s all-red before the next approach. Binaries
include at least 120 s of coordinated states and junction/phase IDs. Signal stop
lines precede the crosswalk; the viewer can play the exported cycle.

Suitable junctions become compact roundabouts with a central island and a shared
counterclockwise circulation lane at 15 km/h. Entries yield to circulating traffic,
following the [Cerema giratory-junction guidance](https://demarches-securite-routes.cerema.fr/IMG/pdf/Audit_SR_PMS_Urbain_Fiches_techniques_mars_2012_cle2651e9-1.pdf).
Entry yield lines sit 0.8 m before the outer ring edge. Roundabouts use the remaining
large-junction allowance; some small layouts have no suitable junction.

Carpark surfaces and driveways are included; lane routing covers the street network.
Maps contain no parked vehicles. The simulator supplies parked and moving vehicles.
In PufferDinde, enable `yield_signs_enabled` and `priority_right_enabled` to enforce
those rules. Its current procedural mode replaces exported light states: the
pedestrian cycle uses junction phases, while the non-pedestrian mode randomizes
lights independently. Map phase data alone does not change that simulator behavior.

The default 500 m extent is comparable to the bundled CARLA Town03/Town05 road
reference extents (approximately 577 × 575 m and 473 × 399 m). `--size` accepts
200–1200 m. Geometry stays flat. Output uses the existing binary format and viewer;
PufferDinde supplies moving agents. Generation with identical options is repeatable.

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

Binary format v0.4 uses a versioned header, a section directory, and indexed records.
Validity uses one byte per frame. The writer omits repeated constant dimensions and zero
elevation, and readers restore them. Curvature and all other sections remain available.
Agent records carry per-lane route spans.

The map has two road sections. `ROADPOLY` holds lanes (centerlines), road lines, and road
edges; `ROADSURF` holds surface outlines: crosswalks, walkways, carparks, drivable areas,
speed bumps, shoulders, intersections, and lane groups. Each element stores its py123d
`MapLayer` value, and polylines also store the layer's py123d type (`LaneType`,
`RoadLineType`, or `RoadEdgeType`) unchanged. Polylines and surfaces share one id space.

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
