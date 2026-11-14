# 123Drive

123Drive converts [123D](https://github.com/autonomousvision/py123d) Arrow
datasets into `.bin` scenarios for Macadam. Python 3.12 to 3.13 is required.

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

Towns are synthetic European town centres. Each seed draws its targets from the range
measured on OpenStreetMap in central Paris, Lyon, Munich and Rome, two 1.2 km windows
per city: junction arm counts, one-way share, lanes per direction, speed limits,
signalized junctions, roundabouts, and bus and bike lane shares. Layout controls are
tuned so towns also fall in the measured range of junction density, street length
between junctions, junction angles, and street orientation entropy. Towns match these
statistics, not the layout of any city. `EUROPE` and `LAYOUT` in `src/mapforge/town.py`
hold the ranges.

Streets come from rotated district lattices with uneven gaps between parallel streets,
straight or jittered, crossed by boulevards that radiate from plazas and sometimes by a
ring. A bend inside a street is not a junction. Every directed street stays reachable
without a U-turn, so towns have no dead ends; the measured windows have 1–18%.
At the extremes, generated towns run a little denser and their junctions a little
sharper than the measured windows. Approach setbacks grow with the width and angle of
the crossing streets.

Major streets get 50 km/h, two or three lanes per direction with lane changes, and bus
lanes, each in its measured share. Other streets use 30 km/h, with some 20 km/h shared
zones. Right turns leave from the kerb lane and left turns from the inner lane;
straight-through connectors keep the street speed and turns use 20 km/h. Car lanes are
2.6–3.2 m wide, bus lanes 3.3 m, and bike lanes 1.6 m, informed by the
[Certu urban-road guide's 3 m lane recommendation](https://www.msr25.doubs.developpement-durable.gouv.fr/IMG/pdf/guide_certu_te-et-amenagements-de-voirie-en-milieu-urbain.pdf).

Bus lanes (`BUS_LANE`) run along the kerb of major streets and also serve bikes. Bike
lanes (`BIKE_LANE`) run along the kerb, against traffic on many one-way streets. Both
connect through junctions to lanes of their own kind; where none continues, they join
the straightest car lane. Car lanes never list bus or bike connectors, so car routing
and the lane-distance matrix stay on car lanes. Bus and bike lanes stop one street
before a roundabout. Macadam reads both lane types but does not drive cars on them.

Maps include traffic controls, crosswalks, 2.2 m sidewalks, kerb parking away from bus
lanes, and small carparks with driveway openings. STOP/YIELD lines sit 0.8 m before
junction entries, after the pedestrian crossing. Crossings leave room for a waiting
car; parking starts farther back for visibility.

Unsignalized junctions give priority to the most important street that continues
through them. Only minor approaches receive STOP or YIELD controls; there are no
all-way stops or all-way yields at ordinary intersections. Some quiet crossings stay
unsigned for
[priority to the right](https://www.onisr.securite-routiere.gouv.fr/politique-de-securite-routiere/code-de-la-route).
STOP placement is a seeded variation, not a visibility assessment of real buildings.

The measured share of junctions receives traffic lights, junctions on major streets
first. Facing approaches share a phase ID, so left turns are unprotected; bus and bike
lanes share their approach's phase, and contraflow bike approaches yield. Binaries carry
junction and phase IDs but no light states: the simulator generates them. Signal stop
lines precede the crosswalk.

Junctions whose arms are at least 47° apart and 42 m long can become roundabouts at the
measured density, plazas with many arms first. Each has a central island and a shared
counterclockwise circulation lane at 15 km/h. Entries yield to circulating traffic,
following the [Cerema giratory-junction guidance](https://demarches-securite-routes.cerema.fr/IMG/pdf/Audit_SR_PMS_Urbain_Fiches_techniques_mars_2012_cle2651e9-1.pdf).
Entry yield lines sit 0.8 m before the outer ring edge. Arms narrow to one lane per
direction. Entries leave the straight arm on a right-hand curve that meets the ring
tangentially, and exits mirror it, leaving a small splitter island where space allows.

Carpark surfaces and driveways are included; lane routing covers the street network.
Maps contain no parked vehicles. The simulator supplies parked and moving vehicles.
In Macadam, enable `yield_signs_enabled` and `priority_right_enabled` to enforce
those rules. Macadam generates light states: its pedestrian cycle uses the junction
phases, while the non-pedestrian mode randomizes lights independently.

The default 500 m extent is comparable to the bundled CARLA Town03/Town05 road
reference extents (approximately 577 × 575 m and 473 × 399 m). `--size` accepts
200–1200 m. Geometry stays flat. Output uses the existing binary format and viewer;
Macadam supplies moving agents. Generation with identical options is repeatable.

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
Macadam chooses the controller and whether to replay, freeze, or omit background
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
