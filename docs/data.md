# Data Surface

This project intentionally exposes a small subset of `123D` into the PufferDrive binary.

## Included in v0.1

| Area | Included |
|------|----------|
| Ego track | position, heading, velocity, box dimensions, validity |
| Dynamic agents | position, heading, velocity, box dimensions, validity, mapped type |
| Agent route | one best lane-id route per eligible vehicle |
| Lanes | centerline, speed limit, predecessor/successor links |
| Road lines | polyline + mapped line type |
| Road edges | polyline + mapped edge type |
| Crosswalks | polygon |
| Traffic lights | per-timestep state + controlled lane ids |
| Metadata | scenario id, dataset name, scenario length, ego index |

## Not included in v0.1

| Area | Not included |
|------|--------------|
| Agent dynamics | acceleration, angular rates, steering, wheelbase |
| Map detail | lane boundaries, lane groups, intersections, walkways, carparks, generic drivable areas |
| Rich traffic controls | stop signs, yield signs, speed-limit signs, cones, barriers |
| Multiple route hypotheses | only one route is stored today |
