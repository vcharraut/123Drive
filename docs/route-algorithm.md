# Route Computation Algorithm

Routes assign each vehicle agent an ordered sequence of lane IDs representing the path it follows. The algorithm lives in `src/bin_factory/transforms/routes.py`.

## Pipeline

```
Per-scenario cache → Per-agent offroad check → GT candidates → DP sequence search → Dead-end extension
```

### 1. Route Cache

`build_route_cache()` precomputes shared lane data:
- Trimmed lane centerlines and bounding boxes
- Lane connectivity graph from `exit_lanes`
- Lane start/end tangents for downstream extension
- Road edge segments for offroad detection

### 2. Offroad Check

`_is_offroad_at_timestep()` keeps the current agent selection logic:
- If the agent's position is farther than 5m from all lane centerlines, or 1m when nearly stationary, skip it
- If the agent's footprint intersects a road edge boundary, skip it

### 3. GT Lane Candidates

`_build_point_observations()` builds a small lane set per GT point:
1. Prefilter lanes by the GT trajectory bounding box
2. Compute point-to-lane distance and local lane tangent alignment
3. Keep only candidates with distance `<= 7m` and heading alignment `> 0.3`
4. Keep up to 7 candidates per point, ranked by distance, alignment, and lane ID
5. Compute projected arc-length on each candidate lane

### 4. DP Sequence Search

`_select_candidate_path()` solves the route jointly across GT:
1. Use a weighted path score: `50 * skipped_points + 1 * hops + 6 * lane_changes + point_distance`
2. Break ties by skipped points, hops, lane changes, then total distance
3. Allow same-lane transitions only if projected progress stays forward within a 2m tolerance
4. Allow lane changes only if the exit-graph shortest path needs at most 3 hops
5. Backtrack the best candidate path
6. Expand connector lanes between consecutive GT-supported lanes

### 5. Extension Beyond GT

`_extend_route_to_dead_end()` continues the route after GT ends:
1. Start from the last GT-supported lane
2. At each branch, choose the exit whose start tangent is most aligned with the current lane end tangent
3. Break ties by smaller lane ID
4. Stop at dead-end, revisit, or map-size safety cap

### 6. Output

Each eligible vehicle gets one route `list[int]` and `route_gt_len`, the count of leading route lanes supported by GT before extension. Agents without a valid route get an empty list and are marked offroad in the binary output.
