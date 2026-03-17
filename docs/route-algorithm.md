# Route Computation Algorithm

Routes assign each vehicle agent an ordered sequence of lane IDs representing the path it follows. The algorithm lives in `src/bin_factory/transforms/routes.py`.

## Pipeline

```
Per-scenario cache → Per-agent offroad check → Root lane selection → Beam search → Best route
```

### 1. Route Cache (once per scenario)

`build_route_cache()` precomputes shared data:
- Trimmed lane centerline polylines and bounding boxes
- Lane connectivity graph (exit lanes)
- Road edge segments for offroad detection

### 2. Offroad Check

`_is_offroad_at_timestep()` skips agents that start off drivable road:
- If the agent's position is farther than 5m (1m if stationary) from all lane centerlines → offroad
- If the agent's bounding box intersects a road edge boundary → offroad

### 3. Root Lane Selection

`_select_root_lane_candidates()` finds the most likely starting lane:
1. Sample up to 5 points from the first 8 trajectory positions
2. Prefilter lanes by bounding-box distance (12m margin)
3. For each candidate lane, compute point-to-polyline distance and heading alignment
4. Score = 0.7 × alignment + 0.3 × inverse_distance, summed over valid samples
5. Return top 3 lanes with score ≥ 0.3

### 4. Beam Search

`_search_route_beam()` extends root lanes through the lane graph:
1. Start with scored root candidates (beam width = 3)
2. At each step, expand each beam state through exit lanes
3. Score each candidate route by concatenating lane centerlines and measuring:
   - **Coverage**: fraction of trajectory points within 6m of the route polyline
   - **Distance**: average distance of covered points
   - **Final score**: coverage / (1 + avg_distance)
4. Routes with poor heading alignment (< 70% of samples aligned) are rejected early
5. Keep top 3 candidates per step, up to 10 lanes deep
6. Return the highest-scoring route across all steps

### 5. Output

Each agent gets at most one route (list of lane IDs). Agents without a valid route get an empty list and are marked as `mark_as_expert=1` in the binary output.
