Conversion pipeline
===================

One worker converts one discovered 123D object. The worker binds dataset and scenario context to
logging, then executes extraction, validation, transforms, serialization, and atomic output.

Extraction
----------

``extract_scenario`` first distinguishes logged scenes from map-only inputs, resolves the map,
and chooses the requested scenario identity.

For a logged scene it then:

#. caches every ego state and rejects missing frames;
#. computes the scene centroid;
#. queries map content around the ego path;
#. extracts ego and detected tracks into fixed-length arrays;
#. reconstructs missing velocities;
#. maps 123D labels to PufferDrive enums;
#. preserves Waymo prediction targets when auxiliary metadata exists;
#. extracts per-lane traffic-light detections.

Lanes are extracted before other map elements because topology and traffic controls reference
lane IDs. Undefined lane types are inferred only when all connected known lanes agree.
Predecessor and successor references whose geometry points in the opposite direction are swapped;
references to missing lanes are removed.

Pre-transform validation
------------------------

Validation checks the extracted scenario and its temporary traffic-light and stop-zone data.
Failure stops only the current scene. See :doc:`validation`.

Ordered transforms
------------------

The following stages run in exactly this order:

.. list-table::
   :header-rows: 1
   :widths: 5 27 68

   * - #
     - Stage
     - Effect
   * - 1
     - Traffic-light interpolation
     - Optional ``--interpolate_tl`` inference from map topology and vehicle motion
   * - 2
     - Reverse road edges
     - Optional ``--reverse_road_edges`` reversal for source conventions opposite to PufferDrive
   * - 3
     - Process polylines
     - Remove duplicate points, simplify, then enforce maximum segment length
   * - 4
     - Interpolate polygons
     - Close polygons and densify their boundary to at most 5 metre spacing
   * - 5
     - Prune invalid map elements
     - Remove geometry too short to serialize and clean every affected reference
   * - 6
     - Build traffic controls
     - Convert source lights and stop zones into final stop lines, headings, states, and lane refs
   * - 7
     - Compute agent routes
     - Match vehicle trajectories to the directed lane network
   * - 8
     - Invalidate overlapping log agents
     - Optional ``--invalid_agent_overlap`` removal of unrouted actors intersecting active actors
   * - 9
     - Compute lane lengths
     - Store total and cumulative arc lengths over final lane geometry
   * - 10
     - Build lane graph
     - Precompute all-pairs directed distances for freeway and surface-street lanes
   * - 11
     - Reindex
     - Remap IDs and every surviving cross-reference unless ``--no_reindex`` is set

Geometry processing
-------------------

Polyline processing first removes consecutive points closer than ``1e-9``. Shapely simplification
uses ``--area_threshold`` as its tolerance and retains the original three-dimensional points.
Distance-based interpolation then subdivides every segment longer than
``--max_segment_length``.

Polygon processing closes open outlines and uses Shapely ``segmentize`` with 5 metre spacing.
Elements with fewer than two polyline points or three polygon points are removed.

Traffic controls
----------------

Traffic-light detections already reference a lane. The transform creates a stop line across the
start of that lane. Its heading is averaged from valid incoming-lane headings, falling back to
the controlled lane direction. Lane boundaries determine stop-line width; missing or invalid
boundaries use 3.5 metres.

Stop-zone polygons become stop, yield, or traffic-light controls. Their stop lines are the
intersection of the polygon with a line perpendicular to travel. A light detection takes
precedence over a stop zone controlling the same lane.

Lane distances
--------------

Only freeway and surface-street lanes participate in the serialized lane graph. A directed edge
follows each source lane's ``exit_lanes`` and is weighted by that source lane's final arc length.
SciPy Dijkstra produces the all-pairs matrix; unreachable destinations remain infinity.

Overlap filtering
-----------------

When enabled, routed agents are considered active and unrouted agents log-only. For each frame,
the transform uses axis-aligned boxes as a fast prefilter, then exact oriented Shapely polygons.
The first intersection with any active agent clears the log-only actor's entire ``valid`` array
and recomputes its control state.

Reindexing
----------

Map, agent, object, and traffic-control IDs become contiguous ranges in iteration order. The
stage also remaps lane topology, routes, traffic controls, lane-graph rows and columns, objects of
interest, and tracks to predict. Unknown references are dropped.

Post-transform validation and serialization
-------------------------------------------

The final scenario is validated again without extraction extras. Successful scenarios are encoded
by :doc:`binary-format`, written to a temporary file in the destination directory, and moved over
the final path atomically.
