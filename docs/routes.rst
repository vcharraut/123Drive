Route computation
=================

Routes assign each vehicle an ordered lane-ID sequence representing its expected path. The
algorithm uses the full observed trajectory, lane geometry, directed topology, heading, elevation,
and road-edge boundaries.

.. code-block:: text

   scenario cache -> eligibility -> off-road check -> point candidates
                  -> dynamic-programming path -> connector expansion -> dead-end extension

Scenario cache
--------------

The converter precomputes shared data once:

* trimmed XYZ lane centerlines and XY bounding boxes;
* directed connectivity from ``exit_lanes``;
* start and end lane tangents;
* padded lane arrays and cumulative segment lengths;
* road-edge segments for off-road detection;
* a cache of short topological paths.

Eligibility
-----------

Ego is always considered when it has a trajectory and the map has lanes. A non-ego vehicle must:

* be valid at ``--route_check_timestep``;
* be on-road at the check frame.

Pedestrians, cyclists,
objects, and other non-vehicle agents do not receive routes. Conversion fails when ego exists but
no ego route can be computed.

Off-road and parked classification
----------------------------------

A moving vehicle must be within 5 metres of an elevation-compatible lane centerline. A nearly
stationary vehicle uses a stricter 0.3 metre threshold. A vehicle is also off-road if its oriented
footprint intersects a road edge within 2 metres of its elevation.

Static agents are frozen during self-play. A track shorter than 1.5 seconds is static when its
median speed is at most 1 m/s. Otherwise, static classification median-smooths the XY trajectory.
An extent at or below 1.5 metres is parked. Larger accumulated jitter is still considered parked when peak motion over 1.5 seconds
stays below 1.5 metres and net displacement is less than 80% of the smoothed path length.

Ground-truth lane candidates
----------------------------

For every valid trajectory point:

#. prefilter lanes with the trajectory bounding box plus a 12 metre margin;
#. project the point to nearby lane centerlines;
#. require distance at most 7 metres;
#. require heading dot product greater than 0.3;
#. require projected lane elevation within 2 metres;
#. rank by distance, alignment, and lane ID;
#. retain at most seven candidates and their projected lane arc-length.

Path search
-----------

A dynamic program selects candidates jointly across the entire trajectory. Its score is:

.. code-block:: text

   50 * skipped_points + 1 * topology_hops + 6 * lane_changes + point_distance

Ties resolve by skipped points, hops, lane changes, then total point distance. Remaining ties are
deterministic because candidates and lane exits are ordered.

Repeated observations on one lane may move backward by at most 2 metres to tolerate projection
noise. A transition between lanes must have a directed path of at most three hops. A breadth-first
search supplies connector lanes between ground-truth-supported candidates.

Extension and output
--------------------

``route_gt_len`` records the ground-truth-supported route length. The route then continues until a
dead end or revisit. At a branch, the exit whose start tangent best matches the current lane's end
tangent wins; smaller lane ID breaks ties.

Control state follows the result:

.. list-table::
   :header-rows: 1

   * - Condition
     - State
   * - Vehicle has a route
     - ``CONTROLLABLE``
   * - Vehicle is moving without a route
     - ``NON_CONTROLLABLE_MOVING``
   * - Vehicle is parked
     - ``NON_CONTROLLABLE_STATIC``

The state is a flag read at simulator initialisation: controllable vehicles may be driven by the
policy or IDM, moving ones are replayed from the log, static ones are frozen in self-play. No
transform removes or invalidates log data.

A final pass hands a non-ego ``CONTROLLABLE`` vehicle to ``NON_CONTROLLABLE_MOVING`` when its
logged box overlaps another agent at ``--route_check_timestep``, or overlaps a moving or static
agent at any later frame (duplicate tracks, perception noise). Controlling it would start the
policy in, or drive it into, a collision the log already contains. A vehicle handed over counts as
log-driven, so controllable vehicles it overlaps later are handed over too. The route is kept.
