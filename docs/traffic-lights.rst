Traffic-light processing
========================

Source detections
-----------------

123D traffic-light detections identify a controlled lane and one state per observed frame.
123Drive creates a fixed-length state sequence for each valid referenced lane. Missing frames
start as ``UNKNOWN``. States map to unknown, red, yellow, green, or off.

Normal conversion preserves these sequences and turns them into serialized traffic controls
during the pipeline.

Optional interpolation
----------------------

``--interpolate_tl`` enables the trajectory-based algorithm implemented from *Improving Traffic
Signal Data Quality for the Waymo Open Motion Dataset* (Yan et al., 2025). The preset enables it
for ``nuplan-mini`` and ``wod-motion``.

The algorithm:

#. builds a freeway and surface-street lane graph;
#. removes malformed short dead ends and invalid connectivity;
#. reconciles lane neighbors, diverges, and merges;
#. groups connected lanes into signalized intersections;
#. assigns vehicle position, speed, and acceleration to lanes at each frame;
#. combines raw light detections with motion-derived red/green evidence;
#. chooses the closest physically feasible intersection phase;
#. smooths short phase flips and inserts yellow transitions;
#. writes the generated states back to extraction extras.

Only intersections with source signal evidence and enough connected lanes are considered.
Generated results currently support three-way and four-way intersection representations.
When inference cannot form a valid intersection or phase sequence, the source detections remain
unchanged.

Kinematic evidence
------------------

The phase generator uses vehicle motion around the stop line:

* speed above 3 m/s supports green;
* speed below 1 m/s supports red;
* acceleration above 0.5 m/s² supports green;
* deceleration below -1 m/s² supports red.

Raw and estimated evidence are confidence-weighted. Agreement receives high weight; disagreement
receives low weight and is resolved against feasible phase patterns. Internal timing windows scale
from the scenario ``dt``.

Final traffic controls
----------------------

After optional interpolation, every usable light becomes a traffic-control record containing:

* an ID and traffic-light type;
* a two-point stop line;
* travel heading;
* one state per scenario frame;
* the controlled lane ID.

Stop-zone traffic lights without observed detections receive ``UNKNOWN`` for the full scenario.
Stop and yield controls carry no state sequence. Bike-lane lights are skipped.

Interpolation changes the source-level sequences only. The later traffic-control stage remains the
single place that creates serialized controls.
