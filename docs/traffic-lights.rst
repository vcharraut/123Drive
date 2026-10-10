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

``--interpolate_tl`` fills missing light states from detections, intersection geometry and vehicle
motion. The preset enables it for ``nuplan-mini`` and ``wod-motion``.

#. Signalized connectors are the lanes carrying a light track plus the successors of lanes in
   traffic-light stop zones.
#. Connectors whose polylines touch form an intersection. Inside it, connectors with the same
   approach heading and turn (left, straight, right) form a movement group sharing one light,
   unless their detections disagree.
#. Two groups conflict when their connectors cross, they come from different approaches and
   neither is a free turn (right turn, or left turn in left-hand traffic such as Singapore).
#. Each connected set of conflicting or same-approach groups runs a hidden Markov model over joint
   states (every group GO or STOP) where no two conflicting groups are GO together. Sets with more
   than 1024 joint states fall back to conflict links only, then to single groups.
#. Forward-backward gives a GO probability per group and frame. Frames above 0.9 become green,
   below 0.1 red, the rest stay ``UNKNOWN``. Inferred green that switches to red gets a 3 s yellow
   tail.
#. Detected frames are copied to the output, except the last 2 s of a red run followed by a gap:
   detectors switch late to green, so there the detection counts only ±1 and the inferred state
   wins when confident.
#. Connectors without a track get one only when at least one frame is known.

Evidence
--------

Per group and frame, as a log-likelihood ratio of GO over STOP:

* each detection frame: ±4 (green/yellow vs red);
* a vehicle crossing the stop line onto the connector (faster than 1 m/s): +12 over the preceding
  0.5 s, +1 on free turns;
* a vehicle stopped (< 0.5 m/s) within 6 m before the stop line: -0.2 per frame, ignoring the last
  2 s before it starts moving.

The group sum is clipped to ±16. Groups switch on average every 30 s. Each pair of groups from the
same approach gets +0.05 per frame where they show the same state (left and straight of one
approach agree about 80% of the time in nuPlan and WOD). All windows scale with the scenario ``dt``.

Scoring
-------

``scripts/tl_score.py`` scores implementations side by side: coverage, fidelity to detections,
lane and time-gap hold-out accuracy, red-light entries, green stalls, short phases and conflicting
greens.

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
