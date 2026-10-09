Input data
==========

123D boundary
-------------

123Drive does not parse raw nuPlan, Waymo, Argoverse, or nuScenes data directly. The
`py123d <https://github.com/autonomousvision/py123d>`_ dependency exposes each dataset through
one API:

* ``SceneAPI`` supplies scene metadata, ego states, box detections, traffic-light detections,
  and its associated map.
* ``MapAPI`` supplies static map layers and is also used directly for map-only conversion.

The input root follows the 123D layout:

.. code-block:: text

   /data/py123d/
   ├── logs/
   │   └── <dataset-specific scene directories>/...arrow
   └── maps/
       └── <dataset-specific map directories>/...arrow

Scene discovery
---------------

For normal conversion, 123Drive creates a 123D ``SceneFilter``. Dataset, split, log, and UUID
arguments become filter fields. The filter also:

* requests box detections;
* resamples to ``--dt``;
* limits the future duration when ``--duration_s`` is nonzero;
* caps the result with ``--num_scenes``.

Discovery uses 123D's process-pool executor when ``--workers`` is greater than one and its
sequential executor otherwise. Conversion then uses a separate Joblib process pool.

Map-only discovery
------------------

``--map_only`` scans ``maps/`` recursively for ``.arrow`` files and opens each as an
``ArrowMapAPI``. Dataset filters match path components. OpenDRIVE conversion automatically
enables map-only mode when ``opendrive`` is explicitly selected.

Required source data
--------------------

A scene conversion requires:

* a map API;
* an ego state for every frame;
* a positive iteration duration;
* box detections for dynamic actors.

Supported map layers are lanes, road lines, road edges, crosswalks, and stop zones. Unsupported
layers and unrecognized object labels are ignored. Waymo Motion auxiliary metadata is used,
when present, to preserve ``objects_of_interest`` and ``tracks_to_predict``.

Coordinate system
-----------------

All geometry is translated by one three-dimensional scene centroid. For logged scenes the
centroid is the mean ego position. Map-only conversion falls back to the mean of all lane
centerline points, or the origin when the map contains no lanes.

Agent ``z`` is the bottom of its bounding box rather than its center. When the source map has
no elevation, every agent, object, traffic-light, map, and stop-zone ``z`` coordinate is forced
to zero.

Map extent
----------

Per-log maps and map-only inputs are loaded completely. Otherwise, the converter queries map
objects intersecting a 250 metre buffer around the ego trajectory. This bounds memory and file
size while retaining nearby road context.
