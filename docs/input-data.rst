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

CARLA road edges
----------------

Road-edge geometry is computed upstream in py123d. The local OpenDRIVE parser
includes shoulders in generic-drivable surfaces and extends the driving surface
to the road-facing walkway boundary where one exists. Standalone corner sidewalks
can face the road on their outer boundary, so lane ordering alone does not select
the curb. Elsewhere, edges follow the drivable footprint.

The parser compares holes before and after adding shoulders and curb connections.
At junctions, it discards detached fragments of a larger island while retaining
the main island, walkway-supported islands, and mapped medians. Walkway polygons
then clip the resulting footprint; a centimetre grid removes sampling slivers.
Narrow holes introduced only by shoulder seams are also discarded. Elevation
lifting splits an edge when nearest boundaries switch between bridge decks.

Regenerate Arrow maps with the modified sibling checkout before converting them
to binaries. From the 123Drive root:

.. code-block:: bash

   PYTHONPATH=../py123d/src .venv/bin/python -m py123d.script.run_conversion \
     dataset=opendrive execution=sequential_executor \
     dataset_paths.py123d_data_root=output/carla-source
   .venv/bin/python -m bin_factory.main --preset opendrive \
     --py123d_path output/carla-source --output output --workers 1

Existing Arrow maps do not acquire the corrected edges just by rerunning 123Drive.

Required source data
--------------------

A scene conversion requires:

* a map API;
* an ego state for every frame;
* a positive iteration duration;
* box detections for dynamic actors.

All py123d map layers are supported: lanes, lane groups, intersections, crosswalks, walkways,
carparks, generic-drivable areas, stop zones, road edges, road lines, and speed bumps. Unrecognized
object labels are ignored. Waymo Motion auxiliary metadata is used, when present, to preserve
``objects_of_interest`` and ``tracks_to_predict``.

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
