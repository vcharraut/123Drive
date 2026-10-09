Validation
==========

Validation can run before and after transforms. ``--validate_level`` controls its cost and
strictness:

.. list-table::
   :header-rows: 1

   * - Level
     - Checks
   * - ``0``
     - Disabled
   * - ``1``
     - Structural schema, array shapes, trajectory lengths, required fields
   * - ``2``
     - Level 1 plus references, finite values, enum values, sizes, topology, and temporal sanity

Level 1
-------

Structural validation checks:

* non-negative scenario length and finite, positive ``dt`` for dynamic scenes;
* exact track array ranks, channel widths, and timestep counts;
* two or more points for polylines and three or more for polygons;
* lane topology containers and optional boundary shapes;
* stop-zone geometry and lane lists;
* traffic-light position, integer states, state count, and controlled lane;
* final traffic-control IDs, types, stop-line shape, heading, states, and controlled lanes;
* lane-distance matrix shape and lane-ID count.

The pre-transform pass includes ``ExtractionExtras``. The post-transform pass instead validates
the final traffic controls, route fields, lane lengths, and graph created by the pipeline.

Level 2
-------

Semantic validation adds:

* no NaN or infinity in dynamic state or map arrays;
* no NaN in the lane-distance matrix (infinity is valid for unreachable lanes);
* every lane, route, traffic-control, light, stop-zone, and graph reference resolves;
* unique traffic-control and lane-graph IDs;
* valid traffic-light enum values;
* ``route_gt_len`` lies inside its route;
* positive dimensions at valid actor frames;
* ego ID 0 exists, has valid contiguous frames, and does not move faster than the 50 m/s
  displacement threshold;
* crossing road-edge elevations do not differ by the suspicious 0.5--4 metre bridge range.

Failure behavior
----------------

Validation returns all errors found in the current pass. The worker logs each error, records one
failure entry for the scene, and continues processing other scenes. Validation does not repair
data. Repairs that are part of the conversion contract happen in extraction or explicit
transform stages.

Use level 1 for normal bulk conversion. Use level 2 when introducing a dataset, changing
extraction, or investigating malformed output. Level 0 removes the safety boundary and can allow
serialization failures or invalid references into the binary.
