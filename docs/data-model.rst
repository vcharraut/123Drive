Scenario data model
===================

The loader normalizes 123D objects into dataclasses from ``bin_factory.schema``. These are an
internal contract between extraction, transforms, validation, serialization, Mapforge, and the
viewer binary parser.

Scenario
--------

``PufferScenario`` contains:

.. list-table::
   :header-rows: 1

   * - Field
     - Shape
     - Meaning
   * - ``agents``
     - ``dict[int, Track]``
     - Ego, vehicles, pedestrians, cyclists, and other dynamic actors
   * - ``objects``
     - ``dict[int, Track]``
     - Signs, cones, traffic-light objects, barriers, and generic objects
   * - ``map``
     - ``dict[int, MapElement]``
     - Lanes, road lines, road edges, and polygonal road areas
   * - ``metadata``
     - ``ScenarioMetadata``
     - Identity, dataset, timing, location, and prediction targets
   * - ``traffic_controls``
     - ``list[dict]``
     - Final traffic lights, stop signs, and yield controls
   * - ``lane_graph``
     - ``dict | None``
     - Lane IDs and their all-pairs directed distances

Tracks
------

Every ``Track`` stores timestep-aligned NumPy arrays:

* ``position`` is ``T x 3``;
* ``heading`` is ``T`` radians;
* ``velocity`` is ``T x 2``;
* ``valid``, ``length``, ``width``, and ``height`` are each ``T``;
* ``route`` is an ordered list of lane IDs;
* ``route_gt_len`` separates the ground-truth-supported route prefix from its extension;
* ``control_state`` determines whether PufferDrive should control the agent.

Ego is always agent ID 0. Missing source velocities are reconstructed from the nearest valid
positions around each frame. Invalid frames receive zero velocity.

Map elements
------------

Lanes, lines, and edges use a ``polyline``. Crosswalks and other areas use a ``polygon``. Lanes
also keep predecessor and successor IDs, neighbor IDs, boundaries, speed limit, total length,
and cumulative per-point length.

Road types reserve numeric ranges:

.. list-table::
   :header-rows: 1

   * - Range
     - Category
   * - 0--9
     - Lanes
   * - 10--19
     - Road-line markings
   * - 20--29
     - Road edges
   * - 30 and above
     - Polygonal road elements

Source lane IDs are preserved through extraction. Non-lane map elements receive sequential IDs
after the largest lane ID. Reindexing normally converts every surviving map, agent, object, and
traffic-control ID to a contiguous zero-based range before serialization.

Enums
-----

Integer values are defined in ``src/bin_factory/puffer_types.py``:

* agents: other, vehicle, pedestrian, cyclist;
* control state: controllable, non-controllable moving, non-controllable static;
* lanes: unknown, freeway, surface street, bike lane, bus lane;
* traffic-light state: unknown, red, yellow, green, off;
* traffic control: traffic light, stop sign, yield sign;
* objects: traffic sign, traffic cone, traffic light, barrier, generic object.

Treat that module, not this prose, as the source of truth for numeric values.
