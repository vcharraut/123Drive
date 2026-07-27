PufferDrive binary format
=========================

``bin_factory.serialize.scenario_to_binary`` writes a header followed by agents, map elements,
traffic controls, objects, lane-graph distances, and metadata. There is no magic number or
version field.

All multi-byte values and NumPy channels are explicitly little-endian. Integers are signed
32-bit. Floats are IEEE 754 32-bit unless noted. Variable-length arrays are prefixed by an
``int32`` count.

Header
------

.. list-table::
   :header-rows: 1

   * - Offset
     - Type
     - Field
   * - 0
     - ``int32``
     - ``n_agents``
   * - 4
     - ``int32``
     - ``n_road_elements``
   * - 8
     - ``int32``
     - ``n_traffic_controls``
   * - 12
     - ``int32``
     - ``n_objects``

Agents
------

Repeat the following block ``n_agents`` times:

.. list-table::
   :header-rows: 1
   :widths: 26 28 46

   * - Type
     - Field
     - Notes
   * - ``int32``
     - ``id``
     - Track ID
   * - ``int32``
     - ``type``
     - ``AgentType``
   * - ``int32``
     - ``T``
     - Trajectory frame count
   * - ``float32[T]``
     - ``x``, then ``y``, then ``z``
     - Channel-major positions
   * - ``float32[T]``
     - ``heading``
     - Radians
   * - ``float32[T]``
     - ``vx``, then ``vy``
     - Velocity channels
   * - ``float32[T]``
     - ``length``, then ``width``, then ``height``
     - Bounding-box channels
   * - ``int32[T]``
     - ``valid``
     - 1 where the source observation exists
   * - ``int32``
     - ``n_route_lanes``
     - Route length
   * - ``int32[n_route_lanes]``
     - ``route_lane_ids``
     - Ordered lane IDs
   * - ``int32``
     - ``route_gt_len``
     - Ground-truth-supported route prefix length
   * - ``float32[3]``
     - ``goal_x``, ``goal_y``, ``goal_z``
     - Position at the last valid frame; zero if none
   * - ``int32``
     - ``control_state``
     - ``ControlState``

Map elements
------------

Repeat the following common block ``n_road_elements`` times:

.. list-table::
   :header-rows: 1

   * - Type
     - Field
     - Notes
   * - ``int32``
     - ``id``
     - Map element ID
   * - ``int32``
     - ``type``
     - Road type
   * - ``int32``
     - ``N``
     - Geometry point count
   * - ``float32[N]``
     - ``x``, then ``y``, then ``z``
     - Channel-major polyline or polygon
   * - ``float32[N]``
     - ``heading``
     - Segment heading; final point repeats the final segment

Types 0 through 9 are lanes and append:

.. list-table::
   :header-rows: 1

   * - Type
     - Field
     - Notes
   * - ``int32``
     - ``n_entry_lanes``
     - Predecessor count
   * - ``int32[n_entry_lanes]``
     - ``entry_lane_ids``
     - Predecessors
   * - ``int32``
     - ``n_exit_lanes``
     - Successor count
   * - ``int32[n_exit_lanes]``
     - ``exit_lane_ids``
     - Successors
   * - ``float32``
     - ``speed_limit``
     - Metres per second; -1 when unknown
   * - ``float32``
     - ``length``
     - Total three-dimensional arc length
   * - ``float32[N]``
     - ``cum_length``
     - Per-point cumulative arc length

Traffic controls
----------------

Repeat the following block ``n_traffic_controls`` times:

.. list-table::
   :header-rows: 1

   * - Type
     - Field
     - Notes
   * - ``int32``
     - ``id``
     - Control ID
   * - ``int32``
     - ``type``
     - ``TCType``
   * - ``float32[6]``
     - ``stop_line``
     - Two XYZ endpoints
   * - ``float32``
     - ``heading``
     - Incoming travel heading
   * - ``int32``
     - ``n_states``
     - State count
   * - ``int32[n_states]``
     - ``states``
     - ``TLState`` per frame for lights
   * - ``int32``
     - ``n_controlled_lanes``
     - Controlled lane count
   * - ``int32[n_controlled_lanes]``
     - ``controlled_lane_ids``
     - Lane references

Objects
-------

Repeat the same dynamic-state prefix used by agents ``n_objects`` times: ``id``, ``type``, ``T``,
position, heading, velocity, dimensions, and valid channels. Objects do not append route, goal,
or control-state fields.

Lane graph
----------

.. list-table::
   :header-rows: 1

   * - Type
     - Field
     - Notes
   * - ``int32``
     - ``n_lanes_graph``
     - Zero means no graph data
   * - ``int32[n]``
     - ``lane_ids``
     - Matrix row and column order
   * - ``float32[n*n]``
     - ``distances``
     - Row-major all-pairs directed distances

Distances are precomputed over freeway and surface-street lanes. Each edge costs the source
lane's length. Unreachable pairs are IEEE 754 positive infinity.

Metadata
--------

Metadata always ends the file:

.. list-table::
   :header-rows: 1

   * - Type
     - Field
     - Notes
   * - ``char[128]``
     - ``id``
     - UTF-8, truncated and null-padded
   * - ``char[32]``
     - ``dataset``
     - UTF-8, truncated and null-padded
   * - ``int32``
     - ``scenario_length``
     - Frame count
   * - ``float32``
     - ``dt``
     - Seconds per frame
   * - ``int32``
     - ``n_objects_of_interest``
     - Auxiliary target count
   * - ``int32[n]``
     - ``objects_of_interest_ids``
     - Agent IDs
   * - ``int32``
     - ``n_tracks_to_predict``
     - Prediction target count
   * - ``int32[n]``
     - ``tracks_to_predict_ids``
     - Agent IDs

Enum values
-----------

.. list-table::
   :header-rows: 1

   * - Enum
     - Values
   * - ``AgentType``
     - 0 other, 1 vehicle, 2 pedestrian, 3 cyclist
   * - ``ControlState``
     - 0 controllable, 1 non-controllable moving, 2 non-controllable static
   * - ``LaneType``
     - 0 unknown, 1 freeway, 2 surface street, 3 bike lane, 4 bus lane
   * - ``TLState``
     - 0 unknown, 1 red, 2 yellow, 3 green, 4 off
   * - ``TCType``
     - 1 traffic light, 2 stop sign, 3 yield sign
   * - ``ObjectType``
     - 1 sign, 2 cone, 3 traffic light, 4 barrier, 5 generic object

Road lines occupy values 10--18, road edges 20--22, and miscellaneous road elements 30--32.
``src/bin_factory/puffer_types.py`` is the authoritative numeric definition.

Compatibility
-------------

Because the payload has no version marker, producers and consumers must update together when the
layout changes. Keep parsing order identical to serialization order. The browser implementation in
``src/viz/web/static/binary_parser.js`` and the static-map reader in
``src/mapforge/static_binary.py`` are useful independent readers.
