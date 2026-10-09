CLI reference
=============

The repository installs four commands: ``convert``, ``mapforge``, ``build``, and ``web``.

convert
-------

.. code-block:: console

   $ uv run convert --help

Input and selection
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Option
     - Default
     - Meaning
   * - ``--py123d_path PATH``
     - ``PY123D_DATA_ROOT``
     - 123D root containing ``logs/`` and ``maps/``
   * - ``--output PATH``
     - ``./output``
     - Binary output directory
   * - ``--preset NAME``
     - none
     - Defaults from ``presets.toml``; explicit CLI options override them
   * - ``--datasets NAME ...``
     - all
     - Dataset families
   * - ``--split_types TYPE ...``
     - all
     - Train, validation, or test-like split categories
   * - ``--split_names NAME ...``
     - all
     - Exact named splits
   * - ``--log_names NAME ...``
     - all
     - Exact logs
   * - ``--scene_uuids UUID ...``
     - all
     - Exact scene UUIDs
   * - ``--num_scenes N``
     - all
     - Maximum discovered scenes
   * - ``--scenario_id_field FIELD``
     - ``scene_uuid``
     - Metadata identity: ``scene_uuid``, ``log_name``, or ``location``
   * - ``--map_only``
     - off
     - Convert Arrow maps without logs

Timing and execution
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Option
     - Default
     - Meaning
   * - ``--duration_s SECONDS``
     - ``0``
     - Scenario duration; zero requests the full log
   * - ``--dt SECONDS``
     - ``0.1``
     - Target timestep
   * - ``--workers N``
     - ``0``
     - ``0`` = 80% CPUs, ``-1`` = all CPUs, ``1`` = sequential
   * - ``--chunk_target_scenes N``
     - ``10000``
     - Scenes per Joblib dispatch chunk
   * - ``--validate_level {0,1,2}``
     - ``1``
     - Off, schema, or semantic validation
   * - ``--log_level LEVEL``
     - ``INFO``
     - ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, or ``CRITICAL``

Transforms
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Option
     - Default
     - Meaning
   * - ``--max_segment_length METRES``
     - ``10.0``
     - Maximum processed polyline segment length
   * - ``--area_threshold METRES``
     - ``0.1``
     - Shapely simplification tolerance; zero disables it
   * - ``--min_route_valid_points PERCENT``
     - ``0``
     - Minimum post-check-frame valid percentage for non-ego routes
   * - ``--route_check_timestep INDEX``
     - ``0``
     - Required on-road valid frame for non-ego routes
   * - ``--no_reindex``
     - off
     - Preserve surviving source identifiers
   * - ``--interpolate_tl`` / ``--impute_tl``
     - off
     - Infer and correct light phases from vehicle motion
   * - ``--invalid_agent_overlap``
     - off
     - Invalidate unrouted actors overlapping active agents
   * - ``--reverse_road_edges``
     - off
     - Reverse road-edge point order for opposite source conventions

mapforge
--------

.. code-block:: console

   $ uv run mapforge --input_dir INPUT --output_dir OUTPUT [--groups GROUP ...] [--overwrite]

``GROUP`` is ``scale``, ``shear``, or ``flip``. See :doc:`mapforge`.

build
-----

.. code-block:: console

   $ uv run build py123d --dataset DATASET [--no_cache] [--dry_run] [--push REGISTRY]
   $ uv run build 123drive [--no_cache] [--dry_run] [--push REGISTRY]

The py123d dataset is ``nuplan``, ``nuplan-mini``, ``wod-motion``, or ``av2-sensor``.
See :doc:`docker`.

web
---

.. code-block:: console

   $ uv run web --dir OUTPUT [--port 8080]

``--dir`` is required and must exist. The server binds to localhost. See :doc:`viewer`.
