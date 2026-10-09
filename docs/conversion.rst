Running conversions
===================

Basic command
-------------

.. code-block:: console

   $ uv run convert --py123d_path /data/py123d --output ./output

``PY123D_DATA_ROOT`` can replace ``--py123d_path``:

.. code-block:: console

   $ PY123D_DATA_ROOT=/data/py123d uv run convert --output ./output

Filtering
---------

Filters can be combined. List-valued options accept one or more values:

.. code-block:: console

   $ uv run convert \
       --py123d_path /data/py123d \
       --output ./output \
       --datasets wod-motion \
       --split_types val \
       --log_names 0123 0456 \
       --num_scenes 100

Use ``--scene_uuids`` for exact debugging targets. ``--split_names`` is more precise than
``--split_types`` when a dataset exposes several named splits.

Timing
------

``--dt`` controls the requested sample period. Its default, ``0.1``, produces 10 Hz
trajectories. ``--duration_s`` limits scene duration; zero means the full scene.

Raw nuPlan logs can span minutes. Converting them without a duration can exhaust memory. The
``nuplan`` and ``nuplan-mini`` presets use 20 seconds.

Parallel execution
------------------

``--workers`` has four modes:

.. list-table::
   :header-rows: 1

   * - Value
     - Behavior
   * - ``0``
     - 80% of available CPU cores
   * - ``1``
     - Sequential conversion
   * - ``-1``
     - Every available CPU core
   * - ``N``
     - Exactly ``N`` workers

The converter reduces the worker count when ``--num_scenes`` is smaller. Scenes are dispatched
in chunks of ``--chunk_target_scenes`` (default 10,000) to avoid constructing one unbounded
Joblib task graph. Results are consumed out of order and update a shared progress bar.

Output identity
---------------

Names include the dataset so outputs from several sources can share one directory:

.. list-table::
   :header-rows: 1

   * - Mode
     - Filename
   * - Default scene identity
     - ``<dataset>__<scene_uuid>.bin``
   * - ``log_name`` or ``location`` identity
     - ``<dataset>__<selected_id>__<scene_uuid>.bin``
   * - Map-only
     - ``<dataset>__<location>.bin``

Unsafe filename characters become underscores. Before conversion starts, every destination is
computed and duplicate destinations abort the run.

Each completed payload is written to a process-specific temporary file beside its destination,
then atomically replaces the final path. Existing final files are replaced; unrelated files are
untouched.

Failures and exit status
------------------------

Worker exceptions do not stop other scenes. Each failure produces one JSON object in
``failures.jsonl`` with ``dataset``, ``log_name``, ``scenario_id``, and ``error`` fields.

The exit status is:

* ``0`` when at least one scene succeeds, including partial success;
* ``0`` when discovery finds no scenes;
* ``1`` when scenes were discovered but none succeeded;
* nonzero immediately for invalid arguments, output-name collisions, or other setup failures.
