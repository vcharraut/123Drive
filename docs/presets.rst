Dataset presets
===============

Presets apply dataset-specific defaults from ``src/bin_factory/presets.toml``. They pin a dataset
family and known corrections, not a split:

.. code-block:: console

   $ uv run convert \
       --preset nuplan \
       --split_names nuplan-mini_val \
       --py123d_path /data/py123d \
       --output ./output

Explicit CLI values override preset defaults. Boolean ``store_true`` options enabled by a preset
cannot currently be disabled from the same command line.

.. list-table::
   :header-rows: 1
   :widths: 16 20 16 48

   * - Preset
     - Dataset
     - Identity
     - Additional defaults
   * - ``av2``
     - ``av2-sensor``
     - ``log_name``
     - 15 s duration; overlap invalidation
   * - ``carla``
     - ``carla``
     - ``log_name``
     - Reverse road edges
   * - ``nuplan``
     - ``nuplan``
     - ``scene_uuid``
     - 20 s duration; reverse edges; overlap invalidation
   * - ``nuplan-mini``
     - ``nuplan-mini``
     - ``scene_uuid``
     - nuPlan corrections plus traffic-light interpolation
   * - ``nuscenes``
     - ``nuscenes``
     - ``scene_uuid``
     - Reverse edges; overlap invalidation
   * - ``opendrive``
     - ``opendrive``
     - ``location``
     - Reverse edges; explicit selection forces map-only conversion
   * - ``wod-motion``
     - ``wod-motion``
     - ``log_name``
     - Overlap invalidation; traffic-light interpolation

Why presets matter
------------------

Raw nuPlan logs can be much longer than simulator scenarios; the duration limit prevents excessive
memory use. Road-edge direction differs for several sources. Overlap invalidation removes
log-only actors that conflict with controllable replay actors. Traffic-light interpolation fills
known signal gaps for selected datasets.

Use a preset whenever one exists. Override only the values required by a specific experiment.
