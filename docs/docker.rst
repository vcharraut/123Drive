Docker workflow
===============

The ``build`` command produces two reusable images for the complete raw-data pipeline:

.. code-block:: text

   raw dataset -- py123d-<dataset> --> 123D Arrow -- 123drive --> PufferDrive .bin
                  /input  /output                    /input       /output

Docker builds require BuildKit because both Dockerfiles use cache mounts.

Build images
------------

.. code-block:: console

   $ uv sync
   $ uv run build py123d --dataset nuplan-mini
   $ uv run build 123drive

The supported extractor images are:

.. list-table::
   :header-rows: 1

   * - Dataset
     - Image
     - 123D extra
   * - ``nuplan``
     - ``py123d-nuplan``
     - ``nuplan``
   * - ``nuplan-mini``
     - ``py123d-nuplan-mini``
     - ``nuplan``
   * - ``wod-motion``
     - ``py123d-wod-motion``
     - ``waymo``
   * - ``av2-sensor``
     - ``py123d-av2-sensor``
     - ``av2``

``--no_cache`` disables Docker's build cache. ``--dry_run`` prints commands without executing
them. ``--push REGISTRY`` tags the image under that registry and pushes it after a successful
build.

123D extractor
--------------

The extractor image pins 123D to ``v0.6.0``, matching the project's Python dependency. It forces
log and map conversion, disables shuffling, requires maps, and excludes camera, lidar, and custom
modalities for a BEV-oriented output.

.. code-block:: console

   $ docker run --rm \
       -v /data/nuplan:/input \
       -v /data/py123d_out:/output \
       --shm-size=10g \
       --ulimit nofile=1048576:1048576 \
       py123d-nuplan-mini \
       --splits nuplan-mini_train nuplan-mini_val

Runtime options are:

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Meaning
   * - ``--input`` / ``-i``
     - ``/input``
     - Raw dataset root
   * - ``--output`` / ``-o``
     - ``/output``
     - 123D Arrow root
   * - ``--splits NAME ...``
     - dataset defaults
     - Exact splits
   * - ``--worker_type``
     - ``ray``
     - ``ray``, ``process_pool``, or ``thread_pool``
   * - ``--workers``
     - 80% CPUs
     - Worker count for non-Ray executors

For nuPlan, mount the directory containing both ``maps/`` and ``nuplan-v1.1/``.

123Drive converter
------------------

The converter image installs the locked project into ``/app/123Drive``. Its entrypoint fixes the
input and output roots, then forwards all remaining arguments to ``convert``:

.. code-block:: console

   $ docker run --rm \
       -v /data/py123d_out:/input \
       -v /data/bins:/output \
       --shm-size=10g \
       --ulimit nofile=1048576:1048576 \
       123drive \
       --preset nuplan \
       --workers 8 \
       --validate_level 1

The image build context excludes repository datasets and outputs through ``.dockerignore``. The
runtime image contains the package, lockfile, license, and converter entrypoint.
