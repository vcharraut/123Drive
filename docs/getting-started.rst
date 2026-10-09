Getting started
===============

Requirements
------------

123Drive supports Python 3.11 through 3.13 and uses `uv <https://docs.astral.sh/uv/>`_ for
environment and command management. Install from a local checkout:

.. code-block:: console

   $ git clone https://github.com/vcharraut/123Drive.git
   $ cd 123Drive
   $ uv sync

The base environment contains the converter, Mapforge, and Docker build command. Install the
viewer dependencies only when needed:

.. code-block:: console

   $ uv sync --extra viz

First conversion
----------------

The input path must be a 123D output root containing ``logs/`` and ``maps/``:

.. code-block:: console

   $ uv run convert \
       --py123d_path /data/py123d \
       --output ./output

For a known dataset, prefer a :doc:`preset <presets>`:

.. code-block:: console

   $ uv run convert \
       --preset nuplan \
       --split_names nuplan-mini_val \
       --py123d_path /data/py123d \
       --output ./output

The command discovers matching scenes, converts them in parallel, and writes ``.bin`` files to
``./output``. Individual failures are recorded in ``output/failures.jsonl``. A partially
successful run exits successfully; a run where every scene fails exits with status 1.

Inspect the result
------------------

.. code-block:: console

   $ uv sync --extra viz
   $ uv run web --dir ./output --port 8080

Open ``http://localhost:8080``. The local viewer lists binaries recursively and renders maps,
agents, routes, objects, and traffic controls. See :doc:`viewer`.

Next steps
----------

* :doc:`input-data` explains the expected 123D data surface.
* :doc:`conversion` covers filters, parallelism, outputs, and failures.
* :doc:`pipeline` follows one scene through every processing stage.
* :doc:`docker` covers the raw-dataset-to-binary container workflow.
