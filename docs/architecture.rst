Architecture
============

123Drive is a pipeline rather than a general-purpose framework. Four small packages own distinct
parts of the workflow:

.. list-table::
   :header-rows: 1

   * - Package
     - Responsibility
   * - ``bin_factory.loader``
     - Discover 123D inputs, map source types, extract the in-memory scenario, and validate it
   * - ``bin_factory.transforms``
     - Run the fixed, order-dependent scenario processing stages
   * - ``bin_factory.serialize``
     - Encode the processed scenario as the PufferDrive binary layout
   * - ``mapforge``
     - Read static binaries, apply affine map augmentation, and write variants
   * - ``viz``
     - Serve binaries and render them in the local browser viewer
   * - ``docker_tools``
     - Build the 123D extractor and 123Drive converter images

End-to-end flow
---------------

.. code-block:: text

   convert CLI
      |
      +-- apply preset, validate arguments
      +-- discover SceneAPI / MapAPI objects
      +-- precompute output paths and reject collisions
      |
      +-- worker process --------------------------------------------+
      |      extract_scenario()                                     |
      |          |                                                   |
      |          +-- PufferScenario + ExtractionExtras               |
      |          +-- pre-transform validation                        |
      |          +-- ordered transform pipeline                      |
      |          +-- post-transform validation                       |
      |          +-- little-endian serialization                     |
      |          +-- atomic destination replacement                  |
      +--------------------------------------------------------------+
      |
      +-- failures.jsonl + process exit status

Why two extraction outputs?
---------------------------

``PufferScenario`` contains data that survives into the binary. ``ExtractionExtras`` contains
source-level traffic-light tracks and stop zones needed while transforming the scenario.
Traffic-control processing consumes those extras and creates the final serialized control
records. This prevents temporary source representations from leaking into the output schema.

Mutation and ordering
---------------------

Transforms mutate one scenario in place. Their order is load-bearing:

* traffic-light interpolation needs original lane topology and source detections;
* lane lengths must describe the processed, serialized geometry;
* route computation must happen before overlap filtering;
* lane-graph distances depend on final lane lengths;
* reindexing changes every identifier and therefore runs last.

The exact sequence is documented in :doc:`pipeline`.

External dependencies
---------------------

The runtime uses:

* 123D for dataset-independent scene and map access;
* NumPy for trajectory and geometry arrays;
* SciPy for sparse lane graphs and Dijkstra distances;
* Shapely for geometry queries, simplification, interpolation, and intersections;
* Joblib for parallel conversion;
* tqdm for progress display.

The optional viewer adds FastAPI, Uvicorn, and a vendored deck.gl browser bundle. It does not
load scripts, fonts, or other assets from the network.
