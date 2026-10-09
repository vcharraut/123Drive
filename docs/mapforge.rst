Static map augmentation
=======================

Mapforge creates affine variants of static PufferDrive map binaries. It accepts only binaries with
zero agents and zero objects.

.. code-block:: console

   $ uv run mapforge \
       --input_dir data/static_maps \
       --output_dir data/static_maps_aug

Transform catalog
-----------------

Transforms are grouped into:

.. list-table::
   :header-rows: 1

   * - Group
     - Variants
     - Matrices
   * - ``scale``
     - ``Sc10``, ``ScX10``, ``ScY10``
     - Uniform 10%, X-only 10%, and Y-only 10% scaling
   * - ``shear``
     - ``ShXP``, ``ShXN``, ``ShYP``, ``ShYN``
     - Positive and negative 0.17 X/Y shear
   * - ``flip``
     - ``FlipX``
     - Reflection across the global Y axis

When all groups are selected, ``FlipX`` is also composed with the seven scale and shear
transforms. Each input therefore produces 15 variants plus a copy of the original.

Select a subset:

.. code-block:: console

   $ uv run mapforge \
       --groups scale shear \
       --input_dir data/static_maps \
       --output_dir data/static_maps_warp

Processing
----------

For each source map, Mapforge:

#. parses map elements, traffic controls, lane graph, and metadata;
#. computes the mean XY coordinate across all map geometry;
#. applies each 2x2 matrix around that centroid;
#. transforms stop lines and recomputes control headings;
#. resamples geometry when a transform expands distances beyond 10 metre segments;
#. recomputes lane lengths and the all-pairs lane graph;
#. changes the scenario ID to the output stem;
#. serializes the result with the normal binary writer.

The source ``z`` coordinates, topology, types, speed limits, states, and remaining metadata stay
unchanged.

Output policy
-------------

The original is copied beside its variants:

.. code-block:: text

   town.bin
   town_Sc10.bin
   town_ShXP.bin
   town_FlipX.bin
   town_FlipX_Sc10.bin
   ...

Mapforge computes the full output plan before writing. Duplicate names or any existing destination
abort the run. Use ``--overwrite`` to replace existing outputs explicitly.

The static reader rejects truncated payloads, trailing bytes, and binaries containing agents or
objects. It does not support augmenting dynamic scenes.
