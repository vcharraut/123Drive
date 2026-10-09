123Drive
========

123Drive converts autonomous-driving datasets exposed through `123D
<https://github.com/autonomousvision/py123d>`_ into the compact binary format consumed by
`PufferDrive <https://github.com/Emerge-Lab/PufferDrive>`_.

The converter extracts trajectories and maps, normalizes them into a common scenario model,
computes simulator-specific data such as routes and lane distances, validates the result, and
writes one ``.bin`` file per scenario.

.. code-block:: text

   raw dataset -> 123D Arrow -> extraction -> transforms -> validation -> PufferDrive .bin
                                       |                         |
                                       +---- in-memory model ----+

Start with :doc:`getting-started`. Read :doc:`pipeline` to understand the implementation and
:doc:`binary-format` when consuming the generated files.

.. toctree::
   :maxdepth: 2
   :caption: Use 123Drive

   getting-started
   input-data
   conversion
   cli-reference
   presets

.. toctree::
   :maxdepth: 2
   :caption: Understand the converter

   architecture
   data-model
   pipeline
   validation
   routes
   traffic-lights
   binary-format

.. toctree::
   :maxdepth: 2
   :caption: Tools

   mapforge
   viewer
   docker
   development
