Development
===========

Repository layout
-----------------

.. code-block:: text

   src/bin_factory/
     loader/          123D discovery, mapping, extraction, validation
     transforms/      ordered scenario transformations
     main.py          convert CLI and worker orchestration
     schema.py        in-memory data model
     serialize.py     binary writer
     presets.toml     dataset defaults
   src/mapforge/      static binary reader and affine augmentation
   src/viz/           FastAPI server and bundled browser frontend
   docker_tools/      image builder, Dockerfiles, entrypoints
   docs/              Sphinx source
   tests/             focused package and end-to-end coverage

Environment
-----------

.. code-block:: console

   $ uv sync --extra all

Useful commands:

.. code-block:: console

   $ uv run pytest
   $ uv run ty check src/
   $ uv run ruff check .

Keep conversion changes in the stage that owns them. The transform order is part of behavior, not
presentation. When changing a stage, verify its required predecessors and all references consumed
by later validation and serialization.

Build the documentation
-----------------------

.. code-block:: console

   $ uv sync --extra docs
   $ uv run sphinx-build -W --keep-going -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` locally. ``-W`` turns Sphinx warnings into build failures;
``--keep-going`` reports all warnings in one pass.

Publishing
----------

Pushing documentation changes to ``main`` runs ``.github/workflows/docs.yml``. It installs the
locked documentation environment, builds HTML, uploads the static artifact, and deploys it to
GitHub Pages.

Repository administrators must select **GitHub Actions** once under **Settings > Pages > Build and
deployment > Source**. The resulting site is:

``https://vcharraut.github.io/123Drive/``

Documentation policy
--------------------

Keep the root ``README.md`` as a short project entry point. All guides, architecture notes, CLI
details, algorithms, and format contracts belong in these reStructuredText pages. ``CHANGELOG.md``
remains Markdown because it is consumed directly from the repository and release tooling.

When behavior changes:

* update the owning page in the same change;
* update :doc:`binary-format` for layout changes;
* update :doc:`routes` or :doc:`traffic-lights` for algorithm changes;
* update :doc:`cli-reference` and :doc:`presets` for user-facing defaults.
