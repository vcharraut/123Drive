Local web viewer
================

The viewer is a local FastAPI application with a plain JavaScript/deck.gl frontend. It reads
PufferDrive binaries in the browser and requires no external web assets.

Start it
--------

.. code-block:: console

   $ uv sync --extra viz
   $ uv run web --dir ./output --port 8080

Open ``http://localhost:8080``. The server binds to localhost and recursively lists ``.bin`` files
under ``--dir``.

Interface
---------

The viewer provides:

* scenario filtering and selection;
* map, agent, object, trajectory, traffic-control, and identifier layer toggles;
* 2D and 3D views, fit-to-scene, follow-ego, zoom, orbit, and theme controls;
* timeline playback at 0.5x, 1x, 2x, or 4x;
* element selection and ID-based search;
* lane path finding using the serialized lane-distance graph;
* a distance ruler and point/line geometry rendering;
* MP4 recording when browser recording support and server-side FFmpeg are available.

The browser fetches enum names and colors from the Python source, so displayed types remain aligned
with the serializer.

HTTP surface
------------

.. list-table::
   :header-rows: 1

   * - Endpoint
     - Purpose
   * - ``GET /api/types``
     - Enum names, numeric ranges, and display colors
   * - ``GET /api/scenarios``
     - Sorted paths relative to the configured scenario directory
   * - ``GET /api/scenario/{path}``
     - One binary payload
   * - ``POST /api/export/mp4``
     - Transcode a browser recording with FFmpeg

Scenario paths are resolved beneath the configured directory and must end in ``.bin``. Static
HTML, CSS, JavaScript, and scenario responses disable browser caching during development.
A content-security policy permits only bundled assets and same-origin requests.

Recording
---------

The browser captures the rendered canvas, then uploads WebM or MP4 data for H.264 transcoding.
The server requires ``ffmpeg`` on ``PATH``. Uploads are limited to 250 MiB, encoding is limited to
120 seconds, and only one recording can encode at a time.

Recording is an optional inspection feature. Normal viewing does not require FFmpeg.
