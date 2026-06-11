# Changelog

Generated from tags on `main`.

## v0.3.2 - 2026-06-11

Compared with `v0.3.1`.
Source: `v0.3.1..v0.3.2` - 28 commits, 54 files changed.

### Added

- Dataset presets: `--preset` flag backed by `src/bin_factory/presets.toml`
  (`av2`/`carla`/`nuplan`/`nuscenes`/`opendrive`/`wod-motion`). A preset pins the
  dataset family and reproducible defaults; explicit CLI flags still override.
- `--dt` flag for iteration timestep (default `0.1` = 10 Hz), threaded into scene
  discovery as `target_iteration_duration_s` and `timestamp_threshold_s`.
- `--reverse_road_edges` flag and transform reversing road-edge polyline order
  (Waymo convention) for nuplan/carla/opendrive.
- `--scenario_id_field` flag (`scene_uuid`/`log_name`/`location`, default
  `scene_uuid`) selecting the py123d attribute used as the scenario id for both
  `metadata.id` and the output filename; each preset pins a per-dataset default.
- Prediction targets: extract `objects_of_interest` and `tracks_to_predict` from the
  WOD-Motion `aux` modality into `ScenarioMetadata`, remap them during reindex, and
  serialize them into binary metadata (previously hard-coded empty lists).
- `transforms/pipeline.py`: ordered, config-gated processing pipeline with
  `build_stages` / `run` as the single `process_scenario` entry point.
- Geometry length primitives `arc_length` and `polyline_length`.
- Docker build `--push` to tag and push either image to a registry.
- CI workflow (`pre-commit` + `pytest` on Python 3.11/3.13) and a ruff
  `.pre-commit-config.yaml`.
- Test suite under `tests/`: pipeline, schema, serialize, geometry, reindex,
  sanitize, signal-phase, traffic-controls, traffic-light interpolation, plus
  real-data end-to-end tests with Arrow fixtures.

### Changed

- Renamed `--impute_tl` to `--interpolate_tl` (`--impute_tl` kept as an alias);
  module `traffic_lights_imputation.py` → `traffic_light_interpolation.py` and
  `impute_traffic_lights` → `interpolate_traffic_lights`.
- Map elements are now `MapElement` / `StopZone` dataclasses (with `is_lane`/
  `is_line`/`is_edge`/`is_crosswalk`/`uses_polyline` predicates and `geometry`/
  `min_points` helpers) instead of dicts; extras are an `ExtractionExtras`
  dataclass. Threaded through extractor, validation, every transform, and serialize.
- Map extraction queries an ego-path corridor buffer (`SCENE_MAP_MARGIN`, via
  shapely `intersects`) instead of a fixed radius around the centroid; ego states
  are now required for non-map-only scenarios.
- Polygon densification uses shapely `segmentize`; default spacing `3.0` → `5.0` m.
- `AGENT_TYPE_MAP` maps the `TWO_WHEELER` label (was `BICYCLE`) to `CYCLIST`.
- Traffic-control processing skips traffic lights whose controlled lane is a bike lane.
- Lane arc-length consolidated: `compute_lane_arc_lengths` removed in favor of the
  shared `geometry.arc_length`.
- `_convert_one` now delegates all processing to `transforms.run`.
- Scenario id derivation is driven by `--scenario_id_field` (and per-preset
  defaults) instead of the hard-coded dataset-specific fields added in v0.3.1;
  `metadata.id` now matches the output filename, and map-only scenarios use
  `location`.
- Conversion tolerates partial failures: exit `0` if any scenario converted, `1`
  only if none did (was `1` whenever any scenario failed).
- Docker entrypoint exports `PY123D_DATA_ROOT=/input` so convert workers resolve maps.
- Pinned `py123d` `0.2.1` → `0.5.1` (pyproject, Dockerfile, docker README default);
  bumped project version to `0.3.2` and packaged `presets.toml` as package data.

### Removed

- `docs/data.md` (supported data surface) and its README link.

### Docs

- Binary format doc now points to `puffer_types.py` as the single source of truth
  for enum values instead of inlining them.
- README: added a Presets section and `--preset` / `--dt` / `--interpolate_tl` flags.

## v0.3.1 - 2026-05-21

Compared with `v0.3`.
Source: `v0.3..v0.3.1` - 2 commits, 2 files changed.

### Changed

- Scenario output filenames now derive identity from dataset-specific fields:
  `scene_uuid` for nuPlan, `location` for OpenDRIVE, and `log_name` otherwise.

## v0.3 - 2026-05-13

Compared with `v0.2`.
Source: `v0.2..v0.3` - 14 commits, 24 files changed.

### Added

- Added per-lane `length` and `cum_length` values after polyline processing.
- Serialized lane arc-length data directly in each lane record.
- Added viewer parsing and info-panel display for lane length and cumulative point distance.
- Added `--log_level` CLI flag.
- Added `--invalid_agent_overlap` CLI flag to zero log-only agents whose bbox overlaps an active replay agent.
- Added road-edge Z-overlap validation for sub-4m bridge-like crossings.
- Added shared `MAP_REF_KEYS` schema constant for map reference remapping.

### Changed

- Changed `--min_route_valid_points` from an absolute sample count to a `0`-`100` percentage.
- Route eligibility now computes the minimum valid sample count from the checked horizon.
- Stationary off-road route filtering is stricter: threshold changed from `1.0m` to `0.3m`.
- Stationary/moving classification now uses trajectory extent around the median instead of summed displacement.
- Lane graph serialization no longer stores a separate `lane_lengths` array; consumers use each lane's `length`.
- Refactored dynamic-state binary writing into a shared serializer helper.
- Centralized logging on the `bin_factory` logger with dataset/scenario context.
- Traffic-light imputation now logs at debug level for generated lights.
- Traffic-light imputation no longer hard-gates on dataset name.
- Traffic-light topology symmetrization now uses original neighbor/diverge/merge snapshots.
- Traffic-light vehicle-to-lane assignment was vectorized and made tolerant of missing phases.
- Reindexing now remaps references without rewriting embedded object IDs.
- Scenario conversion workers now use parsed args directly instead of the removed `ConvertConfig` wrapper.
- Viewer scenario fetches now bypass browser/server cache.
- Docker docs now recommend a high `nofile` ulimit for conversion runs.

### Docs

- Updated binary format docs for lane `length`/`cum_length` and removed lane graph `lane_lengths`.
- Updated route algorithm docs for the eligibility gate and percentage-based route validity.
- Updated README flag tables for `--log_level`, `--min_route_valid_points`, and `--invalid_agent_overlap`.

## v0.2 - 2026-05-13

Compared with `v0.1`.
Source: `v0.1..v0.2` - 1 commit, 4 files changed.

### Added

- Passed `PYTHON_VERSION` as a build argument when building the `123drive` Docker image.

### Changed

- Updated both `123drive` and `py123d` Docker image defaults from Python `3.12` to `3.13`.
- Removed redundant `pytest` installation from the `py123d` Dockerfile's `nuplan` extras path.
- Simplified scenario identity fields from `scene_uuid`, `scenario_id`, `log_name`, `location`, `map_id` to `log_name`, `scene_uuid`, `location`.

## v0.1 - 2026-04-21

Initial tagged release.
Source: repository start through `v0.1`.

### Added

- Added Arrow-to-PufferDrive binary conversion pipeline under `src/bin_factory`.
- Added py123d loading, scene discovery, extraction, validation, route computation, traffic-control conversion, reindexing, sanitization, and serialization.
- Added local FastAPI viewer under `src/viz` with static JS/CSS frontend and binary parsing.
- Added Docker tooling for `py123d` extraction and `123drive` conversion images.
- Added documentation for the binary format, data surface, route algorithm, and Docker workflows.
- Added CLI entry points for conversion, viewer serving, and Docker image builds.
- Added MIT license and uv project metadata/lockfile.
