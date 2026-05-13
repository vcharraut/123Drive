# Changelog

Generated from tags on `main`.

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
