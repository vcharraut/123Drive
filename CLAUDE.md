# 123Drive

123Drive converts 123D Arrow datasets into `.bin` scenarios for the Macadam simulator.

## Map

- `src/bin_factory/loader/` extracts and validates source data; `preparation.py` owns transform order and the configuration gates in `ConversionSettings`.
- `src/bin_factory/schema.py` defines the scenario model; lane `length` and `cum_length` derive from the polyline. `serialize.py` owns the binary: `scenario_to_binary` writes it and derives the lane graph, and `binary_to_scenario` reads any `.bin` back. Read bins through these.
- `src/bin_factory/conversion/` owns geometry, sanitization, and traffic controls; `src/bin_factory/generators/` owns routes, lane distances, shoulder edges, and traffic-light inference. Use the map-element predicates (`is_lane`, `is_line`, `is_edge`) instead of repeating type-range checks.
- `src/mapforge/` builds static maps: `affine.py` augments them, `town.py` generates procedural towns. `src/viz/` serves the scenario viewer. `docker_tools/` builds extractor and converter images.
- `external/py123d/` is the py123d submodule, described below.
- `tests/bin_factory/` and `tests/mapforge/` hold the tests, with committed Arrow fixtures in `tests/py123d_data/`.
- `docs/index.md`, `docs/presets.md`, and `docs/cli-reference.md` are the user documentation: brief, plain Markdown, updated when behavior or options change. CLI entry points are `[project.scripts]` in `pyproject.toml`.
- `DATA_REVIEW.md` tracks open work on the converted binaries: open items only, so delete an item when it is done.

## py123d

`external/py123d` is a git submodule: the fork `vcharraut/py123d` on branch `vcha`, installed editable, so an edit there takes effect at once. Edit it when a task needs py123d to change; read it when investigating upstream behavior.

- Commit py123d changes inside the submodule on `vcha` (a fresh clone starts detached: `git -C external/py123d checkout vcha`), then stage the new pointer here with `git add external/py123d`.
- Push py123d before 123Drive, so the recorded commit exists on the fork.
- A workaround kept in 123Drive for dataset-specific source behavior is marked `# PY123D-REPORT[dataset]: description`. A map correction goes in `clean_map` (`conversion/sanitize.py`), tested on hand-built lanes.

## Binary contract

- A nonempty `route` means eligible for control; an empty route means background. The binary has no `control_state` or `route_gt_len`.
- Macadam chooses controllers and background behavior. Conversion preserves recorded trajectories and validity masks.
- A binary layout change touches `serialize.py` (writer and reader) and the viewer parser (`src/viz/web/static/binary_parser.js`) together; the Node test in `tests/bin_factory/test_serialize.py` compares the two readers. Macadam's reader and existing bins need the matching update.

## Working rules

- Add logic as small functions in the existing pipeline stages rather than new wrappers or utility modules.
- `src/` and `docker_tools/` carry type hints; `tests/` is exempt. CI enforces this with ruff `ANN` and `uv check --extra all`, and `.pre-commit-config.yaml` formats and lints with ruff on commit.
- A check worth running twice becomes a test in `tests/`; a defect found by eye in the viewer gets a test with its fix.
- When a change alters map or scenario output, write sample bins to `output/<topic>/` and report the viewer command: `uv run web --dir output/<topic>`.

## Verification

Use `uv` with Python 3.12 or 3.13: `uv sync --extra all`.

After every change, including documentation changes, run `uv run --extra all pytest -q` before reporting completion. Fix failures caused by the change. Report pre-existing failures and blockers with the command and its outcome. State that tests pass only after a passing run.

For pull requests, describe the resulting behavior and verification, with viewer screenshots when the UI changes.
