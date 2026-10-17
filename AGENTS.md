# Repository instructions

123Drive converts 123D Arrow datasets into `.bin` scenarios for PufferDinde.

## Working rules

- Keep changes scoped and preserve unrelated work.
- Prefer small functions in existing pipeline stages over new wrappers or utility modules.
- Match the edited file's style: four spaces, double quotes, and a 120-character line limit.
- Run formatting and lint tools only when requested.

## Code and documentation

- `src/bin_factory/loader/` extracts and validates source data. `preparation.py` owns transform order and configuration gates.
- `src/bin_factory/schema.py` defines the scenario model; `serialize.py` defines binary encoding and static decoding.
- `src/bin_factory/transforms/` owns routes, traffic lights, geometry, and lane distances. Use the existing map-element predicates instead of repeating type-range checks.
- `src/mapforge/` augments static maps; `src/viz/` serves the scenario viewer; `docker_tools/` builds extractor and converter images.
- `tests/` contains unit and integration tests, with committed Arrow fixtures in `tests/py123d_data/`.
- `docs/index.md`, `docs/presets.md`, and `docs/cli-reference.md` are the user documentation. Keep them brief and update them when behavior or options change. Code is the source for implementation details. Documentation is plain Markdown without a website build.

## Agent eligibility and binary compatibility

- A nonempty `route` means eligible for control; an empty route means background. The binary has no `control_state` or `route_gt_len`.
- PufferDinde chooses controllers and background behavior. Conversion preserves recorded trajectories and validity masks.
- When changing the binary layout, keep the writer, viewer parser, and test readers consistent. Existing bins and external simulator readers may need corresponding updates.

## Development and verification

Use `uv` with Python 3.11 to 3.13:

```bash
uv sync --extra all
uv run convert --preset nuplan --py123d_path /data/123d --output ./output
uv run web --dir ./output --port 8080
```

After every change, including documentation changes, run `uv run --extra all pytest -q` before reporting completion.
Fix failures caused by the change. Report existing failures or execution blockers with the command and outcome.
Never claim tests passed without a passing run.

The `py123d` dependency currently uses the editable sibling checkout `../py123d` through `pyproject.toml`.
Inspect that source when investigating upstream behavior. Keep edits within this repository unless asked otherwise.
Mark dataset-specific source workarounds with `# PY123D-REPORT[dataset]: description`.

For pull requests, describe the resulting behavior and verification. Include viewer screenshots when the UI changes.
