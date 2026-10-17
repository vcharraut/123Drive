# Command options

Run commands with `uv run COMMAND`. Every command accepts `--help` or `-h`.
Defaults below apply without a [preset](presets.md).

## convert

```bash
uv run convert --preset nuplan --py123d_path /data/123d --output ./output
```

### Input and selection

| Option | Default | Meaning |
| --- | --- | --- |
| `--py123d_path PATH` | `PY123D_DATA_ROOT` | 123D root containing logs/ and maps/; required if the environment variable is unset |
| `--output PATH` | `./output` | Binary output directory |
| `--preset NAME` | none | Dataset and processing defaults |
| `--datasets NAME ...` | all | Dataset families |
| `--split_types TYPE ...` | all | Split categories such as train, val, or test |
| `--split_names NAME ...` | all | Exact split names |
| `--log_names NAME ...` | all | Exact log names |
| `--scene_uuids UUID ...` | all | Exact scene UUIDs |
| `--num_scenes N` | all | Maximum number of scenes |
| `--scenario_id_field FIELD` | `scene_uuid` | Metadata identity and filename component: scene_uuid, log_name, or location |
| `--map_only` | off | Convert static maps without logs; automatic when selecting the opendrive dataset |

### Timing and execution

| Option | Default | Meaning |
| --- | --- | --- |
| `--duration_s SECONDS` | `0` | Scenario duration; zero keeps the full available scene |
| `--dt SECONDS` | `0.1` | Target timestep |
| `--workers N` | `0` | 0 uses 80% of CPUs; -1 uses all; 1 runs sequentially; positive N uses N workers |
| `--chunk_target_scenes N` | `10000` | Scenes per conversion dispatch chunk |
| `--validate_level {0,1,2}` | `1` | 0 disables validation; 1 checks schema; 2 adds semantic checks |
| `--log_level LEVEL` | `INFO` | DEBUG, INFO, WARNING, ERROR, or CRITICAL |

### Transforms

| Option | Default | Meaning |
| --- | --- | --- |
| `--max_segment_length METRES` | `10.0` | Maximum processed polyline segment length |
| `--edge_max_segment_length METRES` | `1.0` | Maximum road-edge segment length; zero disables interpolation |
| `--area_threshold METRES` | `0.1` | Polyline simplification tolerance; zero disables simplification |
| `--route_check_timestep INDEX` | `0` | Reference frame for route eligibility, remaining distance, and initial overlaps |
| `--no_reindex` | off | Keep surviving source identifiers instead of assigning contiguous IDs |
| `--interpolate_tl` / `--impute_tl` | off | Infer traffic-light states from observations, topology, and vehicle motion |
| `--reverse_road_edges` | off | Reverse road-edge point order |
| `--speed_limit_source {road,signs}` | `road` | Use road speed limits or the last speed sign |
| `--static_objects {all,onroad}` | `onroad` | Keep every static map object or only those on the road |
| `--shoulders {drivable,offroad}` | `drivable` | With offroad, add road edges beside shoulders outside junctions |
| `--ramp_widening_m METRES` | `0.0` | With off-road shoulders, move ramp edges outward by up to this distance, retaining 1.5 m to the far shoulder boundary; allowed range 0 to 5 |

## web

Install with `uv sync --extra viz`. The viewer serves binaries at <http://localhost:8080>.

| Option | Default | Meaning |
| --- | --- | --- |
| `--dir PATH` | required | Directory containing binaries |
| `--port N` | `8080` | Local server port |

## mapforge

Generate affine variations of static map binaries.

| Option | Default | Meaning |
| --- | --- | --- |
| `--input_dir PATH` | required | Source map binaries |
| `--output_dir PATH` | required | Augmented map binaries |
| `--groups GROUP ...` | all | Transform groups: scale, shear, flip |
| `--overwrite` | off | Replace existing output files |

## build

`uv run build py123d --dataset NAME` builds an extractor image.
`uv run build 123drive` builds the converter image.

| Option | Default | Meaning |
| --- | --- | --- |
| `--dataset NAME` | required for py123d | nuplan, nuplan-mini, wod-motion, or av2-sensor |
| `--no_cache` | off | Disable Docker build cache |
| `--dry_run` | off | Print commands without running them |
| `--push REGISTRY` | none | Tag and push the built image to this registry |

### Extractor container

The py123d image converts raw datasets into Arrow inputs. Mount the raw dataset at
`/input` and the Arrow output directory at `/output`. For nuPlan, the input
must contain both `maps/` and `nuplan-v1.1/`.

| Runtime option | Default | Meaning |
| --- | --- | --- |
| `--input PATH` / `-i PATH` | `/input` | Raw dataset root |
| `--output PATH` / `-o PATH` | `/output` | Arrow output root |
| `--splits NAME ...` | dataset defaults | Exact splits to extract |
| `--worker_type TYPE` | `ray` | ray, process_pool, or thread_pool |
| `--workers N` | 80% of CPUs | Worker count for non-Ray executors |

The 123drive container uses the same mount paths and forwards options to `convert`.
See the [command parsers] and [container entrypoints] for details.

[command parsers]: https://github.com/vcharraut/123Drive/blob/main/src/bin_factory/main.py
[container entrypoints]: https://github.com/vcharraut/123Drive/tree/main/docker_tools
