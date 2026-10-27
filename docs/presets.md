# Dataset differences

`--preset NAME` selects a dataset and applies the defaults below. Other options
keep their [CLI defaults](cli-reference.md). Explicit arguments override preset
values. Boolean flags enabled by a preset cannot currently be disabled from the CLI.

| Preset | Dataset | Scenario identity | Processing defaults |
| --- | --- | --- | --- |
| `av2` | `av2-sensor` | `log_name` | 15 s duration |
| `carla` | `carla` | `log_name` | Reverse road edges; speed limits from signs |
| `griffin` | `griffin` | `log_name` | Reverse road edges; speed limits from signs; infer traffic lights |
| `nuplan` | `nuplan` | `scene_uuid` | 20 s duration; reverse road edges; infer traffic lights |
| `nuplan-map` | `nuplan` | `location` | Map-only city maps; reverse road edges |
| `nuplan-mini` | `nuplan-mini` | `scene_uuid` | 20 s duration; reverse road edges; infer traffic lights |
| `nurec` | `nurec-2604` | `log_name` | Infer traffic lights |
| `nuscenes` | `nuscenes` | `scene_uuid` | Reverse road edges |
| `opendrive` | `opendrive` | `location` | Map-only; reverse road edges; speed limits from signs |
| `wod-motion` | `wod-motion` | `log_name` | Infer traffic lights |

Road-edge reversal adapts the source direction convention. CARLA and OpenDRIVE use
the last speed sign and default to 30 km/h before the first sign. OpenDRIVE towns
without exported signs (Town13) keep road speed limits. OpenDRIVE keeps on-road
static objects. Off-road shoulder edges follow
lane boundaries; ramp widening gives vehicles more room on tight curves.

NuRec scenes start at their second frame, since py123d places the first one before the first
recorded pose.

Waymo Motion also preserves `objects_of_interest` and `tracks_to_predict`
when the source provides them. Every dataset uses the same route eligibility rules.

Dataset extraction and source map construction belong to py123d. Changes there
require regenerating Arrow inputs before converting binaries.
See [preset values] and [extraction code] for implementation details.

[preset values]: https://github.com/vcharraut/123Drive/blob/main/src/bin_factory/presets.toml
[extraction code]: https://github.com/vcharraut/123Drive/blob/main/src/bin_factory/loader/extractor.py
