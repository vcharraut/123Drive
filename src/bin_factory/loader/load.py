from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from py123d.api import MapAPI, SceneFilter, get_filtered_scenes
from py123d.api.map.arrow.arrow_map_api import ArrowMapAPI
from py123d.common.execution import SequentialExecutor


@dataclass(frozen=True)
class MapOnlyScenario:
    scenario_id: str
    map_api: MapAPI
    split_name: str | None = None


def get_py123d_scenarios(
    py123_data_root: str,
    num_scenes: int | None = None,
    datasets: list[str] | None = None,
    split_types: list[str] | None = None,
    split_names: list[str] | None = None,
    log_names: list[str] | None = None,
    duration_s: float | None = None,
    history_s: float | None = 0.0,
    map_api_required: bool = True,
    map_only: bool = False,
) -> list[Any]:
    """Load py123d scenarios from Arrow logs and/or maps.

    Args:
        py123_data_root: Root path to py123d_data (contains logs/ and maps/) or a directory of .arrow maps.
        num_scenes: Optional cap on number of scenes.
        datasets: Optional list of dataset names to include (e.g. ["nuplan", "wod-motion"]).
        split_types: Optional list of split types (train/val/test).
        split_names: Optional list of split names (e.g. ["nuplan-mini_val"]).
        log_names: Optional list of log names to include.
        duration_s: Optional duration for scene extraction; None uses full log.
        history_s: Optional history duration (seconds).
        map_api_required: Whether to only include scenes with map APIs.
        map_only: If True, load map-only scenarios (no logs).

    Returns:
        List of ArrowSceneAPI or MapOnlyScenario.
    """
    data_root = Path(py123_data_root)

    if map_only:
        return _load_map_only_scenarios(data_root)

    scene_filter = SceneFilter(
        datasets=datasets,
        split_types=split_types,
        split_names=split_names,
        log_names=log_names,
        duration_s=duration_s,
        history_s=history_s,
        map_api_required=map_api_required,
        max_num_scenes=num_scenes,
    )

    return get_filtered_scenes(
        scene_filter=scene_filter,
        data_root=data_root,
        executor=SequentialExecutor(),
    )


def _load_map_only_scenarios(
    data_root: Path,
    datasets: list[str] | None = None,
    num_scenes: int | None = None,
) -> list[MapOnlyScenario]:
    maps_root = data_root / "maps"
    map_paths = _discover_map_arrow_paths(maps_root, datasets, num_scenes)

    scenarios: list[MapOnlyScenario] = []
    for map_path in map_paths:
        scenario_id = map_path.stem
        split_name = None if map_path.parent == data_root else map_path.parent.name
        map_api = ArrowMapAPI(map_path)
        scenarios.append(MapOnlyScenario(scenario_id=scenario_id, map_api=map_api, split_name=split_name))

    return scenarios


def _discover_map_arrow_paths(
    maps_root: Path,
    datasets: list[str] | None = None,
    num_scenes: int | None = None,
) -> list[Path]:
    if not maps_root.exists():
        return []

    map_paths = sorted(path for path in maps_root.rglob("*.arrow") if path.is_file())

    if datasets is not None:
        map_paths = [path for path in map_paths if any(dataset in path.parts for dataset in datasets)]

    if num_scenes is not None:
        map_paths = map_paths[:num_scenes]

    return map_paths
