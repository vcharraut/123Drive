from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from py123d.api.map.arrow_map_api import ArrowMapAPI
from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.execution.sequential_executor import SequentialExecutor


@dataclass(frozen=True)
class MapOnlyScenario:
    scenario_id: str
    map_api: ArrowMapAPI
    split_name: str | None = None


def get_py123d_scenarios(
    dataset_path: str | None,
    max_scenarios: int | None = None,
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
        dataset_path: Root path to py123d_data (contains logs/ and maps/) or a directory of .arrow maps.
        max_scenarios: Optional cap on number of scenes.
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
    data_root = Path(dataset_path)

    if map_only:
        return _load_map_only_scenarios(data_root)

    logs_root = data_root / "logs"
    maps_root = data_root / "maps"

    filter_cfg = SceneFilter(
        datasets=datasets,
        split_types=split_types,
        split_names=split_names,
        log_names=log_names,
        duration_s=duration_s,
        history_s=history_s,
        map_api_required=map_api_required,
        max_num_scenes=max_scenarios,
    )

    builder = ArrowSceneBuilder(logs_root=logs_root, maps_root=maps_root)
    return builder.get_scenes(filter_cfg, SequentialExecutor())


def _load_map_only_scenarios(data_root: Path) -> list[MapOnlyScenario]:
    map_paths = _discover_map_arrow_paths(data_root)
    if not map_paths:
        maps_root = data_root / "maps"
        map_paths = _discover_map_arrow_paths(maps_root)

    scenarios: list[MapOnlyScenario] = []
    for map_path in map_paths:
        scenario_id = map_path.stem
        split_name = None if map_path.parent == data_root else map_path.parent.name
        map_api = ArrowMapAPI(map_path)
        scenarios.append(MapOnlyScenario(scenario_id=scenario_id, map_api=map_api, split_name=split_name))

    return scenarios


def _discover_map_arrow_paths(maps_root: Path) -> list[Path]:
    if not maps_root.exists():
        return []

    map_paths: list[Path] = []
    for map_file in maps_root.iterdir():
        if map_file.is_file() and map_file.suffix == ".arrow":
            map_paths.append(map_file)

    return map_paths
