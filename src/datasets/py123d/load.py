from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.datasets.py123d.utils import ensure_py123d_on_path, resolve_py123d_data_root


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from py123d.api.map.arrow_map_api import ArrowMapAPI  # type: ignore[import-not-found]



@dataclass(frozen=True)
class MapOnlyScenario:
    scenario_id: str
    map_api: ArrowMapAPI
    split_name: str | None = None


def _import_py123d():
    ensure_py123d_on_path()

    from py123d.api.map.arrow_map_api import ArrowMapAPI  # type: ignore[import-not-found]
    from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder  # type: ignore[import-not-found]
    from py123d.api.scene.scene_filter import SceneFilter  # type: ignore[import-not-found]
    from py123d.common.multithreading.worker_sequential import Sequential  # type: ignore[import-not-found]

    return ArrowMapAPI, ArrowSceneBuilder, SceneFilter, Sequential


def get_py123d_scenarios(
    dataset_path: str | None,
    num_files: int | None = None,
    datasets: list[str] | None = None,
    split_types: list[str] | None = None,
    split_names: list[str] | None = None,
    log_names: list[str] | None = None,
    duration_s: float | None = None,
    history_s: float | None = 0.0,
    map_api_required: bool = True,
    map_only: bool = False,
) -> list[object]:
    """Load py123d scenarios from Arrow logs and/or maps.

    Args:
        dataset_path: Root path to py123d_data (contains logs/ and maps/).
        num_files: Optional cap on number of scenes.
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
    data_root = resolve_py123d_data_root(dataset_path)

    if map_only:
        return _load_map_only_scenarios(data_root, datasets, split_types, split_names)

    logs_root = data_root / "logs"
    maps_root = data_root / "maps"

    _, ArrowSceneBuilder, SceneFilter, Sequential = _import_py123d()

    filter_cfg = SceneFilter(
        datasets=datasets,
        split_types=split_types,
        split_names=split_names,
        log_names=log_names,
        duration_s=duration_s,
        history_s=history_s,
        map_api_required=map_api_required,
        max_num_scenes=num_files,
    )

    builder = ArrowSceneBuilder(logs_root=logs_root, maps_root=maps_root)
    worker = Sequential()
    return builder.get_scenes(filter_cfg, worker)


def _load_map_only_scenarios(
    data_root: Path,
    datasets: list[str] | None,
    split_types: list[str] | None,
    split_names: list[str] | None,
) -> list[MapOnlyScenario]:
    maps_root = data_root / "maps"
    map_paths = _discover_map_arrow_paths(maps_root, datasets, split_types, split_names)

    ArrowMapAPI, _, _, _ = _import_py123d()

    scenarios: list[MapOnlyScenario] = []
    for map_path in map_paths:
        scenario_id = map_path.stem
        split_name = map_path.parent.name
        map_api = ArrowMapAPI(map_path)
        scenarios.append(MapOnlyScenario(scenario_id=scenario_id, map_api=map_api, split_name=split_name))

    return scenarios


def _discover_map_arrow_paths(
    maps_root: Path,
    datasets: list[str] | None,
    split_types: list[str] | None,
    split_names: list[str] | None,
) -> list[Path]:
    if not maps_root.exists():
        return []

    dataset_filter = set(datasets) if datasets else None
    split_name_filter = set(split_names) if split_names else None
    split_type_filter = set(split_types) if split_types else None

    map_paths: list[Path] = []
    for subdir in maps_root.iterdir():
        if not subdir.is_dir():
            continue

        subdir_name = subdir.name
        split_type = None
        if "_" in subdir_name:
            split_type = subdir_name.split("_")[-1]

        if split_name_filter is not None and subdir_name not in split_name_filter:
            continue

        if split_type_filter is not None and (split_type is None or split_type not in split_type_filter):
            continue

        if dataset_filter is not None and not _matches_dataset_filter(subdir_name, dataset_filter):
            continue

        for map_file in subdir.iterdir():
            if map_file.is_file() and map_file.suffix == ".arrow":
                map_paths.append(map_file)

    return map_paths


def _matches_dataset_filter(name: str, dataset_filter: Iterable[str]) -> bool:
    return any(name == dataset or name.startswith(dataset) or dataset.startswith(name) for dataset in dataset_filter)
