from __future__ import annotations

import logging
import pathlib
from typing import Any

from py123d import api as py123d_api
from py123d.api.map.arrow import arrow_map_api
from py123d.common import execution


logger = logging.getLogger(__name__)


def load_scenes(
    py123d_data_root: str,
    workers: int,
    num_scenes: int | None = None,
    datasets: list[str] | None = None,
    split_types: list[str] | None = None,
    split_names: list[str] | None = None,
    log_names: list[str] | None = None,
    duration_s: float | None = None,
    history_s: float | None = 0.0,
    map_only: bool = False,
) -> list[Any]:
    """Load 123D scenarios from Arrow logs and/or maps.

    Args:
        py123d_data_root: Root path to 123D data (contains logs/ and maps/) or a directory of .arrow maps.
        workers: Number of parallel workers to use for loading scenarios.
        num_scenes: Optional cap on number of scenes.
        datasets: Optional list of dataset names to include (e.g. ["nuplan", "wod-motion"]).
        split_types: Optional list of split types (train/val/test).
        split_names: Optional list of split names (e.g. ["nuplan-mini_val"]).
        log_names: Optional list of log names to include.
        duration_s: Optional duration for scene extraction; None uses full log.
        history_s: Optional history duration (seconds).
        map_only: If True, load map-only scenarios (no logs).

    Returns:
        List of ArrowSceneAPI or ArrowMapAPI.
    """
    data_root = pathlib.Path(py123d_data_root)

    if map_only:
        maps_root = data_root / "maps"
        map_paths = _discover_map_arrow_paths(maps_root, datasets, num_scenes)

        maps: list[arrow_map_api.ArrowMapAPI] = []
        for map_path in map_paths:
            maps.append(arrow_map_api.ArrowMapAPI(map_path))

        return maps

    scene_filter = py123d_api.SceneFilter(
        datasets=datasets,
        split_types=split_types,
        split_names=split_names,
        log_names=log_names,
        future_duration_s=duration_s,
        history_duration_s=history_s,
        max_num_scenes=num_scenes,
    )

    return py123d_api.get_filtered_scenes(
        scene_filter=scene_filter,
        data_root=data_root,
        executor=execution.ProcessPoolExecutor(max_workers=workers) if workers > 1 else execution.SequentialExecutor(),
    )


def _discover_map_arrow_paths(
    maps_root: pathlib.Path,
    datasets: list[str] | None = None,
    num_scenes: int | None = None,
) -> list[pathlib.Path]:
    if not maps_root.exists():
        return []

    map_paths = sorted(path for path in maps_root.rglob("*.arrow") if path.is_file())

    if datasets is not None:
        map_paths = [path for path in map_paths if any(dataset in path.parts for dataset in datasets)]

    if num_scenes is not None:
        map_paths = map_paths[:num_scenes]

    logger.debug("Discovered %d map files", len(map_paths))
    return map_paths
