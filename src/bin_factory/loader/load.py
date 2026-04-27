from __future__ import annotations

import logging
import pathlib

from py123d import api as py123d_api
from py123d.api.map.arrow import arrow_map_api
from py123d.common import execution


logger = logging.getLogger(__name__)


def discover_scenes(
    py123d_data_root: str,
    workers: int,
    num_scenes: int | None = None,
    datasets: list[str] | None = None,
    split_types: list[str] | None = None,
    split_names: list[str] | None = None,
    log_names: list[str] | None = None,
    scene_uuids: list[str] | None = None,
    duration_s: float | None = None,
    map_only: bool = False,
) -> list:
    data_root = pathlib.Path(py123d_data_root)
    if map_only:
        return _discover_maps(data_root / "maps", datasets, num_scenes)

    scene_filter = py123d_api.SceneFilter(
        datasets=datasets,
        split_types=split_types,
        split_names=split_names,
        log_names=log_names,
        scene_uuids=scene_uuids,
        future_duration_s=duration_s,
        max_num_scenes=num_scenes,
        has_map=True,
        required_scene_modalities=["box_detections_se3:any"],
    )
    executor = execution.ProcessPoolExecutor(max_workers=workers) if workers > 1 else execution.SequentialExecutor()
    return py123d_api.get_filtered_scenes(scene_filter=scene_filter, data_root=data_root, executor=executor)


def _discover_maps(maps_root: pathlib.Path, datasets: list[str] | None, num_scenes: int | None) -> list:
    if not maps_root.exists():
        return []
    paths = sorted(p for p in maps_root.rglob("*.arrow") if p.is_file())
    if datasets:
        paths = [p for p in paths if any(d in p.parts for d in datasets)]
    if num_scenes is not None:
        paths = paths[:num_scenes]
    return [arrow_map_api.ArrowMapAPI(p) for p in paths]
