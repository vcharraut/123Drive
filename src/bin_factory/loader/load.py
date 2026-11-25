from __future__ import annotations

import dataclasses
import pathlib

from py123d import api as py123d_api
from py123d.api.map.arrow import arrow_map_api
from py123d.api.scene.arrow import arrow_scene_api
from py123d.api.scene.arrow.modalities import sync_utils
from py123d.common import execution
from py123d.common.utils import uuid_utils


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
    dt: float = 0.1,
    map_only: bool = False,
) -> list:
    data_root = pathlib.Path(py123d_data_root)
    if map_only:
        maps_root = data_root / "maps"
        paths = sorted(p for p in maps_root.rglob("*.arrow") if p.is_file()) if maps_root.exists() else []
        paths = [p for p in paths if not datasets or any(d in p.parts for d in datasets)]
        return [arrow_map_api.ArrowMapAPI(p) for p in paths[:num_scenes]]

    scene_filter = py123d_api.SceneFilter(
        datasets=datasets,
        split_types=split_types,
        split_names=split_names,
        log_names=log_names,
        scene_uuids=scene_uuids,
        target_iteration_duration_s=dt,
        future_duration_s=duration_s,
        timestamp_threshold_s=duration_s,
        max_num_scenes=num_scenes,
        required_scene_modalities=["box_detections_se3:any"],
    )
    executor = execution.ProcessPoolExecutor(max_workers=workers) if workers > 1 else execution.SequentialExecutor()
    scenes = py123d_api.get_filtered_scenes(scene_filter=scene_filter, data_root=data_root, executor=executor)
    return [_skip_first_frame(s) if s.dataset == "nurec" and s.scene_metadata.initial_idx == 0 else s for s in scenes]


def _skip_first_frame(scene: py123d_api.SceneAPI) -> py123d_api.SceneAPI:
    # PY123D-REPORT[nurec]: the 10 Hz grid starts one step before the second rig pose, so log frame 0 precedes the
    # first pose: the ego is extrapolated there and almost no track has started yet.
    # Start the scene at frame 1, keeping its end.
    _, (log_dir, _, maps_root) = scene.__reduce__()
    log_dir = pathlib.Path(log_dir)
    metadata = scene.scene_metadata
    start = metadata.initial_idx + metadata.target_iteration_stride
    metadata = dataclasses.replace(
        metadata,
        initial_uuid=uuid_utils.convert_to_str_uuid(sync_utils.get_sync_table(log_dir)["sync.uuid"][start].as_py()),
        initial_idx=start,
        num_future_iterations=metadata.num_future_iterations - 1,
        future_duration_s=metadata.future_duration_s - metadata.iteration_duration_s,
    )
    return arrow_scene_api.ArrowSceneAPI(log_dir, metadata, maps_root)
