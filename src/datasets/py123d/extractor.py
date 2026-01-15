from __future__ import annotations

# ruff: noqa: I001

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from src.core import types
from src.core.unified import new_unified_scenario
from src.datasets.py123d.utils import ensure_py123d_on_path, safe_id_to_int


if TYPE_CHECKING:
    from collections.abc import Iterable

    from py123d.api.map.arrow_map_api import ArrowMapAPI  # type: ignore[import-not-found]
    from py123d.api.map.map_api import MapAPI  # type: ignore[import-not-found]
    from py123d.api.scene.scene_api import SceneAPI  # type: ignore[import-not-found]
    from py123d.datatypes.map_objects.map_layer_types import RoadEdgeType, RoadLineType  # type: ignore[import-not-found]



@dataclass(frozen=True)
class _MapObjectBundle:
    layer: object
    object_id: object
    obj: object


def convert_py123d_scenario(raw: object) -> dict:
    """Convert py123d SceneAPI or MapOnlyScenario to unified format.

    Args:
        raw: Arrow SceneAPI or MapOnlyScenario.

    Returns:
        Unified scenario dict.
    """
    ensure_py123d_on_path()

    from src.datasets.py123d.load import MapOnlyScenario

    if isinstance(raw, MapOnlyScenario):
        map_api = raw.map_api
        scenario_id = f"map:{raw.scenario_id}"
        timesteps = np.array([0.0], dtype=np.float32)
        scenario_length = len(timesteps)
        log_metadata = None
        scene = None
    else:
        scene: SceneAPI = raw  # type: ignore[assignment]
        map_api = scene.map_api
        scenario_id = f"{scene.log_name}:{scene.scene_uuid}"
        timesteps = _extract_timesteps(scene)
        scenario_length = len(timesteps)
        log_metadata = scene.log_metadata

    scenario = new_unified_scenario(scenario_id=scenario_id, dataset_name="py123d")

    if map_api is not None:
        center = _compute_map_centroid(map_api)
        scenario["static_map_elements"] = extract_static_map_elements(map_api, center)
        map_metadata = map_api.map_metadata
    else:
        center = np.array([0.0, 0.0], dtype=np.float32)
        scenario["static_map_elements"] = {}
        map_metadata = None

    if scene is not None:
        scenario["dynamic_agents"] = extract_dynamic_agents(scene, center)
        scenario["dynamic_map_elements"] = extract_dynamic_map_elements(scene, map_api, center)
    else:
        scenario["dynamic_agents"] = {}
        scenario["dynamic_map_elements"] = {}

    scenario["metadata"].update(
        {
            "scenario_id": scenario_id,
            "scenario_length": scenario_length,
            "sdc_index": 0,
            "timesteps": timesteps,
            "map_center": center.tolist(),
        },
    )

    if map_metadata is not None:
        scenario["metadata"].update(
            {
                "map_dataset": map_metadata.dataset,
                "map_location": map_metadata.location,
                "map_is_local": map_metadata.map_is_local,
                "map_has_z": map_metadata.map_has_z,
                "map_version": map_metadata.version,
            },
        )

    if log_metadata is not None:
        scenario["metadata"].update(
            {
                "log_dataset": log_metadata.dataset,
                "log_split": log_metadata.split,
                "log_name": log_metadata.log_name,
                "log_location": log_metadata.location,
                "log_version": log_metadata.version,
            },
        )

    return scenario


def extract_dynamic_agents(scene: SceneAPI, center: np.ndarray) -> dict[int, dict]:
    """Extract dynamic agents from py123d box detections."""
    ensure_py123d_on_path()

    episode_length = scene.number_of_iterations
    agents: dict[int, dict] = {}

    for frame_idx in range(episode_length):
        detections = _get_box_detections(scene, frame_idx)
        if not detections:
            continue

        for detection in detections:
            track_token = detection.metadata.track_token
            agent_id = safe_id_to_int(track_token)

            if agent_id not in agents:
                agents[agent_id] = {
                    "type": _convert_default_label_to_agent_type(detection.metadata.default_label),
                    "states": {
                        "position": np.zeros((episode_length, 3), dtype=np.float32),
                        "heading": np.zeros((episode_length,), dtype=np.float32),
                        "velocity": np.zeros((episode_length, 2), dtype=np.float32),
                        "valid": np.zeros((episode_length,), dtype=np.bool8),
                        "length": np.zeros((episode_length,), dtype=np.float32),
                        "width": np.zeros((episode_length,), dtype=np.float32),
                        "height": np.zeros((episode_length,), dtype=np.float32),
                    },
                }

            bbox = detection.bounding_box_se3
            center_se3 = bbox.center_se3
            heading = center_se3.pose_se2.yaw

            x = float(center_se3.x) - float(center[0])
            y = float(center_se3.y) - float(center[1])
            z = float(center_se3.z)

            states = agents[agent_id]["states"]
            states["position"][frame_idx] = [x, y, z]
            states["heading"][frame_idx] = heading
            states["valid"][frame_idx] = True
            states["length"][frame_idx] = float(bbox.length)
            states["width"][frame_idx] = float(bbox.width)
            states["height"][frame_idx] = float(bbox.height)

            if detection.velocity_2d is not None:
                states["velocity"][frame_idx] = [
                    float(detection.velocity_2d.x),
                    float(detection.velocity_2d.y),
                ]

    for agent in agents.values():
        states = agent["states"]
        if np.any(states["valid"]):
            pos = states["position"]
            valid = states["valid"]
            velocity = states["velocity"]

            if not np.any(velocity[valid]):
                diffs = np.zeros_like(velocity)
                diffs[1:] = pos[1:, :2] - pos[:-1, :2]
                diffs[0] = diffs[1]
                timestep = scene.log_metadata.timestep_seconds
                if timestep > 0:
                    velocity[:] = diffs / float(timestep)

    return agents


def extract_dynamic_map_elements(
    scene: SceneAPI,
    map_api: MapAPI | None,
    center: np.ndarray,
) -> dict[int, dict]:
    """Extract dynamic traffic light states from py123d logs."""
    ensure_py123d_on_path()

    from py123d.datatypes.detections.traffic_light_detections import (  # type: ignore[import-not-found]
        TrafficLightStatus,
    )

    episode_length = scene.number_of_iterations
    elements: dict[int, dict] = {}

    for frame_idx in range(episode_length):
        traffic_lights = _get_traffic_light_detections(scene, frame_idx)
        if not traffic_lights:
            continue

        for detection in traffic_lights:
            lane_id = int(detection.lane_id)
            if lane_id not in elements:
                position = _get_lane_position(map_api, lane_id, center)
                elements[lane_id] = {
                    "type": types.TRAFFIC_LIGHT,
                    "position": np.array(position, dtype=np.float32),
                    "states": [types.TRAFFIC_LIGHT_UNKNOWN] * episode_length,
                    "controlled_lane": lane_id,
                }

            elements[lane_id]["states"][frame_idx] = _convert_traffic_light_status(
                detection.status,
                TrafficLightStatus,
            )

    return elements


def _get_box_detections(scene: SceneAPI, frame_idx: int):
    """Get box detections, with fallback when timestamp column is missing."""
    try:
        detections = scene.get_box_detections_at_iteration(frame_idx)
        return detections.box_detections if detections is not None else []
    except AssertionError:
        return _get_box_detections_from_arrow(scene, frame_idx)


def _get_box_detections_from_arrow(scene: SceneAPI, frame_idx: int):
    ensure_py123d_on_path()

    from py123d.common.utils.arrow_column_names import (  # type: ignore[import-not-found]
        BOX_DETECTIONS_BOUNDING_BOX_SE3_COLUMN,
        BOX_DETECTIONS_LABEL_COLUMN,
        BOX_DETECTIONS_NUM_LIDAR_POINTS_COLUMN,
        BOX_DETECTIONS_SE3_COLUMNS,
        BOX_DETECTIONS_TOKEN_COLUMN,
        BOX_DETECTIONS_VELOCITY_3D_COLUMN,
    )
    from py123d.common.utils.arrow_helper import get_lru_cached_arrow_table  # type: ignore[import-not-found]
    from py123d.datatypes.detections.box_detections import (  # type: ignore[import-not-found]
        BoxDetectionMetadata,
        BoxDetectionSE3,
    )
    from py123d.geometry import BoundingBoxSE3, Vector3D  # type: ignore[import-not-found]

    arrow_path = getattr(scene, "_arrow_file_path", None)
    if arrow_path is None:
        return []

    table = get_lru_cached_arrow_table(str(arrow_path))
    if not all(column in table.schema.names for column in BOX_DETECTIONS_SE3_COLUMNS):
        return []

    idx = scene.scene_metadata.initial_idx + frame_idx
    if idx >= table.num_rows:
        return []

    label_class = scene.log_metadata.box_detection_label_class
    if label_class is None:
        return []

    boxes = table[BOX_DETECTIONS_BOUNDING_BOX_SE3_COLUMN][idx].as_py()
    tokens = table[BOX_DETECTIONS_TOKEN_COLUMN][idx].as_py()
    labels = table[BOX_DETECTIONS_LABEL_COLUMN][idx].as_py()
    velocities = table[BOX_DETECTIONS_VELOCITY_3D_COLUMN][idx].as_py()
    num_points = table[BOX_DETECTIONS_NUM_LIDAR_POINTS_COLUMN][idx].as_py()

    detections = []
    for box, token, label, velocity, points in zip(boxes, tokens, labels, velocities, num_points):
        metadata = BoxDetectionMetadata(
            label=label_class(label),
            track_token=token,
            num_lidar_points=points,
            timepoint=None,
        )
        bbox = BoundingBoxSE3.from_list(box)
        velocity_3d = Vector3D.from_list(velocity) if velocity is not None else None
        detections.append(BoxDetectionSE3(metadata=metadata, bounding_box_se3=bbox, velocity_3d=velocity_3d))

    return detections


def _get_traffic_light_detections(scene: SceneAPI, frame_idx: int):
    """Get traffic light detections, with fallback when timestamp column is missing."""
    try:
        detections = scene.get_traffic_light_detections_at_iteration(frame_idx)
        return detections.traffic_light_detections if detections is not None else []
    except AssertionError:
        return _get_traffic_light_detections_from_arrow(scene, frame_idx)


def _get_traffic_light_detections_from_arrow(scene: SceneAPI, frame_idx: int):
    ensure_py123d_on_path()

    from py123d.common.utils.arrow_column_names import (  # type: ignore[import-not-found]
        TRAFFIC_LIGHTS_LANE_ID_COLUMN,
        TRAFFIC_LIGHTS_STATUS_COLUMN,
    )
    from py123d.common.utils.arrow_helper import get_lru_cached_arrow_table  # type: ignore[import-not-found]
    from py123d.datatypes.detections.traffic_light_detections import (  # type: ignore[import-not-found]
        TrafficLightDetection,
        TrafficLightStatus,
    )

    arrow_path = getattr(scene, "_arrow_file_path", None)
    if arrow_path is None:
        return []

    table = get_lru_cached_arrow_table(str(arrow_path))
    if TRAFFIC_LIGHTS_LANE_ID_COLUMN not in table.schema.names:
        return []
    if TRAFFIC_LIGHTS_STATUS_COLUMN not in table.schema.names:
        return []

    idx = scene.scene_metadata.initial_idx + frame_idx
    if idx >= table.num_rows:
        return []

    lane_ids = table[TRAFFIC_LIGHTS_LANE_ID_COLUMN][idx].as_py()
    statuses = table[TRAFFIC_LIGHTS_STATUS_COLUMN][idx].as_py()

    detections = []
    for lane_id, status in zip(lane_ids, statuses):
        detections.append(
            TrafficLightDetection(
                lane_id=int(lane_id),
                status=TrafficLightStatus(status),
                timepoint=None,
            ),
        )
    return detections


def _extract_timesteps(scene: SceneAPI) -> np.ndarray:
    try:
        timepoints = [scene.get_timepoint_at_iteration(i).time_s for i in range(scene.number_of_iterations)]
        if not timepoints:
            return np.array([], dtype=np.float32)

        t0 = timepoints[0]
        return np.array([t - t0 for t in timepoints], dtype=np.float32)
    except AssertionError:
        timestep = scene.log_metadata.timestep_seconds
        return np.array(
            [i * timestep for i in range(scene.number_of_iterations)],
            dtype=np.float32,
        )


def _compute_map_centroid(map_api: MapAPI) -> np.ndarray:
    """Compute map centroid using available lane/road geometry."""
    ensure_py123d_on_path()

    from py123d.api.map.arrow_map_api import ArrowMapAPI  # type: ignore[import-not-found]
    from py123d.datatypes.map_objects.map_layer_types import MapLayer  # type: ignore[import-not-found]

    if not isinstance(map_api, ArrowMapAPI):
        return np.array([0.0, 0.0], dtype=np.float32)

    points: list[np.ndarray] = []
    for layer in [MapLayer.LANE, MapLayer.ROAD_LINE, MapLayer.ROAD_EDGE, MapLayer.CROSSWALK]:
        if layer not in map_api._occupancy_maps:
            continue
        ids = map_api._occupancy_maps[layer].ids
        for object_id in ids:
            obj = map_api.get_map_object(object_id, layer)
            if obj is None:
                continue
            coords = _get_object_xy_points(obj)
            if coords is not None and len(coords) > 0:
                points.append(coords)

    if not points:
        return np.array([0.0, 0.0], dtype=np.float32)

    all_xy = np.vstack(points)
    return all_xy.mean(axis=0).astype(np.float32)


def extract_static_map_elements(map_api: MapAPI, center: np.ndarray) -> dict[int, dict]:
    """Extract static map elements from a py123d MapAPI."""
    ensure_py123d_on_path()

    from py123d.api.map.arrow_map_api import ArrowMapAPI  # type: ignore[import-not-found]
    from py123d.datatypes.map_objects.map_layer_types import MapLayer  # type: ignore[import-not-found]

    if not isinstance(map_api, ArrowMapAPI):
        return {}

    static_map_elements: dict[int, dict] = {}
    for bundle in _iter_map_objects(
        map_api,
        [MapLayer.LANE, MapLayer.ROAD_LINE, MapLayer.ROAD_EDGE, MapLayer.CROSSWALK],
    ):
        element = _convert_map_object(bundle.obj, center)
        if element is None:
            continue
        static_map_elements[safe_id_to_int(bundle.object_id)] = element

    return static_map_elements


def _iter_map_objects(map_api: ArrowMapAPI, layers: Iterable) -> list[_MapObjectBundle]:
    bundles: list[_MapObjectBundle] = []
    for layer in layers:
        if layer not in map_api._occupancy_maps:
            continue
        for object_id in map_api._occupancy_maps[layer].ids:
            obj = map_api.get_map_object(object_id, layer)
            if obj is None:
                continue
            bundles.append(_MapObjectBundle(layer=layer, object_id=object_id, obj=obj))
    return bundles


def _convert_map_object(map_object: object, center: np.ndarray) -> dict | None:
    ensure_py123d_on_path()

    from py123d.datatypes.map_objects.map_objects import (  # type: ignore[import-not-found]
        Crosswalk,
        Lane,
        RoadEdge,
        RoadLine,
    )

    if isinstance(map_object, Lane):
        polyline = _centered_array(map_object.centerline.array, center)
        speed_limit_kmh = _mps_to_kmh(map_object.speed_limit_mps)
        left_neighbor = [safe_id_to_int(map_object.left_lane_id)] if map_object.left_lane_id else []
        right_neighbor = [safe_id_to_int(map_object.right_lane_id)] if map_object.right_lane_id else []
        entry_lanes = [safe_id_to_int(lane_id) for lane_id in map_object.predecessor_ids]
        exit_lanes = [safe_id_to_int(lane_id) for lane_id in map_object.successor_ids]

        return {
            "type": types.LANE_SURFACE_STREET,
            "polyline": polyline,
            "speed_limit_mph": _kmh_to_mph(speed_limit_kmh) if speed_limit_kmh >= 0 else -1,
            "speed_limit_kmh": speed_limit_kmh,
            "entry_lanes": entry_lanes,
            "exit_lanes": exit_lanes,
            "left_neighbor": left_neighbor,
            "right_neighbor": right_neighbor,
            "left_boundaries": [],
            "right_boundaries": [],
        }

    if isinstance(map_object, RoadLine):
        polyline = _centered_array(map_object.polyline_3d.array, center)
        return {
            "type": _convert_road_line_type(map_object.road_line_type),
            "polyline": polyline,
        }

    if isinstance(map_object, RoadEdge):
        polyline = _centered_array(map_object.polyline_3d.array, center)
        return {
            "type": _convert_road_edge_type(map_object.road_edge_type),
            "polyline": polyline,
        }

    if isinstance(map_object, Crosswalk):
        polygon = _centered_array(map_object.outline_3d.array, center)
        return {
            "type": types.CROSSWALK,
            "polygon": polygon,
        }

    return None


def _get_object_xy_points(map_object: object) -> np.ndarray | None:
    if hasattr(map_object, "centerline"):
        return map_object.centerline.array[:, :2]
    if hasattr(map_object, "polyline_3d"):
        return map_object.polyline_3d.array[:, :2]
    if hasattr(map_object, "outline_3d"):
        return map_object.outline_3d.array[:, :2]
    return None


def _centered_array(array: np.ndarray, center: np.ndarray) -> np.ndarray:
    centered = array.astype(np.float32, copy=True)
    if centered.shape[1] >= 2:
        centered[:, 0] -= center[0]
        centered[:, 1] -= center[1]
    return centered


def _mps_to_kmh(speed_mps: float | None) -> float:
    if speed_mps is None:
        return -1
    return float(speed_mps) * 3.6


def _kmh_to_mph(speed_kmh: float) -> float:
    return speed_kmh / 1.609344


def _convert_road_edge_type(edge_type: RoadEdgeType | None) -> str:
    ensure_py123d_on_path()

    from py123d.datatypes.map_objects.map_layer_types import RoadEdgeType  # type: ignore[import-not-found]

    if edge_type is None:
        return types.ROAD_EDGE_UNKNOWN
    if edge_type == RoadEdgeType.ROAD_EDGE_BOUNDARY:
        return types.ROAD_EDGE_BOUNDARY
    if edge_type == RoadEdgeType.ROAD_EDGE_MEDIAN:
        return types.ROAD_EDGE_MEDIAN
    return types.ROAD_EDGE_UNKNOWN


def _convert_road_line_type(line_type: RoadLineType | None) -> str:
    ensure_py123d_on_path()

    from py123d.datatypes.map_objects.map_layer_types import RoadLineType  # type: ignore[import-not-found]

    if line_type is None:
        return types.ROAD_LINE_UNKNOWN

    mapping = {
        RoadLineType.DASHED_WHITE: types.ROAD_LINE_BROKEN_SINGLE_WHITE,
        RoadLineType.DASHED_YELLOW: types.ROAD_LINE_BROKEN_SINGLE_YELLOW,
        RoadLineType.DOUBLE_DASH_YELLOW: types.ROAD_LINE_BROKEN_DOUBLE_YELLOW,
        RoadLineType.DOUBLE_DASH_WHITE: types.ROAD_LINE_BROKEN_SINGLE_WHITE,
        RoadLineType.DOUBLE_SOLID_WHITE: types.ROAD_LINE_SOLID_DOUBLE_WHITE,
        RoadLineType.DOUBLE_SOLID_YELLOW: types.ROAD_LINE_SOLID_DOUBLE_YELLOW,
        RoadLineType.SOLID_WHITE: types.ROAD_LINE_SOLID_SINGLE_WHITE,
        RoadLineType.SOLID_YELLOW: types.ROAD_LINE_SOLID_SINGLE_YELLOW,
        RoadLineType.DASH_SOLID_WHITE: types.ROAD_LINE_SOLID_SINGLE_WHITE,
        RoadLineType.DASH_SOLID_YELLOW: types.ROAD_LINE_PASSING_DOUBLE_YELLOW,
        RoadLineType.SOLID_DASH_WHITE: types.ROAD_LINE_SOLID_SINGLE_WHITE,
        RoadLineType.SOLID_DASH_YELLOW: types.ROAD_LINE_PASSING_DOUBLE_YELLOW,
        RoadLineType.SOLID_BLUE: types.ROAD_LINE_SOLID_SINGLE_WHITE,
    }

    return mapping.get(line_type, types.ROAD_LINE_UNKNOWN)


def _convert_default_label_to_agent_type(label) -> str:
    ensure_py123d_on_path()

    from py123d.conversion.registry.box_detection_label_registry import (  # type: ignore[import-not-found]
        DefaultBoxDetectionLabel,
    )

    if label in (DefaultBoxDetectionLabel.VEHICLE, DefaultBoxDetectionLabel.TRAIN, DefaultBoxDetectionLabel.EGO):
        return types.VEHICLE
    if label == DefaultBoxDetectionLabel.BICYCLE:
        return types.CYCLIST
    if label == DefaultBoxDetectionLabel.PERSON:
        return types.PEDESTRIAN
    return types.OTHER


def _get_lane_position(map_api: MapAPI | None, lane_id: int, center: np.ndarray) -> list[float]:
    if map_api is None:
        return [0.0, 0.0, 0.0]

    ensure_py123d_on_path()

    from py123d.datatypes.map_objects.map_layer_types import MapLayer  # type: ignore[import-not-found]

    lane = map_api.get_map_object(lane_id, MapLayer.LANE)
    if lane is None or not hasattr(lane, "centerline"):
        return [0.0, 0.0, 0.0]

    point = lane.centerline.array[0]
    x = float(point[0]) - float(center[0])
    y = float(point[1]) - float(center[1])
    z = float(point[2]) if len(point) > 2 else 0.0
    return [x, y, z]


def _convert_traffic_light_status(status, status_enum) -> str:
    if status == status_enum.GREEN:
        return types.TRAFFIC_LIGHT_GREEN
    if status == status_enum.YELLOW:
        return types.TRAFFIC_LIGHT_YELLOW
    if status == status_enum.RED:
        return types.TRAFFIC_LIGHT_RED
    return types.TRAFFIC_LIGHT_UNKNOWN
