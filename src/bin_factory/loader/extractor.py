from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import shapely.geometry as geom
from py123d.datatypes.detections import DefaultBoxDetectionLabel
from py123d.datatypes.map_objects import Lane, LaneType, MapLayer
from py123d.geometry import Point2D

from bin_factory import types
from bin_factory.loader.load import MapOnlyScenario


if TYPE_CHECKING:
    from py123d.api import MapAPI, SceneAPI

_AGENT_LABELS = {
    DefaultBoxDetectionLabel.EGO,
    DefaultBoxDetectionLabel.VEHICLE,
    # DefaultBoxDetectionLabel.TRAIN,
    DefaultBoxDetectionLabel.BICYCLE,
    DefaultBoxDetectionLabel.PERSON,
    # DefaultBoxDetectionLabel.ANIMAL,
}


def convert_py123d_scenario(raw: SceneAPI | MapOnlyScenario) -> dict:
    """Convert py123d SceneAPI or MapOnlyScenario to intermediate format."""
    # Build base scenario
    if isinstance(raw, MapOnlyScenario):
        scene = None
        map_api = raw.map_api
        scenario_id = raw.scenario_id
        dataset_name = map_api.dataset
    else:
        scene = raw
        map_api = scene.map_api
        scenario_id = f"{scene.log_name}"
        dataset_name = scene.dataset

    if map_api is None:
        raise ValueError("Map API is required to convert scenario")

    py123d_dict = {
        "id": scenario_id,
        "agents": {},
        "map": {},
        "traffic_lights": {},
        "dataset_name": dataset_name,
        "scenario_length": 0,
        "sdc_index": 0,
        "timestep_seconds": 0.0,
    }

    # Compute scene centroid
    centroid = np.zeros(2, dtype=np.float64)
    if scene is not None:
        episode_length = scene.number_of_iterations
        positions = np.array([[0.0, 0.0]] * episode_length, dtype=np.float64)
        valid = np.zeros((episode_length,), dtype=np.int32)

        for frame_idx in range(episode_length):
            ego_state = scene.get_ego_state_se3_at_iteration(frame_idx)
            if ego_state is None:
                continue
            positions[frame_idx] = [float(ego_state.center_se3.x), float(ego_state.center_se3.y)]
            valid[frame_idx] = 1

        valid_positions = positions[valid]
        if len(valid_positions) > 0:
            centroid = valid_positions.mean(axis=0)

    if np.all(centroid == 0.0):
        # Fallback: use road geometry centroid
        points: list[np.ndarray] = []
        road_layers = [MapLayer.LANE, MapLayer.ROAD_LINE, MapLayer.ROAD_EDGE]
        for obj in _get_map_objects(map_api, road_layers):
            coords = _get_object_xy_points(obj)
            if coords is not None and len(coords) > 0:
                points.append(coords)

        if points:
            centroid = np.vstack(points).mean(axis=0)

    map_only = scene is None
    py123d_dict["map"] = _extract_map(map_api, centroid, map_only)

    if scene is not None:
        agents = _extract_objects(scene, centroid)
        # filter to agent-like labels only
        agents = {oid: obj for oid, obj in agents.items() if obj["type"] in _AGENT_LABELS}

        py123d_dict["agents"] = agents
        py123d_dict["traffic_lights"] = _extract_traffic_lights(scene, map_api, centroid)
        py123d_dict["scenario_length"] = scene.number_of_iterations
        py123d_dict["timestep_seconds"] = scene.log_metadata.timestep_seconds

    return py123d_dict


def _extract_objects(scene: SceneAPI, centroid: np.ndarray) -> dict[int, dict]:
    """Extract dynamic objects from py123d box detections and ego state."""
    episode_length = scene.number_of_iterations
    objects: dict[int, dict] = {0: _make_empty_agent(episode_length, DefaultBoxDetectionLabel.EGO)}
    tokens_to_object_id: dict[str, int] = {}

    # -- ego track --
    ego = objects[0]
    timestep = scene.log_metadata.timestep_seconds

    for frame_idx in range(episode_length):
        ego_state = scene.get_ego_state_se3_at_iteration(frame_idx)
        if ego_state is None:
            raise ValueError(f"Missing ego state at frame {frame_idx}")

        _apply_detection_state(ego, frame_idx, ego_state.center_se3, ego_state.bounding_box_se3, centroid)

        if ego_state.dynamic_state_se3 is not None:
            vel = ego_state.dynamic_state_se3.velocity_3d
            ego["velocity"][frame_idx] = [float(vel.x), float(vel.y)]
            continue

        if frame_idx == 0 or not ego["valid"][frame_idx - 1] or timestep <= 0:
            continue

        delta_pos = ego["position"][frame_idx, :2] - ego["position"][frame_idx - 1, :2]
        ego["velocity"][frame_idx] = delta_pos / timestep

    # -- detection tracks --
    next_object_id = 0
    for frame_idx in range(episode_length):
        detections = scene.get_box_detections_se3_at_iteration(frame_idx)
        if not detections:
            continue
        for detection in detections:
            track_token = detection.metadata.track_token
            if track_token in tokens_to_object_id:
                object_id = tokens_to_object_id[track_token]
            else:
                next_object_id += 1
                object_id = next_object_id
                tokens_to_object_id[track_token] = object_id

            if object_id not in objects:
                objects[object_id] = _make_empty_agent(episode_length, detection.metadata.default_label)

            bbox = detection.bounding_box_se3  # type: ignore[attr-defined]
            obj = objects[object_id]
            _apply_detection_state(obj, frame_idx, bbox.center_se3, bbox, centroid)

            if detection.velocity_2d is None:
                continue

            obj["velocity"][frame_idx] = [float(detection.velocity_2d.x), float(detection.velocity_2d.y)]

    return objects


def _extract_traffic_lights(scene: SceneAPI, map_api: MapAPI, centroid: np.ndarray) -> dict[int, dict]:
    """Extract dynamic traffic light states from py123d logs."""
    episode_length = scene.number_of_iterations
    elements: dict[int, dict] = {}

    for frame_idx in range(episode_length):
        traffic_lights = scene.get_traffic_light_detections_at_iteration(frame_idx)
        if not traffic_lights:
            continue

        for detection in traffic_lights:
            lane_id = int(detection.lane_id)
            if lane_id not in elements:
                try:
                    position = _get_lane_position(map_api, lane_id, centroid)
                except ValueError:
                    # Some traffic light detections can be on map borders where lane geometry is missing;
                    # skip these with a warning.
                    # NOTE: This happens on Waymo maps due to weird map filtering
                    continue
                elements[lane_id] = {
                    "position": np.array(position, dtype=np.float64),
                    "states": [None] * episode_length,
                    "controlled_lane": lane_id,
                }

            elements[lane_id]["states"][frame_idx] = detection.status

    return elements


def _extract_map(map_api: MapAPI, centroid: np.ndarray, map_only: bool = False) -> dict[int, dict]:
    """Extract static map elements from a py123d MapAPI."""
    result = {}
    non_lane_objects = []
    undefined_lane = []
    skipped_layers = {
        MapLayer.LANE_GROUP,
        MapLayer.INTERSECTION,
        MapLayer.WALKWAY,
        MapLayer.CARPARK,
        MapLayer.GENERIC_DRIVABLE,
    }

    # Extract all map objects if map-only scenario, otherwise extract objects within a radius around ego
    all_map_layers = map_api.get_available_map_layers()
    if map_only or map_api.map_is_per_log:
        map_objects = _get_map_objects(map_api, all_map_layers)
    else:
        map_objects_by_layer = map_api.get_map_objects_in_radius(
            Point2D(centroid[0], centroid[1]),
            radius=250.0,
            layers=all_map_layers,
        )
        map_objects = [obj for layer in all_map_layers for obj in map_objects_by_layer.get(layer, [])]

    # Lanes are processed first to ensure their IDs are included in the result for reference by other map elements.
    for obj in map_objects:
        if obj.layer == MapLayer.LANE:
            result[obj.object_id] = _convert_map_object_to_static_element(obj, centroid)
            if obj.lane_type == LaneType.UNDEFINED:
                undefined_lane.append(obj.object_id)
        else:
            non_lane_objects.append(obj)

    # Infer undefined lane types based on connected lanes if possible
    # NOTE: Problem exists in nuPlan where some lanes in intersections are marked as UNDEFINED but are actually drivable
    for lane_id in undefined_lane:
        lane = result[lane_id]
        entry_types = {result[entry_id]["type"] for entry_id in lane.get("entry_lanes", []) if entry_id in result}
        exit_types = {result[exit_id]["type"] for exit_id in lane.get("exit_lanes", []) if exit_id in result}
        connected_types = entry_types.union(exit_types)

        if len(connected_types) == 1:
            lane["type"] = connected_types.pop()
        else:
            pass

    # Flip entry/exit lane directions if they are reverse of lane geometry direction.
    # Entry lanes should connect to lane start; exit lanes should connect to lane end.
    for lane in result.values():
        lane_start = lane["polyline"][0]
        lane_end = lane["polyline"][-1]

        entry_lanes = lane.get("entry_lanes", [])
        exit_lanes = lane.get("exit_lanes", [])

        for entry_id in entry_lanes:
            if entry_id not in result:
                continue
            entry_lane = result[entry_id]
            if "polyline" not in entry_lane:
                continue
            entry_end = entry_lane["polyline"][-1]
            if np.linalg.norm(entry_end - lane_start) > np.linalg.norm(entry_end - lane_end):
                lane["entry_lanes"].remove(entry_id)
                lane["exit_lanes"].append(entry_id)

        for exit_id in exit_lanes:
            if exit_id not in result:
                continue
            exit_lane = result[exit_id]
            if "polyline" not in exit_lane:
                continue
            exit_start = exit_lane["polyline"][0]
            if np.linalg.norm(exit_start - lane_end) > np.linalg.norm(exit_start - lane_start):
                lane["exit_lanes"].remove(exit_id)
                lane["entry_lanes"].append(exit_id)

    # Filter dangling lane topology references
    # NOTE: Happens on nuPlan after filtering to map objects within a radius.
    lane_ids = set(result.keys())
    for element in result.values():
        element["entry_lanes"] = [lid for lid in element["entry_lanes"] if lid in lane_ids]
        element["exit_lanes"] = [lid for lid in element["exit_lanes"] if lid in lane_ids]

    # Non-lane elements get sequential IDs after max lane ID to avoid collisions
    # (object_id namespaces can overlap across map layers)
    next_id = max(result.keys(), default=-1) + 1
    for obj in non_lane_objects:
        if obj.layer not in skipped_layers:
            element = _convert_map_object_to_static_element(obj, centroid)
            if element is None:
                continue
            result[next_id] = element
            next_id += 1

    return result


def _convert_map_object_to_static_element(map_object, centroid: np.ndarray) -> dict | None:
    """Convert py123d map object to unified static element dict."""
    if map_object.layer == MapLayer.LANE:
        polyline = _centered_array(map_object.centerline.array, centroid)
        if not map_object.speed_limit_mps or np.isnan(map_object.speed_limit_mps):
            speed_limit_mps = -1.0
        else:
            speed_limit_mps = float(map_object.speed_limit_mps)
        return {
            "layer": map_object.layer,
            "type": map_object.lane_type,
            "polyline": polyline,
            "speed_limit_mps": speed_limit_mps,
            "entry_lanes": map_object.predecessor_ids,
            "exit_lanes": map_object.successor_ids,
        }

    if map_object.layer == MapLayer.ROAD_LINE:
        polyline = _centered_array(map_object.polyline_3d.array, centroid)
        return {
            "layer": map_object.layer,
            "type": map_object.road_line_type,
            "polyline": polyline,
        }

    if map_object.layer == MapLayer.ROAD_EDGE:
        polyline = _centered_array(map_object.polyline_3d.array, centroid)
        return {
            "layer": map_object.layer,
            "type": map_object.road_edge_type,
            "polyline": polyline,
        }

    if map_object.layer == MapLayer.CROSSWALK:
        polygon = _centered_array(map_object.outline_3d.array, centroid)
        return {
            "layer": map_object.layer,
            "type": 0,
            "polygon": polygon,
        }

    if map_object.layer == MapLayer.STOP_ZONE:
        polygon = _centered_array(map_object.outline_3d.array, centroid)
        return {
            "layer": map_object.layer,
            "type": map_object.stop_zone_type,
            "polygon": polygon,
            "controlled_lanes": map_object.lane_ids,
        }

    raise ValueError(f"Unsupported map object layer: {map_object.layer}")


def _get_map_objects(map_api: MapAPI, layers: list[MapLayer]) -> list[Any]:
    objects_by_layer = map_api.query(geom.box(-1e9, -1e9, 1e9, 1e9), layers=layers, predicate="intersects")

    return [obj for layer in layers for obj in objects_by_layer.get(layer, [])]


def _make_empty_agent(episode_length, agent_type):
    return {
        "type": agent_type,
        "position": np.zeros((episode_length, 3), dtype=np.float64),
        "heading": np.zeros((episode_length,), dtype=np.float64),
        "velocity": np.zeros((episode_length, 2), dtype=np.float64),
        "valid": np.zeros((episode_length,), dtype=np.int32),
        "length": np.zeros((episode_length,), dtype=np.float64),
        "width": np.zeros((episode_length,), dtype=np.float64),
        "height": np.zeros((episode_length,), dtype=np.float64),
    }


def _apply_detection_state(obj: dict, frame_idx: int, center_se3, bbox, centroid: np.ndarray):
    obj["position"][frame_idx] = [
        float(center_se3.x) - float(centroid[0]),
        float(center_se3.y) - float(centroid[1]),
        float(center_se3.z),
    ]
    obj["heading"][frame_idx] = center_se3.pose_se2.yaw
    obj["valid"][frame_idx] = 1
    obj["length"][frame_idx] = float(bbox.length)
    obj["width"][frame_idx] = float(bbox.width)
    obj["height"][frame_idx] = float(bbox.height)


def _get_object_xy_points(map_object: object) -> np.ndarray | None:
    if hasattr(map_object, "centerline"):
        return map_object.centerline.array[:, :2].astype(np.float64)
    if hasattr(map_object, "polyline_3d"):
        return map_object.polyline_3d.array[:, :2].astype(np.float64)
    if hasattr(map_object, "outline_3d"):
        return map_object.outline_3d.array[:, :2].astype(np.float64)
    return None


def _centered_array(array: np.ndarray, center: np.ndarray) -> np.ndarray:
    centered = array.astype(np.float64, copy=True)
    if centered.shape[1] >= 2:
        centered[:, 0] -= center[0]
        centered[:, 1] -= center[1]
    return centered


def _get_lane_position(map_api: MapAPI, lane_id: int, center: np.ndarray) -> list[float]:
    lane = map_api.get_map_object(lane_id, MapLayer.LANE)
    if lane is None or not isinstance(lane, Lane):
        raise ValueError(f"Lane {lane_id} not found or has no centerline")

    if len(lane.centerline.array) == 0:
        raise ValueError(f"Lane {lane_id} has empty centerline")

    point = lane.centerline.array[0]
    x = float(point[0]) - float(center[0])
    y = float(point[1]) - float(center[1])
    z = float(point[2]) if len(point) > 2 else 0.0

    return [x, y, z]
