from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from py123d.conversion.registry.box_detection_label_registry import DefaultBoxDetectionLabel
from py123d.datatypes.map_objects.map_layer_types import LaneType, MapLayer
from py123d.geometry import Point2D

from src.bin_factory import types
from src.bin_factory.py123d_loader.load import MapOnlyScenario
from src.bin_factory.py123d_loader.utils import (
    centered_array,
    get_lane_position,
    get_object_xy_points,
    kmh_to_mph,
    mps_to_kmh,
)


if TYPE_CHECKING:
    from py123d.api.map.map_api import MapAPI
    from py123d.api.scene.scene_api import SceneAPI


def convert_py123d_scenario(raw: SceneAPI | MapOnlyScenario) -> dict:
    """Convert py123d SceneAPI or MapOnlyScenario to intermediate format.

    Args:
        raw: Arrow SceneAPI or MapOnlyScenario.

    Returns:
        Intermediate scenario dict.
    """

    scene: SceneAPI | None = None
    if isinstance(raw, MapOnlyScenario):
        map_api = raw.map_api
        scenario_id = raw.scenario_id
        dataset_name = map_api.dataset
        map_only = True
    else:
        scene = raw
        map_api = scene.map_api
        scenario_id = f"{scene.log_name}"  # scene.scene_uuid
        dataset_name = scene.dataset
        map_only = False

    scenario = {
        "id": scenario_id,
        "agents": {},
        "map": {},
        "traffic_lights": {},
        "objects": {},
        "dataset_name": dataset_name,
        "scenario_length": 0,
        "sdc_index": 0,
        "timestep_seconds": 0.0,
    }

    if map_api is None:
        raise ValueError("Map API is required to convert scenario")

    if scene:
        centroid = _compute_map_centroid_from_ego_positions(scene)
    else:
        centroid = _compute_map_centroid_from_road_layers(map_api)

    scenario["map"] = extract_map(map_api, centroid, map_only)

    if scene is not None:
        objects = extract_objects(scene, centroid)
        scenario["agents"], scenario["objects"] = _split_agents_and_objects(objects)
        scenario["traffic_lights"] = extract_traffic_lights(scene, map_api, centroid)
        scenario["scenario_length"] = scene.number_of_iterations
        scenario["timestep_seconds"] = scene.log_metadata.timestep_seconds

    return scenario


def extract_objects(scene: SceneAPI, centroid: np.ndarray) -> dict[int, dict]:
    """Extract dynamic objects from py123d box detections and ego state.

    Returns:
        Dict mapping agent_id to agent data dict.
    """

    episode_length = scene.number_of_iterations
    objects: dict[int, dict] = {}
    object_id = 0
    tokens_to_object_id: dict[str, int] = {}

    # Extract ego vehicle
    objects[object_id] = {
        "type": DefaultBoxDetectionLabel.EGO,
        "position": np.zeros((episode_length, 3), dtype=np.float32),
        "heading": np.zeros((episode_length,), dtype=np.float32),
        "velocity": np.zeros((episode_length, 2), dtype=np.float32),
        "valid": np.zeros((episode_length,), dtype=np.bool_),
        "length": np.zeros((episode_length,), dtype=np.float32),
        "width": np.zeros((episode_length,), dtype=np.float32),
        "height": np.zeros((episode_length,), dtype=np.float32),
    }

    for frame_idx in range(episode_length):
        ego_state = scene.get_ego_state_at_iteration(frame_idx)

        center_se3 = ego_state.center_se3
        heading = center_se3.pose_se2.yaw
        x = float(center_se3.x) - float(centroid[0])
        y = float(center_se3.y) - float(centroid[1])
        z = float(center_se3.z)

        obj = objects[object_id]
        obj["position"][frame_idx] = [x, y, z]
        obj["heading"][frame_idx] = heading
        obj["valid"][frame_idx] = True
        obj["length"][frame_idx] = float(ego_state.vehicle_parameters.length)
        obj["width"][frame_idx] = float(ego_state.vehicle_parameters.width)
        obj["height"][frame_idx] = float(ego_state.vehicle_parameters.height)

        if ego_state.dynamic_state_se3 is not None:
            vel = ego_state.dynamic_state_se3.velocity_3d
            obj["velocity"][frame_idx] = [float(vel.x), float(vel.y)]

    # Extract other objects from box detections
    for frame_idx in range(episode_length):
        detections = scene.get_box_detections_at_iteration(frame_idx)
        if not detections:
            continue

        for detection in detections:
            track_token = detection.metadata.track_token
            if track_token in tokens_to_object_id:
                object_id = tokens_to_object_id[track_token]
            else:
                object_id += 1
                tokens_to_object_id[track_token] = object_id

            if object_id not in objects:
                objects[object_id] = {
                    "type": detection.metadata.default_label,
                    "position": np.zeros((episode_length, 3), dtype=np.float32),
                    "heading": np.zeros((episode_length,), dtype=np.float32),
                    "velocity": np.zeros((episode_length, 2), dtype=np.float32),
                    "valid": np.zeros((episode_length,), dtype=np.bool_),
                    "length": np.zeros((episode_length,), dtype=np.float32),
                    "width": np.zeros((episode_length,), dtype=np.float32),
                    "height": np.zeros((episode_length,), dtype=np.float32),
                }

            bbox = detection.bounding_box_se3  # type: ignore[attr-defined]
            center_se3 = bbox.center_se3
            heading = center_se3.pose_se2.yaw

            x = float(center_se3.x) - float(centroid[0])
            y = float(center_se3.y) - float(centroid[1])
            z = float(center_se3.z)

            obj = objects[object_id]
            obj["position"][frame_idx] = [x, y, z]
            obj["heading"][frame_idx] = heading
            obj["valid"][frame_idx] = True
            obj["length"][frame_idx] = float(bbox.length)
            obj["width"][frame_idx] = float(bbox.width)
            obj["height"][frame_idx] = float(bbox.height)

            if detection.velocity_2d is not None:
                obj["velocity"][frame_idx] = [
                    float(detection.velocity_2d.x),
                    float(detection.velocity_2d.y),
                ]

    return objects


def _split_agents_and_objects(objects: dict[int, dict]) -> tuple[dict[int, dict], dict[int, dict]]:
    """Split agents and objects based on type."""
    agents = {}
    non_agents = {}
    for object_id, obj in objects.items():
        if obj["type"] in [
            DefaultBoxDetectionLabel.EGO,
            DefaultBoxDetectionLabel.VEHICLE,
            DefaultBoxDetectionLabel.TRAIN,
            DefaultBoxDetectionLabel.BICYCLE,
            DefaultBoxDetectionLabel.PERSON,
            DefaultBoxDetectionLabel.ANIMAL,
        ]:
            agents[object_id] = obj
        else:
            non_agents[object_id] = obj
    return agents, non_agents


def extract_traffic_lights(scene: SceneAPI, map_api: MapAPI | None, centroid: np.ndarray) -> dict[int, dict]:
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
                position = get_lane_position(map_api, lane_id, centroid)
                elements[lane_id] = {
                    "position": np.array(position, dtype=np.float32),
                    "states": [None] * episode_length,
                    "controlled_lane": lane_id,
                }

            elements[lane_id]["states"][frame_idx] = detection.status

    return elements


def _compute_map_centroid_from_ego_positions(scene: SceneAPI) -> np.ndarray:
    """Compute map centroid using ego vehicle position from py123d SceneAPI."""
    episode_length = scene.number_of_iterations
    positions = np.array([[0.0, 0.0]] * episode_length, dtype=np.float32)

    for frame_idx in range(episode_length):
        ego_state = scene.get_ego_state_at_iteration(frame_idx)

        center_se3 = ego_state.center_se3
        x = float(center_se3.x)
        y = float(center_se3.y)
        positions[frame_idx] = [x, y]

    return positions.mean(axis=0)


def _compute_map_centroid_from_road_layers(map_api: MapAPI) -> np.ndarray:
    """Compute map centroid using available lane/road geometry."""

    points: list[np.ndarray] = []
    for layer in [MapLayer.LANE, MapLayer.ROAD_LINE, MapLayer.ROAD_EDGE]:
        if layer not in map_api._occupancy_maps:
            raise ValueError(f"Map layer {layer} not found in MapAPI")
        ids = map_api._occupancy_maps[layer].ids
        for object_id in ids:
            obj = map_api.get_map_object(object_id, layer)
            if obj is None:
                raise ValueError(f"Map object with ID {object_id} not found in layer {layer}")
            coords = get_object_xy_points(obj)
            if coords is not None and len(coords) > 0:
                points.append(coords)

    if not points:
        raise ValueError("No map geometry found to compute centroid")

    return np.vstack(points).mean(axis=0)


def _iter_map_objects(map_api, centroid, map_only: bool = False):
    all_map_layers = map_api.get_available_map_layers()

    if map_only or map_api.map_is_local:
        for layer in all_map_layers:
            for oid in map_api._occupancy_maps[layer].ids:
                obj = map_api.get_map_object(oid, layer)
                if obj:
                    yield obj
    else:
        layers = map_api.get_map_objects_in_radius(
            Point2D(centroid[0], centroid[1]),
            radius=250.0,
            layers=all_map_layers,
        )
        for layer in all_map_layers:
            yield from layers.get(layer, [])


def extract_map(map_api: MapAPI, centroid: np.ndarray, map_only: bool = False) -> dict[int, dict]:
    """Extract static map elements from a py123d MapAPI."""
    result = {}
    non_lane_objects = []

    for obj in _iter_map_objects(map_api, centroid, map_only):
        if int(obj.layer) == 0:  # LANE
            result[obj.object_id] = convert_map_object_to_static_element(obj, centroid)
        else:
            non_lane_objects.append(obj)

    # Non-lane elements get sequential IDs after max lane ID to avoid collisions
    # (object_id namespaces overlap across map layers)
    next_id = max(result.keys(), default=-1) + 1
    for obj in non_lane_objects:
        if int(obj.layer) not in [1, 2, 4, 5, 6]:
            element = convert_map_object_to_static_element(obj, centroid)
            if element is None:
                continue
            result[next_id] = element
            next_id += 1

    return result


def convert_map_object_to_static_element(map_object, centroid):
    if int(map_object.layer) == 0:  # LANE
        polyline = centered_array(map_object.centerline.array, centroid)
        speed_limit_kmh = mps_to_kmh(map_object.speed_limit_mps)
        return {
            "type": LaneType.SURFACE_STREET,  # TODO: map lane types properly
            "polyline": polyline,
            "speed_limit_mph": kmh_to_mph(speed_limit_kmh) if speed_limit_kmh >= 0 else -1,
            "speed_limit_kmh": speed_limit_kmh,
            "entry_lanes": map_object.predecessor_ids,
            "exit_lanes": map_object.successor_ids,
            "left_neighbor": [map_object.left_lane_id],
            "right_neighbor": [map_object.right_lane_id],
            "left_boundaries": [],
            "right_boundaries": [],
        }

    if int(map_object.layer) == 9:  # ROAD_LINE
        polyline = centered_array(map_object.polyline_3d.array, centroid)
        return {
            "type": map_object.road_line_type,
            "polyline": polyline,
        }

    if int(map_object.layer) == 8:  # ROAD_EDGE
        polyline = centered_array(map_object.polyline_3d.array, centroid)
        return {
            "type": map_object.road_edge_type,
            "polyline": polyline,
        }

    if int(map_object.layer) == 3:  # CROSSWALK
        polygon = centered_array(map_object.outline_3d.array, centroid)
        return {
            "type": types.CROSSWALK,
            "polygon": polygon,
        }

    if int(map_object.layer) == 7:  # STOP_ZONE
        polygon = centered_array(map_object.outline_3d.array, centroid)
        return {
            "type": map_object.stop_zone_type,
            "polygon": polygon,
            "controlled_lanes": map_object.lane_ids,
        }
