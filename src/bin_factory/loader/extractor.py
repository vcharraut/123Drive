from __future__ import annotations

import logging

import numpy as np
from py123d import api as py123d_api
from py123d import geometry as py123d_geometry
from py123d.datatypes import detections, map_objects

from bin_factory import schema
from bin_factory import types as puffer_types
from bin_factory.loader import mapping


logger = logging.getLogger(__name__)


def extract_scenario(py123_arrow: py123d_api.SceneAPI | py123d_api.MapAPI) -> tuple[schema.PufferScenario, dict]:
    """Convert 123D SceneAPI or MapAPI to (PufferScenario, extras).

    extras = {"traffic_lights": ..., "stop_zones": ...} — consumed by traffic_controls processor.
    """
    if isinstance(py123_arrow, py123d_api.MapAPI):
        scene_api = None
        map_api = py123_arrow
        scenario_id = py123_arrow.location
    else:
        scene_api = py123_arrow
        map_api = scene_api.get_map_api()
        scenario_id = scene_api.scene_uuid

    if map_api is None:
        raise ValueError("Map API is required to convert scenario")

    agents = {}
    traffic_lights = {}
    objects = {}
    metadata = schema.ScenarioMetadata(
        id=scenario_id,
        dataset=py123_arrow.dataset,
        scenario_length=0,
        timestep_seconds=0.0,
    )

    map_only = scene_api is None
    ego_states = None
    if not map_only:
        ego_states = [scene_api.get_ego_state_se3_at_iteration(i) for i in range(scene_api.number_of_iterations)]

    centroid = _compute_centroid(ego_states, map_api)
    map_elements, stop_zones, map_lane_ids = _extract_map(map_api, centroid, map_only)

    if not map_only:
        all_objects = _extract_objects(scene_api, centroid, ego_states)
        for oid, obj in all_objects.items():
            label = obj.type
            if label in mapping.AGENT_TYPE_MAP:
                obj.type = mapping.AGENT_TYPE_MAP[label]
                agents[oid] = obj
            elif label in mapping.OBJECT_TYPE_MAP:
                obj.type = mapping.OBJECT_TYPE_MAP[label]
                objects[oid] = obj

        traffic_lights = _extract_traffic_lights(scene_api, map_api, centroid, map_lane_ids)
        metadata.scenario_length = scene_api.number_of_iterations
        metadata.timestep_seconds = scene_api.log_metadata.timestep_seconds

    scenario = schema.PufferScenario(
        agents=agents,
        map=map_elements,
        objects=objects,
        metadata=metadata,
    )
    extras = {"traffic_lights": traffic_lights, "stop_zones": stop_zones}
    return scenario, extras


def _extract_objects(scene_api: py123d_api.SceneAPI, centroid: np.ndarray, ego_states: list) -> dict[int, dict]:
    """Extract dynamic objects from 123D box detections and ego state."""
    episode_length = scene_api.number_of_iterations
    objects: dict[int, dict] = {0: _make_empty_track(episode_length, detections.DefaultBoxDetectionLabel.EGO)}
    tokens_to_object_id: dict[str, int] = {}

    # Ego agent is always object ID 0, built from cached ego states
    ego = objects[0]
    timestep = scene_api.log_metadata.timestep_seconds

    for frame_idx, ego_state in enumerate(ego_states):
        if ego_state is None:
            raise ValueError(f"Missing ego state at frame {frame_idx}")

        _write_detection_frame(ego, frame_idx, ego_state.center_se3, ego_state.bounding_box_se3, centroid)

        if ego_state.dynamic_state_se3 is not None:
            vel = ego_state.dynamic_state_se3.velocity_3d
            ego.velocity[frame_idx] = [float(vel.x), float(vel.y)]
            continue

        if frame_idx == 0 or not ego.valid[frame_idx - 1] or timestep <= 0:
            continue

        delta_pos = ego.position[frame_idx, :2] - ego.position[frame_idx - 1, :2]
        ego.velocity[frame_idx] = delta_pos / timestep

    # Detections for all other agents
    next_object_id = 0
    for frame_idx in range(episode_length):
        detections_list = scene_api.get_box_detections_se3_at_iteration(frame_idx)
        if not detections_list:
            continue
        for detection in detections_list:
            track_token = detection.attributes.track_token
            if track_token in tokens_to_object_id:
                object_id = tokens_to_object_id[track_token]
            else:
                next_object_id += 1
                object_id = next_object_id
                tokens_to_object_id[track_token] = object_id

            if object_id not in objects:
                objects[object_id] = _make_empty_track(episode_length, detection.attributes.default_label)

            bbox = detection.bounding_box_se3
            obj = objects[object_id]
            _write_detection_frame(obj, frame_idx, bbox.center_se3, bbox, centroid)

            if detection.velocity_2d is None:
                continue

            obj.velocity[frame_idx] = [float(detection.velocity_2d.x), float(detection.velocity_2d.y)]

    return objects


def _extract_traffic_lights(
    scene_api: py123d_api.SceneAPI,
    map_api: py123d_api.MapAPI,
    centroid: np.ndarray,
    lane_ids: set[int],
) -> dict[int, dict]:
    """Extract dynamic traffic light states from 123D logs."""
    elements: dict[int, schema.TrafficLightTrack] = {}

    for frame_idx in range(scene_api.number_of_iterations):
        traffic_lights = scene_api.get_traffic_light_detections_at_iteration(frame_idx)
        if not traffic_lights:
            continue

        for detection in traffic_lights:
            lane_id = int(detection.lane_id)
            if lane_id not in lane_ids:
                logger.debug("TL detection references unknown lane %d, skipping", lane_id)
                continue
            if lane_id not in elements:
                position = _get_lane_position(map_api, lane_id, centroid)
                elements[lane_id] = schema.TrafficLightTrack(
                    position=np.array(position, dtype=np.float64),
                    states=[puffer_types.TLState.UNKNOWN] * scene_api.number_of_iterations,
                    controlled_lane=lane_id,
                )

            elements[lane_id].states[frame_idx] = mapping.TL_STATE_MAP.get(
                detection.status, puffer_types.TLState.UNKNOWN
            )

    return elements


def _compute_centroid(ego_states: list | None, map_api: py123d_api.MapAPI) -> np.ndarray:
    """Compute scene centroid from ego trajectory, falling back to road geometry."""
    if ego_states is not None:
        positions = np.array(
            [[float(s.center_se3.x), float(s.center_se3.y)] for s in ego_states if s is not None],
            dtype=np.float64,
        )
        if len(positions) > 0:
            return positions.mean(axis=0)

    # Fallback: road geometry centroid
    road_layers = [map_objects.MapLayer.LANE, map_objects.MapLayer.ROAD_LINE, map_objects.MapLayer.ROAD_EDGE]
    points = [
        coords
        for obj in _get_map_objects(map_api, road_layers)
        if (coords := _get_object_xy_points(obj)) is not None and len(coords) > 0
    ]
    if points:
        return np.vstack(points).mean(axis=0)
    return np.zeros(2, dtype=np.float64)


def _extract_map(map_api, centroid: np.ndarray, map_only: bool = False) -> tuple[dict[int, dict], list[dict], set[int]]:
    """Extract static map elements from a 123D MapAPI. Returns (elements, stop_zones, lane_ids)."""
    result = {}
    stop_zones = []
    non_lane_objects = []
    undefined_lane = []
    all_map_layers = {
        map_objects.MapLayer.LANE,
        # map_objects.MapLayer.LANE_GROUP,
        # map_objects.MapLayer.INTERSECTION,
        map_objects.MapLayer.CROSSWALK,
        # map_objects.MapLayer.WALKWAY,
        # map_objects.MapLayer.CARPARK,
        # map_objects.MapLayer.GENERIC_DRIVABLE,
        map_objects.MapLayer.STOP_ZONE,
        map_objects.MapLayer.ROAD_EDGE,
        map_objects.MapLayer.ROAD_LINE,
    }

    if map_only or map_api.map_is_per_log:
        map_objs = _get_map_objects(map_api, all_map_layers)
    else:
        map_objects_by_layer = map_api.get_map_objects_in_radius(
            py123d_geometry.Point2D(centroid[0], centroid[1]),
            radius=250.0,
            layers=all_map_layers,
        )
        map_objs = [obj for layer in all_map_layers for obj in map_objects_by_layer.get(layer, [])]

    # Lanes first — other elements reference lane IDs
    for obj in map_objs:
        if obj.layer == map_objects.MapLayer.LANE:
            element = _convert_map_object_to_static_element(obj, centroid)
            if element is not None:
                result[obj.object_id] = element
                if obj.lane_type == map_objects.LaneType.UNDEFINED:
                    undefined_lane.append(obj.object_id)
        else:
            non_lane_objects.append(obj)

    lane_ids = set(result.keys())
    _fix_lane_topology(result, undefined_lane, lane_ids)

    # Non-lane elements get sequential IDs after max lane ID to avoid collisions
    next_id = max(result.keys(), default=-1) + 1
    for obj in non_lane_objects:
        element = _convert_map_object_to_static_element(obj, centroid)
        if element is None:
            continue
        if obj.layer == map_objects.MapLayer.STOP_ZONE:
            stop_zones.append(element)
        else:
            result[next_id] = element
            next_id += 1

    return result, stop_zones, lane_ids


def _convert_map_object_to_static_element(map_object, centroid: np.ndarray) -> dict | None:
    """Convert 123D map object to unified static element dict with puffer types."""
    layer = map_object.layer
    type_map_for_layer = mapping.ROAD_TYPE_MAP.get(layer)
    if type_map_for_layer is None:
        raise ValueError(f"Unsupported map object layer: {layer}")

    if layer == map_objects.MapLayer.LANE:
        puffer_type = type_map_for_layer.get(map_object.lane_type, -1)
        if puffer_type == -1:
            return None
        if not map_object.speed_limit_mps or np.isnan(map_object.speed_limit_mps):
            speed_limit_mps = -1.0
        else:
            speed_limit_mps = float(map_object.speed_limit_mps)
        return {
            "type": puffer_type,
            "polyline": _centered_array(map_object.centerline.array, centroid),
            "speed_limit_mps": speed_limit_mps,
            "entry_lanes": map_object.predecessor_ids,
            "exit_lanes": map_object.successor_ids,
            "left_boundary": map_object.left_boundary,
            "right_boundary": map_object.right_boundary,
            "left_lane": map_object.left_lane,
            "right_lane": map_object.right_lane,
        }

    if layer in (map_objects.MapLayer.ROAD_LINE, map_objects.MapLayer.ROAD_EDGE):
        subtype = map_object.road_line_type if layer == map_objects.MapLayer.ROAD_LINE else map_object.road_edge_type
        puffer_type = type_map_for_layer.get(subtype, -1)
        if puffer_type == -1:
            return None
        return {
            "type": puffer_type,
            "polyline": _centered_array(map_object.polyline_3d.array, centroid),
        }

    if layer == map_objects.MapLayer.CROSSWALK:
        puffer_type = type_map_for_layer.get(None, -1)
        if puffer_type == -1:
            return None
        return {
            "type": puffer_type,
            "polygon": _centered_array(map_object.outline_3d.array, centroid),
        }

    if layer == map_objects.MapLayer.STOP_ZONE:
        puffer_type = mapping.STOP_ZONE_TYPE_MAP.get(map_object.stop_zone_type)
        if puffer_type is None:
            return None
        return {
            "type": puffer_type,
            "polygon": _centered_array(map_object.outline_3d.array, centroid),
            "controlled_lanes": map_object.lane_ids,
        }

    raise ValueError(f"Unsupported map object layer: {layer}")


def _get_map_objects(map_api: py123d_api.MapAPI, layers: list[map_objects.MapLayer]) -> list:
    return [obj for layer in layers for obj in map_api.get_all_map_objects_in_layer(layer)]


def _make_empty_track(episode_length, agent_type):
    return schema.Track(
        type=agent_type,
        position=np.zeros((episode_length, 3), dtype=np.float64),
        heading=np.zeros((episode_length,), dtype=np.float64),
        velocity=np.zeros((episode_length, 2), dtype=np.float64),
        valid=np.zeros((episode_length,), dtype=np.int32),
        length=np.zeros((episode_length,), dtype=np.float64),
        width=np.zeros((episode_length,), dtype=np.float64),
        height=np.zeros((episode_length,), dtype=np.float64),
    )


def _write_detection_frame(obj, frame_idx, center_se3, bbox, centroid):
    obj.position[frame_idx] = [
        float(center_se3.x) - float(centroid[0]),
        float(center_se3.y) - float(centroid[1]),
        float(center_se3.z),
    ]
    obj.heading[frame_idx] = center_se3.pose_se2.yaw
    obj.valid[frame_idx] = 1
    obj.length[frame_idx] = float(bbox.length)
    obj.width[frame_idx] = float(bbox.width)
    obj.height[frame_idx] = float(bbox.height)


def _fix_lane_topology(lanes, undefined_lane_ids, valid_lane_ids):
    """Infer undefined lane types from neighbors + fix reversed entry/exit refs (nuPlan bandage)."""
    for lane_id, lane in lanes.items():
        if lane_id in undefined_lane_ids:
            connected_types = {
                lanes[nid]["type"] for key in ("entry_lanes", "exit_lanes") for nid in lane.get(key, []) if nid in lanes
            }
            if len(connected_types) == 1:
                lane["type"] = connected_types.pop()

        lane_start, lane_end = lane["polyline"][0], lane["polyline"][-1]

        for entry_id in list(lane.get("entry_lanes", [])):
            if entry_id not in lanes or "polyline" not in lanes[entry_id]:
                continue
            entry_end = lanes[entry_id]["polyline"][-1]
            if np.linalg.norm(entry_end - lane_start) > np.linalg.norm(entry_end - lane_end):
                lane["entry_lanes"].remove(entry_id)
                lane["exit_lanes"].append(entry_id)

        for exit_id in list(lane.get("exit_lanes", [])):
            if exit_id not in lanes or "polyline" not in lanes[exit_id]:
                continue
            exit_start = lanes[exit_id]["polyline"][0]
            if np.linalg.norm(exit_start - lane_end) > np.linalg.norm(exit_start - lane_start):
                lane["exit_lanes"].remove(exit_id)
                lane["entry_lanes"].append(exit_id)

    for element in lanes.values():
        element["entry_lanes"] = [lid for lid in element["entry_lanes"] if lid in valid_lane_ids]
        element["exit_lanes"] = [lid for lid in element["exit_lanes"] if lid in valid_lane_ids]


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


def _get_lane_position(map_api: py123d_api.MapAPI, lane_id: int, center: np.ndarray) -> list[float]:
    lane = map_api.get_map_object_in_layer(lane_id, map_objects.MapLayer.LANE)
    if lane is None or not isinstance(lane, map_objects.Lane):
        raise ValueError(f"Lane {lane_id} not found or has no centerline")

    if len(lane.centerline.array) == 0:
        raise ValueError(f"Lane {lane_id} has empty centerline")

    point = lane.centerline.array[0]
    x = float(point[0]) - float(center[0])
    y = float(point[1]) - float(center[1])
    z = float(point[2]) if len(point) > 2 else 0.0

    return [x, y, z]
