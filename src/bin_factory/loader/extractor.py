from __future__ import annotations

import logging
from typing import Any

import numpy as np
from py123d import api as py123d_api
from py123d import geometry as py123d_geometry
from py123d.datatypes import detections, map_objects

from bin_factory import puffer_types, schema
from bin_factory.loader import mapping


logger = logging.getLogger(__name__)

SCENE_MAP_RADIUS = 250.0  # Max distance from ego to map elements for non-map-only scenarios


def extract_scenario(
    py123_arrow: py123d_api.SceneAPI | py123d_api.MapAPI,
) -> tuple[schema.PufferScenario, dict[str, Any]]:
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
    if scenario_id is None:
        raise ValueError("Scenario ID is required to convert scenario")

    agents: dict[int, schema.Track] = {}
    traffic_lights: dict[int, schema.TrafficLightTrack] = {}
    objects: dict[int, schema.Track] = {}
    metadata = schema.ScenarioMetadata(
        id=scenario_id,
        dataset=py123_arrow.dataset,
        scenario_length=0,
        dt=0.0,
        location=map_api.location or "",
    )

    map_only = scene_api is None
    ego_states = (
        [scene_api.get_ego_state_se3_at_iteration(i) for i in range(scene_api.number_of_iterations)]
        if scene_api is not None
        else None
    )

    centroid = _compute_centroid(ego_states, map_api)
    map_elements, stop_zones, map_lane_ids = _extract_map(map_api, centroid, map_only)

    if scene_api is not None and ego_states is not None:
        dt = round(scene_api.scene_metadata.iteration_duration_s, 3)
        if dt <= 0:
            raise ValueError(f"Invalid time step dt={dt} computed from scene metadata")

        all_objects = _extract_objects(scene_api, centroid, ego_states, dt)
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
        metadata.dt = dt

    if not map_api.map_has_z:
        _zero_all_z(agents, objects, traffic_lights, map_elements, stop_zones)

    scenario = schema.PufferScenario(
        agents=agents,
        map=map_elements,
        objects=objects,
        metadata=metadata,
    )
    extras = {"traffic_lights": traffic_lights, "stop_zones": stop_zones}
    return scenario, extras


# ── Main Extraction Functions ───────────────────────


def _extract_objects(
    scene_api: py123d_api.SceneAPI,
    centroid: np.ndarray,
    ego_states: list[Any],
    dt: float,
) -> dict[int, schema.Track]:
    """Extract dynamic objects from 123D box detections and ego state."""
    episode_length = scene_api.number_of_iterations
    objects: dict[int, schema.Track] = {0: _make_empty_track(episode_length, detections.DefaultBoxDetectionLabel.EGO)}
    tokens_to_object_id: dict[str, int] = {}

    # Ego agent is always object ID 0, built from cached ego states
    ego = objects[0]

    for frame_idx, ego_state in enumerate(ego_states):
        if ego_state is None:
            raise ValueError(f"Missing ego state at frame {frame_idx}")

        _write_detection_frame(ego, frame_idx, ego_state.center_se3, ego_state.bounding_box_se3, centroid)

        if ego_state.dynamic_state_se3:
            vel = ego_state.dynamic_state_se3.velocity_3d
            ego.velocity[frame_idx] = [float(vel.x), float(vel.y)]
        # Fallback to finite difference
        elif frame_idx > 0 and ego.valid[frame_idx - 1]:
            delta_pos = ego.position[frame_idx, :2] - ego.position[frame_idx - 1, :2]
            ego.velocity[frame_idx] = delta_pos / dt
        else:
            continue

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

            if detection.velocity_3d is None:
                continue

            obj.velocity[frame_idx] = [float(detection.velocity_3d.x), float(detection.velocity_3d.y)]

    # Backfill first valid frame velocity from next valid frame (no delta available at frame 0)
    for obj in objects.values():
        first_valid = np.argmax(obj.valid)
        if obj.valid[first_valid] and np.all(obj.velocity[first_valid] == 0) and first_valid + 1 < episode_length:
            next_valid = first_valid + 1 + np.argmax(obj.valid[first_valid + 1 :])
            if obj.valid[next_valid] and np.any(obj.velocity[next_valid] != 0):
                obj.velocity[first_valid] = obj.velocity[next_valid]

    return objects


def _extract_traffic_lights(
    scene_api: py123d_api.SceneAPI,
    map_api: py123d_api.MapAPI,
    centroid: np.ndarray,
    lane_ids: set[int],
) -> dict[int, schema.TrafficLightTrack]:
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
                lane = map_api.get_map_object_in_layer(lane_id, map_objects.MapLayer.LANE)
                if lane is None or not isinstance(lane, map_objects.Lane) or len(lane.centerline.array) == 0:
                    logger.debug("TL lane %d has no centerline, skipping", lane_id)
                    continue

                elements[lane_id] = schema.TrafficLightTrack(
                    position=_centered_array(lane.centerline.array[0], centroid).flatten(),
                    states=[puffer_types.TLState.UNKNOWN] * scene_api.number_of_iterations,
                    controlled_lane=lane_id,
                )

            elements[lane_id].states[frame_idx] = mapping.TL_STATE_MAP.get(
                detection.status,
                puffer_types.TLState.UNKNOWN,
            )

    return elements


def _extract_map(
    map_api: py123d_api.MapAPI,
    centroid: np.ndarray,
    map_only: bool = False,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], set[int]]:
    """Extract static map elements from a 123D MapAPI. Returns (elements, stop_zones, lane_ids)."""
    result: dict[int, dict[str, Any]] = {}
    stop_zones: list[dict[str, Any]] = []
    non_lane_objects: list[Any] = []
    undefined_lane: list[int] = []
    layers = [layer for layer in map_api.available_map_layers if layer in mapping.SUPPORTED_MAP_LAYERS]

    if map_only or map_api.map_is_per_log:
        map_objs = map_api.get_all_map_objects_in_layers(layers)
    else:
        map_objects_by_layer = map_api.get_map_objects_in_radius(
            py123d_geometry.Point3D(centroid[0], centroid[1], centroid[2]),
            radius=SCENE_MAP_RADIUS,
            layers=layers,
        )
        map_objs = [obj for layer in layers for obj in map_objects_by_layer.get(layer, [])]

    # Lanes first — other elements reference lane IDs
    for obj in map_objs:
        if obj.layer == map_objects.MapLayer.LANE:
            element = _write_map_object(obj, centroid)
            if element is None:
                continue
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
        element = _write_map_object(obj, centroid)
        if element is None:
            continue
        if obj.layer == map_objects.MapLayer.STOP_ZONE:
            stop_zones.append(element)
        else:
            result[next_id] = element
            next_id += 1

    return result, stop_zones, lane_ids


# ── Writer functions ───────────────────────


def _write_map_object(map_object: Any, centroid: np.ndarray) -> dict[str, Any] | None:
    """Convert 123D map object to unified static element dict with puffer types."""
    layer = map_object.layer
    if mapping.MAP_TYPE_MAP.get(layer) is None:
        return None

    if layer == map_objects.MapLayer.LANE:
        puffer_type = mapping.LANE_TYPE_MAP.get(map_object.lane_type)
        if puffer_type is None:
            return None
        if not map_object.speed_limit_mps or np.isnan(map_object.speed_limit_mps):
            speed_limit_mps = -1.0
        else:
            speed_limit_mps = float(map_object.speed_limit_mps)
        left_neighbor = [lid] if (lid := getattr(map_object, "left_lane_id", None)) is not None else []
        right_neighbor = [rid] if (rid := getattr(map_object, "right_lane_id", None)) is not None else []
        return {
            "type": puffer_type,
            "polyline": _centered_array(map_object.centerline.array, centroid),
            "speed_limit_mps": speed_limit_mps,
            "entry_lanes": map_object.predecessor_ids,
            "exit_lanes": map_object.successor_ids,
            "left_boundary": _centered_array(map_object.left_boundary.array, centroid),
            "right_boundary": _centered_array(map_object.right_boundary.array, centroid),
            "left_neighbor": left_neighbor,
            "right_neighbor": right_neighbor,
        }

    if layer == map_objects.MapLayer.ROAD_LINE:
        puffer_type = mapping.ROAD_LINE_TYPE_MAP.get(map_object.road_line_type)
        if puffer_type is None:
            return None
        return {
            "type": puffer_type,
            "polyline": _centered_array(map_object.polyline_3d.array, centroid),
        }

    if layer == map_objects.MapLayer.ROAD_EDGE:
        puffer_type = mapping.ROAD_EDGE_TYPE_MAP.get(map_object.road_edge_type)
        if puffer_type is None:
            return None
        return {
            "type": puffer_type,
            "polyline": _centered_array(map_object.polyline_3d.array, centroid),
        }

    if layer == map_objects.MapLayer.CROSSWALK:
        puffer_type = mapping.CROSSWALK_TYPE
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

    raise ValueError(f"Unsupported map layer {layer} for object ID {map_object.object_id}")


def _write_detection_frame(
    obj: schema.Track,
    frame_idx: int,
    center_se3: Any,
    bbox: Any,
    centroid: np.ndarray,
) -> None:
    obj.position[frame_idx] = [
        float(center_se3.x) - float(centroid[0]),
        float(center_se3.y) - float(centroid[1]),
        float(center_se3.z) - float(centroid[2]) - float(bbox.height) / 2.0,
    ]
    obj.heading[frame_idx] = center_se3.pose_se2.yaw
    obj.valid[frame_idx] = 1
    obj.length[frame_idx] = float(bbox.length)
    obj.width[frame_idx] = float(bbox.width)
    obj.height[frame_idx] = float(bbox.height)


def _make_empty_track(episode_length: int, agent_type: object) -> schema.Track:
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


def _compute_centroid(ego_states: list[Any] | None, map_api: py123d_api.MapAPI) -> np.ndarray:
    """Compute 3D scene centroid from ego trajectory, falling back to road geometry mean."""
    if ego_states is not None:
        positions = np.array(
            [
                [float(s.center_se3.x), float(s.center_se3.y), float(s.center_se3.z)]
                for s in ego_states
                if s is not None
            ],
            dtype=np.float64,
        )
        if len(positions) > 0:
            return positions.mean(axis=0)

    # Fallback: road geometry centroid
    points = [
        coords
        for obj in map_api.get_all_map_objects_in_layer(map_objects.MapLayer.LANE)
        if (coords := obj.centerline.array) is not None and len(coords) > 0
    ]
    if points:
        return np.vstack(points).mean(axis=0)
    return np.zeros(3, dtype=np.float64)


# ── Corrective functions ───────────────────────


def _zero_all_z(agents, objects, traffic_lights, map_elements, stop_zones):
    """Force all Z values to 0 when map has no Z data."""
    for track in [*agents.values(), *objects.values()]:
        track.position[:, 2] = 0.0
    for tl in traffic_lights.values():
        tl.position[2] = 0.0
    for element in [*map_elements.values(), *stop_zones]:
        for key in ("polyline", "left_boundary", "right_boundary", "polygon"):
            if key in element:
                element[key][:, 2] = 0.0


def _fix_lane_topology(
    lanes: dict[int, dict[str, Any]],
    undefined_lane_ids: list[int],
    valid_lane_ids: set[int],
) -> None:
    """Infer undefined lane types from neighbors + fix reversed entry/exit refs (nuPlan bandage)."""
    for lane_id, lane in lanes.items():
        if lane_id in undefined_lane_ids:
            connected_types = {
                lanes[nid]["type"] for key in ("entry_lanes", "exit_lanes") for nid in lane.get(key, []) if nid in lanes
            }
            if len(connected_types) == 1:
                lane["type"] = connected_types.pop()

        lane_start, lane_end = lane["polyline"][0], lane["polyline"][-1]

        new_entry, new_exit = [], []
        for entry_id in lane.get("entry_lanes", []):
            if entry_id not in lanes or "polyline" not in lanes[entry_id]:
                new_entry.append(entry_id)
                continue
            entry_end = lanes[entry_id]["polyline"][-1]
            if np.linalg.norm(entry_end - lane_start) > np.linalg.norm(entry_end - lane_end):
                new_exit.append(entry_id)
            else:
                new_entry.append(entry_id)

        for exit_id in lane.get("exit_lanes", []):
            if exit_id not in lanes or "polyline" not in lanes[exit_id]:
                new_exit.append(exit_id)
                continue
            exit_start = lanes[exit_id]["polyline"][0]
            if np.linalg.norm(exit_start - lane_end) > np.linalg.norm(exit_start - lane_start):
                new_entry.append(exit_id)
            else:
                new_exit.append(exit_id)

        lane["entry_lanes"] = new_entry
        lane["exit_lanes"] = new_exit

    for element in lanes.values():
        element["entry_lanes"] = [lid for lid in element["entry_lanes"] if lid in valid_lane_ids]
        element["exit_lanes"] = [lid for lid in element["exit_lanes"] if lid in valid_lane_ids]


def _centered_array(array: np.ndarray, center: np.ndarray) -> np.ndarray:
    return array.astype(np.float64) - center
