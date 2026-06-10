from __future__ import annotations

from typing import Any

import numpy as np
import shapely
from py123d import api as py123d_api
from py123d.datatypes import detections, map_objects

from bin_factory import puffer_types, schema
from bin_factory.loader import mapping
from bin_factory.log_context import log


SCENE_MAP_MARGIN = 250.0  # Lateral buffer (m) around the ego path for non-map-only scenarios


def extract_scenario(
    py123_arrow: py123d_api.SceneAPI | py123d_api.MapAPI,
    scenario_id_field: str = "scene_uuid",
) -> tuple[schema.PufferScenario, schema.ExtractionExtras]:
    """Convert 123D SceneAPI or MapAPI to (PufferScenario, extras).

    scenario_id_field picks the py123d attribute used as metadata.id (scene_uuid|log_name|location).
    extras is an ExtractionExtras (traffic_lights, stop_zones) — consumed by the traffic_controls processor.
    """
    if isinstance(py123_arrow, py123d_api.MapAPI):
        scene_api = None
        map_api = py123_arrow
        scenario_id = py123_arrow.location  # map-only objects only expose `location`
    else:
        scene_api = py123_arrow
        map_api = scene_api.get_map_api()
        scenario_id = getattr(scene_api, scenario_id_field)

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

    if not map_only and ego_states is None:
        raise ValueError("Ego states are required to convert scenario with SceneAPI")

    centroid = _compute_centroid(ego_states, map_api)
    map_elements, stop_zones, map_lane_ids = _extract_map(map_api, centroid, ego_states, map_only)

    if scene_api is not None and ego_states is not None:
        dt = round(scene_api.scene_metadata.iteration_duration_s, 3)
        if dt <= 0:
            raise ValueError(f"Invalid time step dt={dt} computed from scene metadata")

        all_objects, tokens_to_object_id = _extract_objects(scene_api, centroid, ego_states)
        for oid, obj in all_objects.items():
            label = obj.type
            if label in mapping.AGENT_TYPE_MAP:
                obj.type = mapping.AGENT_TYPE_MAP[label]
                agents[oid] = obj
            elif label in mapping.OBJECT_TYPE_MAP:
                obj.type = mapping.OBJECT_TYPE_MAP[label]
                objects[oid] = obj

        _extract_prediction_targets(scene_api, tokens_to_object_id, agents, metadata)

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
    extras = schema.ExtractionExtras(traffic_lights=traffic_lights, stop_zones=stop_zones)
    return scenario, extras


# ── Main Extraction Functions ───────────────────────


def _extract_objects(
    scene_api: py123d_api.SceneAPI,
    centroid: np.ndarray,
    ego_states: list[Any],
) -> tuple[dict[int, schema.Track], dict[str, int]]:
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
        ego.velocity[frame_idx] = [
            float(ego_state.box_detection_se3.velocity_3d.x),
            float(ego_state.box_detection_se3.velocity_3d.y),
        ]

    # Detections for all other agents
    next_object_id = 0
    for frame_idx in range(episode_length):
        detections_list = scene_api.get_box_detections_se3_at_iteration(frame_idx)
        if not detections_list:
            continue
        for detection in detections_list:
            track_token = detection.attributes.track_token
            object_id = tokens_to_object_id.get(track_token)
            if object_id is None:
                next_object_id += 1
                object_id = next_object_id
                tokens_to_object_id[track_token] = object_id
                objects[object_id] = _make_empty_track(episode_length, detection.attributes.default_label)

            obj = objects[object_id]
            _write_detection_frame(
                obj,
                frame_idx,
                detection.bounding_box_se3.center_se3,
                detection.bounding_box_se3,
                centroid,
            )

            if detection.velocity_3d is not None:
                obj.velocity[frame_idx] = [float(detection.velocity_3d.x), float(detection.velocity_3d.y)]

    return objects, tokens_to_object_id


def _extract_prediction_targets(
    scene_api: py123d_api.SceneAPI,
    tokens_to_object_id: dict[str, int],
    agents: dict[int, schema.Track],
    metadata: schema.ScenarioMetadata,
) -> None:
    """Populate metadata objects_of_interest / tracks_to_predict from the WOD-Motion aux modality."""
    aux = scene_api.get_all_custom_modality_metadatas().get("aux")
    if aux is None:
        return
    meta = aux.metadata
    idx2tok = meta["track_index_to_token"]

    def _resolve(track_indices):
        ids = (tokens_to_object_id.get(idx2tok.get(idx)) for idx in track_indices)
        return [oid for oid in ids if oid in agents]

    metadata.objects_of_interest = _resolve(meta["objects_of_interest"])
    metadata.tracks_to_predict = _resolve(entry["track_index"] for entry in meta["tracks_to_predict"])


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
                log.debug("TL detection references unknown lane %d, skipping", lane_id)
                continue
            if lane_id not in elements:
                lane = map_api.get_map_object_in_layer(lane_id, map_objects.MapLayer.LANE)
                if lane is None or not isinstance(lane, map_objects.Lane) or len(lane.centerline.array) == 0:
                    log.debug("TL lane %d has no centerline, skipping", lane_id)
                    continue

                elements[lane_id] = schema.TrafficLightTrack(
                    position=_centered_array(lane.centerline_3d.array[0], centroid).flatten(),
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
    ego_states: list[Any] | None = None,
    map_only: bool = False,
) -> tuple[dict[int, schema.MapElement], list[schema.StopZone], set[int]]:
    """Extract static map elements from a 123D MapAPI. Returns (elements, stop_zones, lane_ids)."""
    result: dict[int, schema.MapElement] = {}
    stop_zones: list[schema.StopZone] = []
    non_lane_objects: list[Any] = []
    undefined_lane: list[int] = []
    layers = [layer for layer in map_api.available_map_layers if layer in mapping.SUPPORTED_MAP_LAYERS]

    if map_only or map_api.map_is_per_log:
        map_objs = map_api.get_all_map_objects_in_layers(layers)
    else:
        corridor = shapely.LineString([(s.center_se3.x, s.center_se3.y) for s in ego_states]).buffer(SCENE_MAP_MARGIN)
        map_objects_by_layer = map_api.query(corridor, layers, predicate="intersects")
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


def _write_map_object(map_object: Any, centroid: np.ndarray) -> schema.MapElement | schema.StopZone | None:
    """Convert 123D map object to a MapElement (or StopZone) with puffer types."""
    layer = map_object.layer
    if layer not in mapping.SUPPORTED_MAP_LAYERS:
        return None

    if layer == map_objects.MapLayer.LANE:
        puffer_type = mapping.LANE_TYPE_MAP.get(map_object.lane_type)
        if puffer_type is None:
            return None
        return schema.MapElement(
            type=puffer_type,
            polyline=_centered_array(map_object.centerline_3d.array, centroid),
            speed_limit_mps=_speed_limit(map_object.speed_limit_mps),
            entry_lanes=map_object.predecessor_ids,
            exit_lanes=map_object.successor_ids,
            left_boundary=_centered_array(map_object.left_boundary_3d.array, centroid),
            right_boundary=_centered_array(map_object.right_boundary_3d.array, centroid),
            left_neighbor=_optional_id_list(getattr(map_object, "left_lane_id", None)),
            right_neighbor=_optional_id_list(getattr(map_object, "right_lane_id", None)),
        )

    if layer in (map_objects.MapLayer.ROAD_LINE, map_objects.MapLayer.ROAD_EDGE):
        if layer == map_objects.MapLayer.ROAD_LINE:
            puffer_type = mapping.ROAD_LINE_TYPE_MAP.get(map_object.road_line_type)
        else:
            puffer_type = mapping.ROAD_EDGE_TYPE_MAP.get(map_object.road_edge_type)
        if puffer_type is None:
            return None
        return schema.MapElement(
            type=puffer_type,
            polyline=_centered_array(map_object.polyline_3d.array, centroid),
        )

    if layer == map_objects.MapLayer.CROSSWALK:
        return schema.MapElement(
            type=mapping.CROSSWALK_TYPE,
            polygon=_centered_array(map_object.outline_3d.array, centroid),
        )

    if layer == map_objects.MapLayer.STOP_ZONE:
        puffer_type = mapping.STOP_ZONE_TYPE_MAP.get(map_object.stop_zone_type)
        if puffer_type is None:
            return None
        return schema.StopZone(
            type=puffer_type,
            polygon=_centered_array(map_object.outline_3d.array, centroid),
            controlled_lanes=map_object.lane_ids,
        )

    return None


def _speed_limit(speed_limit_mps):
    return -1.0 if not speed_limit_mps or np.isnan(speed_limit_mps) else float(speed_limit_mps)


def _optional_id_list(value):
    return [value] if value is not None else []


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
        if (coords := obj.centerline_3d.array) is not None and len(coords) > 0
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
            arr = getattr(element, key, None)
            if arr is not None:
                arr[:, 2] = 0.0


def _fix_lane_topology(
    lanes: dict[int, schema.MapElement],
    undefined_lane_ids: list[int],
    valid_lane_ids: set[int],
) -> None:
    """Infer undefined lane types from neighbors + fix reversed entry/exit refs (nuPlan bandage)."""
    for lane_id, lane in lanes.items():
        if lane_id in undefined_lane_ids:
            connected_types = {
                lanes[nid].type for key in ("entry_lanes", "exit_lanes") for nid in getattr(lane, key) if nid in lanes
            }
            if len(connected_types) == 1:
                lane.type = connected_types.pop()

        lane_start, lane_end = lane.polyline[0], lane.polyline[-1]

        new_entry, new_exit = [], []
        for entry_id in lane.entry_lanes:
            if entry_id not in lanes or lanes[entry_id].polyline is None:
                new_entry.append(entry_id)
                continue
            entry_end = lanes[entry_id].polyline[-1]
            if np.linalg.norm(entry_end - lane_start) > np.linalg.norm(entry_end - lane_end):
                new_exit.append(entry_id)
            else:
                new_entry.append(entry_id)

        for exit_id in lane.exit_lanes:
            if exit_id not in lanes or lanes[exit_id].polyline is None:
                new_exit.append(exit_id)
                continue
            exit_start = lanes[exit_id].polyline[0]
            if np.linalg.norm(exit_start - lane_end) > np.linalg.norm(exit_start - lane_start):
                new_entry.append(exit_id)
            else:
                new_exit.append(exit_id)

        lane.entry_lanes = new_entry
        lane.exit_lanes = new_exit

    for element in lanes.values():
        element.entry_lanes = [lid for lid in element.entry_lanes if lid in valid_lane_ids]
        element.exit_lanes = [lid for lid in element.exit_lanes if lid in valid_lane_ids]


def _centered_array(array: np.ndarray, center: np.ndarray) -> np.ndarray:
    return array.astype(np.float64) - center
