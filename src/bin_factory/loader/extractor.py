from __future__ import annotations

import collections
import dataclasses
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import shapely
from py123d import api as py123d_api
from py123d.datatypes import detections, map_objects

from bin_factory import puffer_types, schema
from bin_factory.loader import mapping
from bin_factory.log_context import log


if TYPE_CHECKING:
    from collections.abc import Iterable


SCENE_MAP_MARGIN = 250.0  # Lateral buffer (m) around the ego path for non-map-only scenarios
MIN_LINE_LENGTH = 0.1  # m; shorter road lines are dropped
LINE_DUPLICATE_DISTANCE = 0.05  # m; a road line this close (Hausdorff) to another of its type is a duplicate
HOOK_MAX_LENGTH = 0.5  # m; an end segment shorter than this that turns sharply is a hook, not lane geometry
HOOK_MIN_TURN = np.radians(30.0)


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

    if not map_only and (ego_states is None or any(state is None for state in ego_states)):
        raise ValueError("Ego states are required at every frame to convert scenario with SceneAPI")

    centroid = _compute_centroid(ego_states, map_api)
    map_elements, stop_zones, map_lane_ids = _extract_map(map_api, centroid, ego_states, map_only)

    if scene_api is not None and ego_states is not None:
        dt = float(scene_api.scene_metadata.iteration_duration_s)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"Invalid time step dt={dt} computed from scene metadata")

        all_objects, labels, tokens_to_object_id = _extract_objects(scene_api, centroid, ego_states)
        for obj in all_objects.values():
            _fill_missing_velocities(obj, dt)
        for oid, obj in all_objects.items():
            label = labels[oid]
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
) -> tuple[dict[int, schema.Track], dict[int, detections.DefaultBoxDetectionLabel], dict[str, int]]:
    """Extract dynamic objects from 123D box detections and ego state.

    Returns (tracks, labels, tokens_to_object_id) — labels carry the 123D detection
    label per object id until extract_scenario maps it to a puffer type.
    """
    episode_length = scene_api.number_of_iterations
    objects: dict[int, schema.Track] = {0: _make_empty_track(episode_length)}
    labels: dict[int, detections.DefaultBoxDetectionLabel] = {0: detections.DefaultBoxDetectionLabel.EGO}
    tokens_to_object_id: dict[str, int] = {}

    # Ego agent is always object ID 0, built from cached ego states
    ego = objects[0]

    for frame_idx, ego_state in enumerate(ego_states):
        if ego_state is None:
            raise ValueError(f"Missing ego state at frame {frame_idx}")

        _write_detection_frame(ego, frame_idx, ego_state.center_se3, ego_state.bounding_box_se3, centroid)
        # PY123D-REPORT[nuplan]: raw ego_pose velocity has a lateral bias (body vy ~ -2% of vx) and reads ~1.4%
        # fast against the logged poses. Leave it NaN so _fill_missing_velocities derives it from positions.
        if scene_api.dataset.startswith("nuplan"):
            continue
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
                objects[object_id] = _make_empty_track(episode_length)
                labels[object_id] = detection.attributes.default_label

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

    return objects, labels, tokens_to_object_id


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

    def _resolve(track_indices: Iterable[int]) -> list[int]:
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
    layers: list[str | map_objects.MapLayer] = [
        layer for layer in map_api.available_map_layers if layer in mapping.SUPPORTED_MAP_LAYERS
    ]

    map_objs: list[Any]
    cropped = not (map_only or map_api.map_is_per_log or ego_states is None)
    if not cropped:
        map_objs = list(map_api.get_all_map_objects_in_layers(layers))
    else:
        ego_xy = [(s.center_se3.x, s.center_se3.y) for s in ego_states]
        centerline = shapely.Point(ego_xy[0]) if len(ego_xy) == 1 else shapely.LineString(ego_xy)
        corridor = centerline.buffer(SCENE_MAP_MARGIN)
        map_objects_by_layer = map_api.query(corridor, layers, predicate="intersects")
        map_objs = cast("list[Any]", [obj for layer in layers for obj in map_objects_by_layer.get(layer, [])])

    # Lanes first — other elements reference lane IDs
    for obj in map_objs:
        if obj.layer == map_objects.MapLayer.LANE:
            element = _write_map_object(obj, centroid)
            if not isinstance(element, schema.MapElement):
                continue
            result[obj.object_id] = element
            if obj.lane_type == map_objects.LaneType.UNDEFINED:
                undefined_lane.append(obj.object_id)
        else:
            non_lane_objects.append(obj)

    lane_ids = set(result.keys())
    _fix_lane_topology(result, undefined_lane, lane_ids)
    if cropped and any(lane.speed_limit_mps <= 0 for lane in result.values()):
        # Fill over the whole map: the crop often cuts a lane off every lane that has a limit (Boston: 6% vs 99%)
        map_limits = _map_speed_limits(map_api)
        for lane_id, lane in result.items():
            if lane.speed_limit_mps <= 0:
                lane.speed_limit_mps = map_limits.get(lane_id, -1.0)
    _fill_missing_speed_limits(result)

    # Non-lane elements get sequential IDs after max lane ID to avoid collisions
    next_id = max(result.keys(), default=-1) + 1
    intersection_element_ids: dict[int, int] = {}
    for obj in non_lane_objects:
        element = _write_map_object(obj, centroid)
        if element is None:
            continue
        # PY123D-REPORT[opendrive]: 36% of CARLA road lines are shorter than 10 cm.
        if (
            isinstance(element, schema.MapElement)
            and element.is_line
            and np.linalg.norm(np.diff(element.polyline[:, :2], axis=0), axis=1).sum() < MIN_LINE_LENGTH
        ):
            continue
        if isinstance(element, schema.StopZone):
            controlled_lanes = [lane_id for lane_id in element.controlled_lanes if lane_id in lane_ids]
            if controlled_lanes:
                stop_zones.append(dataclasses.replace(element, controlled_lanes=controlled_lanes))
        else:
            if obj.layer == map_objects.MapLayer.INTERSECTION:
                intersection_element_ids[obj.object_id] = next_id
            result[next_id] = element
            next_id += 1
    _drop_duplicate_lines(result)

    # Stop zones reference intersections by their 123D id, map elements use sequential ids
    stop_zones = [
        dataclasses.replace(stop_zone, intersection_id=intersection_element_ids.get(stop_zone.intersection_id, -1))
        for stop_zone in stop_zones
    ]
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
            polyline=_trim_end_hooks(_centered_array(map_object.centerline_3d.array, centroid)),
            speed_limit_mps=float(speed) if (speed := map_object.speed_limit_mps) and not np.isnan(speed) else -1.0,
            entry_lanes=map_object.predecessor_ids,
            exit_lanes=map_object.successor_ids,
            left_boundary=_centered_array(map_object.left_boundary_3d.array, centroid),
            right_boundary=_centered_array(map_object.right_boundary_3d.array, centroid),
            left_neighbor=[i for i in [getattr(map_object, "left_lane_id", None)] if i is not None],
            right_neighbor=[i for i in [getattr(map_object, "right_lane_id", None)] if i is not None],
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

    if layer in mapping.SURFACE_TYPE_MAP:
        return schema.MapElement(
            type=mapping.SURFACE_TYPE_MAP[layer],
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
            # NOTE: py123d releases before the signal groups have no such attributes.
            intersection_id=getattr(map_object, "intersection_id", None),  # 123D id, mapped to element id later
            signal_group_id=-1 if (group := getattr(map_object, "signal_group_id", None)) is None else int(group),
            signal_sequence=-1 if (seq := getattr(map_object, "signal_sequence", None)) is None else int(seq),
        )

    return None


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


def _make_empty_track(episode_length: int) -> schema.Track:
    return schema.Track(
        type=-1,  # placeholder until the 123D label is mapped to a puffer type
        position=np.zeros((episode_length, 3), dtype=np.float64),
        heading=np.zeros((episode_length,), dtype=np.float64),
        velocity=np.full((episode_length, 2), np.nan, dtype=np.float64),
        valid=np.zeros((episode_length,), dtype=np.int32),
        length=np.zeros((episode_length,), dtype=np.float64),
        width=np.zeros((episode_length,), dtype=np.float64),
        height=np.zeros((episode_length,), dtype=np.float64),
    )


def _fill_missing_velocities(track: schema.Track, dt: float) -> None:
    valid_indices = np.flatnonzero(track.valid)
    observed = np.all(np.isfinite(track.velocity), axis=1)
    track.velocity[~track.valid.astype(bool)] = 0.0

    for index in valid_indices[~observed[valid_indices]]:
        neighbors = valid_indices[valid_indices != index]
        previous = neighbors[neighbors < index]
        following = neighbors[neighbors > index]
        if len(previous) and len(following):
            left, right = previous[-1], following[0]
        elif len(previous):
            left, right = previous[-1], index
        elif len(following):
            left, right = index, following[0]
        else:
            track.velocity[index] = 0.0
            continue
        track.velocity[index] = (track.position[right, :2] - track.position[left, :2]) / ((right - left) * dt)


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
        if (coords := obj.centerline_3d.array) is not None and len(coords) > 0  # ty: ignore[unresolved-attribute]
    ]
    if points:
        return np.vstack(points).mean(axis=0)
    return np.zeros(3, dtype=np.float64)


# ── Corrective functions ───────────────────────


def _zero_all_z(
    agents: dict[int, schema.Track],
    objects: dict[int, schema.Track],
    traffic_lights: dict[int, schema.TrafficLightTrack],
    map_elements: dict[int, schema.MapElement],
    stop_zones: list[schema.StopZone],
) -> None:
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
    """Infer undefined lane types from connected lanes + drop refs to lanes outside the extracted map.

    Entry/exit refs are trusted as-is: py123d already orients nuPlan connections, and the old
    endpoint-distance reversal only misfired on broken WOMD links (creating cycles).
    """
    # PY123D-REPORT[nuplan]: most nuPlan lanes (lane connectors) come out of py123d as LaneType.UNDEFINED.
    # PY123D-REPORT[wod-motion]: some WOMD entry/exit refs point at lanes up to ~100 m away; kept as-is.
    for lane_id in undefined_lane_ids:
        lane = lanes[lane_id]
        connected_types = {
            lanes[nid].type for key in ("entry_lanes", "exit_lanes") for nid in getattr(lane, key) if nid in lanes
        }
        if len(connected_types) == 1:
            lane.type = connected_types.pop()

    for element in lanes.values():
        element.entry_lanes = [lid for lid in element.entry_lanes if lid in valid_lane_ids]
        element.exit_lanes = [lid for lid in element.exit_lanes if lid in valid_lane_ids]
        element.left_neighbor = [lid for lid in element.left_neighbor if lid in valid_lane_ids]
        element.right_neighbor = [lid for lid in element.right_neighbor if lid in valid_lane_ids]


def _fill_missing_speed_limits(lanes: dict[int, schema.MapElement]) -> None:
    """Give each lane with an unknown speed limit the limit of its nearest connected lane (BFS over entry/exit)."""
    # PY123D-REPORT[opendrive]: py123d only copies road-type speeds one hop into junction lanes (and matches the
    # unit "mps" where OpenDRIVE spells "m/s"), leaving ~50% of CARLA Town01-10 lanes without a limit.
    # PY123D-REPORT[wod-motion]: WOMD lanes with speed_limit_mph == 0 come through as unknown.
    queue = collections.deque(lane_id for lane_id, lane in lanes.items() if lane.speed_limit_mps > 0)
    while queue:
        lane = lanes[queue.popleft()]
        for ref in [*lane.entry_lanes, *lane.exit_lanes]:
            if lanes[ref].speed_limit_mps <= 0:
                lanes[ref].speed_limit_mps = lane.speed_limit_mps
                queue.append(ref)


def _trim_end_hooks(line: np.ndarray) -> np.ndarray:
    """Drop the inner points of a centerline's first and last ``HOOK_MAX_LENGTH`` when a segment there turns over
    ``HOOK_MIN_TURN`` off the lane's direction (chord to the first point 1 m+ away). Both endpoints stay put."""
    # PY123D-REPORT[opendrive]: ~2% of Town12/13 junction lane centerlines start or end with a 0.25 m zig-zag
    # (sideways 0.125 m, or 90 degrees out and 178 back), turning the lane's end heading and spiking its curvature.
    for _ in range(2):
        dist = np.linalg.norm(line[:, :2] - line[0, :2], axis=1)
        if (dist >= 1.0).any() and (dist >= HOOK_MAX_LENGTH).any():
            chord = line[np.argmax(dist >= 1.0), :2] - line[0, :2]
            end = int(np.argmax(dist >= HOOK_MAX_LENGTH))  # first point past the end stretch
            steps = np.diff(line[: end + 1, :2], axis=0)
            cos_turn = steps @ chord / np.maximum(np.linalg.norm(steps, axis=1) * np.linalg.norm(chord), 1e-12)
            if (cos_turn < np.cos(HOOK_MIN_TURN)).any():
                line = np.delete(line, np.arange(1, end), axis=0)
        line = line[::-1]
    return line


def _drop_duplicate_lines(elements: dict[int, schema.MapElement]) -> None:
    """Remove road lines that repeat an earlier line of the same type, in either direction or resampled."""
    # PY123D-REPORT[nuplan,opendrive]: ~40% of road lines come twice, once per adjacent lane, often reversed and
    # sometimes sampled differently.
    ids = [eid for eid, element in elements.items() if element.is_line]
    geoms = np.array([shapely.LineString(elements[eid].polyline[:, :2]) for eid in ids], dtype=object)
    types = np.array([elements[eid].type for eid in ids], dtype=int)
    ends = np.array([elements[eid].polyline[[0, -1], :2] for eid in ids]).reshape(-1, 2, 2)
    i, j = shapely.STRtree(geoms).query(geoms, predicate="dwithin", distance=LINE_DUPLICATE_DISTANCE)
    same = np.linalg.norm(ends[i] - ends[j], axis=-1).max(-1) < LINE_DUPLICATE_DISTANCE
    flipped = np.linalg.norm(ends[i] - ends[j, ::-1], axis=-1).max(-1) < LINE_DUPLICATE_DISTANCE
    pairs = (i < j) & (types[i] == types[j]) & (same | flipped)
    pairs[pairs] = shapely.hausdorff_distance(geoms[i[pairs]], geoms[j[pairs]]) < LINE_DUPLICATE_DISTANCE
    for k in np.unique(j[pairs]):
        del elements[ids[k]]


def _map_speed_limits(map_api: py123d_api.MapAPI) -> dict[int, float]:
    """Speed limit of every lane in the whole map, unknown ones filled by ``_fill_missing_speed_limits``."""
    objs = list(map_api.get_all_map_objects_in_layer(map_objects.MapLayer.LANE))
    ids = {obj.object_id for obj in objs}
    lanes = {
        obj.object_id: schema.MapElement(
            type=-1,
            speed_limit_mps=float(speed) if (speed := obj.speed_limit_mps) and not np.isnan(speed) else -1.0,
            entry_lanes=[ref for ref in obj.predecessor_ids if ref in ids],
            exit_lanes=[ref for ref in obj.successor_ids if ref in ids],
        )
        for obj in objs
    }
    _fill_missing_speed_limits(lanes)
    return {lane_id: lane.speed_limit_mps for lane_id, lane in lanes.items()}


def _centered_array(array: np.ndarray, center: np.ndarray) -> np.ndarray:
    return array.astype(np.float64) - center
