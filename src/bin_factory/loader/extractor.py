from __future__ import annotations

import collections
import dataclasses
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import shapely
from py123d import api as py123d_api
from py123d.datatypes import detections, map_objects
from py123d.geometry import EulerAnglesIndex, PoseSE3Index
from py123d.geometry.utils.rotation_utils import get_euler_array_from_quaternion_array

from bin_factory import puffer_types, schema
from bin_factory.loader import mapping
from bin_factory.log_context import log


if TYPE_CHECKING:
    from collections.abc import Iterable


SCENE_MAP_MARGIN = 250.0  # Lateral buffer in metres around the ego path for scene scenarios
MIN_LINE_LENGTH = 0.1  # Minimum road line length in metres
LINE_DUPLICATE_DISTANCE = 0.1  # Hausdorff distance in metres that marks a road line as a duplicate
HOOK_MAX_LENGTH = 0.5  # Maximum end-segment length in metres that counts as a sharp hook
HOOK_MIN_TURN = np.radians(30.0)
NEIGHBOR_MIN_OVERLAP = 10.0  # Metres a neighbour must run alongside a lane, capped at half the shorter lane
NEIGHBOR_GAP = (1.5, 6.0)  # Lateral distance range in metres of a neighbour centerline
NEIGHBOR_MAX_ANGLE = np.radians(25.0)
JUNCTION_SHOULDER_MIN_OVERLAP = 0.5  # Area fraction inside intersections that makes a shoulder a junction shoulder
STATIC_OBJECT_LEVEL_TOL = 2.0  # Maximum height difference in metres between a road surface and an object on it
MAX_LANE_GAP = 3.0  # Maximum endpoint gap in metres for connected entry and exit lanes


def extract_scenario(
    py123_arrow: py123d_api.SceneAPI | py123d_api.MapAPI,
    scenario_id_field: str = "scene_uuid",
    speed_limit_source: str = "road",
    static_objects: str = "all",
) -> tuple[schema.PufferScenario, schema.ExtractionExtras]:
    """Convert a 123D scene or map into a scenario and extraction extras.

    ``scenario_id_field`` selects scene_uuid, log_name, or location for metadata.id.
    ``speed_limit_source`` selects the road limit or the last speed sign's limit.
    ``static_objects`` keeps all static map objects or only those on the road surface.
    Transforms use the extras for traffic controls and shoulder edges.
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
        map_has_z=map_api.map_has_z,
        map_version=map_api.version,
        log_name=map_api.map_metadata.log_name or "",
        split=map_api.map_metadata.split or "",
    )

    map_only = scene_api is None
    ego_states = (
        [scene_api.get_ego_state_se3_at_iteration(i) for i in range(scene_api.number_of_iterations)]
        if scene_api is not None
        else None
    )

    if not map_only and (ego_states is None or any(state is None for state in ego_states)):
        raise ValueError("Ego states are required at every frame to convert scenario with SceneAPI")

    centroid = _compute_centroid(ego_states)
    metadata.origin_xyz = tuple(float(value) for value in centroid)
    polylines, surfaces, stop_zones, map_lane_ids = _extract_map(
        map_api, centroid, ego_states, map_only, speed_limit_source
    )

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
        metadata.scene_uuid = scene_api.scene_uuid
        metadata.log_name = scene_api.log_name
        metadata.split = scene_api.split
        metadata.log_version = scene_api.log_metadata.version
        metadata.timestamps_us = [timestamp.time_us for timestamp in scene_api.get_all_iteration_timestamps()]

    first_id = max([*agents, *objects], default=-1) + 1
    objects.update(
        _extract_static_objects(
            map_api, centroid, ego_states, map_only, metadata.scenario_length, first_id, static_objects == "onroad"
        )
    )

    if not map_api.map_has_z:
        _zero_all_z(agents, objects, traffic_lights, [*polylines.values(), *surfaces.values(), *stop_zones])

    scenario = schema.PufferScenario(
        agents=agents,
        polylines=polylines,
        surfaces=surfaces,
        objects=objects,
        metadata=metadata,
    )
    non_junction_shoulders, junction_drivables = _extract_shoulder_context(map_api, centroid)
    extras = schema.ExtractionExtras(
        traffic_lights=traffic_lights,
        stop_zones=stop_zones,
        non_junction_shoulders=non_junction_shoulders,
        junction_drivables=junction_drivables,
    )
    return scenario, extras


def _mark_junction_lanes(map_api: py123d_api.MapAPI, lanes: dict[int, schema.RoadPolyline]) -> None:
    """Flag lanes whose lane group belongs to an intersection."""
    if map_objects.MapLayer.LANE_GROUP not in map_api.available_map_layers:
        return
    groups: list[Any] = list(map_api.get_all_map_objects_in_layer(map_objects.MapLayer.LANE_GROUP))
    for group in groups:
        if getattr(group, "intersection_id", None) is None:
            continue
        for lane_id in group.lane_ids:
            lane = lanes.get(int(lane_id))
            if lane is not None:
                lane.in_junction = True


def _extract_shoulder_context(
    map_api: py123d_api.MapAPI, centroid: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return centred outlines for the shoulder-edge transform.

    The first list contains shoulders outside intersections. The second contains intersections and their shoulders.
    A shoulder is inside when intersection polygons cover at least JUNCTION_SHOULDER_MIN_OVERLAP of its area.
    """
    layers = map_api.available_map_layers
    if map_objects.MapLayer.SHOULDER not in layers:
        return [], []
    shoulders: list[Any] = list(map_api.get_all_map_objects_in_layer(map_objects.MapLayer.SHOULDER))
    intersections: list[Any] = []
    if map_objects.MapLayer.INTERSECTION in layers:
        intersections = list(map_api.get_all_map_objects_in_layer(map_objects.MapLayer.INTERSECTION))
    raw_intersections = [obj.outline_3d.array for obj in intersections]
    intersection_polygons = [shapely.Polygon(arr[:, :2]).buffer(0) for arr in raw_intersections]
    intersection_tree = shapely.STRtree(intersection_polygons) if intersection_polygons else None
    non_junction: list[np.ndarray] = []
    junction: list[np.ndarray] = [_centered_array(arr, centroid) for arr in raw_intersections]
    for shoulder in shoulders:
        arr = shoulder.outline_3d.array
        polygon = shapely.Polygon(arr[:, :2]).buffer(0)
        if polygon.is_empty:
            continue
        outline = _centered_array(arr, centroid)
        overlap = 0.0
        if intersection_tree is not None:
            overlap = sum(polygon.intersection(intersection_polygons[i]).area for i in intersection_tree.query(polygon))
        in_junction = overlap >= JUNCTION_SHOULDER_MIN_OVERLAP * polygon.area
        (junction if in_junction else non_junction).append(outline)
    return non_junction, junction


def _extract_objects(
    scene_api: py123d_api.SceneAPI,
    centroid: np.ndarray,
    ego_states: list[Any],
) -> tuple[dict[int, schema.Track], dict[int, detections.DefaultBoxDetectionLabel], dict[str, int]]:
    """Extract dynamic objects from 123D box detections and ego state.

    Return tracks, 123D detection labels, and a token-to-object-ID map.
    ``extract_scenario`` maps the labels to PufferDrive types.
    """
    episode_length = scene_api.number_of_iterations
    objects: dict[int, schema.Track] = {0: _make_empty_track(episode_length)}
    labels: dict[int, detections.DefaultBoxDetectionLabel] = {0: detections.DefaultBoxDetectionLabel.EGO}
    tokens_to_object_id: dict[str, int] = {}

    # Ego agent is always object ID 0, built from cached ego states
    ego = objects[0]
    ego.source_id = ego_states[0].box_detection_se3.attributes.track_token
    is_nuplan = scene_api.dataset.startswith("nuplan")

    for frame_idx, ego_state in enumerate(ego_states):
        if ego_state is None:
            raise ValueError(f"Missing ego state at frame {frame_idx}")

        bbox = ego_state.bounding_box_se3
        _write_detection_frame(ego, frame_idx, bbox, centroid, bbox.center_se3.yaw)
        # PY123D-REPORT[nuplan]: raw ego_pose velocity has a lateral bias of -2% of vx and reads 1.4%
        # fast against the logged poses. Leave it NaN so _fill_missing_velocities derives it from positions.
        if is_nuplan:
            continue
        velocity = ego_state.box_detection_se3.velocity_3d
        ego.velocity[frame_idx] = [float(velocity.x), float(velocity.y)]

    # Detections for all other agents
    next_object_id = 0
    for frame_idx in range(episode_length):
        detections_list = scene_api.get_box_detections_se3_at_iteration(frame_idx)
        if not detections_list:
            continue
        quaternions = np.array([d.bounding_box_se3.array[PoseSE3Index.QUATERNION] for d in detections_list])
        headings = get_euler_array_from_quaternion_array(quaternions)[:, EulerAnglesIndex.YAW]
        for detection, heading in zip(detections_list, headings, strict=False):
            track_token = detection.attributes.track_token
            object_id = tokens_to_object_id.get(track_token)
            if object_id is None:
                next_object_id += 1
                object_id = next_object_id
                tokens_to_object_id[track_token] = object_id
                objects[object_id] = _make_empty_track(episode_length)
                objects[object_id].source_id = track_token
                labels[object_id] = detection.attributes.default_label

            obj = objects[object_id]
            _write_detection_frame(
                obj,
                frame_idx,
                detection.bounding_box_se3,
                centroid,
                float(heading),
            )

            if detection.velocity_3d is not None:
                vel = detection.velocity_3d.array
                obj.velocity[frame_idx, 0] = vel[0]
                obj.velocity[frame_idx, 1] = vel[1]

    return objects, labels, tokens_to_object_id


def _extract_prediction_targets(
    scene_api: py123d_api.SceneAPI,
    tokens_to_object_id: dict[str, int],
    agents: dict[int, schema.Track],
    metadata: schema.ScenarioMetadata,
) -> None:
    """Populate metadata objects_of_interest and tracks_to_predict from WOD-Motion aux data."""
    aux = scene_api.get_all_custom_modality_metadatas().get("aux")
    if aux is None:
        return
    meta = aux.metadata
    metadata.timestamps_are_relative = meta["timestamps_are_relative"]
    agents[0].source_id = meta["track_index_to_token"][meta["sdc_track_index"]]
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
    speed_limit_source: str = "road",
) -> tuple[dict[int, schema.RoadPolyline], dict[int, schema.RoadSurface], list[schema.StopZone], set[int]]:
    """Extract static map elements from a 123D MapAPI."""
    result: dict[int, schema.RoadPolyline] = {}
    surfaces: dict[int, schema.RoadSurface] = {}
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
        assert ego_states is not None
        map_objects_by_layer = map_api.query(_ego_corridor(ego_states), layers, predicate="intersects")
        map_objs = [obj for layer in layers for obj in map_objects_by_layer.get(layer, [])]

    # Other elements reference lane IDs, so extract lanes first.
    for obj in map_objs:
        if obj.layer == map_objects.MapLayer.LANE:
            element = _write_map_object(obj, centroid, speed_limit_source)
            if not isinstance(element, schema.RoadPolyline):
                continue
            result[obj.object_id] = element
            if obj.lane_type == map_objects.LaneType.UNDEFINED:
                undefined_lane.append(obj.object_id)
        else:
            non_lane_objects.append(obj)

    lane_ids = set(result.keys())
    _fix_lane_topology(result, undefined_lane, lane_ids)
    _drop_partial_neighbors(result)
    if cropped and any(lane.speed_limit_mps <= 0 for lane in result.values()):
        # Fill over the whole map because the crop often cuts a lane off from every lane with a speed limit.
        map_limits = _map_speed_limits(map_api, speed_limit_source)
        for lane_id, lane in result.items():
            if lane.speed_limit_mps <= 0:
                lane.speed_limit_mps = map_limits.get(lane_id, -1.0)
    _fill_missing_speed_limits(result)

    # Non-lane elements get sequential IDs after max lane ID to avoid collisions
    next_id = max(result.keys(), default=-1) + 1
    intersection_element_ids: dict[int, int] = {}
    for obj in non_lane_objects:
        element = _write_map_object(obj, centroid)
        if element is None or _is_short_road_line(element):
            continue
        if isinstance(element, schema.StopZone):
            controlled_lanes = [lane_id for lane_id in element.controlled_lanes if lane_id in lane_ids]
            if controlled_lanes:
                stop_zones.append(dataclasses.replace(element, controlled_lanes=controlled_lanes))
        elif isinstance(element, schema.RoadSurface):
            if obj.layer == map_objects.MapLayer.INTERSECTION:
                intersection_element_ids[obj.object_id] = next_id
            surfaces[next_id] = element
            next_id += 1
        else:
            result[next_id] = element
            next_id += 1
    drop_duplicate_lines(result)

    _mark_junction_lanes(map_api, result)
    # Stop zones reference intersections by their 123D id, map elements use sequential ids
    stop_zones = [
        dataclasses.replace(stop_zone, intersection_id=intersection_element_ids.get(stop_zone.intersection_id, -1))
        for stop_zone in stop_zones
    ]
    return result, surfaces, stop_zones, lane_ids


def _ego_corridor(ego_states: list[Any]) -> shapely.Geometry:
    ego_xy = [(s.center_se3.x, s.center_se3.y) for s in ego_states]
    centerline = shapely.Point(ego_xy[0]) if len(ego_xy) == 1 else shapely.LineString(ego_xy)
    return centerline.buffer(SCENE_MAP_MARGIN)


def _is_short_road_line(element: schema.RoadPolyline | schema.RoadSurface | schema.StopZone) -> bool:
    # PY123D-REPORT[opendrive]: 36% of CARLA road lines are shorter than 10 cm.
    return (
        isinstance(element, schema.RoadPolyline)
        and element.is_line
        and element.polyline is not None
        and bool(((element.polyline[-1, :2] - element.polyline[0, :2]) ** 2).sum() < MIN_LINE_LENGTH**2)
        and bool(np.linalg.norm(np.diff(element.polyline[:, :2], axis=0), axis=1).sum() < MIN_LINE_LENGTH)
    )


# Writer functions


def _write_map_object(
    map_object: Any, centroid: np.ndarray, speed_limit_source: str = "road"
) -> schema.RoadPolyline | schema.RoadSurface | schema.StopZone | None:
    """Convert a 123D map object into a RoadPolyline, RoadSurface, or StopZone."""
    layer = map_object.layer
    if layer not in mapping.SUPPORTED_MAP_LAYERS:
        return None

    if layer == map_objects.MapLayer.LANE:
        if map_object.lane_type == map_objects.LaneType.SHOULDER:
            return None
        return schema.RoadPolyline(
            layer=int(layer),
            type=int(map_object.lane_type),
            polyline=_trim_end_hooks(_centered_array(map_object.centerline_3d.array, centroid)),
            speed_limit_mps=_lane_speed_limit(map_object, speed_limit_source),
            entry_lanes=map_object.predecessor_ids,
            exit_lanes=map_object.successor_ids,
            left_boundary=_centered_array(map_object.left_boundary_3d.array, centroid),
            right_boundary=_centered_array(map_object.right_boundary_3d.array, centroid),
            left_neighbor=[i for i in [getattr(map_object, "left_lane_id", None)] if i is not None],
            right_neighbor=[i for i in [getattr(map_object, "right_lane_id", None)] if i is not None],
        )

    if layer in (map_objects.MapLayer.ROAD_LINE, map_objects.MapLayer.ROAD_EDGE):
        subtype = map_object.road_line_type if layer == map_objects.MapLayer.ROAD_LINE else map_object.road_edge_type
        return schema.RoadPolyline(
            layer=int(layer),
            type=int(subtype),
            polyline=_centered_array(map_object.polyline_3d.array, centroid),
        )

    if int(layer) in puffer_types.SURFACE_LAYERS:
        return schema.RoadSurface(layer=int(layer), polygon=_centered_array(map_object.outline_3d.array, centroid))

    if layer == map_objects.MapLayer.STOP_ZONE:
        puffer_type = mapping.STOP_ZONE_TYPE_MAP.get(map_object.stop_zone_type)
        if puffer_type is None:
            return None
        return schema.StopZone(
            type=puffer_type,
            polygon=_centered_array(map_object.outline_3d.array, centroid),
            controlled_lanes=map_object.lane_ids,
            # Older py123d releases have no light-phase attributes.
            # 123D id, mapped to element id later
            intersection_id=-1 if (iid := getattr(map_object, "intersection_id", None)) is None else iid,
            phase_idx=-1 if (phase := getattr(map_object, "phase_idx", None)) is None else int(phase),
        )

    return None


def _extract_static_objects(
    map_api: py123d_api.MapAPI,
    centroid: np.ndarray,
    ego_states: list[Any] | None,
    map_only: bool,
    episode_length: int,
    first_id: int,
    onroad_only: bool = False,
) -> dict[int, schema.Track]:
    """Extract static map objects as tracks that stay still throughout the episode.

    With ``onroad_only``, keep only objects touching a generic drivable surface at their elevation.
    """
    if map_objects.MapLayer.STATIC_OBJECT not in map_api.available_map_layers:
        return {}
    static_objects: list[Any]
    if map_only or map_api.map_is_per_log or ego_states is None:
        static_objects = list(map_api.get_all_map_objects_in_layer(map_objects.MapLayer.STATIC_OBJECT))
    else:
        layer = map_objects.MapLayer.STATIC_OBJECT
        found = map_api.query(_ego_corridor(ego_states), [layer], predicate="intersects")
        static_objects = cast("list[Any]", found.get(layer, []))
    if not static_objects:
        return {}
    boxes = [obj.outline.array[:4] for obj in static_objects]
    if onroad_only and map_objects.MapLayer.GENERIC_DRIVABLE in map_api.available_map_layers:
        static_objects, boxes = _filter_onroad_static_objects(map_api, static_objects, boxes)
    tracks: dict[int, schema.Track] = {}
    num_frames = max(episode_length, 1)
    for n, (obj, corners) in enumerate(zip(static_objects, boxes, strict=False)):
        sides = np.diff(corners[:3, :2], axis=0)
        side_norms = np.linalg.norm(sides, axis=1)
        long_side = sides[int(np.argmax(side_norms))]
        track = _make_empty_track(num_frames)  # map-only scenarios allocate one frame
        track.source_id = str(obj.object_id)
        track.type = int(
            puffer_types.ObjectType.BARRIER
            if obj.static_object_type == map_objects.StaticObjectType.BARRIER
            else puffer_types.ObjectType.GENERIC_OBJECT
        )
        track.position[:] = corners[:, :3].mean(axis=0) - centroid
        track.heading[:] = np.arctan2(long_side[1], long_side[0])
        track.velocity[:] = 0.0
        track.valid[:] = 1
        track.length[:] = side_norms.max()
        track.width[:] = side_norms.min()
        track.height[:] = obj.height
        tracks[first_id + n] = track
    return tracks


def _filter_onroad_static_objects(
    map_api: py123d_api.MapAPI,
    static_objects: list[Any],
    boxes: list[np.ndarray],
) -> tuple[list[Any], list[np.ndarray]]:
    # An agent must cross a road edge before colliding with an off-road object.
    surfaces: list[Any] = list(map_api.get_all_map_objects_in_layer(map_objects.MapLayer.GENERIC_DRIVABLE))
    outlines = [surface.outline_3d.array for surface in surfaces]
    tree = shapely.STRtree([shapely.Polygon(outline[:, :2]).buffer(0) for outline in outlines])
    box_polys = [shapely.Polygon(box[:, :2]) for box in boxes]
    q_idx, s_idx = tree.query(box_polys, predicate="intersects")
    hits_by_obj: dict[int, list[int]] = collections.defaultdict(list)
    for qi, si in zip(q_idx.tolist(), s_idx.tolist(), strict=False):
        hits_by_obj[qi].append(si)
    kept_objs = []
    kept_boxes = []
    for idx, (obj, box) in enumerate(zip(static_objects, boxes, strict=False)):
        hits = hits_by_obj.get(idx)
        if not hits:
            continue
        centre = box.mean(axis=0)
        if any(
            abs(outlines[i][int(np.argmin(((outlines[i][:, :2] - centre[:2]) ** 2).sum(axis=1))), 2] - centre[2])
            <= STATIC_OBJECT_LEVEL_TOL
            for i in hits
        ):
            kept_objs.append(obj)
            kept_boxes.append(box)
    return kept_objs, kept_boxes


def _write_detection_frame(
    obj: schema.Track,
    frame_idx: int,
    bbox: Any,
    centroid: np.ndarray,
    heading: float,
) -> None:
    arr = bbox.array
    obj.position[frame_idx, 0] = arr[0] - centroid[0]
    obj.position[frame_idx, 1] = arr[1] - centroid[1]
    obj.position[frame_idx, 2] = arr[2] - centroid[2] - arr[9] / 2.0
    obj.heading[frame_idx] = heading
    obj.valid[frame_idx] = 1
    obj.length[frame_idx] = arr[7]
    obj.width[frame_idx] = arr[8]
    obj.height[frame_idx] = arr[9]


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

    missing = np.flatnonzero(~observed[valid_indices])
    if not len(missing):
        return
    if len(valid_indices) < 2:
        track.velocity[valid_indices[missing]] = 0.0
        return
    left = valid_indices[np.maximum(missing - 1, 0)]
    right = valid_indices[np.minimum(missing + 1, len(valid_indices) - 1)]
    track.velocity[valid_indices[missing]] = (track.position[right, :2] - track.position[left, :2]) / (
        (right - left)[:, None] * dt
    )


def _compute_centroid(ego_states: list[Any] | None) -> np.ndarray:
    if ego_states is not None:
        positions = np.array(
            [s.bounding_box_se3.array[:3] for s in ego_states if s is not None],
            dtype=np.float64,
        )
        if len(positions) > 0:
            return positions.mean(axis=0)
    return np.zeros(3, dtype=np.float64)


# Source data corrections


def _zero_all_z(
    agents: dict[int, schema.Track],
    objects: dict[int, schema.Track],
    traffic_lights: dict[int, schema.TrafficLightTrack],
    map_elements: list[schema.RoadPolyline | schema.RoadSurface | schema.StopZone],
) -> None:
    """Force all Z values to 0 when map has no Z data."""
    for track in [*agents.values(), *objects.values()]:
        track.position[:, 2] = 0.0
    for tl in traffic_lights.values():
        tl.position[2] = 0.0
    for element in map_elements:
        for key in ("polyline", "left_boundary", "right_boundary", "polygon"):
            arr = getattr(element, key, None)
            if arr is not None:
                arr[:, 2] = 0.0


def _fix_lane_topology(
    lanes: dict[int, schema.RoadPolyline],
    undefined_lane_ids: list[int],
    valid_lane_ids: set[int],
) -> None:
    """Infer undefined lane types from connected lanes and drop missing or disconnected references.

    Keep entry and exit directions from py123d, which already orients nuPlan connections.
    The previous endpoint-distance reversal created cycles on broken WOMD links.
    """
    # PY123D-REPORT[nuplan]: most nuPlan lane connectors come out of py123d as LaneType.UNDEFINED.
    # PY123D-REPORT[wod-motion]: some WOMD entry/exit refs point at lanes up to ~100 m away.
    gap_sq = MAX_LANE_GAP**2
    for element in lanes.values():
        line = cast("np.ndarray", element.polyline)
        element.entry_lanes = [
            lid
            for lid in element.entry_lanes
            if lid in valid_lane_ids
            and ((cast("np.ndarray", lanes[lid].polyline)[-1, :2] - line[0, :2]) ** 2).sum() <= gap_sq
        ]
        element.exit_lanes = [
            lid
            for lid in element.exit_lanes
            if lid in valid_lane_ids
            and ((line[-1, :2] - cast("np.ndarray", lanes[lid].polyline)[0, :2]) ** 2).sum() <= gap_sq
        ]
        element.left_neighbor = [lid for lid in element.left_neighbor if lid in valid_lane_ids]
        element.right_neighbor = [lid for lid in element.right_neighbor if lid in valid_lane_ids]

    for lane_id in undefined_lane_ids:
        lane = lanes[lane_id]
        connected_types = {lanes[nid].type for nid in [*lane.entry_lanes, *lane.exit_lanes]}
        if len(connected_types) == 1:
            lane.type = connected_types.pop()


def _drop_partial_neighbors(lanes: dict[int, schema.RoadPolyline]) -> None:
    """Keep a left or right neighbour only if it runs alongside the lane over a lane change's length."""
    # PY123D-REPORT[wod-motion]: WOMD neighbours carry the index range where two lanes are adjacent. py123d keeps one
    # neighbour per side and drops the range, so lanes side by side for a few metres come out as whole-lane neighbours.
    for lane in lanes.values():
        line = cast("np.ndarray", lane.polyline)  # lanes always carry a centerline
        lane.left_neighbor = [
            ref for ref in lane.left_neighbor if _alongside(line, cast("np.ndarray", lanes[ref].polyline), 1.0)
        ]
        lane.right_neighbor = [
            ref for ref in lane.right_neighbor if _alongside(line, cast("np.ndarray", lanes[ref].polyline), -1.0)
        ]


def _alongside(line: np.ndarray, other: np.ndarray, side: float) -> bool:
    """Return True when ``other`` runs parallel to ``line`` on ``side`` (+1 left, -1 right) long enough."""
    line, other = line[:, :2], other[:, :2]
    steps = np.diff(line, axis=0)
    step_norms = np.linalg.norm(steps, axis=1)
    cum = np.r_[0.0, np.cumsum(step_norms)]
    other_steps = np.diff(other, axis=0)
    other_norms = np.linalg.norm(other_steps, axis=1)
    other_length = float(other_norms.sum())
    if cum[-1] <= 0 or other_length <= 0:
        return False
    samples = np.arange(0.0, cum[-1], 1.0)
    points = np.column_stack([np.interp(samples, cum, line[:, 0]), np.interp(samples, cum, line[:, 1])])
    seg_idx = np.clip(np.searchsorted(cum, samples, side="right") - 1, 0, len(steps) - 1)
    direction = steps[seg_idx] / np.maximum(step_norms[seg_idx, None], 1e-9)

    sq = np.maximum((other_steps**2).sum(1), 1e-9)
    dx = points[:, None, 0] - other[:-1, 0]
    dy = points[:, None, 1] - other[:-1, 1]
    t = np.clip((dx * other_steps[:, 0] + dy * other_steps[:, 1]) / sq, 0.0, 1.0)
    rx = t * other_steps[:, 0] - dx
    ry = t * other_steps[:, 1] - dy
    nearest = (rx * rx + ry * ry).argmin(1)
    p_idx = np.arange(len(points))
    rel_x = rx[p_idx, nearest]
    rel_y = ry[p_idx, nearest]
    lateral = side * (direction[:, 0] * rel_y - direction[:, 1] * rel_x)
    other_dir = other_steps[nearest] / np.sqrt(sq[nearest])[:, None]
    parallel = (direction * other_dir).sum(1) > np.cos(NEIGHBOR_MAX_ANGLE)
    adjacent = (lateral > NEIGHBOR_GAP[0]) & (lateral < NEIGHBOR_GAP[1]) & parallel
    return adjacent.sum() >= min(NEIGHBOR_MIN_OVERLAP, 0.5 * min(cum[-1], other_length))


def _fill_missing_speed_limits(lanes: dict[int, schema.RoadPolyline]) -> None:
    """Give each lane with an unknown speed limit the limit of its nearest connected lane via BFS."""
    # PY123D-REPORT[opendrive]: py123d only copies road-type speeds one hop into junction lanes and matches the
    # unit "mps" where OpenDRIVE spells "m/s", leaving ~50% of CARLA Town01-10 lanes without a limit.
    # PY123D-REPORT[wod-motion]: WOMD lanes with speed_limit_mph == 0 come through as unknown.
    queue = collections.deque(lane_id for lane_id, lane in lanes.items() if lane.speed_limit_mps > 0)
    while queue:
        lane = lanes[queue.popleft()]
        for ref in [*lane.entry_lanes, *lane.exit_lanes]:
            if lanes[ref].speed_limit_mps <= 0:
                lanes[ref].speed_limit_mps = lane.speed_limit_mps
                queue.append(ref)


def _trim_end_hooks(line: np.ndarray) -> np.ndarray:
    """Remove points near a centerline's ends when a segment turns more than ``HOOK_MIN_TURN``.

    Check the first and last ``HOOK_MAX_LENGTH`` against the chord to the first point at least 1 m away.
    Keep both endpoints.
    """
    # PY123D-REPORT[opendrive]: ~2% of Town12/13 junction lane centerlines start or end with a 0.25 m zig-zag
    # that turns the lane's end heading and spikes its curvature.
    if len(line) < 3:
        return line
    hook_sq = HOOK_MAX_LENGTH**2
    cos_min = np.cos(HOOK_MIN_TURN)
    for _ in range(2):
        if len(line) >= 3 and ((line[1, :2] - line[0, :2]) ** 2).sum() < hook_sq:
            dist_sq = ((line[:, :2] - line[0, :2]) ** 2).sum(axis=1)
            if (dist_sq >= 1.0).any() and (dist_sq >= hook_sq).any():
                chord = line[np.argmax(dist_sq >= 1.0), :2] - line[0, :2]
                end = int(np.argmax(dist_sq >= hook_sq))  # first point past the end stretch
                steps = np.diff(line[: end + 1, :2], axis=0)
                cos_turn = steps @ chord / np.maximum(np.linalg.norm(steps, axis=1) * np.linalg.norm(chord), 1e-12)
                if (cos_turn < cos_min).any():
                    line = np.delete(line, np.arange(1, end), axis=0)
        line = line[::-1]
    return line


def drop_duplicate_lines(elements: dict[int, schema.RoadPolyline]) -> None:
    """Remove road lines that repeat an earlier line of the same type, in either direction or resampled."""
    # PY123D-REPORT[nuplan,opendrive]: ~40% of road lines come twice, once per adjacent lane, often reversed and
    # sometimes sampled differently. CARLA types the reversed copy of a DASH_SOLID line as SOLID_DASH, so reversed
    # lines also match across that pair.
    ids = [eid for eid, element in elements.items() if element.is_line]
    if len(ids) < 2:
        return
    polylines = [cast("np.ndarray", elements[eid].polyline) for eid in ids]  # road lines always carry one
    types = np.array([elements[eid].type for eid in ids], dtype=int)
    mirrored = types.copy()
    for dash_solid, solid_dash in (
        (puffer_types.RoadLineType.DASH_SOLID_YELLOW, puffer_types.RoadLineType.SOLID_DASH_YELLOW),
        (puffer_types.RoadLineType.DASH_SOLID_WHITE, puffer_types.RoadLineType.SOLID_DASH_WHITE),
    ):
        mirrored[types == dash_solid], mirrored[types == solid_dash] = solid_dash, dash_solid
    ends = np.array([polyline[[0, -1], :2] for polyline in polylines]).reshape(-1, 2, 2)
    pts = shapely.points(ends.min(axis=1))
    i, j = shapely.STRtree(pts).query(pts, predicate="dwithin", distance=LINE_DUPLICATE_DISTANCE * np.sqrt(2.0))
    same_type = types[i] == types[j]
    mask = (i < j) & (same_type | (types[i] == mirrored[j]))
    i, j, same_type = i[mask], j[mask], same_type[mask]
    same = same_type & (np.linalg.norm(ends[i] - ends[j], axis=-1).max(-1) < LINE_DUPLICATE_DISTANCE)
    flipped = np.linalg.norm(ends[i] - ends[j, ::-1], axis=-1).max(-1) < LINE_DUPLICATE_DISTANCE
    cand = same | flipped
    i, j = i[cand], j[cand]
    if len(i) == 0:
        return
    geoms_i = [shapely.LineString(polylines[idx][:, :2]) for idx in i]
    geoms_j = [shapely.LineString(polylines[idx][:, :2]) for idx in j]
    dup = shapely.hausdorff_distance(geoms_i, geoms_j) < LINE_DUPLICATE_DISTANCE
    for k in np.unique(j[dup]):
        del elements[ids[k]]


def _lane_speed_limit(lane: Any, source: str) -> float:
    """Return a lane's speed limit in m/s, or -1 if unknown.

    With ``signs``, use the last speed sign passed, matching CARLA. Without sign data, use the road's limit.
    """
    speed = getattr(lane, "sign_speed_limit_mps", None) if source == "signs" else None
    speed = lane.speed_limit_mps if speed is None else speed
    return float(speed) if speed and not np.isnan(speed) else -1.0


def _map_speed_limits(map_api: py123d_api.MapAPI, speed_limit_source: str = "road") -> dict[int, float]:
    """Return speed limits for all map lanes, filling unknown limits with ``_fill_missing_speed_limits``."""
    objs: list[Any] = list(map_api.get_all_map_objects_in_layer(map_objects.MapLayer.LANE))
    ids = {obj.object_id for obj in objs}
    lanes = {
        obj.object_id: schema.RoadPolyline(
            layer=int(puffer_types.MapLayer.LANE),
            type=-1,
            speed_limit_mps=_lane_speed_limit(obj, speed_limit_source),
            entry_lanes=[ref for ref in obj.predecessor_ids if ref in ids],
            exit_lanes=[ref for ref in obj.successor_ids if ref in ids],
        )
        for obj in objs
    }
    _fill_missing_speed_limits(lanes)
    return {lane_id: lane.speed_limit_mps for lane_id, lane in lanes.items()}


def _centered_array(array: np.ndarray, center: np.ndarray) -> np.ndarray:
    return array.astype(np.float64) - center
