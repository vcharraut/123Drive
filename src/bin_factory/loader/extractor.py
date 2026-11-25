from __future__ import annotations

import collections
import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np
import shapely
from py123d import api as py123d_api
from py123d.datatypes import detections, map_objects
from py123d.geometry import EulerAnglesIndex, PoseSE3Index
from py123d.geometry.utils.rotation_utils import get_euler_array_from_quaternion_array

from bin_factory import puffer_types, schema
from bin_factory.conversion.sanitize import fill_missing_speed_limits
from bin_factory.loader import mapping
from bin_factory.log_context import log


if TYPE_CHECKING:
    from collections.abc import Iterable


SCENE_MAP_MARGIN = 250.0  # Lateral buffer in metres around the ego path for scene scenarios
MIN_LINE_LENGTH = 0.1  # Minimum road line length in metres
JUNCTION_SHOULDER_MIN_OVERLAP = 0.5  # Area fraction inside intersections that makes a shoulder a junction shoulder
STATIC_OBJECT_LEVEL_TOL = 2.0  # Maximum height difference in metres between a road surface and an object on it


def scenario_id(py123d_data: Any, field: str) -> str:
    """Return the scene's ``field`` as its identity. Map-only objects have no other identity than their location."""
    if not hasattr(py123d_data, field):  # map-only objects only expose `location`
        field = "location"
    value = str(getattr(py123d_data, field) or "")
    if not value:
        dataset = getattr(py123d_data, "dataset", "")
        raise ValueError(f"Cannot derive scenario id from py123d_data: missing {field} for dataset {dataset}")
    return value


def extract_scenario(
    py123_arrow: py123d_api.SceneAPI | py123d_api.MapAPI,
    scenario_id_field: str,
    speed_limit_source: str,
    static_objects: str,
) -> tuple[schema.PufferScenario, schema.ExtractionExtras]:
    """Convert a 123D scene or map into a scenario and extraction extras.

    ``scenario_id_field`` selects scene_uuid, log_name, or location for metadata.id.
    ``speed_limit_source`` selects the road limit or the last speed sign's limit.
    ``static_objects`` keeps all static map objects or only those on the road surface.
    Transforms use the extras for traffic controls and shoulder edges. The map comes out as py123d reports it,
    apart from the whole-map speed fill; ``conversion.sanitize.clean_map`` corrects its quirks.
    """
    if isinstance(py123_arrow, py123d_api.MapAPI):
        scene_api = None
        map_api = py123_arrow
    else:
        scene_api = py123_arrow
        map_api = scene_api.get_map_api()

    if map_api is None:
        raise ValueError("Map API is required to convert scenario")

    agents: dict[int, schema.Track] = {}
    traffic_lights: dict[int, schema.TrafficLightTrack] = {}
    objects: dict[int, schema.Track] = {}
    metadata = schema.ScenarioMetadata(
        id=scenario_id(py123_arrow, scenario_id_field),
        dataset=py123_arrow.dataset,
        scenario_length=0,
        dt=0.0,
        location=map_api.location or "",
        map_has_z=map_api.map_has_z,
        map_version=map_api.version,
        log_name=map_api.map_metadata.log_name or "",
        split=map_api.map_metadata.split or "",
    )

    ego_states: list[Any] | None = None
    if scene_api is not None:
        ego_states = [scene_api.get_ego_state_se3_at_iteration(i) for i in range(scene_api.number_of_iterations)]
        if any(state is None for state in ego_states):
            raise ValueError("Ego states are required at every frame to convert scenario with SceneAPI")

    centroid = np.zeros(3, dtype=np.float64)
    if ego_states:
        centroid = np.array([s.bounding_box_se3.array[:3] for s in ego_states], dtype=np.float64).mean(axis=0)
    metadata.origin_xyz = tuple(float(value) for value in centroid)
    # Scenes keep the map near the ego path, unless the map belongs to the log
    corridor = None if ego_states is None or map_api.map_is_per_log else _ego_corridor(ego_states)
    polylines, surfaces, stop_zones, map_lane_ids = _extract_map(map_api, centroid, corridor, speed_limit_source)

    if scene_api is not None and ego_states is not None:
        dt = float(scene_api.scene_metadata.iteration_duration_s)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"Invalid time step dt={dt} computed from scene metadata")

        all_objects, labels, tokens_to_object_id = _extract_objects(scene_api, centroid, ego_states)
        for oid, obj in all_objects.items():
            _fill_missing_velocities(obj, dt)
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
            map_api, centroid, corridor, metadata.scenario_length, first_id, static_objects == "onroad"
        )
    )

    if not map_api.map_has_z:
        _zero_all_z(agents, objects, traffic_lights, [*polylines.values(), *surfaces.values(), *stop_zones])

    scenario = schema.PufferScenario(
        agents=agents, objects=objects, polylines=polylines, metadata=metadata, surfaces=surfaces
    )
    non_junction_shoulders, junction_drivables = _extract_shoulder_context(map_api, centroid)
    return scenario, schema.ExtractionExtras(traffic_lights, stop_zones, non_junction_shoulders, junction_drivables)


def _mark_junction_lanes(
    map_api: py123d_api.MapAPI, lanes: dict[int, schema.RoadPolyline], intersection_element_ids: dict[int, int]
) -> None:
    """Flag lanes whose lane group belongs to an intersection, with that intersection's element id."""
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
                lane.intersection_id = intersection_element_ids.get(group.intersection_id, -1)


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
    # Ego agent is always object ID 0, built from cached ego states
    ego = _make_empty_track(episode_length, ego_states[0].box_detection_se3.attributes.track_token)
    objects: dict[int, schema.Track] = {0: ego}
    labels: dict[int, detections.DefaultBoxDetectionLabel] = {0: detections.DefaultBoxDetectionLabel.EGO}
    tokens_to_object_id: dict[str, int] = {}
    is_nuplan = scene_api.dataset.startswith("nuplan")

    for frame_idx, ego_state in enumerate(ego_states):
        bbox = ego_state.bounding_box_se3
        _write_detection_frame(ego, frame_idx, bbox, centroid, bbox.center_se3.yaw)
        # PY123D-REPORT[nuplan]: raw ego_pose velocity has a lateral bias of -2% of vx and reads 1.4%
        # fast against the logged poses. Leave it NaN so _fill_missing_velocities derives it from positions.
        if not is_nuplan:
            velocity = ego_state.box_detection_se3.velocity_3d
            ego.velocity[frame_idx] = [float(velocity.x), float(velocity.y)]

    # Detections for all other agents
    for frame_idx in range(episode_length):
        detections_list = scene_api.get_box_detections_se3_at_iteration(frame_idx)
        if not detections_list:
            continue
        quaternions = np.array([d.bounding_box_se3.array[PoseSE3Index.QUATERNION] for d in detections_list])
        headings = get_euler_array_from_quaternion_array(quaternions)[:, EulerAnglesIndex.YAW]
        for detection, heading in zip(detections_list, headings, strict=False):
            track_token = detection.attributes.track_token
            object_id = tokens_to_object_id.get(track_token)
            if object_id is None:  # number tracks from 1 in order of appearance
                object_id = tokens_to_object_id[track_token] = len(tokens_to_object_id) + 1
                objects[object_id] = _make_empty_track(episode_length, track_token)
                labels[object_id] = detection.attributes.default_label
            obj = objects[object_id]
            _write_detection_frame(obj, frame_idx, detection.bounding_box_se3, centroid, float(heading))
            if detection.velocity_3d is not None:
                vel = detection.velocity_3d.array
                obj.velocity[frame_idx, 0], obj.velocity[frame_idx, 1] = vel[0], vel[1]

    # PY123D-REPORT[griffin]: Griffin label files include the ego vehicle at (0, 0) in ego coordinates with its actual
    # CARLA box (1.60 x 0.82 m), while EgoStateSE3Metadata hardcodes 4.79 x 1.85 m and leaves the duplicate in
    # box_detections_se3. Copy its dimensions onto ego and drop the duplicate track.
    if scene_api.dataset == "griffin":
        ego_dups = [
            oid
            for oid, obj in objects.items()
            if oid and obj.valid[0] and np.hypot(*(obj.position[0, :2] - ego.position[0, :2])) < 0.5
        ]
        for oid in ego_dups:
            ego.source_id = objects[oid].source_id
            ego.length[:], ego.width[:], ego.height[:] = objects[oid].length, objects[oid].width, objects[oid].height
            tokens_to_object_id.pop(objects.pop(oid).source_id, None)
            labels.pop(oid, None)

    return objects, labels, tokens_to_object_id


def _extract_prediction_targets(
    scene_api: py123d_api.SceneAPI,
    tokens_to_object_id: dict[str, int],
    agents: dict[int, schema.Track],
    metadata: schema.ScenarioMetadata,
) -> None:
    """Populate metadata objects_of_interest and tracks_to_predict from WOD-Motion and Shifts aux data."""
    aux = scene_api.get_all_custom_modality_metadatas().get("aux")
    if aux is None:
        return
    meta = aux.metadata
    metadata.timestamps_are_relative = meta["timestamps_are_relative"]
    if "prediction_requests" in meta:
        metadata.tracks_to_predict = [
            oid for token in meta["prediction_requests"] if (oid := tokens_to_object_id.get(token)) in agents
        ]
        return
    agents[0].source_id = meta["track_index_to_token"][meta["sdc_track_index"]]
    idx2tok = meta["track_index_to_token"]
    # Ego has no detection token; WOMD lists it among the tracks to predict in about one scenario in ten.
    object_ids = {**tokens_to_object_id, agents[0].source_id: 0}

    def _resolve(tokens: Iterable[str]) -> list[int]:
        ids = (object_ids.get(token) for token in tokens)
        return [oid for oid in ids if oid in agents]

    # WOMD stores objects of interest as track ids and tracks to predict as track indices.
    metadata.objects_of_interest = _resolve(str(track_id) for track_id in meta["objects_of_interest"])
    metadata.tracks_to_predict = _resolve(idx2tok[entry["track_index"]] for entry in meta["tracks_to_predict"])


def _extract_traffic_lights(
    scene_api: py123d_api.SceneAPI,
    map_api: py123d_api.MapAPI,
    centroid: np.ndarray,
    lane_ids: set[int],
) -> dict[int, schema.TrafficLightTrack]:
    """Extract dynamic traffic light states from 123D logs."""
    elements: dict[int, schema.TrafficLightTrack] = {}
    for frame_idx in range(scene_api.number_of_iterations):
        for detection in scene_api.get_traffic_light_detections_at_iteration(frame_idx) or []:
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
                detection.status, puffer_types.TLState.UNKNOWN
            )
    return elements


def _map_objects(map_api: py123d_api.MapAPI, layers: list[Any], corridor: shapely.Geometry | None) -> list[Any]:
    """Return the map objects of ``layers``, layer by layer: all of them, or those intersecting ``corridor``."""
    if corridor is None:
        return list(map_api.get_all_map_objects_in_layers(layers))
    found = map_api.query(corridor, layers, predicate="intersects")
    return [obj for layer in layers for obj in found.get(layer, [])]


def _extract_map(
    map_api: py123d_api.MapAPI,
    centroid: np.ndarray,
    corridor: shapely.Geometry | None = None,
    speed_limit_source: str = "road",
) -> tuple[dict[int, schema.RoadPolyline], dict[int, schema.RoadSurface], list[schema.StopZone], set[int]]:
    """Extract static map elements from a 123D MapAPI, within ``corridor`` when given."""
    result: dict[int, schema.RoadPolyline] = {}
    surfaces: dict[int, schema.RoadSurface] = {}
    stop_zones: list[schema.StopZone] = []
    non_lane_objects: list[Any] = []
    layers = [layer for layer in map_api.available_map_layers if layer in mapping.SUPPORTED_MAP_LAYERS]

    # Other elements reference lane IDs, so extract lanes first.
    for obj in _map_objects(map_api, layers, corridor):
        if obj.layer == map_objects.MapLayer.LANE:
            element = _write_map_object(obj, centroid, speed_limit_source)
            if isinstance(element, schema.RoadPolyline):
                result[obj.object_id] = element
        else:
            non_lane_objects.append(obj)

    lane_ids = set(result.keys())
    if corridor is not None and any(lane.speed_limit_mps <= 0 for lane in result.values()):
        # Fill over the whole map because the crop often cuts a lane off from every lane with a speed limit.
        map_limits = _map_speed_limits(map_api, speed_limit_source)
        for lane_id, lane in result.items():
            if lane.speed_limit_mps <= 0:
                lane.speed_limit_mps = map_limits.get(lane_id, -1.0)

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

    _mark_junction_lanes(map_api, result, intersection_element_ids)
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


def _write_map_object(
    map_object: Any, centroid: np.ndarray, speed_limit_source: str = "road"
) -> schema.RoadPolyline | schema.RoadSurface | schema.StopZone | None:
    """Convert a 123D map object of a supported layer into a RoadPolyline, RoadSurface, or StopZone."""
    layer = map_object.layer
    if layer == map_objects.MapLayer.LANE:
        if map_object.lane_type == map_objects.LaneType.SHOULDER:
            return None
        return schema.RoadPolyline(
            layer=int(layer),
            type=int(map_object.lane_type),
            polyline=_centered_array(map_object.centerline_3d.array, centroid),
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
            layer=int(layer), type=int(subtype), polyline=_centered_array(map_object.polyline_3d.array, centroid)
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
    corridor: shapely.Geometry | None,
    episode_length: int,
    first_id: int,
    onroad_only: bool = False,
) -> dict[int, schema.Track]:
    """Extract static map objects as tracks that stay still throughout the episode.

    With ``onroad_only``, keep only objects touching a generic drivable surface at their elevation.
    """
    if map_objects.MapLayer.STATIC_OBJECT not in map_api.available_map_layers:
        return {}
    static_objects = _map_objects(map_api, [map_objects.MapLayer.STATIC_OBJECT], corridor)
    if not static_objects:
        return {}
    boxes = [obj.outline.array[:4] for obj in static_objects]
    if onroad_only and map_objects.MapLayer.GENERIC_DRIVABLE in map_api.available_map_layers:
        static_objects, boxes = _filter_onroad_static_objects(map_api, static_objects, boxes)
    tracks: dict[int, schema.Track] = {}
    for n, (obj, corners) in enumerate(zip(static_objects, boxes, strict=False)):
        sides = np.diff(corners[:3, :2], axis=0)
        side_norms = np.linalg.norm(sides, axis=1)
        long_side = sides[int(np.argmax(side_norms))]
        track = _make_empty_track(max(episode_length, 1), str(obj.object_id))  # map-only scenarios get one frame
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


def _write_detection_frame(obj: schema.Track, frame_idx: int, bbox: Any, centroid: np.ndarray, heading: float) -> None:
    arr = bbox.array  # scalar writes: this runs per detection, and small array operations cost twice as much
    obj.position[frame_idx, 0] = arr[0] - centroid[0]
    obj.position[frame_idx, 1] = arr[1] - centroid[1]
    obj.position[frame_idx, 2] = arr[2] - centroid[2] - arr[9] / 2.0  # the box center sits half its height up
    obj.heading[frame_idx] = heading
    obj.valid[frame_idx] = 1
    obj.length[frame_idx], obj.width[frame_idx], obj.height[frame_idx] = arr[7], arr[8], arr[9]


def _make_empty_track(episode_length: int, source_id: str = "") -> schema.Track:
    return schema.Track(
        type=-1,  # placeholder until the 123D label is mapped to a puffer type
        position=np.zeros((episode_length, 3), dtype=np.float64),
        heading=np.zeros((episode_length,), dtype=np.float64),
        velocity=np.full((episode_length, 2), np.nan, dtype=np.float64),
        valid=np.zeros((episode_length,), dtype=np.int32),
        length=np.zeros((episode_length,), dtype=np.float64),
        width=np.zeros((episode_length,), dtype=np.float64),
        height=np.zeros((episode_length,), dtype=np.float64),
        source_id=source_id,
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


def _lane_speed_limit(lane: Any, source: str) -> float:
    """Return a lane's speed limit in m/s, or -1 if unknown.

    With ``signs``, use the last speed sign passed, matching CARLA. Without sign data, use the road's limit.
    """
    speed = getattr(lane, "sign_speed_limit_mps", None) if source == "signs" else None
    speed = lane.speed_limit_mps if speed is None else speed
    return float(speed) if speed and not np.isnan(speed) else -1.0


def _map_speed_limits(map_api: py123d_api.MapAPI, speed_limit_source: str = "road") -> dict[int, float]:
    """Return speed limits for all map lanes, filling unknown limits with ``fill_missing_speed_limits``."""
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
    fill_missing_speed_limits(lanes)
    return {lane_id: lane.speed_limit_mps for lane_id, lane in lanes.items()}


def _centered_array(array: np.ndarray, center: np.ndarray) -> np.ndarray:
    return array.astype(np.float64) - center
