import dataclasses

import numpy as np

from bin_factory import puffer_types


MAP_REF_KEYS = ("entry_lanes", "exit_lanes", "left_neighbor", "right_neighbor")


def arc_length(polyline: np.ndarray) -> np.ndarray:
    """Return cumulative arc length per point, from 0 to total length, using all coordinate columns."""
    polyline = np.asarray(polyline)
    if len(polyline) < 2:
        return np.zeros(len(polyline), dtype=np.float64)
    seg = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float64)


def remap_element_refs(element: "RoadPolyline", id_map: dict[int, int]) -> "RoadPolyline":
    """Copy a RoadPolyline with every MAP_REF_KEYS id remapped through ``id_map`` and unknown ids dropped."""
    return dataclasses.replace(
        element,
        **{key: [id_map[ref] for ref in getattr(element, key) if ref in id_map] for key in MAP_REF_KEYS},
    )


@dataclasses.dataclass
class ScenarioMetadata:
    id: str
    dataset: str
    scenario_length: int
    dt: float
    location: str = ""
    objects_of_interest: list = dataclasses.field(default_factory=list)
    tracks_to_predict: list = dataclasses.field(default_factory=list)
    origin_xyz: tuple = (0.0, 0.0, 0.0)
    map_has_z: bool = False
    scene_uuid: str = ""
    log_name: str = ""
    split: str = ""
    log_version: str = ""
    map_version: str = ""
    timestamps_us: list = dataclasses.field(default_factory=list)
    timestamps_are_relative: bool = False


@dataclasses.dataclass
class Track:
    type: int
    position: np.ndarray
    heading: np.ndarray
    velocity: np.ndarray
    valid: np.ndarray
    length: np.ndarray
    width: np.ndarray
    height: np.ndarray
    route: list = dataclasses.field(default_factory=list)
    route_entry_s: list = dataclasses.field(default_factory=list)  # per route lane, cum_length where it is entered
    route_exit_s: list = dataclasses.field(default_factory=list)  # per route lane, cum_length where it is left
    source_id: str = ""


@dataclasses.dataclass
class TrafficLightTrack:
    position: np.ndarray
    states: list
    controlled_lane: int


@dataclasses.dataclass
class RoadPolyline:
    layer: int
    type: int  # py123d subtype of the layer
    polyline: np.ndarray | None = None
    speed_limit_mps: float = -1.0
    entry_lanes: list = dataclasses.field(default_factory=list)
    exit_lanes: list = dataclasses.field(default_factory=list)
    left_boundary: np.ndarray | None = None
    right_boundary: np.ndarray | None = None
    left_neighbor: list = dataclasses.field(default_factory=list)
    right_neighbor: list = dataclasses.field(default_factory=list)
    in_junction: bool = False
    intersection_id: int = -1  # map element id of the lane's intersection, -1 if none or unknown; not serialized

    @property
    def cum_length(self) -> np.ndarray:
        """3D arc length at each point, derived from the polyline: the metric of route spans."""
        return arc_length(self.polyline if self.polyline is not None else np.zeros((0, 3)))

    @property
    def length(self) -> float:
        cum_length = self.cum_length
        return float(cum_length[-1]) if len(cum_length) else 0.0

    @property
    def is_lane(self) -> bool:
        return self.layer == puffer_types.MapLayer.LANE

    @property
    def is_line(self) -> bool:
        return self.layer == puffer_types.MapLayer.ROAD_LINE

    @property
    def is_edge(self) -> bool:
        return self.layer == puffer_types.MapLayer.ROAD_EDGE


@dataclasses.dataclass
class RoadSurface:
    layer: int
    polygon: np.ndarray


@dataclasses.dataclass
class StopZone:
    type: int
    polygon: np.ndarray
    controlled_lanes: list = dataclasses.field(default_factory=list)
    intersection_id: int = -1  # map element id of the intersection, -1 if unknown
    phase_idx: int = -1  # slot in the intersection's light cycle, -1 if not in a cycle


@dataclasses.dataclass
class TrafficControl:
    id: int
    type: int
    stop_line: np.ndarray  # (2, 3)
    heading: float
    states: list
    controlled_lanes: list
    intersection_id: int = -1  # map element id of the intersection, -1 if unknown
    phase_idx: int = -1  # slot in the intersection's light cycle, compact 0..N-1, -1 if not in a cycle


@dataclasses.dataclass
class ExtractionExtras:
    traffic_lights: dict = dataclasses.field(default_factory=dict)
    stop_zones: list = dataclasses.field(default_factory=list)
    non_junction_shoulders: list = dataclasses.field(default_factory=list)  # centred (N, 3) shoulder outlines
    # centred (N, 3) intersection + junction-shoulder outlines
    junction_drivables: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class PufferScenario:
    agents: dict[int, Track]
    objects: dict[int, Track]
    polylines: dict[int, RoadPolyline]
    metadata: ScenarioMetadata
    surfaces: dict[int, RoadSurface] = dataclasses.field(default_factory=dict)  # same id space as polylines
    traffic_controls: list[TrafficControl] = dataclasses.field(default_factory=list)
