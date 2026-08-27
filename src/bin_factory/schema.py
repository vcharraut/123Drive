import dataclasses

import numpy as np

from bin_factory import puffer_types


MAP_REF_KEYS = ("entry_lanes", "exit_lanes", "left_neighbor", "right_neighbor")


def remap_element_refs(element: "MapElement", id_map: dict[int, int]) -> "MapElement":
    """Copy a MapElement with every MAP_REF_KEYS id passed through ``id_map`` (unknown ids dropped)."""
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
    route_gt_len: int = 0
    control_state: int = int(puffer_types.ControlState.NON_CONTROLLABLE_STATIC)


@dataclasses.dataclass
class TrafficLightTrack:
    position: np.ndarray
    states: list
    controlled_lane: int


@dataclasses.dataclass
class MapElement:
    type: int
    polyline: np.ndarray | None = None
    polygon: np.ndarray | None = None
    speed_limit_mps: float = -1.0
    entry_lanes: list = dataclasses.field(default_factory=list)
    exit_lanes: list = dataclasses.field(default_factory=list)
    left_boundary: np.ndarray | None = None
    right_boundary: np.ndarray | None = None
    left_neighbor: list = dataclasses.field(default_factory=list)
    right_neighbor: list = dataclasses.field(default_factory=list)
    length: float = 0.0
    cum_length: np.ndarray | None = None

    @property
    def is_lane(self) -> bool:
        return puffer_types.is_road_lane(self.type)

    @property
    def is_line(self) -> bool:
        return puffer_types.is_road_line(self.type)

    @property
    def is_edge(self) -> bool:
        return puffer_types.is_road_edge(self.type)

    @property
    def is_crosswalk(self) -> bool:
        return puffer_types.is_crosswalk(self.type)

    @property
    def uses_polyline(self) -> bool:
        return self.is_lane or self.is_line or self.is_edge

    @property
    def geometry(self) -> np.ndarray | None:
        return self.polyline if self.uses_polyline else self.polygon

    @property
    def min_points(self) -> int:
        return 2 if self.uses_polyline else 3


@dataclasses.dataclass
class StopZone:
    type: int
    polygon: np.ndarray
    controlled_lanes: list = dataclasses.field(default_factory=list)
    junction_id: int = -1  # -1: not part of a signalized junction cycle
    phase_idx: int = -1


@dataclasses.dataclass
class ExtractionExtras:
    traffic_lights: dict = dataclasses.field(default_factory=dict)
    stop_zones: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class PufferScenario:
    agents: dict[int, Track]
    objects: dict[int, Track]
    map: dict[int, MapElement]
    metadata: ScenarioMetadata
    traffic_controls: list[dict] = dataclasses.field(default_factory=list)
    lane_graph: dict | None = None
