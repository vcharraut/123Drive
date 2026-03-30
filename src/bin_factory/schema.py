import dataclasses

import numpy as np


@dataclasses.dataclass
class ScenarioMetadata:
    id: str
    dataset: str
    scenario_length: int
    dt: float
    location: str = ""


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


@dataclasses.dataclass
class TrafficLightTrack:
    position: np.ndarray
    states: list
    controlled_lane: int


@dataclasses.dataclass
class PufferScenario:
    agents: dict[int, Track]
    objects: dict[int, Track]
    map: dict[int, dict]
    metadata: ScenarioMetadata
    traffic_controls: list[dict] = dataclasses.field(default_factory=list)
    lane_graph: dict | None = None
