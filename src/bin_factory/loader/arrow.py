from dataclasses import dataclass, field

import numpy as np


@dataclass
class ScenarioMetadata:
    id: str
    dataset: str
    scenario_length: int
    timestep_seconds: float


@dataclass
class DynamicState:
    type: object
    position: np.ndarray
    heading: np.ndarray
    velocity: np.ndarray
    valid: np.ndarray
    length: np.ndarray
    width: np.ndarray
    height: np.ndarray
    route: list = field(default_factory=list)


@dataclass
class TrafficLightState:
    position: np.ndarray
    states: list
    controlled_lane: int


@dataclass
class ArrowScenario:
    agents: dict[int, DynamicState]
    map: dict[int, dict]
    traffic_lights: dict[int, TrafficLightState]
    objects: dict[int, DynamicState]
    metadata: ScenarioMetadata
    # Additional fields created in processing
    traffic_controls: list[dict] = field(default_factory=list)
    lane_graph: dict | None = None
