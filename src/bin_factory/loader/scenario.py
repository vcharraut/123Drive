from dataclasses import dataclass, field


@dataclass
class ArrowScenario:
    agents: dict
    map: dict
    traffic_lights: dict
    objects: dict
    metadata: dict
    traffic_controls: list[dict] = field(default_factory=list)
    lane_graph: dict | None = None
