from .generic import TLS, Direction, Pt
from .geometry import classify_direction


class VehicleState:
    def __init__(self, object_id: int, lane_pos_idx: int, speed: float, acceleration: float = None):
        self.object_id: int = object_id
        self.lane_pos_idx: int = lane_pos_idx
        self.speed: float = speed  # m/s
        self.acceleration: float = acceleration  # m/s^2


class InJunctionLane:
    def __init__(
        self,
        shape: list[Pt],
        record_tls: list = None,
        record_vehs: list[dict[int, VehicleState]] = None,
        id=None,
        length: int = 0,
    ) -> None:
        self.id = id
        self.shape: list[Pt] = shape[:]
        self.direction: Direction = classify_direction([(pt.x, pt.y) for pt in shape])

        if not record_vehs:
            record_vehs = [{} for _ in range(length)]

        if not record_tls:
            record_tls = [-1 for _ in range(length)]

        self.record_vehs: list[dict[int, VehicleState]] = record_vehs[:]

        # tls-related
        self.record_tls_waymonic: list = record_tls[:]
        self.record_tls: list[TLS] = [tls.generalize() for tls in record_tls]
        self.new_tls: list[TLS] = [TLS.UNKNOWN for _ in range(length)]


class ApproachingLane:
    def __init__(
        self,
        shape: list[Pt],
        record_vehs: list[dict[int, VehicleState]] = None,
        injunction_lanes: list[InJunctionLane] = None,
        id=None,
        length: int = 0,
    ) -> None:
        self.id = id
        self.shape: list[Pt] = shape[:]

        if not record_vehs:
            record_vehs = [{} for _ in range(length)]

        self.record_vehs: list[dict[int, VehicleState]] = record_vehs[:]
        self.injunction_lanes: list[InJunctionLane] = injunction_lanes[:] if injunction_lanes else []
