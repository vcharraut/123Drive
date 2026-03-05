import numpy as np

from .generic import DetailedTLS
from .geometry import classify_direction


class VehicleState:
    def __init__(self, object_id, lane_pos_idx, speed, acceleration=None):
        self.object_id = object_id
        self.lane_pos_idx = lane_pos_idx
        self.speed = speed
        self.acceleration = acceleration


class InJunctionLane:
    def __init__(self, shape, record_tls=None, record_vehs=None, id=None, length=0):
        self.id = id
        self.shape = shape
        self.direction = classify_direction(np.asarray(shape)[:, :2])

        if not record_vehs:
            record_vehs = [{} for _ in range(length)]

        if not record_tls:
            record_tls = [DetailedTLS.ABSENT] * length

        self.record_vehs = record_vehs[:]
        self.record_tls_detailed = record_tls[:]
        self.record_tls = [tls.generalize() for tls in record_tls]


class ApproachingLane:
    def __init__(self, shape, record_vehs=None, injunction_lanes=None, id=None, length=0):
        self.id = id
        self.shape = shape

        if not record_vehs:
            record_vehs = [{} for _ in range(length)]

        self.record_vehs = record_vehs[:]
        self.injunction_lanes = injunction_lanes[:] if injunction_lanes else []
