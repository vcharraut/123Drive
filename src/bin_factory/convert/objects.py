"""Convert perception objects from intermediate format to Puffer format."""

import numpy as np
from py123d.datatypes.detections import DefaultBoxDetectionLabel

from bin_factory.convert import types as puffer_types


PY123D_TO_PUFFER_OBJECT = {
    DefaultBoxDetectionLabel.TRAFFIC_SIGN: puffer_types.ObjectType.TRAFFIC_SIGN,
    DefaultBoxDetectionLabel.TRAFFIC_CONE: puffer_types.ObjectType.TRAFFIC_CONE,
    DefaultBoxDetectionLabel.TRAFFIC_LIGHT: puffer_types.ObjectType.TRAFFIC_LIGHT,
    DefaultBoxDetectionLabel.BARRIER: puffer_types.ObjectType.BARRIER,
    DefaultBoxDetectionLabel.GENERIC_OBJECT: puffer_types.ObjectType.GENERIC_OBJECT,
}


def convert_objects(py123d_objects: dict) -> list[dict]:
    return [
        {
            "id": object_id,
            "type": _convert_object_type_to_int(object_data["type"]),
            "states": _convert_object_states(object_data),
        }
        for object_id, object_data in py123d_objects.items()
    ]


def _convert_object_type_to_int(object_type) -> int:
    return PY123D_TO_PUFFER_OBJECT.get(object_type, puffer_types.ObjectType.GENERIC_OBJECT)


def _convert_object_states(object_data: dict) -> dict:
    position = object_data["position"]
    xyz = np.column_stack([position, np.zeros(len(position), dtype=np.float64)]) if position.shape[1] == 2 else position
    return {
        "xyz": xyz,
        "heading": object_data["heading"],
        "velocity": object_data["velocity"],
        "length": object_data["length"],
        "width": object_data["width"],
        "height": object_data["height"],
        "valid": object_data["valid"],
    }
