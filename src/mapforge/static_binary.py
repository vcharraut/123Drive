import copy
import dataclasses
import struct
from pathlib import Path

import numpy as np

from bin_factory import puffer_types
from bin_factory.schema import MapElement, PufferScenario, ScenarioMetadata
from bin_factory.serialize import METADATA_DATASET_BYTES, METADATA_ID_BYTES, scenario_to_binary


class StaticBinaryError(ValueError):
    """Raised when a .bin payload is not a supported static map binary."""


@dataclasses.dataclass
class _Reader:
    data: bytes
    offset: int = 0

    def i32(self) -> int:
        value = struct.unpack_from("<i", self.data, self.offset)[0]
        self.offset += 4
        return int(value)

    def f32(self) -> float:
        value = struct.unpack_from("<f", self.data, self.offset)[0]
        self.offset += 4
        return float(value)

    def i32_array(self, n: int) -> np.ndarray:
        if n == 0:
            return np.zeros(0, dtype=np.int32)
        arr = np.frombuffer(self.data, dtype="<i4", count=n, offset=self.offset).copy()
        self.offset += 4 * n
        return arr

    def f32_array(self, n: int) -> np.ndarray:
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        arr = np.frombuffer(self.data, dtype="<f4", count=n, offset=self.offset).copy()
        self.offset += 4 * n
        return arr

    def int_list(self) -> list[int]:
        n = self.i32()
        return self.i32_array(n).astype(np.int64).tolist()

    def string(self, n: int) -> str:
        raw = self.data[self.offset : self.offset + n]
        self.offset += n
        return raw.split(b"\0", 1)[0].decode("utf-8")


def read_static_scenario(path: str | Path) -> PufferScenario:
    """Read a static PufferDrive .bin file into a PufferScenario.

    Accepts only static map binaries: no agents and no objects.
    """
    path = Path(path)
    return static_binary_to_scenario(path.read_bytes(), source=str(path))


def static_binary_to_scenario(data: bytes, source: str = "<bytes>") -> PufferScenario:
    reader = _Reader(data)
    try:
        n_agents = reader.i32()
        n_roads = reader.i32()
        n_traffic = reader.i32()
        n_objects = reader.i32()

        if n_agents or n_objects:
            raise StaticBinaryError(f"{source} is not static: found {n_agents} agents and {n_objects} objects")

        road_map = _read_roads(reader, n_roads)
        traffic_controls = _read_traffic_controls(reader, n_traffic)
        lane_graph = _read_lane_graph(reader)
        metadata = _read_metadata(reader)
    except (struct.error, ValueError) as exc:
        if isinstance(exc, StaticBinaryError):
            raise
        raise StaticBinaryError(f"{source} is not a valid static PufferDrive binary") from exc

    if reader.offset != len(data):
        raise StaticBinaryError(f"{source} has {len(data) - reader.offset} trailing bytes after metadata")

    return PufferScenario(
        agents={},
        objects={},
        map=road_map,
        traffic_controls=traffic_controls,
        lane_graph=lane_graph,
        metadata=metadata,
    )


def write_static_scenario(scenario: PufferScenario, path: str | Path, overwrite: bool = False) -> None:
    if scenario.agents or scenario.objects:
        raise StaticBinaryError("write_static_scenario only supports static scenarios")

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(scenario_to_binary(scenario))


def clone_static_scenario(scenario: PufferScenario) -> PufferScenario:
    return copy.deepcopy(scenario)


def _read_roads(reader: _Reader, n_roads: int) -> dict[int, MapElement]:
    road_map = {}
    for _ in range(n_roads):
        element_id = reader.i32()
        road_type = reader.i32()
        n_points = reader.i32()
        x = reader.f32_array(n_points)
        y = reader.f32_array(n_points)
        z = reader.f32_array(n_points)
        reader.f32_array(n_points)  # headings are recomputed by serialize.scenario_to_binary

        xyz = np.column_stack([x, y, z]).astype(np.float64)
        if (
            puffer_types.is_road_lane(road_type)
            or puffer_types.is_road_line(road_type)
            or puffer_types.is_road_edge(road_type)
        ):
            element = MapElement(type=road_type, polyline=xyz)
        else:
            element = MapElement(type=road_type, polygon=xyz)

        if puffer_types.is_road_lane(road_type):
            element.entry_lanes = reader.int_list()
            element.exit_lanes = reader.int_list()
            element.speed_limit_mps = reader.f32()
            element.length = reader.f32()
            element.cum_length = reader.f32_array(n_points).astype(np.float64)

        road_map[element_id] = element
    return road_map


def _read_traffic_controls(reader: _Reader, n_traffic: int) -> list[dict]:
    traffic_controls = []
    for _ in range(n_traffic):
        control_id = reader.i32()
        control_type = reader.i32()
        stop_line = np.array(
            [
                [reader.f32(), reader.f32(), reader.f32()],
                [reader.f32(), reader.f32(), reader.f32()],
            ],
            dtype=np.float64,
        )
        heading = reader.f32()
        states = reader.int_list()
        controlled_lanes = reader.int_list()
        traffic_controls.append(
            {
                "id": control_id,
                "type": control_type,
                "stop_line": stop_line,
                "heading": heading,
                "states": states,
                "controlled_lanes": controlled_lanes,
            }
        )
    return traffic_controls


def _read_lane_graph(reader: _Reader) -> dict | None:
    n_lanes = reader.i32()
    if n_lanes == 0:
        return None
    lane_ids = reader.i32_array(n_lanes).astype(np.int64).tolist()
    distances = reader.f32_array(n_lanes * n_lanes).reshape(n_lanes, n_lanes).astype(np.float64)
    return {"lane_ids": lane_ids, "distances": distances}


def _read_metadata(reader: _Reader) -> ScenarioMetadata:
    scenario_id = reader.string(METADATA_ID_BYTES)
    dataset = reader.string(METADATA_DATASET_BYTES)
    scenario_length = reader.i32()
    dt = reader.f32()
    objects_of_interest = reader.int_list()
    tracks_to_predict = reader.int_list()
    return ScenarioMetadata(
        id=scenario_id,
        dataset=dataset,
        scenario_length=scenario_length,
        dt=dt,
        objects_of_interest=objects_of_interest,
        tracks_to_predict=tracks_to_predict,
    )
