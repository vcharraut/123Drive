from pathlib import Path

from bin_factory.schema import PufferScenario
from bin_factory.serialize import BufferLike, binary_to_scenario, scenario_to_binary


class StaticBinaryError(ValueError):
    """Raised when a static map binary would carry agents."""


def static_binary_to_scenario(data: BufferLike, source: str = "<bytes>") -> PufferScenario:
    scenario = binary_to_scenario(data, source)
    if scenario.agents:
        raise StaticBinaryError(f"{source} is not static: found {len(scenario.agents)} agents")
    return scenario


def write_static_scenario(scenario: PufferScenario, path: str | Path, overwrite: bool = False) -> None:
    if scenario.agents:
        raise StaticBinaryError("write_static_scenario only supports static scenarios")

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(scenario_to_binary(scenario))
