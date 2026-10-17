from pathlib import Path

from bin_factory.schema import PufferScenario
from bin_factory.serialize import StaticBinaryError, scenario_to_binary
from bin_factory.serialize import static_binary_to_scenario as static_binary_to_scenario


def write_static_scenario(scenario: PufferScenario, path: str | Path, overwrite: bool = False) -> None:
    if scenario.agents:
        raise StaticBinaryError("write_static_scenario only supports static scenarios")

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(scenario_to_binary(scenario))
