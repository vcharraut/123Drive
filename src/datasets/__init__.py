from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class DatasetConfig:
    name: str
    load_func: Callable
    convert_func: Callable
    preprocess_func: Optional[Callable] = None


def get_dataset_config(name: str) -> DatasetConfig:
    if name == "waymo":
        from src.datasets import waymo

        return DatasetConfig(
            name="waymo",
            load_func=waymo.get_waymo_scenarios,
            convert_func=waymo.convert_waymo_scenario,
            preprocess_func=waymo.preprocess_waymo_scenarios,
        )
    if name == "nuplan":
        from src.datasets import nuplan

        return DatasetConfig(
            name="nuplan",
            load_func=nuplan.get_nuplan_scenarios,
            convert_func=nuplan.convert_nuplan_scenario,
        )
    if name == "openscenes":
        from src.datasets import openscenes

        return DatasetConfig(
            name="openscenes",
            load_func=openscenes.get_openscenes_scenarios,
            convert_func=openscenes.convert_openscenes_scenario,
        )

    raise ValueError(f"Unsupported dataset: {name} (supported: waymo|nuplan|openscenes)")
