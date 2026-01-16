from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class DatasetConfig:
    name: str
    load_func: Callable
    convert_func: Callable
    preprocess_func: Optional[Callable] = None


def get_dataset_config(name: str) -> DatasetConfig:
    if name == "py123d":
        from src.datasets import py123d

        return DatasetConfig(
            name="py123d",
            load_func=py123d.get_py123d_scenarios,
            convert_func=py123d.convert_py123d_scenario,
        )

    raise ValueError(f"Unsupported dataset: {name} (supported: py123d)")
