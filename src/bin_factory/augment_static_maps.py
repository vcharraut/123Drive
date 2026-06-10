import argparse
import dataclasses
import logging
import shutil
from pathlib import Path

import numpy as np
import yaml

from bin_factory import static_binary
from bin_factory.schema import PufferScenario
from bin_factory.transforms import build_lane_distance_matrix, compute_lane_lengths


logger = logging.getLogger(__name__)

SINGULAR_VALUE_TOLERANCE = 1e-9
MAX_AUGMENTED_SEGMENT_LENGTH = 10.0
REQUIRED_CONFIG_KEYS = {"limit", "transforms"}


@dataclasses.dataclass(frozen=True)
class AugmentConfig:
    limit: int
    transforms: list[dict]


def build_transform_catalog(transform_configs: list[dict]) -> dict[str, np.ndarray]:
    raw_catalog = _raw_transform_catalog_from_config(transform_configs)
    validate_transform_catalog(raw_catalog)
    return raw_catalog


def _raw_transform_catalog_from_config(transform_configs: list[dict]) -> dict[str, np.ndarray]:
    if not isinstance(transform_configs, list):
        raise ValueError("Config key 'transforms' must be a list")

    raw_catalog = {}
    for idx, transform_config in enumerate(transform_configs):
        if not isinstance(transform_config, dict):
            raise ValueError(f"Transform config at index {idx} must be a mapping")
        transform_keys = set(transform_config)
        if transform_keys != {"name", "matrix"}:
            missing = sorted({"name", "matrix"} - transform_keys)
            extra = sorted(transform_keys - {"name", "matrix"})
            details = []
            if missing:
                details.append(f"missing keys: {missing}")
            if extra:
                details.append(f"unsupported keys: {extra}")
            raise ValueError(
                f"Transform config at index {idx} must contain exactly ['matrix', 'name'] "
                f"({'; '.join(details)})"
            )

        name = transform_config.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Transform config at index {idx} must have a non-empty string name")
        if name in raw_catalog:
            raise ValueError(f"Duplicate transform name in config: {name!r}")
        try:
            matrix = np.asarray(transform_config["matrix"], dtype=np.float64)
        except KeyError as exc:
            raise ValueError(f"Transform {name!r} must define a matrix") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Transform {name!r} matrix must contain numeric values") from exc

        if matrix.shape != (2, 2):
            raise ValueError(f"Transform {name!r} matrix must have shape (2, 2), got {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"Transform {name!r} matrix must contain only finite values")

        raw_catalog[name] = matrix

    return raw_catalog


def _validate_yaml_config_path(config_path: Path) -> Path:
    config_path = Path(config_path)
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("config must point to a .yaml or .yml file")
    if config_path.is_file():
        return config_path
    raise FileNotFoundError(f"YAML config file does not exist: {config_path}")


def load_augment_config(config_path: Path) -> AugmentConfig:
    config_path = _validate_yaml_config_path(config_path)
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML config: {config_path}") from exc
    raw_config = loaded or {}
    if not isinstance(raw_config, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")

    return _augment_config_from_mapping(raw_config)


def _augment_config_from_mapping(raw_config: dict) -> AugmentConfig:
    config_keys = set(raw_config)
    if config_keys != REQUIRED_CONFIG_KEYS:
        missing = sorted(REQUIRED_CONFIG_KEYS - config_keys)
        extra = sorted(config_keys - REQUIRED_CONFIG_KEYS)
        details = []
        if missing:
            details.append(f"missing keys: {missing}")
        if extra:
            details.append(f"unsupported keys: {extra}")
        raise ValueError(f"Config must contain exactly {sorted(REQUIRED_CONFIG_KEYS)} ({'; '.join(details)})")

    limit = raw_config["limit"]
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValueError("Config key 'limit' must be an integer")
    if limit < 0:
        raise ValueError("Config key 'limit' must be >= 0")

    return AugmentConfig(limit=limit, transforms=raw_config["transforms"])


def validate_transform_catalog(catalog: dict[str, np.ndarray]) -> None:
    for name, matrix in catalog.items():
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        smallest = float(np.min(singular_values))
        if smallest <= SINGULAR_VALUE_TOLERANCE:
            raise ValueError(f"Transform {name!r} must be invertible")


def augment_first_static_maps(
    input_dir: Path,
    output_dir: Path,
    limit: int,
    catalog: dict[str, np.ndarray],
) -> list[Path]:
    if limit < 0:
        raise ValueError("limit must be >= 0")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if limit == 0:
        return []

    input_paths = sorted(input_dir.glob("*.bin"))[:limit]
    if not input_paths:
        raise FileNotFoundError(f"No .bin files found in {input_dir}")

    written = []
    for input_path in input_paths:
        output_stem = input_path.stem
        original_output_path = output_dir / f"{output_stem}.bin"
        _copy_original(input_path, original_output_path)
        written.append(original_output_path)
        logger.info("Wrote %s", original_output_path)

        scenario = static_binary.read_static_scenario(input_path)
        xy = _map_xy_points(scenario)
        centroid = xy.mean(axis=0)
        for transform_name, matrix in catalog.items():
            augmented = static_binary.clone_static_scenario(scenario)
            apply_affine_transform(augmented, matrix, centroid)
            _update_metadata_id(augmented, output_stem, transform_name)
            output_path = output_dir / f"{output_stem}_{transform_name}.bin"
            static_binary.write_static_scenario(augmented, output_path, overwrite=True)
            written.append(output_path)
            logger.info("Wrote %s", output_path)

    return written


def _copy_original(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)


def apply_affine_transform(scenario: PufferScenario, matrix: np.ndarray, centroid: np.ndarray) -> None:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 2):
        raise ValueError(f"Affine matrix must have shape (2, 2), got {matrix.shape}")
    centroid = np.asarray(centroid, dtype=np.float64)
    if centroid.shape != (2,):
        raise ValueError(f"Centroid must have shape (2,), got {centroid.shape}")

    resample_points = _max_local_scale(matrix) > 1.0 + SINGULAR_VALUE_TOLERANCE
    for element in scenario.map.values():
        key = "polyline" if element.uses_polyline else "polygon"
        transformed = _transform_xyz(np.asarray(getattr(element, key), dtype=np.float64), matrix, centroid)
        if resample_points:
            transformed = _resample_xyz_segments(transformed, MAX_AUGMENTED_SEGMENT_LENGTH)
        setattr(element, key, transformed)

    for traffic_control in scenario.traffic_controls:
        traffic_control["stop_line"] = _transform_xyz(
            np.asarray(traffic_control["stop_line"], dtype=np.float64),
            matrix,
            centroid,
        )
        traffic_control["heading"] = _transform_heading(float(traffic_control["heading"]), matrix)

    compute_lane_lengths(scenario)
    scenario.lane_graph = build_lane_distance_matrix(scenario.map)


def _map_xy_points(scenario: PufferScenario) -> np.ndarray:
    xy_parts = []
    for element in scenario.map.values():
        points = element.geometry
        if points is not None and len(points):
            xy_parts.append(np.asarray(points, dtype=np.float64)[:, :2])

    if not xy_parts:
        raise ValueError("Cannot compute map centroid: no road geometry")
    return np.vstack(xy_parts)


def _transform_xyz(xyz: np.ndarray, matrix: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    transformed = xyz.copy()
    transformed[:, :2] = (transformed[:, :2] - centroid) @ matrix.T + centroid
    return transformed


def _max_local_scale(matrix: np.ndarray) -> float:
    return float(np.max(np.linalg.svd(matrix, compute_uv=False)))


def _resample_xyz_segments(xyz: np.ndarray, max_segment_length: float) -> np.ndarray:
    if len(xyz) < 2:
        return xyz.copy()

    diffs = np.diff(xyz, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    subdivisions = np.maximum(np.ceil(seg_lengths / max_segment_length).astype(int), 1)

    total_points = int(subdivisions.sum()) + 1
    result = np.empty((total_points, xyz.shape[1]), dtype=xyz.dtype)

    idx = 0
    for i, diff in enumerate(diffs):
        n = subdivisions[i]
        t = np.arange(n, dtype=xyz.dtype).reshape(-1, 1) / n
        result[idx : idx + n] = xyz[i] + t * diff
        idx += n
    result[idx] = xyz[-1]

    return result


def _transform_heading(heading: float, matrix: np.ndarray) -> float:
    unit = np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)
    transformed = matrix @ unit
    return float(np.arctan2(transformed[1], transformed[0]))


def _update_metadata_id(scenario: PufferScenario, output_stem: str, transform_name: str) -> None:
    scenario.metadata.id = f"{output_stem}_{transform_name}"


def _yaml_config_path(value: str) -> Path:
    try:
        return _validate_yaml_config_path(Path(value))
    except (FileNotFoundError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate affine-augmented static map binaries")
    parser.add_argument(
        "--config",
        type=_yaml_config_path,
        required=True,
        help="YAML config for affine map generation",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing static map .bin files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for augmented .bin files",
    )
    return parser


def main() -> int:
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=logging.INFO)
    args = build_parser().parse_args()
    config = load_augment_config(args.config)
    catalog = build_transform_catalog(config.transforms)
    transforms_per_map = len(catalog)
    expected_count = config.limit * (1 + transforms_per_map)
    logger.info(
        "Generating up to %d map binaries from %d input maps: %d originals/map + %d transforms/map",
        expected_count,
        config.limit,
        1,
        transforms_per_map,
    )
    written = augment_first_static_maps(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        limit=config.limit,
        catalog=catalog,
    )
    logger.info("Generated %d augmented map binaries.", len(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
