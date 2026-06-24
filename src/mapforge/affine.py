import argparse
import logging
import shutil
from pathlib import Path

import numpy as np

from bin_factory.schema import PufferScenario
from bin_factory.transforms import build_lane_distance_matrix, compute_lane_lengths
from mapforge import static_binary


logger = logging.getLogger(__name__)

SINGULAR_VALUE_TOLERANCE = 1e-9
MAX_AUGMENTED_SEGMENT_LENGTH = 10.0

# Affine transform catalog, grouped by family. Select families with --groups.
TRANSFORM_GROUPS: dict[str, dict[str, list[list[float]]]] = {
    "scale": {
        "Sc10": [[1.10, 0.0], [0.0, 1.10]],  # uniform 10% scale
        "ScX10": [[1.10, 0.0], [0.0, 1.0]],  # stretch global X
        "ScY10": [[1.0, 0.0], [0.0, 1.10]],  # stretch global Y
    },
    "shear": {
        "ShXP": [[1.0, 0.17], [0.0, 1.0]],  # positive X shear
        "ShXN": [[1.0, -0.17], [0.0, 1.0]],  # negative X shear
        "ShYP": [[1.0, 0.0], [0.17, 1.0]],  # positive Y shear
        "ShYN": [[1.0, 0.0], [-0.17, 1.0]],  # negative Y shear
    },
    "flip": {
        "FlipX": [[-1.0, 0.0], [0.0, 1.0]],  # mirror global X
    },
}


def select_transforms(groups: list[str] | None) -> dict[str, np.ndarray]:
    chosen = list(TRANSFORM_GROUPS) if groups is None else groups
    unknown = sorted({name for name in chosen if name not in TRANSFORM_GROUPS})
    if unknown:
        raise ValueError(f"Unknown transform group(s): {unknown}. Available: {list(TRANSFORM_GROUPS)}")
    return {
        name: np.asarray(matrix, dtype=np.float64)
        for group in chosen
        for name, matrix in TRANSFORM_GROUPS[group].items()
    }


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

    input_paths = sorted(input_dir.glob("*.bin"))
    if limit:
        input_paths = input_paths[:limit]
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate affine-augmented static map binaries")
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=list(TRANSFORM_GROUPS),
        default=None,
        help="Transform groups to run (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Augment only the first N input maps (0 = all)",
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
    catalog = select_transforms(args.groups)
    written = augment_first_static_maps(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        catalog=catalog,
    )
    logger.info(
        "Generated %d map binaries: %d transforms/map across groups %s",
        len(written),
        len(catalog),
        args.groups or list(TRANSFORM_GROUPS),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
