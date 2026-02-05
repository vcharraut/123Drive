from __future__ import annotations

import hashlib
from pathlib import Path


def safe_id_to_int(value: object, bits: int = 32) -> int:
    """Convert an arbitrary ID to a stable int.

    Args:
        value: ID value (int/str/other).
        bits: Hash bit-length to keep.

    Returns:
        Stable integer ID.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value.isdigit():
            return int(value)
        raw = value.encode("utf-8")
    else:
        raw = str(value).encode("utf-8")

    hash_bytes = hashlib.sha256(raw).digest()
    return int.from_bytes(hash_bytes[: bits // 8], "big")


def resolve_py123d_data_root(dataset_path: str | None) -> Path:
    """Resolve py123d data root path.

    Args:
        dataset_path: Optional override for the dataset root.

    Returns:
        Path to py123d_data root.
    """
    if dataset_path:
        return Path(dataset_path)

    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "py123d_data"
