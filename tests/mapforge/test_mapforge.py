"""Tests for the ``mapforge`` package (static-binary IO + affine augmentation).

Data source is the two committed opendrive maps under ``tests/py123d_data/maps/opendrive``.
Each is run once through the real map-only conversion path to produce a static ``.bin``
payload, which is the input mapforge actually consumes in production.
"""

import pathlib

import numpy as np
import pytest

from bin_factory import loader, main, serialize, transforms
from mapforge import affine, static_binary
from mapforge.static_binary import StaticBinaryError


DATA_ROOT = pathlib.Path(__file__).parent.parent / "py123d_data"
MAP_LOCATIONS = ["Town02", "Town10HD"]


def _opendrive_static_bytes(location):
    """Map-only conversion of one opendrive map -> serialized static .bin bytes."""
    config = main.build_parser().parse_args(["--py123d_path", str(DATA_ROOT)])
    config.map_only = True
    maps = loader.discover_scenes(py123d_data_root=str(DATA_ROOT), workers=1, map_only=True, datasets=["opendrive"])
    scene = next(m for m in maps if m.location == location)
    scenario, extras = loader.extract_scenario(scene)
    assert loader.validate_scenario(scenario, extras=extras, level=2) == []
    transforms.run(scenario, extras, config)
    return serialize.scenario_to_binary(scenario)


@pytest.fixture(scope="module")
def opendrive_bins():
    """{location: static .bin bytes} built once for the whole module."""
    return {loc: _opendrive_static_bytes(loc) for loc in MAP_LOCATIONS}


def _all_xy(scenario):
    return np.vstack(
        [
            np.asarray(e.geometry, dtype=np.float64)[:, :2]
            for e in scenario.map.values()
            if e.geometry is not None and len(e.geometry)
        ]
    )


def _centroid(scenario):
    return _all_xy(scenario).mean(axis=0)


# ── static_binary: read ───────────────────────


@pytest.mark.parametrize("location", MAP_LOCATIONS)
def test_read_static_scenario_is_map_only(opendrive_bins, location):
    scenario = static_binary.static_binary_to_scenario(opendrive_bins[location])
    assert scenario.agents == {}
    assert scenario.objects == {}
    assert len(scenario.map) > 0
    assert scenario.metadata.id == location
    assert scenario.metadata.dataset == "opendrive"


@pytest.mark.parametrize("location", MAP_LOCATIONS)
def test_read_static_scenario_from_path(opendrive_bins, tmp_path, location):
    path = tmp_path / f"{location}.bin"
    path.write_bytes(opendrive_bins[location])
    from_path = static_binary.read_static_scenario(path)
    from_bytes = static_binary.static_binary_to_scenario(opendrive_bins[location])
    assert list(from_path.map) == list(from_bytes.map)


@pytest.mark.parametrize("location", MAP_LOCATIONS)
def test_read_preserves_geometry_to_float32(opendrive_bins, location):
    scenario = static_binary.static_binary_to_scenario(opendrive_bins[location])
    # The reader keeps element ids and shapes; geometry survives to float32 precision.
    for element in scenario.map.values():
        assert element.geometry is not None
        assert len(element.geometry) >= element.min_points


@pytest.mark.parametrize("location", MAP_LOCATIONS)
def test_read_serialize_is_idempotent(opendrive_bins, location):
    # First cycle normalizes float32 precision; every later cycle is then byte-exact.
    once = serialize.scenario_to_binary(static_binary.static_binary_to_scenario(opendrive_bins[location]))
    twice = serialize.scenario_to_binary(static_binary.static_binary_to_scenario(once))
    assert once == twice


def test_read_rejects_binary_with_agents():
    # Header: n_agents=1, n_roads=0, n_traffic=0, n_objects=0.
    data = np.array([1, 0, 0, 0], dtype="<i4").tobytes()
    with pytest.raises(StaticBinaryError, match="not static"):
        static_binary.static_binary_to_scenario(data)


def test_read_rejects_trailing_bytes(opendrive_bins):
    with pytest.raises(StaticBinaryError, match="trailing bytes"):
        static_binary.static_binary_to_scenario(opendrive_bins["Town02"] + b"\x00")


def test_read_rejects_truncated_payload(opendrive_bins):
    with pytest.raises(StaticBinaryError):
        static_binary.static_binary_to_scenario(opendrive_bins["Town02"][:-8])


# ── static_binary: write ───────────────────────


def test_write_then_read_round_trips(opendrive_bins, tmp_path):
    scenario = static_binary.static_binary_to_scenario(opendrive_bins["Town02"])
    out = tmp_path / "out.bin"
    static_binary.write_static_scenario(scenario, out)
    assert out.exists()
    assert list(static_binary.read_static_scenario(out).map) == list(scenario.map)


def test_write_refuses_overwrite_without_flag(opendrive_bins, tmp_path):
    scenario = static_binary.static_binary_to_scenario(opendrive_bins["Town02"])
    out = tmp_path / "out.bin"
    static_binary.write_static_scenario(scenario, out)
    with pytest.raises(FileExistsError):
        static_binary.write_static_scenario(scenario, out)
    static_binary.write_static_scenario(scenario, out, overwrite=True)  # explicit overwrite is allowed


def test_write_rejects_non_static(opendrive_bins, tmp_path):
    scenario = static_binary.static_binary_to_scenario(opendrive_bins["Town02"])
    scenario.agents = {0: object()}
    with pytest.raises(StaticBinaryError):
        static_binary.write_static_scenario(scenario, tmp_path / "bad.bin")


# ── static_binary: clone ───────────────────────


def test_clone_is_independent_deep_copy(opendrive_bins):
    scenario = static_binary.static_binary_to_scenario(opendrive_bins["Town02"])
    clone = static_binary.clone_static_scenario(scenario)
    first = next(iter(clone.map.values()))
    key = "polyline" if first.uses_polyline else "polygon"
    getattr(first, key)[:] = 0.0
    clone.metadata.id = "mutated"
    original_first = next(iter(scenario.map.values()))
    assert getattr(original_first, key).any()  # original geometry untouched
    assert scenario.metadata.id == "Town02"


# ── affine: transform selection ───────────────────────


def test_select_transforms_default_is_all_groups():
    catalog = affine.select_transforms(None)
    expected = {name for group in affine.TRANSFORM_GROUPS.values() for name in group}
    assert set(catalog) == expected
    assert all(m.shape == (2, 2) for m in catalog.values())


def test_select_transforms_subset():
    assert set(affine.select_transforms(["flip"])) == {"FlipX"}


def test_select_transforms_unknown_raises():
    with pytest.raises(ValueError, match="Unknown transform group"):
        affine.select_transforms(["warp"])


# ── affine: apply_affine_transform ───────────────────────


def test_apply_affine_rejects_bad_matrix_shape(opendrive_bins):
    scenario = static_binary.static_binary_to_scenario(opendrive_bins["Town02"])
    with pytest.raises(ValueError, match="shape"):
        affine.apply_affine_transform(scenario, np.eye(3), np.zeros(2))


def test_apply_affine_rejects_bad_centroid_shape(opendrive_bins):
    scenario = static_binary.static_binary_to_scenario(opendrive_bins["Town02"])
    with pytest.raises(ValueError, match="Centroid"):
        affine.apply_affine_transform(scenario, np.eye(2), np.zeros(3))


def test_flip_mirrors_x_about_centroid(opendrive_bins):
    scenario = static_binary.static_binary_to_scenario(opendrive_bins["Town02"])
    reference = static_binary.clone_static_scenario(scenario)
    centroid = _centroid(scenario)
    affine.apply_affine_transform(scenario, affine.TRANSFORM_GROUPS["flip"]["FlipX"], centroid)

    for eid, element in scenario.map.items():
        ref = reference.map[eid].geometry
        # FlipX has unit singular values -> no resampling, so point counts are preserved.
        assert len(element.geometry) == len(ref)
        assert np.allclose(element.geometry[:, 0], 2 * centroid[0] - ref[:, 0], atol=1e-3)
        assert np.allclose(element.geometry[:, 1], ref[:, 1], atol=1e-3)


def test_scale_grows_extent_and_resamples(opendrive_bins):
    scenario = static_binary.static_binary_to_scenario(opendrive_bins["Town02"])
    reference = static_binary.clone_static_scenario(scenario)
    centroid = _centroid(scenario)
    before_extent = np.ptp(_all_xy(reference), axis=0)
    before_points = sum(len(e.geometry) for e in reference.map.values())

    affine.apply_affine_transform(scenario, affine.TRANSFORM_GROUPS["scale"]["Sc10"], centroid)

    after_extent = np.ptp(_all_xy(scenario), axis=0)
    after_points = sum(len(e.geometry) for e in scenario.map.values())
    assert np.allclose(after_extent / before_extent, 1.10, atol=1e-2)
    # A scale > 1 lengthens segments, so resampling can only add points, never drop them.
    assert after_points > before_points


def test_traffic_control_heading_rotates_under_flip(opendrive_bins):
    scenario = static_binary.static_binary_to_scenario(opendrive_bins["Town02"])
    assert scenario.traffic_controls, "opendrive fixture should carry traffic controls"
    reference = static_binary.clone_static_scenario(scenario)
    centroid = _centroid(scenario)
    matrix = np.asarray(affine.TRANSFORM_GROUPS["flip"]["FlipX"], dtype=np.float64)
    affine.apply_affine_transform(scenario, matrix, centroid)

    for tc, ref in zip(scenario.traffic_controls, reference.traffic_controls, strict=False):
        h = float(ref["heading"])
        rotated = matrix @ np.array([np.cos(h), np.sin(h)])
        assert tc["heading"] == pytest.approx(float(np.arctan2(rotated[1], rotated[0])))


def test_apply_affine_rebuilds_lane_graph(opendrive_bins):
    scenario = static_binary.static_binary_to_scenario(opendrive_bins["Town02"])
    lane_ids = list(scenario.lane_graph["lane_ids"])
    affine.apply_affine_transform(scenario, affine.TRANSFORM_GROUPS["scale"]["ScX10"], _centroid(scenario))
    # Stretching global X changes inter-lane distances but not the lane set.
    assert list(scenario.lane_graph["lane_ids"]) == lane_ids
    assert scenario.lane_graph["distances"].shape == (len(lane_ids), len(lane_ids))


# ── affine: augment_maps ───────────────────────


def test_augment_maps_writes_original_plus_variants(opendrive_bins, tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    for loc in MAP_LOCATIONS:
        (input_dir / f"opendrive__{loc}.bin").write_bytes(opendrive_bins[loc])

    output_dir = tmp_path / "out"
    catalog = affine.select_transforms(["flip"])  # one transform -> one variant per map
    written = affine.augment_maps(input_dir, output_dir, catalog)

    assert len(written) == len(MAP_LOCATIONS) * (1 + len(catalog))
    for loc in MAP_LOCATIONS:
        assert (output_dir / f"opendrive__{loc}.bin").exists()  # untouched copy of the original
        variant = output_dir / f"opendrive__{loc}_FlipX.bin"
        assert variant.exists()
        assert static_binary.read_static_scenario(variant).metadata.id == f"opendrive__{loc}_FlipX"


def test_augment_maps_original_copy_is_byte_identical(opendrive_bins, tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "opendrive__Town02.bin").write_bytes(opendrive_bins["Town02"])
    output_dir = tmp_path / "out"
    affine.augment_maps(input_dir, output_dir, affine.select_transforms(["flip"]))
    assert (output_dir / "opendrive__Town02.bin").read_bytes() == opendrive_bins["Town02"]


def test_augment_maps_empty_input_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        affine.augment_maps(empty, tmp_path / "out", affine.select_transforms(["flip"]))


# ── affine: CLI ───────────────────────


def test_main_cli_generates_binaries(opendrive_bins, tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "opendrive__Town02.bin").write_bytes(opendrive_bins["Town02"])
    output_dir = tmp_path / "out"

    argv = ["mapforge", "--groups", "flip", "--input_dir", str(input_dir), "--output_dir", str(output_dir)]
    monkeypatch.setattr("sys.argv", argv)
    assert affine.main() == 0
    assert sorted(p.name for p in output_dir.glob("*.bin")) == [
        "opendrive__Town02.bin",
        "opendrive__Town02_FlipX.bin",
    ]
