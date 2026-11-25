"""Test static binary reading, writing, and affine map transforms.

Convert the two committed OpenDRIVE maps in ``tests/py123d_data/maps/opendrive`` once
to produce the static ``.bin`` files that mapforge reads.
"""

import copy
import pathlib

import numpy as np
import pytest

from bin_factory import loader, puffer_types, schema, serialize
from bin_factory.preparation import ConversionSettings, prepare_scenario
from mapforge import affine, static_binary
from mapforge.static_binary import StaticBinaryError


DATA_ROOT = pathlib.Path(__file__).parent.parent / "py123d_data"
MAP_LOCATIONS = ["Town02", "Town10HD"]


def _opendrive_static_bytes(location):
    """Map-only conversion of one opendrive map -> serialized static .bin bytes."""
    maps = loader.discover_scenes(py123d_data_root=str(DATA_ROOT), workers=1, map_only=True, datasets=["opendrive"])
    scene = next(m for m in maps if m.location == location)
    return serialize.scenario_to_binary(prepare_scenario(scene, ConversionSettings(validate_level=2)))


@pytest.fixture(scope="module")
def opendrive_bins():
    """{location: static .bin bytes} built once for the whole module."""
    return {loc: _opendrive_static_bytes(loc) for loc in MAP_LOCATIONS}


@pytest.fixture
def town02(opendrive_bins):
    """A fresh scenario read from the Town02 static binary."""
    return static_binary.static_binary_to_scenario(opendrive_bins["Town02"])


def _input_dir(tmp_path, opendrive_bins, locations):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    for loc in locations:
        (input_dir / f"opendrive__{loc}.bin").write_bytes(opendrive_bins[loc])
    return input_dir


def _geometries(scenario):
    return {
        **{eid: e.polyline for eid, e in scenario.polylines.items()},
        **{eid: e.polygon for eid, e in scenario.surfaces.items()},
    }


def _all_xy(scenario):
    return np.vstack(
        [np.asarray(g, dtype=np.float64)[:, :2] for g in _geometries(scenario).values() if g is not None and len(g)]
    )


def _centroid(scenario):
    return _all_xy(scenario).mean(axis=0)


def _with_object(scenario, x=10.0, y=20.0, heading=0.3, length=4.0, width=1.0):
    """Adds one static object (single valid frame), like a converted CARLA static obstacle."""
    scenario.objects = {
        7: schema.Track(
            type=int(puffer_types.ObjectType.BARRIER),
            position=np.array([[x, y, 1.5]]),
            heading=np.array([heading]),
            velocity=np.zeros((1, 2)),
            valid=np.array([1], dtype=np.int32),
            length=np.array([length]),
            width=np.array([width]),
            height=np.array([2.0]),
        )
    }
    return scenario


def _box_corners(x, y, heading, length, width):
    d = np.array([np.cos(heading), np.sin(heading)])
    n = np.array([-np.sin(heading), np.cos(heading)])
    return np.array([[x, y] + sl * length / 2 * d + sw * width / 2 * n for sl in (-1, 1) for sw in (-1, 1)])


# Read static binaries


@pytest.mark.parametrize("location", MAP_LOCATIONS)
def test_read_static_scenario_is_map_only_with_whole_geometry(opendrive_bins, location):
    scenario = static_binary.static_binary_to_scenario(opendrive_bins[location])
    assert scenario.agents == {}
    assert scenario.objects == {}
    assert len(scenario.polylines) > 0
    assert scenario.metadata.id == location
    assert scenario.metadata.dataset == "opendrive"
    # The reader keeps element ids and shapes; geometry survives to float32 precision.
    for element in scenario.polylines.values():
        assert element.polyline is not None
        assert len(element.polyline) >= 2
    for surface in scenario.surfaces.values():
        assert len(surface.polygon) >= 3


@pytest.mark.parametrize("location", MAP_LOCATIONS)
def test_read_serialize_is_idempotent(opendrive_bins, location):
    # First cycle normalizes float32 precision; every later cycle is then byte-exact.
    once = serialize.scenario_to_binary(static_binary.static_binary_to_scenario(opendrive_bins[location]))
    twice = serialize.scenario_to_binary(static_binary.static_binary_to_scenario(once))
    assert once == twice


def test_read_rejects_binary_with_agents(town02):
    scenario = _with_object(town02)
    scenario.agents, scenario.objects = scenario.objects, {}
    with pytest.raises(StaticBinaryError, match="not static: found 1 agents"):
        static_binary.static_binary_to_scenario(serialize.scenario_to_binary(scenario))


def test_read_keeps_objects(town02):
    data = serialize.scenario_to_binary(_with_object(town02))
    track = static_binary.static_binary_to_scenario(data).objects[7]
    assert track.type == int(puffer_types.ObjectType.BARRIER)
    assert np.allclose(track.position, [[10.0, 20.0, 1.5]])
    assert np.allclose([track.heading[0], track.length[0], track.width[0], track.height[0]], [0.3, 4.0, 1.0, 2.0])
    assert track.valid.tolist() == [1]
    # The reader returns what the writer stored: a second cycle is byte-exact.
    assert serialize.scenario_to_binary(static_binary.static_binary_to_scenario(data)) == data


def test_read_rejects_trailing_bytes(opendrive_bins):
    with pytest.raises(serialize.BinaryFormatError, match="trailing bytes"):
        static_binary.static_binary_to_scenario(opendrive_bins["Town02"] + b"\x00")


def test_read_rejects_truncated_payload(opendrive_bins):
    with pytest.raises(serialize.BinaryFormatError):
        static_binary.static_binary_to_scenario(opendrive_bins["Town02"][:-8])


# Write static binaries


def test_write_round_trips_and_requires_explicit_overwrite(town02, tmp_path):
    scenario = town02
    out = tmp_path / "out.bin"
    static_binary.write_static_scenario(scenario, out)
    assert list(_geometries(static_binary.static_binary_to_scenario(out.read_bytes()))) == list(_geometries(scenario))
    with pytest.raises(FileExistsError):
        static_binary.write_static_scenario(scenario, out)
    static_binary.write_static_scenario(scenario, out, overwrite=True)  # explicit overwrite is allowed


def test_write_rejects_non_static(town02, tmp_path):
    town02.agents = dict(_with_object(town02).objects)
    with pytest.raises(StaticBinaryError):
        static_binary.write_static_scenario(town02, tmp_path / "bad.bin")


# Select transforms


def test_select_transforms_default_is_all_groups():
    catalog = affine.select_transforms(None)
    base = {name for group_name, group in affine.TRANSFORM_GROUPS.items() if group_name != "flip" for name in group}
    expected = base | {"FlipX"} | {f"FlipX_{name}" for name in base}
    assert set(catalog) == expected
    assert len(catalog) == 15
    assert all(m.shape == (2, 2) for m in catalog.values())


def test_select_transforms_subset():
    assert set(affine.select_transforms(["flip"])) == {"FlipX"}


def test_select_transforms_unknown_raises():
    with pytest.raises(ValueError, match="Unknown transform group"):
        affine.select_transforms(["warp"])


# Apply affine transforms


@pytest.mark.parametrize(
    ("matrix", "centroid", "match"), [(np.eye(3), np.zeros(2), "shape"), (np.eye(2), np.zeros(3), "Centroid")]
)
def test_apply_affine_rejects_bad_matrix_or_centroid_shape(town02, matrix, centroid, match):
    with pytest.raises(ValueError, match=match):
        affine.apply_affine_transform(town02, matrix, centroid)


def test_flip_mirrors_x_about_centroid(town02):
    scenario = town02
    reference = copy.deepcopy(scenario)
    centroid = _centroid(scenario)
    affine.apply_affine_transform(scenario, affine.TRANSFORM_GROUPS["flip"]["FlipX"], centroid)

    reference_geometries = _geometries(reference)
    for eid, geometry in _geometries(scenario).items():
        ref = reference_geometries[eid]
        assert geometry is not None
        assert ref is not None
        # FlipX has unit singular values -> no resampling, so point counts are preserved.
        assert len(geometry) == len(ref)
        assert np.allclose(geometry[:, 0], 2 * centroid[0] - ref[:, 0], atol=1e-3)
        assert np.allclose(geometry[:, 1], ref[:, 1], atol=1e-3)


def test_scale_grows_extent_and_resamples(town02):
    scenario = town02
    reference = copy.deepcopy(scenario)
    centroid = _centroid(scenario)
    before_extent = np.ptp(_all_xy(reference), axis=0)
    before_points = sum(len(g) for g in _geometries(reference).values() if g is not None)

    affine.apply_affine_transform(scenario, affine.TRANSFORM_GROUPS["scale"]["Sc10"], centroid)

    after_extent = np.ptp(_all_xy(scenario), axis=0)
    after_points = sum(len(g) for g in _geometries(scenario).values() if g is not None)
    assert np.allclose(after_extent / before_extent, 1.10, atol=1e-2)
    # A scale > 1 lengthens segments, so resampling can only add points, never drop them.
    assert after_points > before_points


def test_traffic_control_heading_rotates_under_flip(town02):
    scenario = town02
    assert scenario.traffic_controls, "opendrive fixture should carry traffic controls"
    reference = copy.deepcopy(scenario)
    centroid = _centroid(scenario)
    matrix = np.asarray(affine.TRANSFORM_GROUPS["flip"]["FlipX"], dtype=np.float64)
    affine.apply_affine_transform(scenario, matrix, centroid)

    for tc, ref in zip(scenario.traffic_controls, reference.traffic_controls, strict=False):
        h = float(ref.heading)
        rotated = matrix @ np.array([np.cos(h), np.sin(h)])
        assert tc.heading == pytest.approx(float(np.arctan2(rotated[1], rotated[0])))


def test_flip_mirrors_objects(town02):
    scenario = _with_object(town02)
    centroid = _centroid(scenario)
    affine.apply_affine_transform(scenario, affine.TRANSFORM_GROUPS["flip"]["FlipX"], centroid)
    track = scenario.objects[7]
    assert np.allclose(track.position, [[2 * centroid[0] - 10.0, 20.0, 1.5]])
    assert np.isclose(np.cos(track.heading[0]), np.cos(np.pi - 0.3))
    assert np.isclose(np.sin(track.heading[0]), np.sin(np.pi - 0.3))
    assert np.allclose([track.length[0], track.width[0]], [4.0, 1.0])


@pytest.mark.parametrize("name", ["ShXP", "Sc10", "ScY10"])
def test_affine_object_box_encloses_mapped_footprint(town02, name):
    group = next(g for g in affine.TRANSFORM_GROUPS.values() if name in g)
    matrix = np.asarray(group[name], dtype=np.float64)
    scenario = _with_object(town02)
    centroid = _centroid(scenario)
    mapped = (_box_corners(10.0, 20.0, 0.3, 4.0, 1.0) - centroid) @ matrix.T + centroid
    affine.apply_affine_transform(scenario, matrix, centroid)
    track = scenario.objects[7]
    # Mapped corners lie inside the new box (in its frame), and the box is no larger than their extent along it.
    d = np.array([np.cos(track.heading[0]), np.sin(track.heading[0])])
    n = np.array([-d[1], d[0]])
    local = mapped - track.position[0, :2]
    assert np.isclose(np.ptp(local @ d), track.length[0])
    assert np.isclose(np.ptp(local @ n), track.width[0])
    assert np.allclose((local @ d).mean(), 0.0, atol=1e-9)
    assert np.allclose((local @ n).mean(), 0.0, atol=1e-9)


# Augment maps


def test_augment_maps_writes_original_plus_variants(opendrive_bins, tmp_path):
    input_dir = _input_dir(tmp_path, opendrive_bins, MAP_LOCATIONS)
    output_dir = tmp_path / "out"
    catalog = affine.select_transforms(["flip"])  # one transform -> one variant per map
    written = affine.augment_maps(input_dir, output_dir, catalog)

    assert len(written) == len(MAP_LOCATIONS) * (1 + len(catalog))
    for loc in MAP_LOCATIONS:
        assert (output_dir / f"opendrive__{loc}.bin").read_bytes() == opendrive_bins[loc]
        variant = output_dir / f"opendrive__{loc}_FlipX.bin"
        assert variant.exists()
        assert static_binary.static_binary_to_scenario(variant.read_bytes()).metadata.id == f"opendrive__{loc}_FlipX"


def test_augment_maps_empty_input_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        affine.augment_maps(empty, tmp_path / "out", affine.select_transforms(["flip"]))


def test_augment_maps_requires_explicit_overwrite(opendrive_bins, tmp_path):
    input_dir = _input_dir(tmp_path, opendrive_bins, ["Town02"])
    output_dir = tmp_path / "out"
    catalog = affine.select_transforms(["flip"])
    affine.augment_maps(input_dir, output_dir, catalog)

    with pytest.raises(FileExistsError):
        affine.augment_maps(input_dir, output_dir, catalog)

    assert len(affine.augment_maps(input_dir, output_dir, catalog, overwrite=True)) == 2


# CLI


def test_main_cli_generates_binaries(opendrive_bins, tmp_path, monkeypatch):
    input_dir = _input_dir(tmp_path, opendrive_bins, ["Town02"])
    output_dir = tmp_path / "out"

    argv = ["mapforge", "--groups", "flip", "--input_dir", str(input_dir), "--output_dir", str(output_dir)]
    monkeypatch.setattr("sys.argv", argv)
    assert affine.main() == 0
    assert sorted(p.name for p in output_dir.glob("*.bin")) == [
        "opendrive__Town02.bin",
        "opendrive__Town02_FlipX.bin",
    ]
