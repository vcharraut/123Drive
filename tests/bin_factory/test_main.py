import dataclasses
import pathlib
import tomllib
from types import SimpleNamespace

import pytest

from bin_factory import loader, main
from bin_factory.preparation import ConversionSettings


PRESETS = tomllib.loads((pathlib.Path(main.__file__).parent / "presets.toml").read_text())


def test_output_path_always_uses_scene_uuid(tmp_path):
    scene = SimpleNamespace(dataset="nuplan", scene_uuid="uuid-1", log_name="log-7", location="boston")

    assert main._build_output_path(scene, tmp_path, "scene_uuid").name == "nuplan__uuid-1.bin"
    assert main._build_output_path(scene, tmp_path, "log_name").name == "nuplan__log-7__uuid-1.bin"


def test_map_output_path_uses_location(tmp_path):
    map_api = SimpleNamespace(dataset="opendrive", location="Town02")

    assert main._build_output_path(map_api, tmp_path, "scene_uuid").name == "opendrive__Town02.bin"


def test_scenario_id_uses_location_for_maps_and_rejects_empty_ids():
    assert loader.scenario_id(SimpleNamespace(dataset="opendrive", location="Town02"), "log_name") == "Town02"
    with pytest.raises(ValueError, match="missing location for dataset wod-motion"):
        loader.scenario_id(SimpleNamespace(dataset="wod-motion", scene_uuid="uuid-1", location=None), "location")


def test_converter_has_no_lane_graph_policy_flags():
    flags = {option for action in main.build_parser()._actions for option in action.option_strings}

    assert "--max_lane_graph_lanes" not in flags
    assert "--no_lane_graph" not in flags


def test_cli_defaults_are_the_settings_defaults():
    _args, settings = main.parse_args(["--py123d_path", "data"])

    assert settings == ConversionSettings()


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_resolves_to_its_values(name):
    args, settings = main.parse_args(["--preset", name, "--py123d_path", "data"])

    setting_names = {field.name for field in dataclasses.fields(ConversionSettings)}
    for key, value in PRESETS[name].items():
        assert getattr(settings if key in setting_names else args, key) == value


def test_explicit_flags_override_preset_values():
    args, settings = main.parse_args(
        ["--preset", "nuplan", "--duration_s", "5", "--static_objects", "all", "--py123d_path", "data"]
    )

    assert args.duration_s == 5.0
    assert settings.static_objects == "all"
    assert settings.reverse_road_edges  # still from the preset


def test_opendrive_dataset_converts_maps_only():
    args, _settings = main.parse_args(["--datasets", "opendrive", "--py123d_path", "data"])

    assert args.map_only
