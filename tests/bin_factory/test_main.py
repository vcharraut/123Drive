from types import SimpleNamespace

from bin_factory import main


def test_output_path_always_uses_scene_uuid(tmp_path):
    scene = SimpleNamespace(dataset="nuplan", scene_uuid="uuid-1", log_name="log-7", location="boston")

    assert main._build_output_path(scene, tmp_path, "scene_uuid").name == "nuplan__uuid-1.bin"
    assert main._build_output_path(scene, tmp_path, "log_name").name == "nuplan__log-7__uuid-1.bin"


def test_map_output_path_uses_location(tmp_path):
    map_api = SimpleNamespace(dataset="opendrive", location="Town02")

    assert main._build_output_path(map_api, tmp_path, "scene_uuid").name == "opendrive__Town02.bin"


def test_converter_has_no_lane_graph_policy_flags():
    flags = {option for action in main.build_parser()._actions for option in action.option_strings}

    assert "--max_lane_graph_lanes" not in flags
    assert "--no_lane_graph" not in flags
