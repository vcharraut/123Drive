import types

from bin_factory.transforms import pipeline


def _config(**overrides):
    base = {
        "interpolate_tl": False,
        "reverse_road_edges": False,
        "invalid_agent_overlap": False,
        "no_reindex": False,
        "max_segment_length": 10.0,
        "area_threshold": 0.1,
        "min_route_valid_points": 0.0,
        "route_check_timestep": 0,
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _names(config):
    return [stage.__name__ for stage in pipeline.build_stages(config)]


def test_default_stage_order():
    assert _names(_config()) == [
        "_process_polylines",
        "_interpolate_polygons",
        "_prune_invalid_map_elements",
        "_process_traffic_controls",
        "_process_agent_routes",
        "_compute_lane_widths",
        "_compute_lane_lengths",
        "_build_lane_graph",
        "_reindex_scenario",
    ]


def test_interpolate_tl_prepends_interpolation():
    assert _names(_config(interpolate_tl=True))[0] == "_interpolate_traffic_lights"
    assert "_interpolate_traffic_lights" not in _names(_config(interpolate_tl=False))


def test_reverse_road_edges_runs_before_polylines():
    names = _names(_config(reverse_road_edges=True))
    assert names.index("_reverse_road_edges") == names.index("_process_polylines") - 1
    assert "_reverse_road_edges" not in _names(_config(reverse_road_edges=False))


def test_invalid_agent_overlap_runs_right_after_routes():
    names = _names(_config(invalid_agent_overlap=True))
    assert names.index("_invalid_agent_overlap") == names.index("_process_agent_routes") + 1
    assert names.index("_invalid_agent_overlap") < names.index("_compute_lane_lengths")


def test_no_reindex_drops_reindex_and_default_runs_it_last():
    assert "_reindex_scenario" not in _names(_config(no_reindex=True))
    assert _names(_config())[-1] == "_reindex_scenario"


def test_lane_graph_runs_immediately_after_lane_lengths():
    names = _names(_config())
    assert names.index("_build_lane_graph") == names.index("_compute_lane_lengths") + 1
