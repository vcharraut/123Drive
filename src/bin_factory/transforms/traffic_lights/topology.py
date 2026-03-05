# Based on: Yan, X., Liang, E., Wang, J., Zhu, H., & Liu, H. X. (2025).
# "Improving Traffic Signal Data Quality for the Waymo Open Motion Dataset."
# arXiv:2506.07150v1. University of Michigan.
# https://github.com/michigan-traffic-lab/WOMD-Traffic-Signal-Data-Improvement

from py123d.datatypes.map_objects.map_layer_types import LaneType

from src.bin_factory import types
from src.bin_factory.transforms.traffic_lights.utils import (
    DetailedTLS,
    UnionFind,
    distance_between_points,
    polyline_length,
    real_neighbor_type,
    two_lines_parallel,
)


LANE_SHORT_THRESHOLD = 2
POINT_CLOSE_THRESHOLD = 5
LINE_PARALLEL_THRESHOLD = 15


def scenario_state_to_detailed(state):
    mapping = {
        types.TRAFFIC_LIGHT_UNKNOWN: DetailedTLS.UNKNOWN,
        types.TRAFFIC_LIGHT_ARROW_RED: DetailedTLS.ARROW_STOP,
        types.TRAFFIC_LIGHT_ARROW_YELLOW: DetailedTLS.ARROW_CAUTION,
        types.TRAFFIC_LIGHT_ARROW_GREEN: DetailedTLS.ARROW_GO,
        types.TRAFFIC_LIGHT_RED: DetailedTLS.STOP,
        types.TRAFFIC_LIGHT_YELLOW: DetailedTLS.CAUTION,
        types.TRAFFIC_LIGHT_GREEN: DetailedTLS.GO,
        types.TRAFFIC_LIGHT_FLASHING_RED: DetailedTLS.FLASHING_STOP,
        types.TRAFFIC_LIGHT_FLASHING_YELLOW: DetailedTLS.FLASHING_CAUTION,
    }
    return mapping.get(state, DetailedTLS.UNKNOWN)


def build_lane_topology(scenario):
    lanes = _load_lanes(scenario)
    _clean_lanes(lanes)
    signalized = _find_special_intersections(lanes, "signalized_intersection")
    return lanes, signalized


def _load_lanes(scenario):
    lanes = {}
    length = scenario["scenario_length"]

    for _id, feature in scenario["map"].items():
        _id = int(_id)
        if feature["type"] in [LaneType.SURFACE_STREET, LaneType.FREEWAY]:
            lanes[_id] = {
                "polyline": feature["polyline"],
                "entry_lanes": list(feature["entry_lanes"]),
                "exit_lanes": list(feature["exit_lanes"]),
                "left_neighbors": [int(n) for n in feature["left_neighbor"] if n is not None],
                "right_neighbors": [int(n) for n in feature["right_neighbor"] if n is not None],
                "diverge_lanes": set(),
                "merge_lanes": set(),
                "needs_stop": False,
                "record_tls": [DetailedTLS.ABSENT] * length,
            }

    for _id, feature in scenario["map"].items():
        if feature["type"] == types.STOP_SIGN:
            for lane_id in feature["lanes"]:
                if lane_id in lanes:
                    lanes[lane_id]["needs_stop"] = True

    for id_, dynamic_state in scenario["traffic_lights"].items():
        lane = dynamic_state["controlled_lane"]
        lane_type = scenario["map"][lane]["type"]
        if lane_type == LaneType.BIKE_LANE:
            continue
        for i, state in enumerate(dynamic_state["states"]):
            lanes[lane]["record_tls"][i] = scenario_state_to_detailed(state)

    return lanes


def _clean_lanes(lanes):
    # I. delete short fringe features
    to_delete = [
        f_id
        for f_id, lane in lanes.items()
        if ((not lane["entry_lanes"]) or (not lane["exit_lanes"]))
        and polyline_length(lane["polyline"]) < LANE_SHORT_THRESHOLD
        or len(lane["polyline"]) <= 1
    ]
    for f_id in to_delete:
        del lanes[f_id]

    # II. clean entry/exit lanes
    for lane in lanes.values():
        _clean_entryexit(lane, lanes)

    # III. clean neighbors
    for lane in lanes.values():
        _clean_neighbors(lane, lanes)

    # ensure neighborhood pairs are consistent
    for f_id, lane in lanes.items():
        lane["left_neighbors"] = [nb for nb in lane["left_neighbors"] if f_id in lanes[nb]["right_neighbors"]]
        lane["right_neighbors"] = [nb for nb in lane["right_neighbors"] if f_id in lanes[nb]["left_neighbors"]]

    # IV. add diverge/merge info
    for lane in lanes.values():
        for i in lane["entry_lanes"]:
            for j in lane["entry_lanes"]:
                if i != j:
                    lanes[i]["merge_lanes"].add(j)
        for i in lane["exit_lanes"]:
            for j in lane["exit_lanes"]:
                if i != j:
                    lanes[i]["diverge_lanes"].add(j)

    # ensure diverge/merge pairs consistent
    for f_id, lane in lanes.items():
        lane["diverge_lanes"] = {id for id in lane["diverge_lanes"] if f_id in lanes[id]["diverge_lanes"]}
        lane["merge_lanes"] = {id for id in lane["merge_lanes"] if f_id in lanes[id]["merge_lanes"]}


def _clean_entryexit(lane, lanes):
    lane["entry_lanes"] = [
        id
        for id in lane["entry_lanes"]
        if id in lanes
        and distance_between_points(lane["polyline"][0], lanes[id]["polyline"][-1]) < POINT_CLOSE_THRESHOLD
    ]
    lane["exit_lanes"] = [
        id
        for id in lane["exit_lanes"]
        if id in lanes
        and distance_between_points(lane["polyline"][-1], lanes[id]["polyline"][0]) < POINT_CLOSE_THRESHOLD
    ]


def _clean_neighbors(lane, lanes):
    def _clean(neighbor_ids):
        new_ids = []
        for nb_id in neighbor_ids:
            if nb_id not in lanes:
                continue
            nbr_type = _neighbor_type(lane["polyline"], lanes[nb_id]["polyline"])
            if nbr_type in ("real", "bifurcated-parallel", "merged-parallel"):
                new_ids.append(nb_id)
            if nbr_type in ("bifurcated", "bifurcated-parallel"):
                lane["diverge_lanes"].add(nb_id)
            if nbr_type in ("merged", "merged-parallel"):
                lane["merge_lanes"].add(nb_id)
        return new_ids

    lane["left_neighbors"] = _clean(lane["left_neighbors"])
    lane["right_neighbors"] = _clean(lane["right_neighbors"])


def _neighbor_type(poly1, poly2):
    line1 = [poly1[0][:2], poly1[-1][:2]]
    line2 = [poly2[0][:2], poly2[-1][:2]]
    parallel = two_lines_parallel(line1, line2, LINE_PARALLEL_THRESHOLD)

    start2start = distance_between_points(poly1[0], poly2[0])
    end2end = distance_between_points(poly1[-1], poly2[-1])

    def level(d):
        if d < 1:
            return "low"
        elif d < 5:
            return "mid"
        return "high"

    levels = [level(start2start), level(end2end)]
    if parallel:
        if "low" in levels:
            return "bifurcated-parallel" if levels[0] == "low" else "merged-parallel"
        return "other"
    else:
        if levels[0] in ("low", "mid"):
            return "bifurcated"
        elif levels[1] in ("low", "mid"):
            return "merged"
        return "other"


def _find_special_intersections(lanes, type):
    def _is_connection_group(elements):
        if any(e not in lanes for e in elements):
            return False
        if len(elements) <= 1:
            return False
        return any(lanes[id]["diverge_lanes"] or lanes[id]["merge_lanes"] for id in elements)

    if not lanes:
        return []
    uf = UnionFind(max(lanes.keys()) + 1)

    for lc_id, lane in lanes.items():
        for nb_id in lane["left_neighbors"] + lane["right_neighbors"]:
            if (
                real_neighbor_type(
                    lane["polyline"],
                    lanes[nb_id]["polyline"],
                    POINT_CLOSE_THRESHOLD=POINT_CLOSE_THRESHOLD,
                )
                == "complete"
            ):
                uf.union(lc_id, nb_id)
        for d_id in lane["diverge_lanes"]:
            uf.union(lc_id, d_id)
        for m_id in lane["merge_lanes"]:
            uf.union(lc_id, m_id)

    groups = [e for e in uf.form_groups() if _is_connection_group(e)]

    # round 2: external lanes bridging groups
    internal = {id for group in groups for id in group}
    for id in set(lanes.keys()) - internal:
        if lanes[id]["exit_lanes"] and lanes[id]["entry_lanes"]:
            exit_id = lanes[id]["exit_lanes"][0]
            entry_id = lanes[id]["entry_lanes"][0]
            if uf.find(exit_id) == uf.find(entry_id):
                uf.union(id, exit_id)
                uf.union(id, entry_id)

    groups = [e for e in uf.form_groups() if _is_connection_group(e)]

    if type == "signalized_intersection":
        return [g for g in groups if _signalized_criteria(g, lanes)]
    return [g for g in groups if _stop_criteria(g, lanes)]


def _signalized_criteria(group, lanes):
    return (
        len(group) >= 4
        and any(tl != DetailedTLS.ABSENT for id in group for tl in lanes[id]["record_tls"])
        and not any(
            len(lanes[id]["entry_lanes"]) > 1 and not any(eid in group for eid in lanes[id]["entry_lanes"])
            for id in group
        )
    )


def _stop_criteria(group, lanes):
    return (
        len(group) >= 3
        and any(lanes[id]["needs_stop"] for id in group)
        and not any(len(lanes[id]["entry_lanes"]) > 1 for id in group)
    )
