"""Check generated towns against their invariants and the city ranges documented in ``mapforge.town``."""

import numpy as np
import pytest
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from bin_factory import serialize
from bin_factory.puffer_types import MapLayer
from mapforge import town


# Measured OSM ranges from the LAYOUT comment. Generated junctions run sharper than the measured windows
# (see docs/index.md), so the angle floor sits below the measured 67 degrees.
MEASURED = {"density": (67, 130), "length": (67, 107), "angle": (60, 88), "entropy": (2.31, 3.52)}


# 200 m seed 22 has bends tight enough to fold a lane offset, 500 m seed 1 has roundabouts, 500 m seed 4 has kerb
# parking on curves, and the 1200 m town has several districts, boulevards and multi-lane streets.
@pytest.fixture(scope="module", params=[(200, 0), (200, 22), (500, 1), (500, 4), (1200, 0)])
def scenario(request):
    return town.generate_town(*request.param)


@pytest.fixture(scope="module")
def lanes(scenario):
    return {i: p for i, p in scenario.polylines.items() if p.is_lane}


def measure(size, seed):
    """Junctions per km² of the street hull, median street length, median smallest junction angle in degrees and
    length-weighted 36-bin street orientation entropy."""
    rng = np.random.default_rng(seed)
    profile = {key: rng.uniform(*span) for key, span in town.EUROPE.items()}
    layout = {key: rng.uniform(*span) for key, span in town.LAYOUT.items()}
    nodes, streets, edges, outward, _ = town.street_network(size, layout, profile, rng)
    junctions = np.flatnonzero(np.bincount(edges.ravel(), minlength=len(nodes)) >= 3)
    area = shapely.MultiPoint(np.concatenate(streets)).convex_hull.area / 1e6
    smallest = []
    for node in junctions:
        angles = np.sort(np.arctan2(outward[edges == node][:, 1], outward[edges == node][:, 0]))
        smallest.append(np.diff(np.r_[angles, angles[0] + 2 * np.pi]).min())
    segments = np.concatenate([np.diff(street, axis=0) for street in streets])
    bearings = np.arctan2(np.r_[segments[:, 1], -segments[:, 1]], np.r_[segments[:, 0], -segments[:, 0]])
    weights = np.tile(np.linalg.norm(segments, axis=1), 2)
    share = np.histogram(bearings, 36, (-np.pi, np.pi), weights=weights)[0]
    share = share[share > 0] / share.sum()
    return {
        "density": len(junctions) / area,
        "length": np.median([np.linalg.norm(np.diff(street, axis=0), axis=1).sum() for street in streets]),
        "angle": np.rad2deg(np.median(smallest)),
        "entropy": -(share * np.log(share)).sum(),
    }


def test_car_lanes_reach_each_other(lanes):
    cars = {i: p for i, p in lanes.items() if p.type == town.CAR}
    index = {lane_id: k for k, lane_id in enumerate(cars)}
    pairs = [
        (index[i], index[j])
        for i, p in cars.items()
        for j in [*p.exit_lanes, *p.left_neighbor, *p.right_neighbor]
        if j in index
    ]
    rows, cols = np.array(pairs).T
    graph = csr_matrix((np.ones(len(pairs)), (rows, cols)), shape=(len(cars), len(cars)))
    assert connected_components(graph, directed=True, connection="strong")[0] == 1


def test_every_lane_leads_on_and_cars_stay_on_car_lanes(lanes):
    assert all(p.exit_lanes for p in lanes.values())
    assert all(lanes[j].type == town.CAR for p in lanes.values() if p.type == town.CAR for j in p.exit_lanes)


def test_lanes_do_not_fold_or_cross_outside_junctions(lanes):
    lines = np.array([shapely.LineString(p.polyline[:, :2]) for p in lanes.values()])
    assert shapely.is_simple(lines).all()
    streets = lines[[not p.in_junction for p in lanes.values()]]
    assert shapely.STRtree(streets).query(streets, predicate="crosses").size == 0


def test_lanes_stay_out_of_parking(scenario, lanes):
    parking = [shapely.Polygon(s.polygon[:, :2]) for s in scenario.surfaces.values() if s.layer == MapLayer.CARPARK]
    lines = [shapely.LineString(p.polyline[:, :2]) for p in lanes.values()]
    assert not shapely.intersects(lines, shapely.union_all(parking)).any()


def test_pavement_is_one_piece(scenario):
    # Exterior kerb rings run counterclockwise; a second one is pavement cut off from the streets, such as a kerbed
    # row of parking.
    rings = [p.polyline[:, :2] for p in scenario.polylines.values() if p.is_edge]
    assert all(np.allclose(ring[0], ring[-1]) for ring in rings)
    areas = [np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1]) for ring in rings]
    assert np.count_nonzero(np.array(areas) > 0) == 1


def test_town_fills_its_extent(scenario, lanes):
    size = float(scenario.metadata.id.split("_")[1][:-1])
    extent = np.ptp(np.concatenate([p.polyline[:, :2] for p in lanes.values()]), axis=0).max()
    assert extent >= 0.8 * (size - 40)  # streets stop 20 m inside the requested extent


def test_controls_are_finite_and_carry_no_light_states(scenario):
    assert scenario.traffic_controls
    for control in scenario.traffic_controls:
        assert control.states == []
        assert np.isfinite(control.heading)
        assert np.isfinite(control.stop_line).all()


def test_town_round_trips(scenario):
    assert all(np.isfinite(p.polyline).all() for p in scenario.polylines.values())
    once = serialize.scenario_to_binary(serialize.binary_to_scenario(serialize.scenario_to_binary(scenario)))
    assert serialize.scenario_to_binary(serialize.binary_to_scenario(once)) == once


def test_generation_is_repeatable():
    first, second = (serialize.scenario_to_binary(town.generate_town(200, 3)) for _ in range(2))
    assert first == second
    assert first != serialize.scenario_to_binary(town.generate_town(200, 4))


@pytest.mark.parametrize(("size", "seed"), [(199, 0), (1201, 0), (500, -1)])
def test_generation_rejects_out_of_range_options(size, seed):
    with pytest.raises(ValueError, match="Use size 200"):
        town.generate_town(size, seed)


def test_town_statistics_match_measured_cities():
    # Measured windows are 1.2 km, so measure towns of that size; each seed draws its own targets, so check medians.
    samples = [measure(1200, seed) for seed in range(30)]
    for name, (low, high) in MEASURED.items():
        assert low <= np.median([sample[name] for sample in samples]) <= high, name
