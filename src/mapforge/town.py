import argparse
from dataclasses import replace
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import split, substring

from bin_factory.conversion.geometry import interpolate_all_polygons, process_polylines
from bin_factory.conversion.traffic_controls import group_facing_phases
from bin_factory.puffer_types import LaneType, MapLayer, RoadEdgeType, RoadLineType, TCType
from bin_factory.schema import PufferScenario, RoadPolyline, RoadSurface, ScenarioMetadata, TrafficControl
from mapforge.static_binary import write_static_scenario


# Pooled ranges over OSM drive networks in 1.2 km windows of Paris (Opéra, Oberkampf), Lyon (Brotteaux,
# Guillotière), Munich (Maxvorstadt, Altstadt) and Rome (Campo Marzio, Monti), fetched 2026-10-30. Lengths count
# each street once. Each town draws every value independently, so seeds cover the range these cities span
# rather than reproducing one of them.
EUROPE = {
    "four": (0.27, 0.58),  # junctions with four or more arms, share of all junctions
    "five": (0.01, 0.07),  # junctions with five or more arms
    "major": (0.27, 0.39),  # primary/secondary/tertiary share of street length
    "oneway": (0.39, 0.93),  # one-way share of street length
    "multi_lane": (0.09, 0.3),  # share of street length with two or more lanes per direction
    "fifty": (0.0, 0.54),  # 50 km/h share of signed street length
    "twenty": (0.0, 0.08),  # 20 km/h share; the rest is 30 km/h
    "signals": (0.09, 0.39),  # signalized share of junctions
    "roundabouts": (0.0, 0.7),  # per km²
    "bus": (0.0, 0.56),  # share of major street length with bus lanes
    "bike": (0.0, 0.48),  # share of street length with bike lanes or tracks
    "contraflow": (0.0, 0.76),  # chance a one-way street's bike lane runs against traffic; OSM: contraflow share
}
# Layout controls, tuned so generated towns fall in the measured ranges of junction density (67-130 per km²),
# street length between junctions (median 67-107 m), smallest junction angle (median 67-88 degrees) and
# street orientation entropy (2.31-3.52).
LAYOUT = {
    "spacing": (82.0, 105.0),  # street lattice spacing in metres
    "jitter": (0.05, 0.2),  # organic districts: node jitter as a fraction of the spacing
    "grid": (0.3, 1.0),  # chance that a district keeps straight streets
    "district": (500.0, 1500.0),  # district size in metres
    "aspect": (1.0, 2.0),  # block aspect ratio
    "variation": (0.0, 0.3),  # district spacing varies by up to this fraction
    "uneven": (0.3, 0.5),  # gaps between parallel streets vary by up to this fraction
    "acute": (25.0, 40.0),  # smallest junction angle in degrees, except where boulevards cross
    "plazas": (0.0, 1.2),  # plazas with radiating boulevards per km²
    "ring": (0.0, 0.5),  # chance of a ring boulevard
}
CAR, BUS, BIKE = LaneType.SURFACE_STREET, LaneType.BUS_LANE, LaneType.BIKE_LANE


def arm_angles(edges: np.ndarray, heading: np.ndarray, live: np.ndarray, node: int) -> tuple[np.ndarray, np.ndarray]:
    arms = np.flatnonzero(live & (edges == node).any(axis=1))
    return arms, np.where(edges[arms, 0] == node, heading[arms], heading[arms] + np.pi)


def street_graph(
    size: float, layout: dict[str, float], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Polygon, float]:
    radius = size / 2 - 20
    spacing = min(layout["spacing"], size / 5)
    theta = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    reach = radius * (1.13 + 0.13 * np.sin(3 * theta + rng.uniform(0, 2 * np.pi)))
    outline = Polygon(np.column_stack((np.cos(theta), np.sin(theta))) * reach[:, None])
    outline = outline.intersection(shapely.box(-radius, -radius, radius, radius))
    # Boulevards radiate from plazas and may follow a ring; their vertices become fixed street nodes.
    boulevards = []
    for plaza in rng.uniform(-0.5, 0.5, (rng.poisson(layout["plazas"] * (size / 1000) ** 2), 2)) * radius:
        count = int(rng.integers(4, 8))
        angles = (np.arange(count) + rng.uniform(-0.2, 0.2, count)) * 2 * np.pi / count + rng.uniform(0, 2 * np.pi)
        ends = plaza + rng.uniform(0.6, 2.0, (count, 1)) * radius * np.column_stack((np.cos(angles), np.sin(angles)))
        for end in ends:
            line, direction = LineString([plaza, end]), (end - plaza) / np.linalg.norm(end - plaza)
            # Boulevards meet at a clear angle or stay a block apart, so their vertices never merge sideways.
            clear = [
                direction @ d < np.cos(np.pi / 6)
                if np.allclose(other.coords[0], plaza)
                else abs(direction @ d) < np.cos(np.pi / 6)
                if line.intersects(other)
                else line.distance(other) > spacing
                for other, d in boulevards
            ]
            if all(clear):
                boulevards.append((line, direction))
    if rng.random() < layout["ring"]:
        center = rng.uniform(-0.2, 0.2, 2) * radius
        ring = Point(center).buffer(rng.uniform(0.35, 0.6) * radius, quad_segs=32).exterior
        crossings = [shapely.get_coordinates(ring.intersection(other)) - center for other, _ in boulevards]
        # The ring crosses boulevards within 60 degrees of square, away from tangents.
        steep = [
            np.all(np.abs(c @ d) >= 0.5 * np.linalg.norm(c, axis=1))
            for c, (_, d) in zip(crossings, boulevards, strict=True)
        ]
        near = [len(c) or ring.distance(other) > spacing for c, (other, _) in zip(crossings, boulevards, strict=True)]
        boulevards += [(ring, None)] if all(steep) and all(near) else []
    boulevards = [line for line, _ in boulevards]
    parts = shapely.get_parts(shapely.union_all(boulevards).intersection(outline)) if boulevards else []
    parts = [part for part in parts if part.geom_type == "LineString" and part.length > 1]
    vertices, chain = [np.zeros((0, 2))], []
    for part in parts:
        count = max(1, round(part.length / spacing))
        start = sum(len(v) for v in vertices)
        spots = shapely.line_interpolate_point(part, np.linspace(0, part.length, count + 1))
        vertices.append(shapely.get_coordinates(spots))
        chain += [(start + i, start + i + 1) for i in range(count)]
    vertices, inverse = np.unique(np.round(np.concatenate(vertices), 3), axis=0, return_inverse=True)
    # Merge boulevard vertices closer than a third of the spacing, such as near-coincident crossings.
    near = np.array(sorted(cKDTree(vertices).query_pairs(spacing / 3)), dtype=int).reshape(-1, 2)
    pairs = csr_matrix((np.ones(len(near)), (near[:, 0], near[:, 1])), shape=(len(vertices), len(vertices)))
    group = connected_components(pairs, directed=False)[1]
    vertices = vertices[np.unique(group, return_index=True)[1]]
    forced = np.sort(group[inverse.reshape(-1)][np.array(chain, dtype=int).reshape(-1, 2)], axis=1)
    forced = np.unique(forced[forced[:, 0] != forced[:, 1]], axis=0)
    # Each district is a rotated street lattice; regular districts keep it straight, the others jitter it.
    seeds = rng.uniform(-radius, radius, (max(1, round((size / layout["district"]) ** 2)), 2))
    grid = rng.random(len(seeds)) < layout["grid"]
    turn = rng.uniform(0, np.pi / 2, len(seeds))
    scale = rng.uniform(1 - layout["variation"], 1 + layout["variation"], len(seeds))
    steps = spacing * np.array([np.sqrt(layout["aspect"]), 1 / np.sqrt(layout["aspect"])])
    extent = int(np.ceil(1.5 * radius / steps.min() / scale.min() / (1 - layout["uneven"])))
    index = np.arange(-extent, extent + 1, dtype=int)
    lattice = np.stack(np.meshgrid(index, index), axis=-1).reshape(-1, 2)
    fabric = []
    for district in range(len(seeds)):
        c, s = np.cos(turn[district]), np.sin(turn[district])
        jitter = (0.05 if grid[district] else layout["jitter"]) * spacing
        # Rows and columns sit at uneven gaps, so parallel streets are closer or farther apart.
        gaps = np.cumsum(rng.uniform(1 - layout["uneven"], 1 + layout["uneven"], (len(index), 2)), axis=0)
        gaps -= gaps[len(index) // 2] + rng.random(2)
        grid_xy = np.column_stack((gaps[lattice[:, 0] + extent, 0], gaps[lattice[:, 1] + extent, 1]))
        local = (grid_xy * steps * scale[district]) @ np.array([[c, s], [-s, c]])
        local += rng.uniform(-jitter, jitter, local.shape)
        mine = np.argmin(np.linalg.norm(local[:, None] - seeds[None], axis=2), axis=1) == district
        fabric.append(np.column_stack((local[mine], np.full(mine.sum(), district), lattice[mine])))
    fabric = np.concatenate(fabric)
    inside = shapely.contains_xy(outline, fabric[:, 0], fabric[:, 1])
    if parts:
        inside &= shapely.distance(shapely.points(fabric[:, :2]), shapely.multilinestrings(parts)) > 0.3 * spacing
    points = np.vstack((vertices, fabric[inside, :2]))
    owner = np.r_[np.full(len(vertices), -1), fabric[inside, 2]].astype(int)
    cell = np.vstack((np.zeros((len(vertices), 2)), fabric[inside, 3:]))
    close = np.array(sorted(cKDTree(points).query_pairs(0.15 * spacing)), dtype=int).reshape(-1, 2)
    kept = np.setdiff1d(np.arange(len(points)), close[close[:, 1] >= len(vertices), 1])
    points, owner, cell = points[kept], owner[kept], cell[kept]
    # Uneven gaps can stop the last street well inside the outline; stretch the town to fill its extent.
    middle = (points.max(axis=0) + points.min(axis=0)) / 2
    stretch = np.min(np.ptp(shapely.get_coordinates(outline), axis=0) / np.ptp(points, axis=0))
    points = middle + (points - middle) * stretch
    # Triangulate, then keep boulevard segments and drop the triangulation edges they cross.
    simplices = Delaunay(points).simplices
    edges = np.unique(np.sort(simplices[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0)
    # Drop slivers across concave parts of the outline, but not the longest lattice streets.
    longest = steps.max() * scale.max() * (1 + layout["uneven"])
    keep = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1) < 1.5 * longest
    middle = points[edges].mean(axis=1)
    keep &= shapely.contains_xy(outline.buffer(5), middle[:, 0], middle[:, 1])
    if len(forced):
        segments = shapely.linestrings(points[edges])
        keep[shapely.STRtree(segments).query(shapely.linestrings(points[forced]), predicate="crosses")[1]] = False
    edges = np.unique(np.vstack((edges[keep], forced)), axis=0)
    protected = np.isin(edges[:, 0] * len(points) + edges[:, 1], forced[:, 0] * len(points) + forced[:, 1])
    # Peel dead ends, and two-arm nodes folding back sharper than 45 degrees, which would become hairpins.
    while True:
        degree = np.bincount(edges.ravel(), minlength=len(points))
        keep = (degree[edges] >= 2).all(axis=1)
        for node in np.flatnonzero(degree == 2):
            arms = np.flatnonzero((edges == node).any(axis=1))
            v = points[edges[arms].sum(axis=1) - node] - points[node]
            keep[arms] &= v[0] @ v[1] < np.cos(np.deg2rad(45)) * np.linalg.norm(v[0]) * np.linalg.norm(v[1])
        if keep.all():
            break
        edges, protected = edges[keep], protected[keep]
    graph = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(points), len(points)))
    labels = connected_components(graph, directed=False)[1]
    keep = labels[edges[:, 0]] == np.bincount(labels[edges[:, 0]]).argmax()
    edges, protected = edges[keep], protected[keep]
    return points, edges, protected, owner, cell, outline, spacing


def street_network(
    size: float, layout: dict[str, float], profile: dict[str, float], rng: np.random.Generator
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    points, edges, protected, owner, cell, outline, spacing = street_graph(size, layout, rng)
    graph = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(points), len(points)))
    # Prune acute angles first, then triangulation edges that are not lattice streets, then edges until the
    # share of four- and five-arm junctions matches the profile. No street becomes a dead end.
    vector = points[edges[:, 1]] - points[edges[:, 0]]
    heading = np.arctan2(vector[:, 1], vector[:, 0])
    length = np.linalg.norm(vector, axis=1)
    district = owner[edges[:, 0]]
    step = np.abs(cell[edges[:, 0]] - cell[edges[:, 1]]).sum(axis=1)
    offaxis = (district == owner[edges[:, 1]]) & (district >= 0) & (step != 1)
    keep = np.ones(len(edges), dtype=bool)
    gap = np.full(len(edges), np.pi)
    for edge, (a, b) in enumerate(edges):
        for node, angle in ((a, heading[edge]), (b, heading[edge] + np.pi)):
            arms, angles = arm_angles(edges, heading, keep, node)
            difference = np.abs((angles[arms != edge] - angle + np.pi) % (2 * np.pi) - np.pi)
            gap[edge] = min(gap[edge], difference.min(initial=np.pi))
    acute = np.deg2rad(layout["acute"])
    tier = np.where(gap < acute, 0, np.where(offaxis, 1, 2))
    # Arm counts are matched away from the town edge, like the measured city windows.
    interior = shapely.distance(shapely.points(points), outline.exterior) > spacing
    interior |= ~interior.any()
    components = connected_components(graph, directed=False)[0]
    # Within a tier, sharper edges go first so surviving junctions lean towards right angles.
    for edge in np.lexsort((gap + np.where(tier == 0, 0, rng.uniform(0, 0.6, len(edges))), tier)):
        a, b = edges[edge]
        degree = np.bincount(edges[keep].ravel(), minlength=len(points))
        if protected[edge] or min(degree[a], degree[b]) < 3:
            continue
        if tier[edge] == 2:
            junctions = np.count_nonzero(interior & (degree >= 3))
            five = np.count_nonzero(interior & (degree >= 5)) / junctions > profile["five"]
            if not five and np.count_nonzero(interior & (degree >= 4)) / junctions <= profile["four"]:
                break
            if max(degree[a] * interior[a], degree[b] * interior[b]) < (5 if five else 4):
                continue
        # Re-check acute edges and refuse removals that leave a hairpin bend.
        bends = []
        for node, angle in ((a, heading[edge]), (b, heading[edge] + np.pi)):
            arms, angles = arm_angles(edges, heading, keep, node)
            others = angles[arms != edge]
            bends.append(np.abs((others - angle + np.pi) % (2 * np.pi) - np.pi).min())
            # A two-arm node becomes a bend: no hairpins, and sharp bends only between long legs.
            bend = np.abs((others[0] - others[1] + np.pi) % (2 * np.pi) - np.pi) if degree[node] == 3 else np.pi
            if bend < np.deg2rad(100) or (bend < np.deg2rad(150) and length[arms[arms != edge]].min() < 40):
                bends[-1] = -1.0
        if min(bends) < 0 or (tier[edge] == 0 and min(bends) >= acute):
            continue
        keep[edge] = False
        pairs = edges[keep]
        graph = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points)))
        if connected_components(graph, directed=False)[0] != components:
            keep[edge] = True
    edges, protected = edges[keep], protected[keep]
    # Chains through two-arm nodes become one street with bends.
    degree = np.bincount(edges.ravel(), minlength=len(points))
    junction = degree > 2
    incident = [[] for _ in points]
    for edge, (a, b) in enumerate(edges):
        incident[a].append(edge)
        incident[b].append(edge)
    seen = np.zeros(len(edges), dtype=bool)
    paths, corridor = [], []
    for first in np.flatnonzero(junction[edges].any(axis=1)):
        if seen[first]:
            continue
        node = edges[first, 0] if junction[edges[first, 0]] else edges[first, 1]
        edge, path, members = first, [node], []
        while True:
            seen[edge] = True
            members.append(edge)
            node = edges[edge].sum() - node
            path.append(node)
            if junction[node]:
                break
            edge = sum(incident[node]) - edge  # the other edge of a two-arm node
        # A loop back to its own junction keeps its middle vertex as a junction.
        halves = [path[: len(path) // 2 + 1], path[len(path) // 2 :]] if path[0] == path[-1] else [path]
        paths += halves
        corridor += [protected[members].all()] * len(halves)
    ends = np.array([(path[0], path[-1]) for path in paths])
    used, ends = np.unique(ends, return_inverse=True)
    raw, smooth = [points[path] for path in paths], []
    for line in raw:
        for _ in range(2):  # Chaikin corner cutting rounds bends and keeps the junction ends
            cut = np.column_stack((0.75 * line[:-1] + 0.25 * line[1:], 0.25 * line[:-1] + 0.75 * line[1:]))
            line = np.vstack((line[0], cut.reshape(-1, 2)[1:-1], line[-1]))
        smooth.append(line)
    # A rounded bend that cuts across a nearby street keeps its sharp corner instead.
    lines = [LineString(line) for line in raw + smooth]
    street, other = shapely.STRtree(lines).query(lines[len(raw) :], predicate="crosses")
    sharp = set(street[other % len(raw) != street])
    streets = [raw[i] if i in sharp else smooth[i] for i in range(len(raw))]
    outward = np.stack([np.array([s[1] - s[0], s[-2] - s[-1]]) for s in streets])
    outward /= np.linalg.norm(outward, axis=2)[:, :, None]
    return points[used], streets, ends.reshape(-1, 2), outward, np.array(corridor)


def street_plan(
    nodes: np.ndarray,
    streets: list[np.ndarray],
    edges: np.ndarray,
    outward: np.ndarray,
    corridor: np.ndarray,
    roundabouts: dict[int, float],
    profile: dict[str, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(edges)
    lines = [LineString(s) for s in streets]
    lengths = shapely.length(lines)
    # The town edge is where the generated map stops, not a ring road.
    blocks = shapely.union_all(shapely.get_parts(shapely.polygonize(lines)))
    outer = ~shapely.contains(blocks, shapely.line_interpolate_point(lines, 0.5, normalized=True))
    # A street continues through a junction when its arms deviate by less than 35 degrees.
    links = []
    for node in np.unique(edges):
        arms = np.argwhere(edges == node)
        vectors = outward[arms[:, 0], arms[:, 1]]
        pairs = sorted(combinations(range(len(arms)), 2), key=lambda p: vectors[p[0]] @ vectors[p[1]])
        used = set()
        for i, j in pairs:
            if vectors[i] @ vectors[j] > -np.cos(np.deg2rad(35)):
                break
            if i not in used and j not in used:
                used |= {i, j}
                links.append((*arms[i], *arms[j]))
    links = np.array(links, dtype=int).reshape(-1, 4)
    # Orient every street of a stroke the same way: equal end indices at a link mean opposite directions.
    relation = {(a, b): 1 if ia != ib else -1 for a, ia, b, ib in links} | {
        (b, a): 1 if ia != ib else -1 for a, ia, b, ib in links
    }
    graph = csr_matrix((np.ones(len(links)), (links[:, 0], links[:, 2])), shape=(count, count))
    stroke = connected_components(graph, directed=False)[1]
    sign = np.ones(count)
    for root in np.unique(stroke, return_index=True)[1]:
        order, parent = breadth_first_order(graph, root, directed=False)
        for street in order[1:]:
            sign[street] = sign[parent[street]] * relation[(parent[street], street)]
    total = lengths.sum()
    stroke_length = np.bincount(stroke, weights=lengths)
    # Boulevards are major first, then inner strokes reaching farthest, so winding lanes and the edge stay minor.
    reach = [np.linalg.norm(np.ptp(nodes[edges[stroke == s]].reshape(-1, 2), axis=0)) for s in range(stroke.max() + 1)]
    rim = np.bincount(stroke, weights=outer) == np.bincount(stroke)
    ranking = np.lexsort((-np.array(reach), rim, -np.bincount(stroke, weights=corridor)))
    cumulative = np.cumsum(stroke_length[ranking])
    major = np.isin(stroke, ranking[: np.searchsorted(cumulative, profile["major"] * total) + 1])
    # One-way strokes must keep every directed street reachable without a U-turn.
    start, end = np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]
    street = np.r_[np.arange(count), np.arange(count)]
    rows, cols = [], []
    for node in np.unique(edges):
        into, out = np.meshgrid(np.flatnonzero(end == node), np.flatnonzero(start == node), indexing="ij")
        turn = street[into] != street[out]
        rows.append(into[turn])
        cols.append(out[turn])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    transitions = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(2 * count, 2 * count))
    allowed = np.ones(2 * count, dtype=bool)
    oneway = 0.0
    for candidate in rng.permutation(stroke.max() + 1):
        members = np.flatnonzero(stroke == candidate)
        if oneway + lengths[members].sum() > profile["oneway"] * total * 1.1:
            continue
        closed = np.where(sign[members] * rng.choice([-1, 1]) > 0, members + count, members)
        allowed[closed] = False
        if connected_components(transitions[allowed][:, allowed], directed=True, connection="strong")[0] != 1:
            allowed[closed] = True
            continue
        oneway += lengths[members].sum()
    directions = np.column_stack((allowed[:count], allowed[count:]))
    lanes = directions.astype(int)
    multi = 0.0
    for candidate in rng.permutation(np.unique(stroke[major])):
        if multi >= profile["multi_lane"] * total:
            break
        members = np.flatnonzero(stroke == candidate)
        lanes[members] *= np.where(directions[members].all(axis=1, keepdims=True), 2, int(rng.integers(2, 4)))
        multi += lengths[members].sum()
    # Roundabouts have a single-lane ring, so their arms narrow to one lane per direction.
    arm = np.isin(edges, list(roundabouts)).any(axis=1)
    lanes[arm] = np.minimum(lanes[arm], 1)
    # 50 km/h goes to major streets first, 20 km/h shared zones to minor ones; the rest is 30 km/h.
    speed = np.full(count, 30.0)
    ranked = np.r_[rng.permutation(np.unique(stroke[major])), rng.permutation(np.unique(stroke[~major]))]
    speed[np.isin(stroke, ranked[::-1][np.cumsum(stroke_length[ranked[::-1]]) <= profile["twenty"] * total])] = 20.0
    speed[np.isin(stroke, ranked[np.cumsum(stroke_length[ranked]) <= profile["fifty"] * total])] = 50.0
    # Bus and bike lanes stop short of roundabouts; bikes share the bus lanes.
    typed = np.zeros((count, 2), dtype=int)
    bus = 0.0
    for candidate in rng.permutation(np.unique(stroke[major])):
        if bus >= profile["bus"] * lengths[major].sum():
            break
        members = np.flatnonzero((stroke == candidate) & ~arm)
        typed[members] = np.where(directions[members], BUS, 0)
        bus += lengths[members].sum()
    bike = 0.0
    for candidate in rng.permutation(stroke.max() + 1):
        if bike >= profile["bike"] * total:
            break
        members = np.flatnonzero((stroke == candidate) & ~arm & (typed == 0).all(axis=1))
        contraflow = (rng.random() < profile["contraflow"]) & ~directions[members].all(axis=1)
        typed[members] = np.where(directions[members] ^ contraflow[:, None], BIKE, 0)
        bike += lengths[members].sum()
    # Keep bus and bike lanes followable: each end needs another street to carry on, in its own lane or with cars.
    changed = True
    while changed:
        changed = False
        for edge, direction in np.argwhere(typed > 0):
            carry = (lanes > 0) | (typed == typed[edge, direction])
            carry[edge] = False
            onward = carry & (edges == edges[edge, 1 - direction])
            inward = carry & (edges[:, ::-1] == edges[edge, direction])
            if not onward.any() or not inward.any():
                typed[edge, direction] = 0
                changed = True
    return stroke, major, lanes, speed, typed


def select_roundabouts(
    nodes: np.ndarray, edges: np.ndarray, outward: np.ndarray, lengths: np.ndarray, count: int, rng: np.random.Generator
) -> dict[int, float]:
    degree = np.bincount(edges.ravel(), minlength=len(nodes))
    candidates = np.flatnonzero(degree >= 3)
    chosen = {}
    # Plazas with many arms come first.
    for node in candidates[np.lexsort((rng.random(len(candidates)), -degree[candidates]))]:
        if len(chosen) >= count:
            break
        arms = np.argwhere(edges == node)
        vectors = outward[arms[:, 0], arms[:, 1]]
        angles = np.sort(np.arctan2(vectors[:, 1], vectors[:, 0]))
        shortest = lengths[arms[:, 0]].min()
        if np.diff(np.r_[angles, angles[0] + 2 * np.pi]).min() < np.deg2rad(47) or shortest < 42:
            continue
        if any(np.linalg.norm(nodes[node] - nodes[other]) < 75 for other in chosen):
            continue
        chosen[int(node)] = min(6 + 1.5 * degree[node], shortest * 0.16)
    return chosen


def xyz(points: Any) -> np.ndarray:
    points = np.asarray(points)
    return np.column_stack((points[:, :2], np.zeros(len(points))))


def lane(points: Any, speed: float, kind: int = CAR, **links: Any) -> Any:
    return RoadPolyline(layer=MapLayer.LANE, type=kind, polyline=xyz(points), speed_limit_mps=speed, **links)


def offset(line: LineString, distance: float) -> np.ndarray:
    # Mitred vertex normals keep one parallel polyline per lane, even around tight bends.
    points = np.asarray(line.coords)
    tangent = np.diff(points, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    mitre = np.vstack((normal[:1], normal[:-1] + normal[1:], normal[-1:]))
    mitre /= np.sum(mitre * np.vstack((normal, normal[-1:])), axis=1)[:, None]
    shifted = points + distance * mitre
    # Inside a tight bend the offset runs backwards; dropping those vertices removes the loop.
    index = np.arange(len(points))
    while True:
        forward = np.sum(np.diff(shifted[index], axis=0) * np.diff(points[index], axis=0), axis=1) > 0
        if forward.all():
            return shifted[index]
        drop = np.where(index[1:] == len(points) - 1, index[:-1], index[1:])[~forward]
        index = index[~np.isin(index, drop)]


def surface_parts(geometry: Any) -> list[Polygon]:
    # Millimetre precision removes seams that would collapse in float32 binary coordinates.
    pending = list(shapely.get_parts(shapely.set_precision(geometry, 0.001)))
    simple = []
    while pending:
        polygon = pending.pop()
        if polygon.geom_type != "Polygon" or polygon.area <= 0.01:
            continue
        if not polygon.interiors:
            simple.append(polygon)
            continue
        x = Polygon(polygon.interiors[0]).representative_point().x
        _, bottom, _, top = polygon.bounds
        pieces = split(polygon, LineString([(x, bottom - 1), (x, top + 1)]))
        pending.extend(shapely.get_parts(shapely.set_precision(pieces, 0.001)))
    return simple


def street_geometry(
    nodes: np.ndarray,
    streets: list[np.ndarray],
    edges: np.ndarray,
    outward: np.ndarray,
    major: np.ndarray,
    lanes: np.ndarray,
    typed: np.ndarray,
    speed: np.ndarray,
    roundabouts: dict[int, float],
    rng: np.random.Generator,
) -> tuple[dict[int, Any], list[tuple], list[Polygon], list[Any], list[LineString], np.ndarray, list[tuple]]:
    count = len(edges)
    oneway = ~lanes.all(axis=1)
    width = np.where(major, 3.2, np.where(oneway, rng.uniform(2.6, 2.9, count), rng.uniform(2.75, 3.05, count)))
    extra = np.select([typed == BUS, typed == BIKE], [3.3, 1.6], 0.0)
    side = lanes * width[:, None] + extra
    half = side.sum(axis=1) / 2
    divider = (side[:, 0] - side[:, 1]) / 2
    lengths = np.array([LineString(s).length for s in streets])
    # Each approach starts where it clears every other arm of its junction.
    setback = np.zeros((count, 2))
    for node in range(len(nodes)):
        arms = np.argwhere(edges == node)
        cos = np.clip(outward[arms[:, 0], arms[:, 1]] @ outward[arms[:, 0], arms[:, 1]].T, -1, 1)
        sin = np.maximum(np.sqrt(1 - cos**2), 0.05)
        own, other = half[arms[:, 0]][:, None], half[arms[:, 0]][None, :]
        clear = (other + own * np.abs(cos)) / sin
        clear = np.where(cos < 0, np.minimum(clear, own * sin / np.maximum(-cos, 0.05)), clear)
        np.fill_diagonal(clear, 0)
        setback[arms[:, 0], arms[:, 1]] = np.maximum(clear.max(axis=1) + 1.5, roundabouts.get(node, -8) + 8)
    setback = np.minimum(np.maximum(setback, 6.0), 0.35 * lengths[:, None])
    trims = zip(streets, setback, lengths, strict=True)
    lines = [substring(LineString(s), a, length - b) for s, (a, b), length in trims]
    coords = [np.asarray(line.coords) for line in lines]
    head = np.array([c[1] - c[0] for c in coords])
    tail = np.array([c[-1] - c[-2] for c in coords])
    head /= np.linalg.norm(head, axis=1)[:, None]
    tail /= np.linalg.norm(tail, axis=1)[:, None]
    first, last = np.array([c[0] for c in coords]), np.array([c[-1] for c in coords])
    across = np.column_stack((-head[:, 1], head[:, 0])) * half[:, None]
    behind = np.column_stack((-tail[:, 1], tail[:, 0])) * half[:, None]
    mouths = np.stack((first - across, first + across, last - behind, last + behind), axis=1)
    corners = [np.concatenate((mouths[edges[:, 0] == n, :2], mouths[edges[:, 1] == n, 2:])) for n in range(len(nodes))]
    junctions = [MultiPoint(c.reshape(-1, 2)).convex_hull for c in corners]
    # Car lanes sit next to the divider; a bus or bike lane runs along the kerb.
    polylines, records = {}, []
    for edge, direction in product(range(count), range(2)):
        sign = -1 if direction == 0 else 1
        source, target = edges[edge] if direction == 0 else edges[edge, ::-1]
        layout = [
            (CAR, divider[edge] + sign * (lanes[edge, direction] - k - 0.5) * width[edge], width[edge])
            for k in range(lanes[edge, direction])
        ]
        if typed[edge, direction]:
            shift = divider[edge] + sign * (lanes[edge, direction] * width[edge] + extra[edge, direction] / 2)
            layout.append((typed[edge, direction], shift, extra[edge, direction]))
        cars = []
        for kind, shift, lane_width in layout:
            path = offset(lines[edge], shift)
            lane_id = len(polylines)
            limit = 20.0 if kind == BIKE else speed[edge]
            polylines[lane_id] = lane(path if direction == 0 else path[::-1], limit / 3.6, kind)
            records.append((lane_id, edge, direction, kind, int(source), int(target), lane_width))
            cars += [lane_id] if kind == CAR else []
        for k, lane_id in enumerate(cars):
            neighbors = {"left_neighbor": cars[k + 1 : k + 2], "right_neighbor": cars[max(k - 1, 0) : k]}
            polylines[lane_id] = replace(polylines[lane_id], **neighbors)
    markings = []
    for edge in range(count):
        offsets = [(divider[edge], RoadLineType.DASHED_WHITE if (lanes[edge] == 1).all() else RoadLineType.SOLID_WHITE)]
        offsets = offsets if lanes[edge].all() else []
        offsets += [
            (divider[edge] + sign * j * width[edge], RoadLineType.DASHED_WHITE)
            for direction, sign in ((0, -1), (1, 1))
            for j in range(1, lanes[edge, direction])
        ]
        offsets += [
            (divider[edge] + sign * lanes[edge, direction] * width[edge], RoadLineType.SOLID_WHITE)
            for direction, sign in ((0, -1), (1, 1))
            if typed[edge, direction]
        ]
        markings += [(kind, LineString(offset(lines[edge], shift))) for shift, kind in offsets]
    roads = [line.buffer(h, cap_style="flat", join_style="mitre") for line, h in zip(lines, half, strict=True)]
    return polylines, records, roads, junctions, lines, half, markings


def turn_curve(a: np.ndarray, b: np.ndarray, heading_a: np.ndarray, heading_b: np.ndarray) -> np.ndarray:
    handle = np.linalg.norm(b - a) * 0.45
    t = np.linspace(0, 1, 32)[:, None]
    curve = (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * (a + heading_a * handle)
    return curve + 3 * (1 - t) * t**2 * (b - heading_b * handle) + t**3 * b


def junction_lanes(
    polylines: dict[int, Any], records: list[tuple], junctions: list[Any], roundabouts: dict[int, float]
) -> tuple[dict[int, Any], Any]:
    groups, arrive, depart = {}, {}, {}
    for lane_id, edge, direction, kind, source, target, _ in records:
        groups.setdefault((edge, direction, kind), []).append(lane_id)
        arrive.setdefault((target, kind), set()).add((edge, direction))
        depart.setdefault((source, kind), set()).add((edge, direction))
    begin = {i: p.polyline[1, :2] - p.polyline[0, :2] for i, p in polylines.items()}
    finish = {i: p.polyline[-1, :2] - p.polyline[-2, :2] for i, p in polylines.items()}
    begin = {i: v / np.linalg.norm(v) for i, v in begin.items()}
    finish = {i: v / np.linalg.norm(v) for i, v in finish.items()}

    def turn(a: int, b: int) -> float:
        return float(np.arctan2(finish[a][0] * begin[b][1] - finish[a][1] * begin[b][0], finish[a] @ begin[b]))

    turns = []
    for (node, kind), approaches in arrive.items():
        if node in roundabouts:
            continue
        for edge, direction in sorted(approaches):
            sources = groups[(edge, direction, kind)]
            options = [groups[(e, d, kind)] for e, d in sorted(depart.get((node, kind), ())) if e != edge]
            if not options and kind != CAR:
                cars = [groups[(e, d, CAR)][0] for e, d in sorted(depart.get((node, CAR), ())) if e != edge]
                options = [[min(cars, key=lambda i: abs(turn(sources[0], i)))]]
            if kind != CAR or not options:
                turns += [(sources[0], option[0], kind) for option in options]
                continue
            angles = np.array([turn(sources[0], option[0]) for option in options])
            options = [options[i] for i in np.argsort(angles)]
            angles = np.sort(angles)
            # Lanes serve turns from right to left; right turns use the kerb lane, left turns the inner one.
            n, m = len(sources), len(options)
            low = np.arange(n) * m // n
            high = np.maximum(low, -(-(np.arange(n) + 1) * m // n) - 1)
            for i, k in product(range(n), range(m)):
                if not low[i] <= k <= high[i]:
                    continue
                served = np.flatnonzero((low <= k) & (high >= k))
                out = options[k]
                left = angles[k] > np.pi / 4
                j = max(len(out) - 1 - (served[-1] - i), 0) if left else min(i - served[0], len(out) - 1)
                turns.append((sources[i], out[j], kind))
    # Bus and bike lanes without a same-kind approach start from the straightest car lane.
    for (node, kind), approaches in depart.items():
        if kind == CAR or node in roundabouts:
            continue
        for edge, direction in sorted(approaches):
            if any(e != edge for e, _ in arrive.get((node, kind), ())):
                continue
            target = groups[(edge, direction, kind)][0]
            cars = [groups[(e, d, CAR)][0] for e, d in sorted(arrive.get((node, CAR), ())) if e != edge]
            turns.append((min(cars, key=lambda i: abs(turn(i, target))), target, kind))
    first = len(polylines)
    exits = {key: [] for key in polylines}
    entries = {key: [] for key in polylines}
    connectors = {}
    for turn_id, (source, target, kind) in enumerate(turns, first):
        a, b = polylines[source].polyline[-1, :2], polylines[target].polyline[0, :2]
        limit = min(polylines[source].speed_limit_mps, polylines[target].speed_limit_mps)
        limit = limit if abs(turn(source, target)) < np.pi / 6 else min(limit, 20 / 3.6)
        curve = turn_curve(a, b, finish[source], begin[target])
        connectors[turn_id] = lane(curve, limit, kind, entry_lanes=[source], exit_lanes=[target], in_junction=True)
        # Car lanes never list bus or bike connectors, so car routing stays on car lanes.
        if polylines[source].type == kind:
            exits[source].append(turn_id)
        if polylines[target].type == kind:
            entries[target].append(turn_id)
    linked = {key: replace(value, entry_lanes=entries[key], exit_lanes=exits[key]) for key, value in polylines.items()}
    linked.update(connectors)
    # Include the swept lane width in the junction pavement, including acute corner turns.
    footprints = [LineString(c.polyline[:, :2]).buffer(0.9 if c.type == BIKE else 1.7) for c in connectors.values()]
    junctions = [p for node, p in enumerate(junctions) if node not in roundabouts]
    return linked, shapely.union_all([*junctions, *footprints], grid_size=0.001)


def roundabout_lanes(
    polylines: dict[int, Any], records: list[tuple], center: np.ndarray, node: int, radius: float
) -> tuple[dict[int, Any], Any, Polygon, list[tuple[int, float, float]]]:
    # Arms narrow to one lane before a roundabout, so each approach and exit has a single lane.
    arms = [
        (i, edge, target == node)
        for i, edge, _, kind, source, target, _ in records
        if kind == CAR and node in (source, target)
    ]
    side = np.array([1 if entering else -1 for *_, entering in arms])
    paths = [polylines[i].polyline[:, :2] if entering else polylines[i].polyline[::-1, :2] for i, _, entering in arms]
    tips = np.array([path[-1] for path in paths])
    u = np.array([path[-2] - path[-1] for path in paths])
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = side[:, None] * np.column_stack((-u[:, 1], u[:, 0]))  # the driver's right
    along, lateral = np.sum((tips - center) * u, axis=1), np.sum((tips - center) * v, axis=1)
    # Entries and exits leave the straight arm on a right-hand arc that meets the ring tangentially. Each takes at
    # most 40% of the angle to the next arm, so ring segments never vanish.
    axis = np.arctan2(u[:, 1], u[:, 0])
    gap = np.abs((axis[:, None] - axis[None, :] + np.pi) % (2 * np.pi) - np.pi)
    gap[np.array([e for _, e, _ in arms])[:, None] == np.array([e for _, e, _ in arms])[None, :]] = np.pi
    widest = np.minimum(0.6, 0.4 * gap.min(axis=1))
    bend = np.clip((radius * np.sin(widest) - lateral) / (1 - np.sin(widest)), 1.0, along - radius - 0.5)
    theta = np.arcsin((lateral + bend) / (radius + bend))
    ports = sorted(zip((axis + side * theta) % (2 * np.pi), range(len(arms)), strict=True))
    angles = np.array([angle for angle, _ in ports])
    positions = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    first, count = len(polylines), len(ports)
    linked = dict(polylines)
    for index in range(count):
        next_index = (index + 1) % count
        sweep = (angles[next_index] - angles[index]) % (2 * np.pi)
        steps = np.linspace(angles[index], angles[index] + sweep, max(3, int(radius * sweep) + 1))
        arc = center + radius * np.column_stack((np.cos(steps), np.sin(steps)))
        arc[0], arc[-1] = positions[index], positions[next_index]
        linked[first + index] = lane(
            arc, 15 / 3.6, entry_lanes=[first + (index - 1) % count], exit_lanes=[first + next_index], in_junction=True
        )
    footprints = []
    yields = []
    outer = Point(center).buffer(radius + 1.9, quad_segs=64)
    island = Point(center).buffer(radius - 1.9, quad_segs=64)
    for index, (_, k) in enumerate(ports):
        lane_id, _, entering = arms[k]
        hub = (radius + bend[k]) * np.array([np.cos(theta[k]), np.sin(theta[k])])
        turn = np.linspace(-np.pi / 2, theta[k] - np.pi, 12)
        local = np.vstack(([along[k], lateral[k]], hub + bend[k] * np.column_stack((np.cos(turn), np.sin(turn)))))
        curve = center + local[:, :1] * u[k] + local[:, 1:] * v[k]
        connector = len(linked)
        if entering:
            source, target = lane_id, first + index
            yields.append((connector, 1.5, LineString(curve).difference(outer).length - 0.8))
        else:
            source, target, curve = first + (index - 1) % count, lane_id, curve[::-1]
        linked[connector] = lane(curve, 15 / 3.6, entry_lanes=[source], exit_lanes=[target], in_junction=True)
        linked[source] = replace(linked[source], exit_lanes=[*linked[source].exit_lanes, connector])
        linked[target] = replace(linked[target], entry_lanes=[*linked[target].entry_lanes, connector])
        footprints.append(LineString(curve).buffer(1.75))
    pavement = shapely.union_all([outer.difference(island), *footprints], grid_size=0.001).difference(island)
    return linked, pavement, island, yields


def streetscape(
    lines: list[LineString],
    half: np.ndarray,
    typed: np.ndarray,
    roads: list[Polygon],
    junction_pavement: Any,
    islands: Any,
    rng: np.random.Generator,
) -> tuple[list[tuple[MapLayer, Polygon]], Any, list[LineString]]:
    surfaces = [(MapLayer.LANE_GROUP, road) for road in roads]
    surfaces += [(MapLayer.INTERSECTION, p) for p in surface_parts(junction_pavement)]
    markings = []
    parking = []
    bay_lines = []
    tree = shapely.STRtree(roads)
    # Kerb parking on both sides, away from bus lanes, other streets and junctions.
    for edge, side in product(range(len(lines)), (0, 1)):
        if typed[edge, side] == BUS or rng.random() > (0.2 if half[edge] < 1.5 else 0.85):
            continue
        s = 15.75 + 5.5 * np.arange(max(0, int((lines[edge].length - 26) // 5.5)))
        centers = shapely.get_coordinates(shapely.line_interpolate_point(lines[edge], s))
        t = shapely.get_coordinates(shapely.line_interpolate_point(lines[edge], s + 0.5))
        t -= shapely.get_coordinates(shapely.line_interpolate_point(lines[edge], s - 0.5))
        t /= np.linalg.norm(t, axis=1)[:, None]
        n = np.column_stack((-t[:, 1], t[:, 0])) * (1 if side else -1)
        centers += n * (half[edge] + 1.0)
        back, front = centers - 2.75 * t, centers + 2.75 * t
        bays = shapely.polygons(np.stack((back - n, front - n, front + n, back + n), axis=1))
        bay, road = tree.query(bays, predicate="intersects")
        clear = ~np.isin(np.arange(len(bays)), bay[road != edge]) & ~shapely.intersects(bays, junction_pavement)
        clear &= shapely.area(shapely.intersection(bays, roads[edge])) < 0.1
        parking += list(bays[clear])
        bay_lines += list(shapely.linestrings(np.stack((back - n, back + n), axis=1)[clear]))
    pavement = shapely.union_all([*roads, junction_pavement, *parking], grid_size=0.001)
    # Small off-street lots, with an open driveway through the sidewalk.
    lots = []
    driveways = []
    for edge in rng.permutation(len(lines)):
        if len(lots) >= max(1, len(lines) // 24):
            break
        if lines[edge].length < 35:
            continue
        middle = np.array(lines[edge].interpolate(0.5, normalized=True).coords[0])
        t = np.array(lines[edge].interpolate(lines[edge].length / 2 + 0.5).coords[0]) - middle
        t /= np.linalg.norm(t)
        n = np.array([-t[1], t[0]])
        center = middle + n * (half[edge] + 15)
        lot = Polygon(
            [center - 12 * t - 8 * n, center + 12 * t - 8 * n, center + 12 * t + 8 * n, center - 12 * t + 8 * n]
        )
        if lot.intersects(pavement.buffer(2.5)) or any(lot.intersects(p) for p in lots):
            continue
        lots.append(lot)
        driveways.append(LineString([middle, center]).buffer(3.0, cap_style="flat"))
        markings.extend(
            LineString([center + offset * t + side * 3 * n, center + offset * t + side * 8 * n])
            for offset, side in product(np.arange(-10, 10.1, 2.5), (-1, 1))
        )
    access = shapely.union_all(driveways, grid_size=0.001)
    markings += [line for bay, line in zip(parking, bay_lines, strict=True) if not bay.intersects(access)]
    parking = [p.difference(access) for p in parking]
    pavement = shapely.union_all([pavement, *lots, *driveways], grid_size=0.001)
    # Close hairline gaps, so kerb parking joins its street instead of becoming a kerbed island, and fill the
    # specks left between turn paths; real islands are larger.
    pavement = pavement.buffer(0.25, join_style="mitre").buffer(-0.25, join_style="mitre")
    specks = [Polygon(ring) for part in shapely.get_parts(pavement) for ring in part.interiors]
    pavement = shapely.union_all([pavement, *[speck for speck in specks if speck.area < 20]], grid_size=0.001)
    surfaces += [(MapLayer.CARPARK, p) for p in surface_parts(shapely.union_all(parking, grid_size=0.001)) + lots]
    surfaces += [(MapLayer.GENERIC_DRIVABLE, p) for p in driveways]
    sidewalks = pavement.buffer(2.2, join_style="round").difference(pavement).difference(islands)
    surfaces += [(MapLayer.WALKWAY, p) for p in surface_parts(sidewalks)]
    for edge, endpoint in product(range(len(lines)), (0, 1)):
        length = lines[edge].length
        if length < 12.4 or (length < 20 and endpoint == 1):
            continue
        # Leave one car's length between the crossing and the yield line.
        distance = 7 if length >= 20 else length / 2
        distance = distance if endpoint == 0 else length - distance
        c = np.array(lines[edge].interpolate(distance).coords[0])
        t = shapely.get_coordinates(shapely.line_interpolate_point(lines[edge], [distance + 0.5, distance - 0.5]))
        t = (t[0] - t[1]) / np.linalg.norm(t[0] - t[1])
        n = np.array([-t[1], t[0]]) * half[edge]
        surfaces.append((MapLayer.CROSSWALK, Polygon([c - t - n, c + t - n, c + t + n, c - t + n])))
    return surfaces, pavement, markings


def intersection_rules(
    nodes: np.ndarray,
    edges: np.ndarray,
    outward: np.ndarray,
    major: np.ndarray,
    stroke: np.ndarray,
    records: list[tuple],
    polylines: dict[int, Any],
    roundabouts: dict[int, float],
    profile: dict[str, float],
    rng: np.random.Generator,
) -> dict[int, tuple[TCType, int, int]]:
    degree = np.bincount(edges.ravel(), minlength=len(nodes))
    approaches = {}
    for lane_id, edge, direction, kind, _, target, _ in records:
        approaches.setdefault(target, {}).setdefault((edge, direction), []).append((lane_id, kind))
    car_arms = {node: [a for a in arms.values() if any(k == CAR for _, k in a)] for node, arms in approaches.items()}
    candidates = [
        node for node in approaches if degree[node] >= 3 and node not in roundabouts and len(car_arms[node]) >= 2
    ]
    eligible = [
        node
        for node in candidates
        if min(polylines[i].length for arm in approaches[node].values() for i, k in arm if k == CAR) >= 20
    ]
    importance = {node: major[(edges == node).any(axis=1)].sum() + degree[node] + rng.random() for node in eligible}
    count = int(profile["signals"] * len(candidates) + rng.random())
    signalized = set(sorted(eligible, key=importance.get, reverse=True)[:count])
    rules = {}
    signed = []
    for node in candidates:
        arms = approaches[node]
        if node in signalized:
            # One phase per arm for now; group_facing_phases merges facing arms, so left turns are unprotected.
            rules.update({i: (TCType.YIELD_SIGN, -1, -1) for arm in arms.values() for i, _ in arm})
            phases = enumerate(car_arms[node])
            rules.update({i: (TCType.TRAFFIC_LIGHT, node, phase) for phase, arm in phases for i, _ in arm})
            continue
        incident = np.argwhere(edges == node)
        # Priority follows the most important street that continues through the junction.
        through = [s for s in np.unique(stroke[incident[:, 0]]) if np.count_nonzero(stroke[incident[:, 0]] == s) >= 2]
        if through:
            best = max(through, key=lambda s: (major[stroke == s].any(), np.count_nonzero(stroke == s)))
            priority = set(incident[stroke[incident[:, 0]] == best, 0])
        else:
            vectors = outward[incident[:, 0], incident[:, 1]]
            left, right = np.triu_indices(len(incident), 1)
            score = -np.sum(vectors[left] * vectors[right], axis=1)
            score += 1.5 * (major[incident[left, 0]].astype(int) + major[incident[right, 0]])
            best = int(np.argmax(score))
            priority = {incident[left[best], 0], incident[right[best], 0]}
        minor = [i for (edge, _), arm in arms.items() if edge not in priority for i, _ in arm]
        if not minor:
            continue
        if not major[incident[:, 0]].any() and rng.random() < 0.35:
            continue  # Quiet unsigned junction: the simulator applies priority to the right.
        signed.append(minor)
    stop_junctions = set(rng.permutation(len(signed))[: max(1, len(signed) // 4)])
    for index, minor in enumerate(signed):
        kind = TCType.STOP_SIGN if index in stop_junctions else TCType.YIELD_SIGN
        rules.update(dict.fromkeys(minor, (kind, -1, -1)))
    return rules


def generate_town(size: float = 500.0, seed: int = 0) -> PufferScenario:
    if not np.isfinite(size) or not 200 <= size <= 1200 or seed < 0:
        raise ValueError("Use size 200..1200 metres and a nonnegative seed")
    rng = np.random.default_rng(seed)
    profile = {key: rng.uniform(*span) for key, span in EUROPE.items()}
    layout = {key: rng.uniform(*span) for key, span in LAYOUT.items()}
    nodes, streets, edges, outward, corridor = street_network(size, layout, profile, rng)
    lengths = np.array([LineString(s).length for s in streets])
    count = rng.poisson(profile["roundabouts"] * 0.8 * (size / 1000) ** 2)
    roundabouts = select_roundabouts(nodes, edges, outward, lengths, count, rng)
    stroke, major, lanes, speed, typed = street_plan(
        nodes, streets, edges, outward, corridor, roundabouts, profile, rng
    )
    polylines, records, roads, junctions, lines, half, markings = street_geometry(
        nodes, streets, edges, outward, major, lanes, typed, speed, roundabouts, rng
    )
    polylines, junction_pavement = junction_lanes(polylines, records, junctions, roundabouts)
    islands = []
    yield_lanes = []
    for node, radius in roundabouts.items():
        polylines, pavement, island, yields = roundabout_lanes(polylines, records, nodes[node], node, radius)
        junction_pavement = shapely.union_all([junction_pavement, pavement], grid_size=0.001)
        islands.append(island)
        yield_lanes.extend(yields)
    islands = shapely.union_all(islands, grid_size=0.001)
    surface_geometry, pavement, kerb_markings = streetscape(lines, half, typed, roads, junction_pavement, islands, rng)
    markings += [(RoadLineType.SOLID_WHITE, line) for line in kerb_markings]
    for i, (line_type, line) in enumerate(markings, len(polylines)):
        polylines[i] = RoadPolyline(layer=MapLayer.ROAD_LINE, type=line_type, polyline=xyz(line.coords))
    # Reverse clockwise exteriors and counterclockwise holes to keep drivable space on the left.
    polygons = shapely.orient_polygons(shapely.set_precision(pavement, 0.001), exterior_cw=True)
    rings = [ring for p in shapely.get_parts(polygons) for ring in [p.exterior, *p.interiors]]
    for i, ring in enumerate(rings, len(polylines)):
        median = ring.is_ccw and Polygon(ring).intersects(islands)
        edge_type = RoadEdgeType.ROAD_EDGE_MEDIAN if median else RoadEdgeType.ROAD_EDGE_BOUNDARY
        polylines[i] = RoadPolyline(layer=MapLayer.ROAD_EDGE, type=edge_type, polyline=xyz(ring.coords[::-1]))
    first = len(polylines)
    pieces = [(layer, p) for layer, geometry in surface_geometry for p in surface_parts(geometry)]
    surfaces = {
        first + i: RoadSurface(layer=layer, polygon=xyz(p.exterior.coords)) for i, (layer, p) in enumerate(pieces)
    }
    scenario = PufferScenario(
        agents={},
        objects={},
        polylines=polylines,
        surfaces=surfaces,
        metadata=ScenarioMetadata(
            id=f"town_{size:g}m_{seed:06d}",
            dataset="mapforge",
            scenario_length=1,
            dt=0.1,
            location="synthetic_europe",
            map_version="town-v3",
        ),
    )
    process_polylines(scenario)
    interpolate_all_polygons(scenario, tolerance=0.1)
    rules = intersection_rules(nodes, edges, outward, major, stroke, records, polylines, roundabouts, profile, rng)
    phases = {}
    for kind, node, phase in rules.values():
        if kind == TCType.TRAFFIC_LIGHT:
            phases[node] = max(phases.get(node, 0), phase + 1)
    intersections = [i for i, s in surfaces.items() if s.layer == MapLayer.INTERSECTION]
    tree = shapely.STRtree([Polygon(surfaces[i].polygon[:, :2]) for i in intersections])
    junction_ids = {node: intersections[tree.query_nearest(Point(nodes[node]))[0]] for node in phases}
    placements = [(lane_id, width, distance, TCType.YIELD_SIGN, -1, -1) for lane_id, width, distance in yield_lanes]
    for lane_id, _, _, _, _, _, lane_width in records:
        if lane_id not in rules:
            continue
        kind, node, phase = rules[lane_id]
        # Signals stop cars before the crossing; STOP/YIELD lines leave a waiting space after it.
        distance = polylines[lane_id].length - (8.8 if kind == TCType.TRAFFIC_LIGHT else 0.8)
        placements.append((lane_id, lane_width / 2, distance, kind, node, phase))
    controls = []
    for lane_id, half_width, distance, kind, node, phase in placements:
        line = LineString(polylines[lane_id].polyline[:, :2])
        center = np.array(line.interpolate(distance).coords[0])
        heading = np.array(line.interpolate(distance + 0.05).coords[0])
        heading -= np.array(line.interpolate(distance - 0.05).coords[0])
        heading /= np.linalg.norm(heading)
        n = np.array([-heading[1], heading[0]]) * half_width
        # The simulator generates light states from the junction and phase IDs.
        controls.append(
            TrafficControl(
                id=len(controls),
                type=kind,
                stop_line=xyz([center - n, center + n]),
                heading=float(np.arctan2(heading[1], heading[0])),
                states=[],
                controlled_lanes=[lane_id],
                intersection_id=junction_ids.get(node, -1),
                phase_idx=phase,
            )
        )
    scenario.traffic_controls = group_facing_phases(controls)
    return scenario


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate European-style towns as static Macadam binaries")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--size", type=float, default=500.0, help="Town extent in metres (200..1200)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=1, help="Generate consecutive seeds")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    for seed in range(args.seed, args.seed + args.count):
        scenario = generate_town(args.size, seed)
        path = args.output_dir / f"{scenario.metadata.id}.bin"
        write_static_scenario(scenario, path, overwrite=args.overwrite)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
