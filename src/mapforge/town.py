import argparse
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import split

from bin_factory.conversion.geometry import interpolate_all_polygons, process_polylines
from bin_factory.generators import build_lane_distance_matrix, compute_lane_lengths
from bin_factory.puffer_types import LaneType, MapLayer, RoadEdgeType, RoadLineType, TCType, TLState
from bin_factory.schema import PufferScenario, RoadPolyline, RoadSurface, ScenarioMetadata, TrafficControl
from mapforge.static_binary import write_static_scenario


def street_network(style: str, size: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    count = max(4, round(size / 85) + 1)
    spacing = (size - 40) / (count - 1)
    axes = np.cumsum(rng.uniform(0.7, 1.3, (2, count)), axis=1)
    axes = (axes - axes[:, :1]) / (axes[:, -1:] - axes[:, :1]) * (size - 40) - (size - 40) / 2
    x, y = np.meshgrid(*axes)
    points = np.column_stack((x.ravel(), y.ravel()))
    polar = np.arctan2(points[:, 1], points[:, 0])
    radius = np.linalg.norm(points, axis=1) / ((size - 40) / 2)
    outline = 1.13 + 0.13 * np.sin(3 * polar + rng.uniform(0, 2 * np.pi))
    inside = radius < outline
    points += rng.uniform(-0.19, 0.19, points.shape) * spacing
    points[:, 0] += rng.uniform(0.05, 0.22) * points[:, 1]
    points[:, 1] += spacing * 0.25 * np.sin(points[:, 0] / size * 5 + rng.uniform(0, 2 * np.pi))
    indices = np.arange(count * count).reshape(count, count)
    horizontal = np.column_stack((indices[:, :-1].ravel(), indices[:, 1:].ravel()))
    vertical = np.column_stack((indices[:-1].ravel(), indices[1:].ravel()))
    edges = np.vstack((horizontal, vertical))
    arterial = np.array([a // count == count // 2 if b - a == 1 else a % count == count // 2 for a, b in edges])
    cells = indices[:-1, :-1].ravel()
    avenue = cells // count == cells % count
    diagonal = avenue | (rng.random(len(cells)) < (0.12 if style == "lyon" else 0.24))
    reverse = (rng.random(len(cells)) < 0.5) & ~avenue
    diagonals = np.column_stack((cells + reverse, cells + count + 1 - reverse))[diagonal]
    diagonal_arterial = avenue[diagonal]
    keep = inside[edges].all(axis=1)
    # Peel exposed tips from the irregular outline; keep loops around the town's edge.
    while True:
        degree = np.bincount(edges[keep].ravel(), minlength=len(points))
        remaining = keep & (degree[edges] >= 2).all(axis=1)
        if np.array_equal(keep, remaining):
            break
        keep = remaining
    edges, arterial = edges[keep], arterial[keep]
    used, edges = np.unique(edges, return_inverse=True)
    edges = edges.reshape(-1, 2)
    remap = np.full(len(points), -1)
    remap[used] = np.arange(len(used))
    diagonals = remap[diagonals]
    retained = (diagonals >= 0).all(axis=1)
    diagonals, diagonal_arterial = diagonals[retained], diagonal_arterial[retained]
    points = points[used]
    angle = rng.uniform(-np.pi, np.pi)
    points = points @ np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points -= (points.max(axis=0) + points.min(axis=0)) / 2
    points *= (size - 40) / np.ptp(points, axis=0).max()
    # Merge some blocks into larger ones and introduce T-junctions.
    keep = np.ones(len(edges), dtype=bool)
    candidates = np.flatnonzero(~arterial & (rng.random(len(edges)) < 0.22))
    for edge in candidates:
        degree = np.bincount(edges[keep].ravel(), minlength=len(points))
        if np.any(degree[edges[edge]] <= 2):
            continue
        keep[edge] = False
        pairs = edges[keep]
        graph = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points)))
        if connected_components(graph, directed=False)[0] != 1:
            keep[edge] = True
    edges, arterial = edges[keep], arterial[keep]
    # Keep local junctions to three/four arms; only one node may become a larger plaza.
    degree = np.bincount(edges.ravel(), minlength=len(points))
    arm_limit = np.full(len(points), 4)
    hubs = np.flatnonzero(degree >= 3)
    if len(hubs):
        arm_limit[rng.choice(hubs)] = 6
    for index in np.argsort(rng.random(len(diagonals)) - diagonal_arterial):
        pair = diagonals[index]
        if np.any(degree[pair] >= arm_limit[pair]):
            continue
        edges = np.vstack((edges, pair))
        arterial = np.r_[arterial, diagonal_arterial[index]]
        degree[pair] += 1
    # Keep every directed street reachable without requiring a U-turn.
    directions = np.ones((len(edges), 2), dtype=bool)
    candidates = np.flatnonzero(~arterial & (rng.random(len(edges)) < 0.7))
    for edge in candidates:
        side = int(rng.integers(2))
        directions[edge, side] = False
        pairs = np.concatenate((edges[directions[:, 0]], edges[directions[:, 1], ::-1]))
        transitions = (pairs[:, 1, None] == pairs[None, :, 0]) & (pairs[:, 0, None] != pairs[None, :, 1])
        graph = csr_matrix(transitions)
        if connected_components(graph, directed=True, connection="strong")[0] != 1:
            directions[edge, side] = True
    return points, edges, arterial, directions


def select_roundabouts(points: np.ndarray, edges: np.ndarray, rng: np.random.Generator) -> dict[int, float]:
    chosen = {}
    degree = np.bincount(edges.ravel(), minlength=len(points))
    budget = 2 - np.count_nonzero(degree > 4)
    for node in rng.permutation(np.flatnonzero((degree >= 3) & (degree <= 4))):
        incident = np.flatnonzero((edges == node).any(axis=1))
        neighbors = np.where(edges[incident, 0] == node, edges[incident, 1], edges[incident, 0])
        vectors = points[neighbors] - points[node]
        angles = np.sort(np.arctan2(vectors[:, 1], vectors[:, 0]))
        separation = np.diff(np.r_[angles, angles[0] + 2 * np.pi]).min()
        shortest = np.linalg.norm(vectors, axis=1).min()
        if separation < np.deg2rad(47) or shortest < 42:
            continue
        if any(np.linalg.norm(points[node] - points[other]) < 75 for other in chosen):
            continue
        chosen[int(node)] = min(11.0, shortest * rng.uniform(0.13, 0.16))
        if len(chosen) >= min(budget, max(1, len(points) // 16)):
            break
    return chosen


def xyz(points: Any) -> np.ndarray:
    points = np.asarray(points)
    return np.column_stack((points[:, :2], np.zeros(len(points))))


def lane(points: Any, speed: float, **links: Any) -> Any:
    return RoadPolyline(
        layer=MapLayer.LANE,
        type=LaneType.SURFACE_STREET,
        polyline=xyz(points),
        speed_limit_mps=speed,
        **links,
    )


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
    points: np.ndarray,
    edges: np.ndarray,
    arterial: np.ndarray,
    directions: np.ndarray,
    roundabouts: dict[int, float],
    rng: np.random.Generator,
) -> tuple[
    dict[int, Any],
    list[Any],
    list[Polygon],
    list[Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    vector = points[edges[:, 1]] - points[edges[:, 0]]
    lengths = np.linalg.norm(vector, axis=1)
    tangent = vector / lengths[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    widths = np.where(arterial, 3.25, rng.uniform(2.75, 3.05, len(edges)))
    widths = np.where(directions.all(axis=1), widths, rng.uniform(2.5, 2.8, len(edges)))
    half = widths * directions.sum(axis=1) / 2
    # Compact local junctions, even where an avenue passes through.
    degree = np.bincount(edges.ravel(), minlength=len(points))
    setback = np.where(degree >= 5, 18.0, 7.0)
    for node, radius in roundabouts.items():
        setback[node] = radius + 8
    start = points[edges[:, 0]] + tangent * np.minimum(setback[edges[:, 0]], lengths * 0.35)[:, None]
    end = points[edges[:, 1]] - tangent * np.minimum(setback[edges[:, 1]], lengths * 0.35)[:, None]
    roads = [LineString([a, b]).buffer(w, cap_style="flat") for a, b, w in zip(start, end, half, strict=True)]
    mouths = np.stack(
        (
            start - normal * half[:, None],
            start + normal * half[:, None],
            end - normal * half[:, None],
            end + normal * half[:, None],
        ),
        axis=1,
    )
    junctions = []
    for node in range(len(points)):
        corners = np.concatenate(
            (mouths[edges[:, 0] == node, :2].reshape(-1, 2), mouths[edges[:, 1] == node, 2:].reshape(-1, 2))
        )
        junctions.append(MultiPoint(corners).convex_hull)
    polylines = {}
    connections = []
    for edge, direction in product(range(len(edges)), range(2)):
        if not directions[edge, direction]:
            continue
        sign = 1 if direction == 0 else -1
        offset = normal[edge] * widths[edge] / 2 * sign if directions[edge].all() else np.zeros(2)
        a, b = (start[edge], end[edge]) if direction == 0 else (end[edge], start[edge])
        source, target = edges[edge] if direction == 0 else edges[edge, ::-1]
        lane_id = len(polylines)
        speed = 50 if arterial[edge] else (30 if directions[edge].all() else 20)
        polylines[lane_id] = lane([a - offset, b - offset], speed / 3.6)
        connections.append((lane_id, edge, source, target, sign * tangent[edge]))
    return polylines, connections, roads, junctions, start, end, tangent, normal, half


def turn_curve(a: np.ndarray, b: np.ndarray, heading_a: np.ndarray, heading_b: np.ndarray) -> np.ndarray:
    handle = np.linalg.norm(b - a) * 0.45
    t = np.linspace(0, 1, 32)[:, None]
    curve = (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * (a + heading_a * handle)
    return curve + 3 * (1 - t) * t**2 * (b - heading_b * handle) + t**3 * b


def junction_lanes(
    polylines: dict[int, Any],
    connections: list[Any],
    junctions: list[Any],
    roundabouts: dict[int, float],
) -> tuple[dict[int, Any], Any]:
    turns = []
    for incoming, outgoing in product(connections, repeat=2):
        source, edge_a, _, node, heading_a = incoming
        target, edge_b, next_node, _, heading_b = outgoing
        if node != next_node or edge_a == edge_b or node in roundabouts:
            continue
        a = polylines[source].polyline[-1, :2]
        b = polylines[target].polyline[0, :2]
        curve = turn_curve(a, b, heading_a, heading_b)
        turns.append(
            (source, target, node, lane(curve, 20 / 3.6, entry_lanes=[source], exit_lanes=[target], in_junction=True))
        )
    first = len(polylines)
    exits = {key: [] for key in polylines}
    entries = {key: [] for key in polylines}
    for turn_id, (source, target, _, _) in enumerate(turns, first):
        exits[source].append(turn_id)
        entries[target].append(turn_id)
    linked = {key: replace(value, entry_lanes=entries[key], exit_lanes=exits[key]) for key, value in polylines.items()}
    linked.update({turn_id: turn[3] for turn_id, turn in enumerate(turns, first)})
    # Include the swept lane width in the junction pavement, including acute corner turns.
    footprints = [LineString(turn.polyline[:, :2]).buffer(1.7) for _, _, _, turn in turns]
    junctions = [p for node, p in enumerate(junctions) if node not in roundabouts]
    return linked, shapely.union_all([*junctions, *footprints], grid_size=0.001)


def roundabout_lanes(
    polylines: dict[int, Any],
    connections: list[Any],
    center: np.ndarray,
    node: int,
    radius: float,
) -> tuple[dict[int, Any], Any, Polygon, list[tuple[int, float, float]]]:
    incoming = [(i, h, True) for i, _, _, target, h in connections if target == node]
    outgoing = [(i, h, False) for i, _, source, _, h in connections if source == node]
    ports = []
    for street, heading, entering in incoming + outgoing:
        radial = -heading if entering else heading
        angle = (np.arctan2(radial[1], radial[0]) + (0.32 if entering else -0.32)) % (2 * np.pi)
        ports.append((angle, street, heading, entering))
    ports.sort(key=lambda p: p[0])
    angles = np.array([p[0] for p in ports])
    positions = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    first, count = len(polylines), len(ports)
    linked = dict(polylines)
    for index in range(count):
        next_index = (index + 1) % count
        sweep = (angles[next_index] - angles[index]) % (2 * np.pi)
        theta = np.linspace(angles[index], angles[index] + sweep, max(3, int(radius * sweep) + 1))
        arc = center + radius * np.column_stack((np.cos(theta), np.sin(theta)))
        arc[0], arc[-1] = positions[index], positions[next_index]
        linked[first + index] = lane(
            arc, 15 / 3.6, entry_lanes=[first + (index - 1) % count], exit_lanes=[first + next_index], in_junction=True
        )
    footprints = []
    yields = []
    outer = Point(center).buffer(radius + 1.9, quad_segs=64)
    island = Point(center).buffer(radius - 1.9, quad_segs=64)
    for index, (angle, street, heading, entering) in enumerate(ports):
        ring_heading = np.array([-np.sin(angle), np.cos(angle)])
        connector = first + count + index
        if entering:
            source, target = street, first + index
            curve = turn_curve(polylines[street].polyline[-1, :2], positions[index], heading, ring_heading)
            yields.append((connector, 1.5, LineString(curve).difference(outer).length - 0.8))
        else:
            source, target = first + (index - 1) % count, street
            curve = turn_curve(positions[index], polylines[street].polyline[0, :2], ring_heading, heading)
        linked[connector] = lane(curve, 15 / 3.6, entry_lanes=[source], exit_lanes=[target], in_junction=True)
        linked[source] = replace(linked[source], exit_lanes=[*linked[source].exit_lanes, connector])
        linked[target] = replace(linked[target], entry_lanes=[*linked[target].entry_lanes, connector])
        footprints.append(LineString(curve).buffer(1.75))
    pavement = shapely.union_all([outer.difference(island), *footprints], grid_size=0.001).difference(island)
    return linked, pavement, island, yields


def streetscape(
    start: np.ndarray,
    end: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    half: np.ndarray,
    roads: list[Polygon],
    junction_pavement: Any,
    islands: Any,
    rng: np.random.Generator,
) -> tuple[list[tuple[MapLayer, Polygon]], Any, list[LineString]]:
    surfaces = [(MapLayer.LANE_GROUP, road) for road in roads]
    surfaces += [(MapLayer.INTERSECTION, p) for p in surface_parts(junction_pavement)]
    markings = []
    bays = []
    parking_sides = rng.random((len(start), 2)) < np.where(half[:, None] < 1.5, 0.2, 0.85)
    for edge, side in product(range(len(start)), (-1, 1)):
        if not parking_sides[edge, (side + 1) // 2]:
            continue
        length = np.linalg.norm(end[edge] - start[edge])
        count = int((length - 26) // 5.5)
        centers = start[edge] + np.arange(count)[:, None] * 5.5 * tangent[edge] + 15.75 * tangent[edge]
        centers += side * normal[edge] * (half[edge] + 1.0)
        bays.extend((center, tangent[edge], normal[edge]) for center in centers)
    parking = [Polygon([c - 2.75 * t - n, c + 2.75 * t - n, c + 2.75 * t + n, c - 2.75 * t + n]) for c, t, n in bays]
    pavement = shapely.union_all([*roads, junction_pavement, *parking], grid_size=0.001)
    # Small off-street lots, with an open driveway through the sidewalk.
    lots = []
    driveways = []
    for edge in rng.permutation(len(start)):
        if len(lots) >= max(1, len(start) // 24):
            break
        if np.linalg.norm(end[edge] - start[edge]) < 35:
            continue
        t, n = tangent[edge], normal[edge]
        middle = (start[edge] + end[edge]) / 2
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
    parking = [p.difference(access) for p in parking]
    pavement = shapely.union_all([pavement, *lots, *driveways], grid_size=0.001)
    surfaces += [(MapLayer.CARPARK, p) for p in surface_parts(shapely.union_all(parking, grid_size=0.001)) + lots]
    surfaces += [(MapLayer.GENERIC_DRIVABLE, p) for p in driveways]
    sidewalks = pavement.buffer(2.2, join_style="round").difference(pavement).difference(islands)
    surfaces += [(MapLayer.WALKWAY, p) for p in surface_parts(sidewalks)]
    for edge, endpoint in product(range(len(start)), (0, 1)):
        length = np.linalg.norm(end[edge] - start[edge])
        if length < 12.4 or (length < 20 and endpoint == 1):
            continue
        t = tangent[edge] * (1 if endpoint == 0 else -1)
        # Leave one car's length between the crossing and the yield line.
        c = (start[edge] if endpoint == 0 else end[edge]) + t * (7 if length >= 20 else length / 2)
        n = normal[edge] * half[edge]
        surfaces.append((MapLayer.CROSSWALK, Polygon([c - t - n, c + t - n, c + t + n, c - t + n])))
    markings += [
        LineString([c - 2.75 * t - n, c - 2.75 * t + n])
        for c, t, n in bays
        if not Polygon([c - 2.75 * t - n, c + 2.75 * t - n, c + 2.75 * t + n, c - 2.75 * t + n]).intersects(access)
    ]
    return surfaces, pavement, markings


def intersection_rules(
    points: np.ndarray,
    edges: np.ndarray,
    arterial: np.ndarray,
    connections: list[Any],
    polylines: dict[int, Any],
    roundabouts: dict[int, float],
    rng: np.random.Generator,
) -> dict[int, tuple[TCType, int, int]]:
    degree = np.bincount(edges.ravel(), minlength=len(points))
    incoming = {
        node: [(lane_id, edge) for lane_id, edge, _, target, _ in connections if target == node]
        for node in range(len(points))
    }
    candidates = [
        node for node in incoming if degree[node] >= 3 and len(incoming[node]) >= 2 and node not in roundabouts
    ]
    signal_candidates = [node for node in candidates if min(polylines[i].length for i, _ in incoming[node]) >= 20]
    signal_candidates.sort(key=lambda node: (arterial[(edges == node).any(axis=1)].sum(), degree[node]), reverse=True)
    signalized = set(signal_candidates[: max(1, min(2, len(points) // 80))])
    rules = {}
    signed = []
    for node in candidates:
        if node in signalized:
            # Protected approach phases: opposing left turns never receive green together.
            rules.update(
                {lane_id: (TCType.TRAFFIC_LIGHT, node, phase) for phase, (lane_id, _) in enumerate(incoming[node])}
            )
            continue
        incident = np.flatnonzero((edges == node).any(axis=1))
        neighbors = np.where(edges[incident, 0] == node, edges[incident, 1], edges[incident, 0])
        vectors = points[neighbors] - points[node]
        vectors /= np.linalg.norm(vectors, axis=1)[:, None]
        left, right = np.triu_indices(len(incident), 1)
        arriving_edges = [edge for _, edge in incoming[node]]
        valid = np.isin(incident[left], arriving_edges) | np.isin(incident[right], arriving_edges)
        score = -np.sum(vectors[left] * vectors[right], axis=1)
        score += 1.5 * (arterial[incident[left]].astype(int) + arterial[incident[right]])
        best = int(np.argmax(np.where(valid, score, -np.inf)))
        priority = {incident[left[best]], incident[right[best]]}
        minor = [lane_id for lane_id, edge in incoming[node] if edge not in priority]
        if not minor:
            continue
        if not arterial[incident].any() and rng.random() < 0.35:
            continue  # Quiet unsigned junction: the simulator applies priority to the right.
        signed.append(minor)
    stop_junctions = set(rng.permutation(len(signed))[: max(1, len(signed) // 4)])
    for index, minor in enumerate(signed):
        kind = TCType.STOP_SIGN if index in stop_junctions else TCType.YIELD_SIGN
        rules.update(dict.fromkeys(minor, (kind, -1, -1)))
    return rules


def generate_town(style: str = "mixed", size: float = 500.0, seed: int = 0) -> PufferScenario:
    if style not in ("mixed", "paris", "lyon") or not np.isfinite(size) or not 200 <= size <= 1200 or seed < 0:
        raise ValueError("Use style mixed/paris/lyon, size 200..1200 metres, and a nonnegative seed")
    rng = np.random.default_rng(seed)
    points, edges, arterial, directions = street_network(style, size, rng)
    roundabouts = select_roundabouts(points, edges, rng)
    polylines, connections, roads, junctions, start, end, tangent, normal, half = street_geometry(
        points,
        edges,
        arterial,
        directions,
        roundabouts,
        rng,
    )
    polylines, junction_pavement = junction_lanes(polylines, connections, junctions, roundabouts)
    islands = []
    yield_lanes = []
    for node, radius in roundabouts.items():
        polylines, pavement, island, yields = roundabout_lanes(polylines, connections, points[node], node, radius)
        junction_pavement = shapely.union_all([junction_pavement, pavement], grid_size=0.001)
        islands.append(island)
        yield_lanes.extend(yields)
    islands = shapely.union_all(islands, grid_size=0.001)
    surface_geometry, pavement, markings = streetscape(
        start,
        end,
        tangent,
        normal,
        half,
        roads,
        junction_pavement,
        islands,
        rng,
    )
    markings = [(RoadLineType.SOLID_WHITE, line) for line in markings]
    markings += [
        (RoadLineType.DASHED_WHITE, LineString([a, b]))
        for a, b, two_way in zip(start, end, directions.all(axis=1), strict=True)
        if two_way
    ]
    first = len(polylines)
    polylines.update(
        {
            first + i: RoadPolyline(layer=MapLayer.ROAD_LINE, type=line_type, polyline=xyz(line.coords))
            for i, (line_type, line) in enumerate(markings)
        }
    )
    # Clockwise exteriors and counterclockwise holes keep drivable space on the right.
    polygons = shapely.orient_polygons(shapely.set_precision(pavement, 0.001), exterior_cw=True)
    rings = [ring for p in shapely.get_parts(polygons) for ring in [p.exterior, *p.interiors]]
    first = len(polylines)
    polylines.update(
        {
            first + i: RoadPolyline(
                layer=MapLayer.ROAD_EDGE,
                type=(
                    RoadEdgeType.ROAD_EDGE_MEDIAN
                    if ring.is_ccw and Polygon(ring).intersects(islands)
                    else RoadEdgeType.ROAD_EDGE_BOUNDARY
                ),
                polyline=xyz(ring.coords),
            )
            for i, ring in enumerate(rings)
        }
    )
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
            id=f"{style}_{size:g}m_{seed:06d}",
            dataset="mapforge",
            scenario_length=1,
            dt=0.1,
            location=f"synthetic_{style}",
            map_version="town-v2",
        ),
    )
    process_polylines(scenario)
    interpolate_all_polygons(scenario, tolerance=0.1)
    compute_lane_lengths(scenario)
    rules = intersection_rules(points, edges, arterial, connections, polylines, roundabouts, rng)
    phases = {}
    for kind, node, phase in rules.values():
        if kind == TCType.TRAFFIC_LIGHT:
            phases[node] = max(phases.get(node, 0), phase + 1)
    junction_ids = {
        node: min(
            (i for i, s in surfaces.items() if s.layer == MapLayer.INTERSECTION),
            key=lambda i: Polygon(surfaces[i].polygon[:, :2]).distance(Point(points[node])),
        )
        for node in phases
    }
    offsets = {node: int(rng.integers(count * 210)) for node, count in phases.items()}
    frames = max(1200, max(phases.values()) * 210) if phases else 1
    placements = [(lane_id, width, distance, TCType.YIELD_SIGN, -1, -1) for lane_id, width, distance in yield_lanes]
    for lane_id, edge, _, node, _ in connections:
        if lane_id not in rules:
            continue
        kind, node, phase = rules[lane_id]
        # Signals stop cars before the crossing; STOP/YIELD lines leave a waiting space after it.
        distance = polylines[lane_id].length - (8.8 if kind == TCType.TRAFFIC_LIGHT else 0.8)
        placements.append((lane_id, half[edge] / directions[edge].sum(), distance, kind, node, phase))
    controls = []
    for lane_id, half_width, distance, kind, node, phase in placements:
        line = LineString(polylines[lane_id].polyline[:, :2])
        center = np.array(line.interpolate(distance).coords[0])
        heading = np.array(line.interpolate(distance + 0.05).coords[0])
        heading -= np.array(line.interpolate(distance - 0.05).coords[0])
        heading /= np.linalg.norm(heading)
        n = np.array([-heading[1], heading[0]]) * half_width
        states = []
        if kind == TCType.TRAFFIC_LIGHT:
            step = (np.arange(frames) + offsets[node]) % (phases[node] * 210)
            own = step // 210 == phase
            states = np.where(
                own & (step % 210 < 160), TLState.GREEN, np.where(own & (step % 210 < 190), TLState.YELLOW, TLState.RED)
            ).tolist()
        controls.append(
            TrafficControl(
                id=len(controls),
                type=kind,
                stop_line=xyz([center - n, center + n]),
                heading=float(np.arctan2(heading[1], heading[0])),
                states=states,
                controlled_lanes=[lane_id],
                intersection_id=junction_ids.get(node, -1),
                phase_idx=phase,
            )
        )
    scenario.metadata.scenario_length = frames
    scenario.traffic_controls = controls
    scenario.lane_graph = build_lane_distance_matrix(polylines)
    return scenario


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate French-inspired towns as static PufferDinde binaries")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--style", choices=("mixed", "paris", "lyon"), default="mixed")
    parser.add_argument("--size", type=float, default=500.0, help="Town extent in metres (200..1200)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=1, help="Generate consecutive seeds")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    for seed in range(args.seed, args.seed + args.count):
        scenario = generate_town(args.style, args.size, seed)
        path = args.output_dir / f"{scenario.metadata.id}.bin"
        write_static_scenario(scenario, path, overwrite=args.overwrite)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
