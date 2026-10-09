"""Road edges along lane boundaries that border shoulder surfaces outside junctions: the shoulder stays in the
map (py123d keeps the layer) but PufferDrive treats it as off-road. Band samples whose probe lies inside an
intersection polygon or a junction shoulder are ignored, so shoulders inside a junction stay drivable and no edge
crosses a junction; samples whose probe lies inside another lane are blocked, so no edge crosses a lane that overlaps
a shoulder (ramp connectors); a transverse cap closes a band wherever it goes on without a following edge."""

import dataclasses

import numpy as np
import shapely
from shapely.strtree import STRtree

from bin_factory import puffer_types, schema

SAMPLE_SPACING_M = 2.0
PROBE_OFFSET_M = 0.3  # just outside the lane boundary
BAND_START_TOL_M = 0.8  # a band may begin this far from the boundary (interpolated curves, lane-line gaps)
MIN_BAND_WIDTH_M = 1.0  # a run starts where the band is at least this wide
KEEP_BAND_WIDTH_M = 0.6  # and goes on through narrowings down to this
MAX_BAND_WIDTH_M = 8.0
MIN_RUN_SAMPLES = 3
MAX_DIP_SAMPLES = 2  # short narrowings inside a band do not split it
LOOKAHEAD_M = (1.0, 2.0, 3.0)
END_INSET_M = 0.3  # lane-end samples are probed this far inside the lane, off shoulder-polygon corners
BISECTION_STEPS = 4  # run ends inside a lane are refined to 2 m / 2^4
Z_TOL_M = 3.0  # surfaces on another level (bridges, underpasses) are ignored
BLOCKED_WIDTH_M = -1.0  # band width where another lane covers the probe: never bridged, never snapped onto
STUB_LANE_LENGTH_M = 0.5  # seam connectors shorter than this carry no run; caps look through them
MAX_STUB_HOPS = 4


@dataclasses.dataclass(frozen=True)
class _Surroundings:
    shoulders: "_Surfaces"
    junctions: "_Surfaces"
    lanes: "_Surfaces"  # keyed by lane id, so a boundary is never tested against its own lane


def add_shoulder_edges(
    scenario: schema.PufferScenario, shoulder_outlines: list[np.ndarray], junction_outlines: list[np.ndarray]
) -> int:
    """Append one ROAD_EDGE_BOUNDARY per stretch of a lane boundary that runs along a shoulder band outside the
    junction surfaces, plus a cap across the band at every run end where the band goes on without a following run.
    Edges keep the drivable side on their right (py123d convention)."""
    shoulders = _Surfaces(shoulder_outlines)
    if not shoulders:
        return 0
    lanes = {i: e for i, e in scenario.map.items() if e.is_lane and e.polyline is not None}
    surroundings = _Surroundings(shoulders, _Surfaces(junction_outlines), _lane_surfaces(lanes))
    next_id = max(scenario.map.keys(), default=-1) + 1
    added = 0
    for lane_id, element in lanes.items():
        for boundary, lane_on_right in ((element.left_boundary, True), (element.right_boundary, False)):
            if boundary is None or len(boundary) < 2:
                continue
            samples, tangents, normals, widths = _boundary_bands(boundary, lane_id, element, surroundings)
            in_band = _band_mask(widths)
            beyond = [_band_ahead(shoulders, samples, tangents, normals, 0, -1.0),
                      _band_ahead(shoulders, samples, tangents, normals, len(samples) - 1, 1.0)]
            in_band = _snap_to_lane_ends(in_band, widths, beyond)
            min_samples = 2 if in_band.all() else MIN_RUN_SAMPLES  # a short lane fully along a band still gets its edge
            for start, end in _runs(in_band, min_samples):
                polyline = _run_polyline(surroundings, lane_id, samples, normals, start, end)
                scenario.map[next_id] = _edge(polyline if lane_on_right else polyline[::-1])
                next_id += 1
                added += 1
                for idx, sign, next_lanes in ((end - 1, 1.0, element.exit_lanes), (start, -1.0, element.entry_lanes)):
                    tangent = sign * tangents[idx]
                    band_ahead = _band_ahead(shoulders, samples, tangents, normals, idx, sign)
                    if band_ahead < KEEP_BAND_WIDTH_M:
                        continue  # the band ends or thins out: the kerb edge closes it
                    at_lane_end = idx == (len(samples) - 1 if sign > 0 else 0)
                    if not at_lane_end:
                        continue  # a run broken inside the lane continues after a gap in the same lane
                    next_ids = _through_stubs(lanes, next_lanes, sign > 0)
                    if any(_lane_has_run(i, lanes.get(i), surroundings) for i in next_ids):
                        continue  # the next lane carries the band on with its own run
                    width = widths[idx] if widths[idx] >= KEEP_BAND_WIDTH_M else band_ahead  # end sample on a corner sliver
                    scenario.map[next_id] = _edge(_cap(samples[idx], normals[idx], width, tangent))
                    next_id += 1
                    added += 1
    return added


def _through_stubs(lanes: dict, ids: list, forward: bool) -> list:
    """Lane ids with stub lanes replaced by their exits (forward) or entries (backward), a few hops deep."""
    result: list = []
    pending = list(ids)
    for _ in range(MAX_STUB_HOPS):
        stubs = [i for i in pending if i in lanes and _lane_length(lanes[i]) < STUB_LANE_LENGTH_M]
        result += [i for i in pending if i not in stubs]
        pending = [j for i in stubs for j in (lanes[i].exit_lanes if forward else lanes[i].entry_lanes)]
        if not pending:
            break
    return result + pending


def _lane_length(lane) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(lane.polyline)[:, :2], axis=0), axis=1).sum())


def _lane_surfaces(lanes: dict) -> "_Surfaces":
    keys, outlines = [], []
    for lane_id, lane in lanes.items():
        if lane.left_boundary is None or lane.right_boundary is None:
            continue
        if len(lane.left_boundary) < 2 or len(lane.right_boundary) < 2:
            continue
        left = np.asarray(lane.left_boundary, dtype=np.float64)
        right = np.asarray(lane.right_boundary, dtype=np.float64)
        columns = min(left.shape[1], right.shape[1])
        keys.append(lane_id)
        outlines.append(np.vstack([left[:, :columns], right[::-1, :columns]]))
    return _Surfaces(outlines, keys)


def _boundary_bands(boundary, lane_id, lane, surroundings):
    """(samples, tangents, outward normals, band width per sample) along a lane boundary; the width is 0 where the
    probe lies inside a junction surface of the same level and BLOCKED_WIDTH_M where it lies inside another lane."""
    samples = _resample(np.asarray(boundary, dtype=np.float64), SAMPLE_SPACING_M)
    tangents = _tangents(samples[:, :2])
    normals = _outward_normals(tangents, samples[:, :2], np.asarray(lane.polyline)[:, :2])
    probes = samples.copy()
    probes[0, :2] += END_INSET_M * tangents[0]
    probes[-1, :2] -= END_INSET_M * tangents[-1]
    widths = np.array([_band_width(surroundings, lane_id, p, n) for p, n in zip(probes, normals)])
    return samples, tangents, normals, widths


def _band_width(surroundings, lane_id, point_xyz: np.ndarray, normal: np.ndarray) -> float:
    """Band width at a boundary point, from the probe just outside the boundary."""
    probe = point_xyz.copy()
    probe[:2] += PROBE_OFFSET_M * normal
    if surroundings.lanes.containing(probe, exclude=lane_id):
        return BLOCKED_WIDTH_M
    if surroundings.junctions.containing(probe):
        return 0.0
    return _band(surroundings.shoulders, point_xyz, normal)[0]


def _lane_has_run(lane_id, lane, surroundings) -> bool:
    """Whether a boundary of `lane` would get a run (same mask rules as add_shoulder_edges)."""
    if lane is None or not lane.is_lane or lane.polyline is None:
        return False
    for boundary in (lane.left_boundary, lane.right_boundary):
        if boundary is None or len(boundary) < 2:
            continue
        samples, tangents, normals, widths = _boundary_bands(boundary, lane_id, lane, surroundings)
        in_band = _band_mask(widths)
        if _runs(in_band, 2 if in_band.all() else MIN_RUN_SAMPLES):
            return True
    return False


def _run_polyline(surroundings, lane_id, samples, normals, start: int, end: int) -> np.ndarray:
    """Samples of a run, extended to where the band actually begins and ends when the run stops inside the lane."""
    points = [samples[i] for i in range(start, end)]
    if start > 0:
        points.insert(0, _band_limit(surroundings, lane_id, samples, normals, start - 1, start))
    if end < len(samples):
        points.append(_band_limit(surroundings, lane_id, samples, normals, end, end - 1))
    return np.array(points)


def _band_limit(surroundings, lane_id, samples, normals, outside: int, inside: int) -> np.ndarray:
    """Point between an out-of-band sample and its in-band neighbour where the band begins, by bisection."""
    low, high = 0.0, 1.0
    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (low + high)
        point = samples[outside] + mid * (samples[inside] - samples[outside])
        normal = normals[outside] + mid * (normals[inside] - normals[outside])
        normal /= max(np.linalg.norm(normal), 1e-9)
        if _band_width(surroundings, lane_id, point, normal) >= KEEP_BAND_WIDTH_M:
            high = mid
        else:
            low = mid
    return samples[outside] + high * (samples[inside] - samples[outside])


def _band_ahead(shoulders, samples, tangents, normals, idx: int, sign: float) -> float:
    """Widest band found 1-3 m beyond sample `idx` in direction `sign` along the boundary."""
    tangent = sign * tangents[idx]
    best = 0.0
    for d in LOOKAHEAD_M:
        probe = samples[idx].copy()
        probe[:2] += d * tangent + PROBE_OFFSET_M * normals[idx]
        best = max(best, _band(shoulders, probe, normals[idx])[0])
    return best


def _edge(polyline: np.ndarray) -> schema.MapElement:
    return schema.MapElement(type=int(puffer_types.RoadEdgeType.BOUNDARY), polyline=np.array(polyline, dtype=np.float64))


class _Surfaces:
    """Polygons of surface outlines with a spatial index, per-vertex heights for the level test and optional keys."""

    def __init__(self, outlines: list[np.ndarray], keys: list | None = None):
        keys = list(keys) if keys is not None else [None] * len(outlines)
        triples = [(shapely.Polygon(o[:, :2]).buffer(0), np.asarray(o, dtype=np.float64), k) for o, k in zip(outlines, keys)]
        triples = [(polygon, outline, key) for polygon, outline, key in triples if not polygon.is_empty]
        self.polygons = [polygon for polygon, _, _ in triples]
        self.outlines = [outline for _, outline, _ in triples]
        self.keys = [key for _, _, key in triples]
        self.tree = STRtree(self.polygons) if self.polygons else None

    def __bool__(self):
        return bool(self.polygons)

    def same_level(self, index: int, point_xyz: np.ndarray) -> bool:
        outline = self.outlines[index]
        if outline.shape[1] < 3:
            return True
        nearest = np.argmin(np.linalg.norm(outline[:, :2] - point_xyz[:2], axis=1))
        return abs(outline[nearest, 2] - point_xyz[2]) <= Z_TOL_M

    def containing(self, point_xyz: np.ndarray, exclude=None) -> bool:
        """Whether a same-level surface other than the one keyed `exclude` contains the point."""
        if not self.polygons:
            return False
        point = shapely.Point(point_xyz[:2])
        return any(
            (exclude is None or self.keys[i] != exclude) and self.polygons[i].contains(point) and self.same_level(i, point_xyz)
            for i in self.tree.query(point)
        )


def _band(shoulders, point_xyz: np.ndarray, normal: np.ndarray):
    """(band width along the outward normal from the lane boundary point, far-side point); (0, None) when no band
    of the same level starts there."""
    point_xy = point_xyz[:2]
    start_xy = point_xy + 0.05 * normal
    ray = shapely.LineString([start_xy, point_xy + MAX_BAND_WIDTH_M * normal])
    candidates = [shoulders.polygons[i] for i in shoulders.tree.query(ray) if shoulders.same_level(i, point_xyz)]
    if not candidates:
        return 0.0, None
    covered = ray.intersection(shapely.union_all(candidates))
    if covered.is_empty:
        return 0.0, None
    parts = list(covered.geoms) if hasattr(covered, "geoms") else [covered]
    start = shapely.Point(start_xy)
    first = min(parts, key=lambda part: part.distance(start))
    if first.distance(start) > BAND_START_TOL_M or first.geom_type != "LineString":
        return 0.0, None
    coords = np.asarray(first.coords)
    far = coords[np.argmax(np.linalg.norm(coords - start_xy, axis=1))]
    return first.length + 0.05, far


def _cap(sample: np.ndarray, normal: np.ndarray, width: float, tangent_out: np.ndarray) -> np.ndarray:
    """Transverse edge across the band at a run end, with the continuing side (tangent_out) on its right."""
    inner = sample.copy()
    outer = sample.copy()
    outer[:2] = sample[:2] + width * normal
    right_of_normal = np.array([normal[1], -normal[0]])
    return np.array([inner, outer]) if np.dot(right_of_normal, tangent_out) > 0.0 else np.array([outer, inner])


def _resample(polyline: np.ndarray, spacing: float) -> np.ndarray:
    """Points every `spacing` metres along the polyline, first and last vertex included; a boundary shorter than one
    spacing gets its midpoint too, since its end points often sit on shoulder-polygon corners."""
    steps = np.linalg.norm(np.diff(polyline[:, :2], axis=0), axis=1)
    station = np.concatenate([[0.0], np.cumsum(steps)])
    count = max(2, int(np.ceil(station[-1] / spacing)) + 1) if station[-1] > spacing else 3
    targets = np.linspace(0.0, station[-1], count)
    return np.column_stack([np.interp(targets, station, polyline[:, k]) for k in range(polyline.shape[1])])


def _tangents(samples_xy: np.ndarray) -> np.ndarray:
    direction = np.gradient(samples_xy, axis=0)
    return direction / np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)


def _outward_normals(tangents: np.ndarray, samples_xy: np.ndarray, centerline_xy: np.ndarray) -> np.ndarray:
    """Unit normals pointing away from the lane centerline at every sample."""
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    distance = lambda points: np.min(np.linalg.norm(points[:, None, :] - centerline_xy[None, :, :], axis=2), axis=1)
    flip = distance(samples_xy + normals) < distance(samples_xy - normals)
    normals[flip] *= -1.0
    return normals


def _band_mask(widths: np.ndarray) -> np.ndarray:
    """Samples along a band: hysteresis on the width, dips bridged; a short boundary (3 samples) follows its midpoint,
    because its end samples sit on shoulder-polygon corners and measure unreliably."""
    if len(widths) == 3:
        return np.full(3, widths[1] >= MIN_BAND_WIDTH_M)
    return _bridge_dips(_hysteresis(widths, MIN_BAND_WIDTH_M, KEEP_BAND_WIDTH_M), MAX_DIP_SAMPLES, widths < 0.0)


def _hysteresis(widths: np.ndarray, start_m: float, keep_m: float) -> np.ndarray:
    """Samples at least `start_m` wide seed runs that extend over neighbours at least `keep_m` wide."""
    seeds = widths >= start_m
    keep = widths >= keep_m
    mask = seeds.copy()
    for i in range(1, len(mask)):
        if keep[i] and mask[i - 1]:
            mask[i] = True
    for i in range(len(mask) - 2, -1, -1):
        if keep[i] and mask[i + 1]:
            mask[i] = True
    return mask


def _bridge_dips(mask: np.ndarray, max_dip: int, blocked: np.ndarray) -> np.ndarray:
    """True runs separated by at most `max_dip` False samples are joined, unless a blocked sample lies between."""
    bridged = mask.copy()
    start = None
    for i, value in enumerate(mask):
        if value:
            if start is not None and 0 < i - start <= max_dip and not blocked[start:i].any():
                bridged[start:i] = True
            start = i + 1
    return bridged


def _snap_to_lane_ends(mask: np.ndarray, widths: np.ndarray, beyond: list[float]) -> np.ndarray:
    """A run reaching the second sample from a lane end covers the end too when a band is there or goes on past it
    (the end sample often sits on a shoulder polygon corner and measures nothing)."""
    snapped = mask.copy()
    if len(mask) < 2:
        return snapped
    for end, inner, ahead in ((0, 1, beyond[0]), (len(mask) - 1, len(mask) - 2, beyond[1])):
        reaches = widths[end] > 0.0 or (widths[end] == 0.0 and ahead >= KEEP_BAND_WIDTH_M)
        if mask[inner] and not mask[end] and reaches:
            snapped[end] = True
    return snapped


def _runs(mask: np.ndarray, min_samples: int) -> list[tuple[int, int]]:
    """(start, end) index pairs of the True runs of `mask` with at least `min_samples` entries."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        if not value and start is not None:
            if i - start >= min_samples:
                runs.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_samples:
        runs.append((start, len(mask)))
    return runs
