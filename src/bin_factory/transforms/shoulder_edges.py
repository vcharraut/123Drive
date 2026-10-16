"""Add road edges between lanes and shoulders outside junctions.

Keep shoulder geometry in the map. The added edges make PufferDrive treat those shoulders as
off-road. Skip probes inside intersections or junction shoulders to keep junctions drivable.
Block probes inside other lanes so edges do not cross overlapping ramp connectors.

Close a shoulder band with a transverse cap when the band continues without a following edge,
including into a junction, or ends at a lane end on road surface no kerb closes (filled corner
pockets). On one-way, single-lane ramps with shoulders on both sides, edges may sit outside the
lane boundary so long and wide vehicles fit through tight curves.
"""

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
MIN_POINT_SPACING_M = 0.05  # run points closer than this are one point
MAX_STUB_HOPS = 4
RAMP_BAND_CLEARANCE_M = 1.5  # a widened edge stays this far inside the band's far side, where posts and rails stand
RAMP_MIN_BAND_COVERAGE = 0.5  # a ramp has shoulder runs along at least this fraction of both boundaries
WIDENING_TAPER_SLOPE = 0.125  # metres of widening change per metre along the boundary


@dataclasses.dataclass(frozen=True)
class _Side:
    lane_on_right: bool
    samples: np.ndarray
    tangents: np.ndarray
    normals: np.ndarray
    widths: np.ndarray
    runs: list

    @property
    def coverage(self) -> float:
        return sum(end - start for start, end in self.runs) / len(self.samples)


@dataclasses.dataclass(frozen=True)
class _RunEnd:
    cap_width_m: float | None  # band width a cap closes at this end; None when no cap is needed
    hold: bool  # a widened edge keeps its offset up to this end (the band goes on into a junction or a widened lane)


@dataclasses.dataclass(frozen=True)
class _Surroundings:
    shoulders: "_Surfaces"
    junctions: "_Surfaces"
    lanes: "_Surfaces"  # keyed by lane id, so a boundary is never tested against its own lane
    kerbs: STRtree  # the map's road edges


def add_shoulder_edges(
    scenario: schema.PufferScenario,
    shoulder_outlines: list[np.ndarray],
    junction_outlines: list[np.ndarray],
    ramp_widening_m: float = 0.0,
) -> int:
    """Append ROAD_EDGE_BOUNDARY elements along shoulder bands outside junctions.

    Cap run ends where the band continues without another run. Ramp edges may sit up to
    ``ramp_widening_m`` outside the lane boundary. Keep the drivable side on the right,
    following the py123d convention.
    """
    shoulders = _Surfaces(shoulder_outlines)
    if not shoulders:
        return 0
    lanes = {i: e for i, e in scenario.map.items() if e.is_lane and e.polyline is not None}
    kerbs = STRtree(
        [shapely.LineString(e.polyline[:, :2]) for e in scenario.map.values() if e.is_edge and e.polyline is not None]
    )
    surroundings = _Surroundings(shoulders, _Surfaces(junction_outlines), _lane_surfaces(lanes), kerbs)
    continuing = _Surfaces(shoulder_outlines + junction_outlines)  # a band goes on through junction surfaces too
    sides = {lane_id: _lane_sides(lane_id, element, surroundings, continuing) for lane_id, element in lanes.items()}
    ramps = {lane_id for lane_id, element in lanes.items() if _is_ramp(element, sides[lane_id])}
    next_id = max(scenario.map.keys(), default=-1) + 1
    added = 0
    caps: set = set()  # two lanes ending on one band close it with the same cap
    for lane_id, element in lanes.items():
        widening_m = ramp_widening_m if lane_id in ramps else 0.0
        for side in sides[lane_id]:
            samples, tangents, normals = side.samples, side.tangents, side.normals
            for start, end in side.runs:
                run_ends = [
                    _run_end(lanes, ramps, surroundings, side, idx, sign, next_lanes)
                    for idx, sign, next_lanes in (
                        (start, -1.0, element.entry_lanes),
                        (end - 1, 1.0, element.exit_lanes),
                    )
                ]
                points, point_normals, point_widths = _run_points(surroundings, lane_id, side, start, end)
                offsets = _widening_targets(point_widths, widening_m, run_ends[0].hold, run_ends[1].hold)
                offsets = _tapered(points, _clear_of_other_lanes(surroundings, lane_id, points, point_normals, offsets))
                polyline = _distinct_points(_offset_points(points, point_normals, offsets))
                if len(polyline) < 2:
                    continue  # the run collapsed onto one point at the band limits
                scenario.map[next_id] = _edge(polyline if side.lane_on_right else polyline[::-1])
                next_id += 1
                added += 1
                run_end_sides = ((end - 1, 1.0, run_ends[1], offsets[-1]), (start, -1.0, run_ends[0], offsets[0]))
                for idx, sign, run_end, offset_m in run_end_sides:
                    if run_end.cap_width_m is None or run_end.cap_width_m - offset_m < MIN_POINT_SPACING_M:
                        continue
                    cap = _cap(samples[idx], normals[idx], offset_m, run_end.cap_width_m, sign * tangents[idx])
                    cap_key = frozenset(tuple(np.round(point[:2] / MIN_POINT_SPACING_M)) for point in cap)
                    if cap_key in caps:
                        continue
                    caps.add(cap_key)
                    scenario.map[next_id] = _edge(cap)
                    next_id += 1
                    added += 1
    return added


def _lane_sides(
    lane_id: int, lane: schema.MapElement, surroundings: _Surroundings, continuing: "_Surfaces"
) -> list[_Side]:
    """Band samples and shoulder runs along the boundaries of a lane."""
    sides = []
    for boundary, lane_on_right in ((lane.left_boundary, True), (lane.right_boundary, False)):
        if boundary is None or len(boundary) < 2:
            continue
        samples, tangents, normals, widths = _boundary_bands(boundary, lane_id, lane, surroundings)
        in_band = _band_mask(widths)
        beyond = [
            _band_ahead(continuing, samples, tangents, normals, 0, -1.0),
            _band_ahead(continuing, samples, tangents, normals, len(samples) - 1, 1.0),
        ]
        in_band = _snap_to_lane_ends(in_band, widths, beyond)
        min_samples = 2 if in_band.all() else MIN_RUN_SAMPLES  # a short lane fully along a band still gets its edge
        sides.append(_Side(lane_on_right, samples, tangents, normals, widths, _runs(in_band, min_samples)))
    return sides


def _is_ramp(lane: schema.MapElement, sides: list[_Side]) -> bool:
    """A one-way single-lane carriageway outside junctions: no lateral neighbours, shoulders along both boundaries
    (a two-way street with one lane per direction has the oncoming lane on one side instead)."""
    if lane.in_junction or lane.left_neighbor or lane.right_neighbor or len(sides) != 2:
        return False
    return all(side.coverage >= RAMP_MIN_BAND_COVERAGE for side in sides)


def _run_end(
    lanes: dict, ramps: set, surroundings: _Surroundings, side: _Side, idx: int, sign: float, next_lanes: list
) -> _RunEnd:
    """What happens to the band beyond the run end at sample `idx` (sign +1: run end, -1: run start)."""
    samples, tangents, normals, widths = side.samples, side.tangents, side.normals, side.widths
    band_ahead = _band_ahead(surroundings.shoulders, samples, tangents, normals, idx, sign)
    if band_ahead < KEEP_BAND_WIDTH_M:
        band_ahead = _band_ahead(surroundings.junctions, samples, tangents, normals, idx, sign)
    at_lane_end = idx == (len(samples) - 1 if sign > 0 else 0)
    width = widths[idx] if widths[idx] >= KEEP_BAND_WIDTH_M else _run_end_width(widths, idx, sign)
    if band_ahead < KEEP_BAND_WIDTH_M:
        if at_lane_end and not _kerb_ahead(surroundings, side, idx, sign, width):
            return _RunEnd(float(width), False)  # the band ends on other road surface (filled corner pocket)
        return _RunEnd(None, False)  # the band ends or thins out: the kerb edge closes it
    if not at_lane_end:  # a run broken inside the lane continues after a gap in the same lane
        outside = idx + int(sign)
        probe = samples[outside].copy()
        probe[:2] += PROBE_OFFSET_M * normals[outside]
        return _RunEnd(None, surroundings.junctions.containing(probe))
    next_ids = _through_stubs(lanes, next_lanes, sign > 0)
    carrying = [i for i in next_ids if _lane_has_run(i, lanes.get(i), surroundings)]
    if carrying:  # the next lane carries the band on with its own run
        return _RunEnd(None, all(i in ramps for i in carrying))
    return _RunEnd(float(width), True)


def _kerb_ahead(surroundings: _Surroundings, side: _Side, idx: int, sign: float, width: float) -> bool:
    """Whether a road edge crosses the middle of the band from END_INSET_M before to LOOKAHEAD_M beyond sample `idx`."""
    tangent = sign * side.tangents[idx]
    middle = side.samples[idx, :2] + 0.5 * width * side.normals[idx]
    probe = shapely.LineString([middle - END_INSET_M * tangent, middle + LOOKAHEAD_M[-1] * tangent])
    return len(surroundings.kerbs.query(probe, predicate="intersects")) > 0


def _run_end_width(widths: np.ndarray, idx: int, sign: float) -> float:
    """Band width of the last run sample before `idx` that measured a band (the end sample sits on a corner sliver)."""
    step = -1 if sign > 0 else 1
    for i in range(idx + step, -1 if step < 0 else len(widths), step):
        if widths[i] >= KEEP_BAND_WIDTH_M:
            return float(widths[i])
    return KEEP_BAND_WIDTH_M


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


def _lane_length(lane: schema.MapElement) -> float:
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


def _boundary_bands(
    boundary: np.ndarray, lane_id: int, lane: schema.MapElement, surroundings: _Surroundings
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(samples, tangents, outward normals, band width per sample) along a lane boundary; the width is 0 where the
    probe lies inside a junction surface of the same level and BLOCKED_WIDTH_M where it lies inside another lane."""
    samples = _resample(np.asarray(boundary, dtype=np.float64), SAMPLE_SPACING_M)
    tangents = _tangents(samples[:, :2])
    normals = _outward_normals(tangents, samples[:, :2], np.asarray(lane.polyline)[:, :2])
    probes = samples.copy()
    probes[0, :2] += END_INSET_M * tangents[0]
    probes[-1, :2] -= END_INSET_M * tangents[-1]
    widths = np.array([_band_width(surroundings, lane_id, p, n) for p, n in zip(probes, normals, strict=True)])
    return samples, tangents, normals, widths


def _band_width(surroundings: _Surroundings, lane_id: int, point_xyz: np.ndarray, normal: np.ndarray) -> float:
    """Band width at a boundary point, from the probe just outside the boundary."""
    probe = point_xyz.copy()
    probe[:2] += PROBE_OFFSET_M * normal
    if surroundings.lanes.containing(probe, exclude=lane_id):
        return BLOCKED_WIDTH_M
    if surroundings.junctions.containing(probe):
        return 0.0
    return _band(surroundings.shoulders, point_xyz, normal)[0]


def _lane_has_run(lane_id: int, lane: schema.MapElement | None, surroundings: _Surroundings) -> bool:
    """Whether a boundary of `lane` would get a run (same mask rules as add_shoulder_edges)."""
    if lane is None or not lane.is_lane or lane.polyline is None:
        return False
    for boundary in (lane.left_boundary, lane.right_boundary):
        if boundary is None or len(boundary) < 2:
            continue
        widths = _boundary_bands(boundary, lane_id, lane, surroundings)[3]
        in_band = _band_mask(widths)
        if _runs(in_band, 2 if in_band.all() else MIN_RUN_SAMPLES):
            return True
    return False


def _run_points(
    surroundings: _Surroundings, lane_id: int, side: _Side, start: int, end: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points, outward normals, band widths) of a run: its samples, extended to where the band actually begins and
    ends when the run stops inside the lane; band-limit points get a NaN width."""
    samples, normals, widths = side.samples, side.normals, side.widths
    points = [samples[i] for i in range(start, end)]
    point_normals = [normals[i] for i in range(start, end)]
    point_widths = [float(widths[i]) for i in range(start, end)]
    if start > 0:
        points.insert(0, _band_limit(surroundings, lane_id, samples, normals, start - 1, start))
        point_normals.insert(0, normals[start])
        point_widths.insert(0, np.nan)
    if end < len(samples):
        points.append(_band_limit(surroundings, lane_id, samples, normals, end, end - 1))
        point_normals.append(normals[end - 1])
        point_widths.append(np.nan)
    return np.array(points), np.array(point_normals), np.array(point_widths)


def _widening_targets(point_widths: np.ndarray, widening_m: float, hold_start: bool, hold_end: bool) -> np.ndarray:
    """Outward offset of every run point: up to widening_m but RAMP_BAND_CLEARANCE_M short of the band's far
    side, 0 where another lane blocks the band and at run ends where the band closes. Points that measured no band
    (corner slivers, junction probes, band limits) take the nearest measured value."""
    offsets = np.zeros(len(point_widths))
    if widening_m <= 0.0:
        return offsets
    measured = point_widths >= KEEP_BAND_WIDTH_M
    if not measured.any():
        return offsets
    offsets = np.full(len(point_widths), np.nan)
    offsets[measured] = np.clip(point_widths[measured] - RAMP_BAND_CLEARANCE_M, 0.0, widening_m)
    offsets[point_widths < 0.0] = 0.0
    known = np.flatnonzero(~np.isnan(offsets))
    nearest = known[np.abs(np.arange(len(offsets))[:, None] - known[None, :]).argmin(axis=1)]
    offsets = offsets[nearest]
    if not hold_start:
        offsets[0] = 0.0
    if not hold_end:
        offsets[-1] = 0.0
    return offsets


def _clear_of_other_lanes(
    surroundings: _Surroundings, lane_id: int, points: np.ndarray, point_normals: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """Offsets dropped to 0 where the widened point would lie inside another lane of the same level."""
    cleared = offsets.copy()
    for i in np.flatnonzero(offsets > 0.0):
        target = points[i].copy()
        target[:2] += offsets[i] * point_normals[i]
        if surroundings.lanes.containing(target, exclude=lane_id):
            cleared[i] = 0.0
    return cleared


def _tapered(points: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Offsets limited to change by WIDENING_TAPER_SLOPE per metre along the run, lowering the larger side."""
    if not (offsets > 0.0).any():
        return offsets
    steps = WIDENING_TAPER_SLOPE * np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    tapered = offsets.copy()
    for i in range(1, len(tapered)):
        tapered[i] = min(tapered[i], tapered[i - 1] + steps[i - 1])
    for i in range(len(tapered) - 2, -1, -1):
        tapered[i] = min(tapered[i], tapered[i + 1] + steps[i])
    return tapered


def _offset_points(points: np.ndarray, point_normals: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    shifted = points.copy()
    shifted[:, :2] += offsets[:, None] * point_normals
    return shifted


def _distinct_points(polyline: np.ndarray) -> np.ndarray:
    """Polyline without consecutive points closer than MIN_POINT_SPACING_M."""
    kept = [polyline[0]]
    for point in polyline[1:]:
        if np.linalg.norm(point[:2] - kept[-1][:2]) >= MIN_POINT_SPACING_M:
            kept.append(point)
    return np.array(kept)


def _band_limit(
    surroundings: _Surroundings, lane_id: int, samples: np.ndarray, normals: np.ndarray, outside: int, inside: int
) -> np.ndarray:
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


def _band_ahead(
    shoulders: "_Surfaces", samples: np.ndarray, tangents: np.ndarray, normals: np.ndarray, idx: int, sign: float
) -> float:
    """Widest band found 1-3 m beyond sample `idx` in direction `sign` along the boundary."""
    tangent = sign * tangents[idx]
    best = 0.0
    for d in LOOKAHEAD_M:
        probe = samples[idx].copy()
        probe[:2] += d * tangent + PROBE_OFFSET_M * normals[idx]
        best = max(best, _band(shoulders, probe, normals[idx])[0])
    return best


def _edge(polyline: np.ndarray) -> schema.MapElement:
    edge_type = int(puffer_types.RoadEdgeType.BOUNDARY)
    return schema.MapElement(type=edge_type, polyline=np.array(polyline, dtype=np.float64))


class _Surfaces:
    """Polygons of surface outlines with a spatial index, per-vertex heights for the level test and optional keys."""

    def __init__(self, outlines: list[np.ndarray], keys: list | None = None) -> None:
        keys = list(keys) if keys is not None else [None] * len(outlines)
        triples = [
            (shapely.Polygon(o[:, :2]).buffer(0), np.asarray(o, dtype=np.float64), k)
            for o, k in zip(outlines, keys, strict=True)
        ]
        triples = [(polygon, outline, key) for polygon, outline, key in triples if not polygon.is_empty]
        self.polygons = [polygon for polygon, _, _ in triples]
        self.outlines = [outline for _, outline, _ in triples]
        self.keys = [key for _, _, key in triples]
        self.tree = STRtree(self.polygons) if self.polygons else None

    def __bool__(self) -> bool:
        return bool(self.polygons)

    def same_level(self, index: int, point_xyz: np.ndarray) -> bool:
        outline = self.outlines[index]
        if outline.shape[1] < 3:
            return True
        nearest = np.argmin(np.linalg.norm(outline[:, :2] - point_xyz[:2], axis=1))
        return abs(outline[nearest, 2] - point_xyz[2]) <= Z_TOL_M

    def containing(self, point_xyz: np.ndarray, exclude: int | None = None) -> bool:
        """Whether a same-level surface other than the one keyed `exclude` contains the point."""
        if self.tree is None:
            return False
        point = shapely.Point(point_xyz[:2])
        return any(
            (exclude is None or self.keys[i] != exclude)
            and self.polygons[i].contains(point)
            and self.same_level(i, point_xyz)
            for i in self.tree.query(point)
        )


def _band(shoulders: _Surfaces, point_xyz: np.ndarray, normal: np.ndarray) -> tuple[float, np.ndarray | None]:
    """(band width along the outward normal from the lane boundary point, far-side point); (0, None) when no band
    of the same level starts there."""
    if shoulders.tree is None:
        return 0.0, None
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


def _cap(sample: np.ndarray, normal: np.ndarray, inner_m: float, outer_m: float, tangent_out: np.ndarray) -> np.ndarray:
    """Transverse edge across the band at a run end, from inner_m to outer_m out of the boundary, with the continuing
    side (tangent_out) on its right."""
    inner = sample.copy()
    outer = sample.copy()
    inner[:2] = sample[:2] + inner_m * normal
    outer[:2] = sample[:2] + outer_m * normal
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

    def distance(points: np.ndarray) -> np.ndarray:
        return np.min(np.linalg.norm(points[:, None, :] - centerline_xy[None, :, :], axis=2), axis=1)

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
