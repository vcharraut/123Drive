"""Add road edges between lanes and shoulders outside junctions.

Keep shoulder geometry in the map. The added edges make PufferDrive treat those shoulders as
off-road. Skip probes inside intersections or junction shoulders to keep junctions drivable.
Block probes inside other lanes so edges do not cross overlapping ramp connectors.

Close a shoulder band with a transverse cap if it continues without a following edge, including
into a junction. Also cap bands that end at a lane end on road surface without a closing kerb.
On one-way, single-lane ramps with shoulders on both sides, edges may sit outside the lane
boundary so long and wide vehicles fit through tight curves.
"""

import dataclasses

import numpy as np
import shapely
from shapely.strtree import STRtree

from bin_factory import puffer_types, schema


SAMPLE_SPACING_M = 2.0
PROBE_OFFSET_M = 0.3  # just outside the lane boundary
BAND_START_TOL_M = 0.8  # a band may begin this far from the boundary on interpolated curves or lane-line gaps
MIN_BAND_WIDTH_M = 1.0  # a run starts where the band is at least this wide
KEEP_BAND_WIDTH_M = 0.6  # and goes on through narrowings down to this
MAX_BAND_WIDTH_M = 8.0
MIN_RUN_SAMPLES = 3
MAX_DIP_SAMPLES = 2  # short narrowings inside a band do not split it
LOOKAHEAD_M = (1.0, 2.0, 3.0)
END_INSET_M = 0.3  # lane-end samples are probed this far inside the lane, off shoulder-polygon corners
BISECTION_STEPS = 4  # run ends inside a lane are refined to 2 m / 2^4
Z_TOL_M = 3.0  # ignore surfaces on another level such as bridges or underpasses
BLOCKED_WIDTH_M = -1.0  # band width where another lane covers the probe, never bridged or snapped onto
STUB_LANE_LENGTH_M = 0.5  # seam connectors shorter than this carry no run and caps look through them
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
    has_unsnapped_run: bool = False

    @property
    def coverage(self) -> float:
        return sum(end - start for start, end in self.runs) / len(self.samples)


@dataclasses.dataclass(frozen=True)
class _RunEnd:
    cap_width_m: float | None  # band width a cap closes at this end, or None when no cap is needed
    hold: bool  # keep a widened edge's offset up to this end when the band continues into a junction or widened lane


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
    lanes = {i: e for i, e in scenario.polylines.items() if e.is_lane and e.polyline is not None}
    kerbs = STRtree(
        [
            shapely.LineString(e.polyline[:, :2])
            for e in scenario.polylines.values()
            if e.is_edge and e.polyline is not None
        ]
    )
    surroundings = _Surroundings(shoulders, _Surfaces(junction_outlines), _lane_surfaces(lanes), kerbs)
    continuing = _Surfaces(shoulder_outlines + junction_outlines)  # a band goes on through junction surfaces too
    sides = {lane_id: _lane_sides(lane_id, element, surroundings, continuing) for lane_id, element in lanes.items()}
    lanes_with_runs = {lane_id for lane_id, s_list in sides.items() if any(s.has_unsnapped_run for s in s_list)}
    ramps = {lane_id for lane_id, element in lanes.items() if _is_ramp(element, sides[lane_id])}
    first_id = next_id = max([*scenario.polylines, *scenario.surfaces], default=-1) + 1
    caps: set = set()  # two lanes ending on one band close it with the same cap
    runs = [
        (lane_id, element, side, run)
        for lane_id, element in lanes.items()
        for side in sides[lane_id]
        for run in side.runs
    ]
    for lane_id, element, side, (start, end) in runs:
        run_ends = [
            _run_end(lanes, ramps, surroundings, side, idx, sign, next_lanes, lanes_with_runs)
            for idx, sign, next_lanes in ((start, -1.0, element.entry_lanes), (end - 1, 1.0, element.exit_lanes))
        ]
        points, point_normals, point_widths = _run_points(surroundings, lane_id, side, start, end)
        offsets = _widening_targets(
            point_widths, ramp_widening_m if lane_id in ramps else 0.0, run_ends[0].hold, run_ends[1].hold
        )
        offsets = _tapered(points, _clear_of_other_lanes(surroundings, lane_id, points, point_normals, offsets))
        shifted = points.copy()
        shifted[:, :2] += offsets[:, None] * point_normals
        polyline = _distinct_points(shifted)
        if len(polyline) < 2:
            continue  # the run collapsed onto one point at the band limits
        scenario.polylines[next_id] = _edge(polyline if side.lane_on_right else polyline[::-1])
        next_id += 1
        # Cap the run's end, then its start
        for idx, sign, run_end, offset_m in (
            (end - 1, 1.0, run_ends[1], offsets[-1]),
            (start, -1.0, run_ends[0], offsets[0]),
        ):
            if run_end.cap_width_m is None or run_end.cap_width_m - offset_m < MIN_POINT_SPACING_M:
                continue
            cap = _cap(side.samples[idx], side.normals[idx], offset_m, run_end.cap_width_m, sign * side.tangents[idx])
            cap_key = frozenset(tuple(np.round(point[:2] / MIN_POINT_SPACING_M)) for point in cap)
            if cap_key not in caps:
                caps.add(cap_key)
                scenario.polylines[next_id] = _edge(cap)
                next_id += 1
    return next_id - first_id


def _lane_sides(
    lane_id: int, lane: schema.RoadPolyline, surroundings: _Surroundings, continuing: "_Surfaces"
) -> list[_Side]:
    """Band samples and shoulder runs along the boundaries of a lane."""
    sides = []
    for boundary, lane_on_right in ((lane.left_boundary, True), (lane.right_boundary, False)):
        if boundary is None or len(boundary) < 2:
            continue
        samples, tangents, normals, widths = _boundary_bands(boundary, lane_id, lane, surroundings)
        in_band = _band_mask(widths)
        has_unsnapped = bool(_runs(in_band, 2 if in_band.all() else MIN_RUN_SAMPLES))
        beyond = [
            _band_ahead(continuing, samples, tangents, normals, 0, -1.0)
            if len(in_band) >= 2 and in_band[1] and not in_band[0] and widths[0] == 0.0
            else 0.0,
            _band_ahead(continuing, samples, tangents, normals, len(samples) - 1, 1.0)
            if len(in_band) >= 2 and in_band[-2] and not in_band[-1] and widths[-1] == 0.0
            else 0.0,
        ]
        in_band = _snap_to_lane_ends(in_band, widths, beyond)
        min_samples = 2 if in_band.all() else MIN_RUN_SAMPLES  # a short lane fully along a band still gets its edge
        sides.append(
            _Side(lane_on_right, samples, tangents, normals, widths, _runs(in_band, min_samples), has_unsnapped)
        )
    return sides


def _is_ramp(lane: schema.RoadPolyline, sides: list[_Side]) -> bool:
    """Check for a one-way, single-lane carriageway outside junctions.

    It must have no lateral neighbours and shoulders along both boundaries.
    A two-way street with one lane per direction has an oncoming lane on one side instead.
    """
    if lane.in_junction or lane.left_neighbor or lane.right_neighbor or len(sides) != 2:
        return False
    return all(side.coverage >= RAMP_MIN_BAND_COVERAGE for side in sides)


def _run_end(
    lanes: dict,
    ramps: set,
    surroundings: _Surroundings,
    side: _Side,
    idx: int,
    sign: float,
    next_lanes: list,
    lanes_with_runs: set[int],
) -> _RunEnd:
    """Determine whether to cap the band and keep its widening at sample `idx`.

    A `sign` of +1 checks the run end, and -1 checks the run start.
    """
    samples, tangents, normals, widths = side.samples, side.tangents, side.normals, side.widths
    band_ahead = _band_ahead(surroundings.shoulders, samples, tangents, normals, idx, sign)
    if band_ahead < KEEP_BAND_WIDTH_M:
        band_ahead = _band_ahead(surroundings.junctions, samples, tangents, normals, idx, sign)
    at_lane_end = idx == (len(samples) - 1 if sign > 0 else 0)
    width = widths[idx] if widths[idx] >= KEEP_BAND_WIDTH_M else _run_end_width(widths, idx, sign)
    if band_ahead < KEEP_BAND_WIDTH_M:
        if at_lane_end and not _kerb_ahead(surroundings, side, idx, sign, width):
            return _RunEnd(float(width), False)  # the band ends on other road surface such as a filled corner pocket
        return _RunEnd(None, False)  # the band ends or thins out and the kerb edge closes it
    if not at_lane_end:  # a run broken inside the lane continues after a gap in the same lane
        outside = idx + int(sign)
        probe = samples[outside].copy()
        probe[:2] += PROBE_OFFSET_M * normals[outside]
        return _RunEnd(None, surroundings.junctions.containing(probe))
    carrying = [i for i in _through_stubs(lanes, next_lanes, sign > 0) if i in lanes_with_runs]
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
    """Return the last measured band width before `idx`. The end sample may lie on a corner sliver."""
    step = -1 if sign > 0 else 1
    for i in range(idx + step, -1 if step < 0 else len(widths), step):
        if widths[i] >= KEEP_BAND_WIDTH_M:
            return float(widths[i])
    return KEEP_BAND_WIDTH_M


def _through_stubs(lanes: dict, ids: list, forward: bool) -> list:
    """Replace stub lane IDs with their exits or entries, up to MAX_STUB_HOPS forward or backward."""
    result: list = []
    pending = list(ids)
    for _ in range(MAX_STUB_HOPS):
        stubs = [i for i in pending if i in lanes and _lane_length(lanes[i]) < STUB_LANE_LENGTH_M]
        result += [i for i in pending if i not in stubs]
        pending = [j for i in stubs for j in (lanes[i].exit_lanes if forward else lanes[i].entry_lanes)]
        if not pending:
            break
    return result + pending


def _lane_length(lane: schema.RoadPolyline) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(lane.polyline)[:, :2], axis=0), axis=1).sum())


def _lane_surfaces(lanes: dict) -> "_Surfaces":
    keys, outlines = [], []
    for lane_id, lane in lanes.items():
        left, right = lane.left_boundary, lane.right_boundary
        if left is None or right is None or len(left) < 2 or len(right) < 2:
            continue
        left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
        columns = min(left.shape[1], right.shape[1])
        keys.append(lane_id)
        outlines.append(np.vstack([left[:, :columns], right[::-1, :columns]]))
    return _Surfaces(outlines, keys)


def _boundary_bands(
    boundary: np.ndarray, lane_id: int, lane: schema.RoadPolyline, surroundings: _Surroundings
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return samples, tangents, outward normals, and band widths along a lane boundary.

    Set width to 0 inside junction surfaces at the same elevation, or BLOCKED_WIDTH_M inside another lane.
    """
    samples = _resample(np.asarray(boundary, dtype=np.float64), SAMPLE_SPACING_M)
    tangents = _tangents(samples[:, :2])
    normals = _outward_normals(tangents, samples[:, :2], np.asarray(lane.polyline)[:, :2])
    probes = samples.copy()
    probes[0, :2] += END_INSET_M * tangents[0]
    probes[-1, :2] -= END_INSET_M * tangents[-1]
    probe_pts = probes.copy()
    probe_pts[:, :2] += PROBE_OFFSET_M * normals
    blocked = surroundings.lanes.containing_mask(probe_pts, exclude=lane_id)
    widths = np.where(blocked, BLOCKED_WIDTH_M, 0.0)
    unblocked = np.flatnonzero(~blocked)
    if len(unblocked) == 0 or surroundings.shoulders.tree is None:
        return samples, tangents, normals, widths
    in_junction = surroundings.junctions.containing_mask(probe_pts[unblocked])
    active = unblocked[~in_junction]
    if len(active) == 0:
        return samples, tangents, normals, widths
    starts_xy = probes[active, :2] + 0.05 * normals[active]
    ends_xy = probes[active, :2] + MAX_BAND_WIDTH_M * normals[active]
    rays = shapely.linestrings(np.stack([starts_xy, ends_xy], axis=1))
    r_idx, s_idx = surroundings.shoulders.tree.query(rays, predicate="intersects")
    hits_by_ray: dict[int, list[int]] = {}
    for r_i, s_i in zip(r_idx.tolist(), s_idx.tolist(), strict=False):
        hits_by_ray.setdefault(r_i, []).append(s_i)
    for r_i, s_list in hits_by_ray.items():
        idx = int(active[r_i])
        widths[idx] = _band(surroundings.shoulders, probes[idx], normals[idx], rays[r_i], s_list)[0]
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


def _run_points(
    surroundings: _Surroundings, lane_id: int, side: _Side, start: int, end: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a run's points, outward normals, and band widths.

    Extend runs that stop inside a lane to the band's limits. Assign NaN widths to those limit points.
    """
    samples, normals, widths = side.samples, side.normals, side.widths
    points, point_normals, point_widths = list(samples[start:end]), list(normals[start:end]), widths[start:end].tolist()
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
    """Return outward offsets up to widening_m, keeping RAMP_BAND_CLEARANCE_M from the band's far side.

    Set offsets to 0 where another lane blocks the band and at run ends where the band closes.
    Use the nearest measured offset for corner slivers, junction probes, and band limits with no measurement.
    """
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
    """Set offsets to 0 where widening would put a point inside another lane at the same elevation."""
    cleared = offsets.copy()
    for i in np.flatnonzero(offsets > 0.0):
        target = points[i].copy()
        target[:2] += offsets[i] * point_normals[i]
        if surroundings.lanes.containing(target, exclude=lane_id):
            cleared[i] = 0.0
    return cleared


def _tapered(points: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Limit offset changes to WIDENING_TAPER_SLOPE per metre by lowering the larger offset."""
    if not (offsets > 0.0).any():
        return offsets
    steps = WIDENING_TAPER_SLOPE * np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    tapered = offsets.copy()
    for i in range(1, len(tapered)):
        tapered[i] = min(tapered[i], tapered[i - 1] + steps[i - 1])
    for i in range(len(tapered) - 2, -1, -1):
        tapered[i] = min(tapered[i], tapered[i + 1] + steps[i])
    return tapered


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
    """Bisect the interval between an out-of-band sample and its in-band neighbour to find the band's start."""
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


def _edge(polyline: np.ndarray) -> schema.RoadPolyline:
    return schema.RoadPolyline(
        layer=int(puffer_types.MapLayer.ROAD_EDGE),
        type=int(puffer_types.RoadEdgeType.ROAD_EDGE_BOUNDARY),
        polyline=np.array(polyline, dtype=np.float64),
    )


class _Surfaces:
    """Polygons of surface outlines with a spatial index, per-vertex heights for the level test and optional keys."""

    def __init__(self, outlines: list[np.ndarray], keys: list | None = None) -> None:
        keys = list(keys) if keys is not None else [None] * len(outlines)
        polygons = [shapely.Polygon(o[:, :2]).buffer(0) for o in outlines]
        kept = [i for i, polygon in enumerate(polygons) if not polygon.is_empty]
        self.polygons = [polygons[i] for i in kept]
        self.outlines = [np.asarray(outlines[i], dtype=np.float64) for i in kept]
        self.keys = [keys[i] for i in kept]
        self.z_bounds = [
            (float(o[:, 2].min()), float(o[:, 2].max())) if o.shape[1] >= 3 else None for o in self.outlines
        ]
        self.tree = STRtree(self.polygons) if self.polygons else None

    def __bool__(self) -> bool:
        return bool(self.polygons)

    def same_level(self, index: int, point_xyz: np.ndarray) -> bool:
        bounds = self.z_bounds[index]
        if bounds is None:
            return True
        z = float(point_xyz[2])
        z_min, z_max = bounds
        if z < z_min - Z_TOL_M or z > z_max + Z_TOL_M:
            return False
        if z >= z_max - Z_TOL_M and z <= z_min + Z_TOL_M:
            return True
        outline = self.outlines[index]
        diff = outline[:, :2] - point_xyz[:2]
        nearest = int(np.argmin(diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]))
        return abs(outline[nearest, 2] - z) <= Z_TOL_M

    def containing(self, point_xyz: np.ndarray, exclude: int | None = None) -> bool:
        """Whether a same-level surface other than the one keyed `exclude` contains the point."""
        if self.tree is None:
            return False
        point = shapely.Point(point_xyz[:2])
        return any(
            (exclude is None or self.keys[i] != exclude)
            and self.same_level(i, point_xyz)
            and self.polygons[i].contains(point)
            for i in self.tree.query(point)
        )

    def containing_mask(self, points_xyz: np.ndarray, exclude: int | None = None) -> np.ndarray:
        mask = np.zeros(len(points_xyz), dtype=bool)
        if self.tree is None or len(points_xyz) == 0:
            return mask
        pts = shapely.points(points_xyz[:, :2])
        pt_idx, poly_idx = self.tree.query(pts, predicate="within")
        for p_i, s_i in zip(pt_idx.tolist(), poly_idx.tolist(), strict=False):
            if (
                not mask[p_i]
                and (exclude is None or self.keys[s_i] != exclude)
                and self.same_level(s_i, points_xyz[p_i])
            ):
                mask[p_i] = True
        return mask


def _band(
    shoulders: _Surfaces,
    point_xyz: np.ndarray,
    normal: np.ndarray,
    ray: shapely.LineString | None = None,
    hit_indices: list[int] | None = None,
) -> tuple[float, np.ndarray | None]:
    """Return the band width along the outward normal and the far-side point.

    Return (0, None) if no band at the same elevation starts at this boundary point.
    """
    if shoulders.tree is None:
        return 0.0, None
    point_xy = point_xyz[:2]
    start_xy = point_xy + 0.05 * normal
    if ray is None:
        ray = shapely.LineString([start_xy, point_xy + MAX_BAND_WIDTH_M * normal])
    indices = hit_indices if hit_indices is not None else shoulders.tree.query(ray, predicate="intersects").tolist()
    candidates = [shoulders.polygons[i] for i in indices if shoulders.same_level(i, point_xyz)]
    if not candidates:
        return 0.0, None
    covered = (
        ray.intersection(candidates[0]) if len(candidates) == 1 else ray.intersection(shapely.union_all(candidates))
    )
    if covered.is_empty:
        return 0.0, None
    parts = list(covered.geoms) if hasattr(covered, "geoms") else [covered]
    start = shapely.Point(start_xy)
    first = min(parts, key=lambda part: part.distance(start))
    if first.distance(start) > BAND_START_TOL_M or first.geom_type != "LineString":
        return 0.0, None
    coords = np.asarray(first.coords)
    far = coords[np.argmax(((coords - start_xy) ** 2).sum(axis=1))]
    return first.length + 0.05, far


def _cap(sample: np.ndarray, normal: np.ndarray, inner_m: float, outer_m: float, tangent_out: np.ndarray) -> np.ndarray:
    """Return an edge across the band at a run end, from inner_m to outer_m outside the boundary.

    Orient the edge so the continuing side, tangent_out, lies on its right.
    """
    inner = sample.copy()
    outer = sample.copy()
    inner[:2] = sample[:2] + inner_m * normal
    outer[:2] = sample[:2] + outer_m * normal
    right_of_normal = np.array([normal[1], -normal[0]])
    return np.array([inner, outer]) if np.dot(right_of_normal, tangent_out) > 0.0 else np.array([outer, inner])


def _resample(polyline: np.ndarray, spacing: float) -> np.ndarray:
    """Sample the polyline at intervals of at most `spacing` metres, including both endpoints.

    Include the midpoint when the boundary is no longer than `spacing`, since endpoints often lie on polygon corners.
    """
    station = schema.arc_length(polyline[:, :2])
    count = max(2, int(np.ceil(station[-1] / spacing)) + 1) if station[-1] > spacing else 3
    targets = np.linspace(0.0, station[-1], count)
    return np.column_stack([np.interp(targets, station, polyline[:, k]) for k in range(polyline.shape[1])])


def _tangents(samples_xy: np.ndarray) -> np.ndarray:
    direction = np.gradient(samples_xy, axis=0)
    return direction / np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)


def _outward_normals(tangents: np.ndarray, samples_xy: np.ndarray, centerline_xy: np.ndarray) -> np.ndarray:
    """Unit normals pointing away from the lane centerline at every sample."""
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    def dist_sq(points: np.ndarray) -> np.ndarray:
        diff = points[:, None, :] - centerline_xy[None, :, :]
        return np.min(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2, axis=1)

    flip = dist_sq(samples_xy + normals) < dist_sq(samples_xy - normals)
    normals[flip] *= -1.0
    return normals


def _band_mask(widths: np.ndarray) -> np.ndarray:
    """Mark band samples using width hysteresis, bridging short gaps.

    For a three-sample boundary, use its midpoint width. Endpoints on shoulder-polygon corners give unreliable widths.
    """
    if len(widths) == 3:
        return np.full(3, widths[1] >= MIN_BAND_WIDTH_M)
    return _bridge_dips(_hysteresis(widths, MIN_BAND_WIDTH_M, KEEP_BAND_WIDTH_M), MAX_DIP_SAMPLES, widths < 0.0)


def _hysteresis(widths: np.ndarray, start_m: float, keep_m: float) -> np.ndarray:
    """Start runs at widths of at least `start_m` and extend through neighbours at least `keep_m` wide."""
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
    """Join True runs separated by at most `max_dip` False samples, unless a blocked sample lies between."""
    bridged = mask.copy()
    start = None
    for i, value in enumerate(mask):
        if value:
            if start is not None and 0 < i - start <= max_dip and not blocked[start:i].any():
                bridged[start:i] = True
            start = i + 1
    return bridged


def _snap_to_lane_ends(mask: np.ndarray, widths: np.ndarray, beyond: list[float]) -> np.ndarray:
    """Extend a run from the second sample to the lane end if a band exists there or continues beyond it.

    The end sample often lies on a shoulder polygon corner and measures no width.
    """
    snapped = mask.copy()
    if len(mask) < 2:
        return snapped
    for end, inner, ahead in ((0, 1, beyond[0]), (len(mask) - 1, len(mask) - 2, beyond[1])):
        reaches = widths[end] > 0.0 or (widths[end] == 0.0 and ahead >= KEEP_BAND_WIDTH_M)
        if mask[inner] and not mask[end] and reaches:
            snapped[end] = True
    return snapped


def _runs(mask: np.ndarray, min_samples: int) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for True runs in `mask` with at least `min_samples` entries."""
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
