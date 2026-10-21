import itertools
import struct


class _Reader:
    def __init__(self, data):
        self.data, self.off = data, 0

    def ints(self, n):
        values = struct.unpack_from(f"<{n}i", self.data, self.off)
        self.off += 4 * n
        return list(values)

    def floats(self, n):
        values = struct.unpack_from(f"<{n}f", self.data, self.off)
        self.off += 4 * n
        return list(values)

    def raw(self, n):
        result = self.data[self.off : self.off + n]
        assert len(result) == n
        self.off += n
        return result

    def string(self):
        value = self.raw(self.ints(1)[0]).decode("utf-8")
        self.raw(-self.off % 4)
        return value

    def finish(self):
        assert len(self.data) - self.off < 8
        assert not any(self.data[self.off :])

    def records(self, count):
        offsets = struct.unpack(f"<{count + 1}Q", self.raw(8 * (count + 1)))
        assert offsets[0] == self.off
        assert offsets[-1] == len(self.data)
        assert all(start <= stop and start % 8 == 0 for start, stop in itertools.pairwise(offsets))
        return [_Reader(self.data[start:stop]) for start, stop in itertools.pairwise(offsets)]

    def dynamic(self, frames):
        first, n, flags = self.ints(3)
        t = self.ints(1)[0] if flags & 16 else frames
        assert 0 <= first <= first + n <= t
        self.floats((5 if flags & 8 else 6) * n)
        for bit in range(3):
            self.floats(1 if flags & (1 << bit) else n)
        self.raw(n)
        self.raw(-self.off % 4)
        return t, first, n


def parse_binary(data):
    magic, version, count = struct.unpack_from("<8sII", data)
    assert magic == b"123DRIVE"
    assert version == 1
    sections = {}
    end = 16 + 32 * count
    for index in range(count):
        tag, start, size, n, frames = struct.unpack_from("<8sQQII", data, 16 + 32 * index)
        assert start % 8 == 0
        assert start >= end
        assert start + size <= len(data)
        assert not any(data[end:start])
        sections[tag.rstrip(b"\0")] = (_Reader(data[start : start + size]), n, frames)
        end = start + size
    assert end == len(data)

    agents = []
    r, n_agents, frames = sections[b"AGENTS"]
    for record in r.records(n_agents):
        eid, kind = record.ints(2)
        span = record.dynamic(frames)
        route = record.ints(record.ints(1)[0])
        entry_s, exit_s = record.floats(len(route)), record.floats(len(route))
        record.finish()
        agents.append(
            {
                "id": eid,
                "type": kind,
                "span": span,
                "n_points": span[0],
                "route": route,
                "route_entry_s": entry_s,
                "route_exit_s": exit_s,
            }
        )

    roads, lanes = [], []
    r, n_roads, _ = sections[b"ROADPOLY"]
    for record in r.records(n_roads):
        eid, layer, kind, npts, flags = record.ints(5)
        record.floats((3 if flags & 1 else 4) * npts)
        is_lane = layer == 0  # py123d MapLayer.LANE
        entry, exit_, speed = [], [], 0
        if is_lane:
            entry = record.ints(record.ints(1)[0])
            exit_ = record.ints(record.ints(1)[0])
            record.ints(record.ints(1)[0])
            record.ints(record.ints(1)[0])
            speed = record.floats(1)[0]
            record.floats(2 * npts)
        record.finish()
        road = {
            "id": eid,
            "layer": layer,
            "type": kind,
            "npts": npts,
            "lane": is_lane,
            "entry": entry,
            "exit": exit_,
            "speed": speed,
        }
        roads.append(road)
        if is_lane:
            lanes.append(road)

    surfaces = []
    r, n_surfaces, _ = sections[b"ROADSURF"]
    for record in r.records(n_surfaces):
        eid, layer, npts, flags = record.ints(4)
        record.floats((2 if flags & 1 else 3) * npts)
        record.finish()
        surfaces.append({"id": eid, "layer": layer, "npts": npts})

    tcs = []
    r, n_tc, _ = sections[b"TRAFFIC"]
    for record in r.records(n_tc):
        eid, kind = record.ints(2)
        record.floats(7)
        states = record.ints(record.ints(1)[0])
        controlled = record.ints(record.ints(1)[0])
        record.finish()
        tcs.append({"id": eid, "type": kind, "states": states, "lanes": controlled})

    objects, object_spans = [], []
    r, n_objects, frames = sections[b"OBJECTS"]
    for record in r.records(n_objects):
        eid, kind = record.ints(2)
        objects.append(eid)
        object_spans.append(record.dynamic(frames))
        record.finish()

    r, n_graph, _ = sections[b"LANEGRPH"]
    graph_ids = r.ints(n_graph)
    r.floats(n_graph * n_graph)
    r.finish()
    r = sections[b"METADATA"][0]
    meta_id, dataset, location = r.string(), r.string(), r.string()
    scenario_length = r.ints(1)[0]
    dt = r.floats(1)[0]
    ooi = r.ints(r.ints(1)[0])
    ttp = r.ints(r.ints(1)[0])
    r.finish()
    r = sections[b"TLPHASE1"][0]
    r.ints(2 * n_tc)
    r.finish()
    r = sections[b"SOURCE01"][0]
    origin_xyz = struct.unpack("<3d", r.raw(24))
    map_has_z, relative, count, reserved = r.ints(4)
    assert reserved == 0
    timestamps = list(struct.unpack(f"<{count}q", r.raw(8 * count)))
    source = {name: r.string() for name in ("scene_uuid", "log_name", "split", "log_version", "map_version")}
    source.update(
        origin_xyz=origin_xyz,
        map_has_z=bool(map_has_z),
        timestamps_are_relative=bool(relative),
        timestamps_us=timestamps,
    )
    for agent in agents:
        agent["source_id"] = r.string()
    object_sources = [r.string() for _ in objects]
    r.finish()
    return {
        "counts": (n_agents, n_roads, n_surfaces, n_tc, n_objects),
        "agents": agents,
        "roads": roads,
        "lanes": lanes,
        "surfaces": surfaces,
        "tcs": tcs,
        "objects": objects,
        "object_spans": object_spans,
        "lane_graph_ids": graph_ids,
        "id": meta_id,
        "dataset": dataset,
        "location": location,
        "scenario_length": scenario_length,
        "dt": dt,
        "ooi": ooi,
        "ttp": ttp,
        "source": source,
        "object_sources": object_sources,
        "consumed": end,
    }
