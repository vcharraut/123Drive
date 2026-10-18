/* 123Drive v2 decoder. The layout is defined in src/bin_factory/serialize.py. */
'use strict';

window.parsePufferBinary = function parsePufferBinary(buffer, {laneGraph = true} = {}) {
  try {
    const view = new DataView(buffer);
    const decoder = new TextDecoder();
    let off = 0;
    const i32 = () => { const v = view.getInt32(off, true); off += 4; return v; };
    const f32 = () => { const v = view.getFloat32(off, true); off += 4; return v; };
    const f64 = () => { const v = view.getFloat64(off, true); off += 8; return v; };
    const u64 = () => {
      const v = view.getUint32(off, true) + 4294967296 * view.getUint32(off + 4, true);
      off += 8;
      if (!Number.isSafeInteger(v)) throw new Error('Binary offset exceeds supported size');
      return v;
    };
    const i32arr = (n) => { const a = new Int32Array(buffer, off, n); off += n * 4; return a; };
    const f32arr = (n) => { const a = new Float32Array(buffer, off, n); off += n * 4; return a; };
    const intList = () => Array.from(i32arr(i32()));
    const str = () => {
      const n = i32();
      const value = decoder.decode(new Uint8Array(buffer, off, n));
      off = Math.ceil((off + n) / 4) * 4;
      return value;
    };
    const magic = decoder.decode(new Uint8Array(buffer, 0, 8));
    off = 8;
    const version = i32();
    if (magic !== '123DRIVE' || version !== 2) {
      throw new Error('Unsupported binary format; regenerate bins with the current converter');
    }
    const sectionCount = i32();
    const sections = new Map();
    let end = 16 + 32 * sectionCount;
    for (let s = 0; s < sectionCount; s++) {
      const tag = decoder.decode(new Uint8Array(buffer, off, 8)).replace(/\0+$/, '');
      off += 8;
      const start = u64(), size = u64(), count = i32(), frames = i32();
      if (start % 8 || start < end || start + size > buffer.byteLength || sections.has(tag)) {
        throw new Error('Invalid binary section directory');
      }
      sections.set(tag, {start, size, count, frames});
      end = start + size;
    }
    if (end !== buffer.byteLength) throw new Error('Binary has trailing bytes after sections');

    const records = (tag, read) => {
      const section = sections.get(tag);
      off = section.start;
      const offsets = Array.from({length: section.count + 1}, u64);
      if (offsets[0] !== 8 * (section.count + 1) || offsets[section.count] !== section.size) {
        throw new Error('Invalid binary record offsets');
      }
      return Array.from({length: section.count}, (_, index) => {
        const start = offsets[index], stop = offsets[index + 1];
        if (start % 8 || stop < start || stop > section.size) throw new Error('Invalid binary record offsets');
        off = section.start + start;
        const value = read(section.frames);
        if (off > section.start + stop) throw new Error('Binary record exceeds its indexed size');
        return value;
      });
    };
    const colsToRows = (channels) => {
      const rows = new Array(channels[0].length);
      for (let t = 0; t < rows.length; t++) {
        const row = new Array(channels.length);
        for (let c = 0; c < channels.length; c++) row[c] = channels[c][t];
        rows[t] = row;
      }
      return rows;
    };
    const readTrack = (frameCount, hasRoute) => {
      const id = i32(), type = i32(), first = i32(), n = i32(), flags = i32();
      const T = flags & 16 ? i32() : frameCount;
      const pad = (stored, Arr) => {
        if (first === 0 && n === T) return stored;
        const result = new Arr(T);
        result.set(stored, first);
        return result;
      };
      const x = pad(f32arr(n), Float32Array);
      const y = pad(f32arr(n), Float32Array);
      const z = flags & 8 ? new Float32Array(T) : pad(f32arr(n), Float32Array);
      const heading = pad(f32arr(n), Float32Array);
      const vx = pad(f32arr(n), Float32Array);
      const vy = pad(f32arr(n), Float32Array);
      const dimensions = [0, 1, 2].map(bit => {
        if (!(flags & (1 << bit))) return pad(f32arr(n), Float32Array);
        return new Float32Array(T).fill(f32(), first, first + n);
      });
      const valid = pad(new Uint8Array(buffer, off, n), Uint8Array);
      off = Math.ceil((off + n) / 4) * 4;
      const track = {
        id, type, xyz: colsToRows([x, y, z]), heading, velocity: colsToRows([vx, vy]),
        length: dimensions[0], width: dimensions[1], height: dimensions[2], valid,
      };
      if (hasRoute) track.route = intList();
      return track;
    };
    const agents = records('AGENTS', frames => readTrack(frames, true));
    const road_map_elements = records('ROADS', () => {
      const id = i32(), type = i32(), S = i32(), flags = i32();
      const x = f32arr(S), y = f32arr(S);
      const z = flags & 1 ? new Float32Array(S) : f32arr(S);
      const heading = f32arr(S);
      let entry_lanes = [], exit_lanes = [], left_neighbors = [], right_neighbors = [];
      let speed_limit = 0, length = 0, cum_length = [], curvature = [];
      if (type >= TYPES.LANE_RANGE[0] && type <= TYPES.LANE_RANGE[1]) {
        entry_lanes = intList();
        exit_lanes = intList();
        left_neighbors = intList();
        right_neighbors = intList();
        speed_limit = f32();
        cum_length = f32arr(S);
        length = cum_length[S - 1];
        curvature = f32arr(S);
      }
      return {
        id, type, xyz: colsToRows([x, y, z]), heading, entry_lanes, exit_lanes, left_neighbors, right_neighbors,
        speed_limit, length, cum_length, curvature,
      };
    });
    const traffic_control_elements = records('TRAFFIC', () => {
      const id = i32(), type = i32();
      const stop_line = [[f32(), f32(), f32()], [f32(), f32(), f32()]];
      const heading = f32(), states = intList(), controlled_lanes = intList();
      return {id, type, stop_line, heading, states, controlled_lanes};
    });
    const objects = records('OBJECTS', frames => readTrack(frames, false));
    const graph = sections.get('LANEGRPH');
    let lane_graph = null;
    if (laneGraph && graph.count) {
      off = graph.start;
      lane_graph = {lane_ids: i32arr(graph.count), distances: f32arr(graph.count * graph.count), n: graph.count};
    }
    off = sections.get('METADATA').start;
    const id = str(), dataset = str(), location = str();
    const scenario_length = i32(), dt = f32();
    const objects_of_interest = intList(), tracks_to_predict = intList();
    const metadata = {id, dataset, location, scenario_length, dt, num_objects: objects.length,
                      objects_of_interest, tracks_to_predict};
    if (sections.has('TLPHASE1')) {
      off = sections.get('TLPHASE1').start;
      for (const control of traffic_control_elements) {
        control.junction_id = i32();
        control.phase_idx = i32();
      }
    }
    if (sections.has('SOURCE01')) {
      off = sections.get('SOURCE01').start;
      metadata.origin_xyz = [f64(), f64(), f64()];
      metadata.map_has_z = Boolean(i32());
      metadata.timestamps_are_relative = Boolean(i32());
      const count = i32();
      i32();
      metadata.timestamps_us = new BigInt64Array(buffer, off, count);
      off += count * 8;
      for (const name of ['scene_uuid', 'log_name', 'split', 'log_version', 'map_version']) metadata[name] = str();
      for (const track of [...agents, ...objects]) track.source_id = str();
    }
    return {agents, road_map_elements, traffic_control_elements, objects, lane_graph, metadata};
  } catch (error) {
    if (error instanceof RangeError) throw new Error('Unexpected end of binary payload');
    throw error;
  }
};
