/* Parse PufferDrive .bin format into the scenario object expected by app.js.
 *
 * Binary layout (all little-endian):
 *   Header: num_agents(i32), num_roads(i32), num_traffic(i32), num_objects(i32)
 *   Agents[]:  id(i32), type(i32), T(i32), first(i32), n(i32),
 *              x[n](f32), y[n](f32), z[n](f32),
 *              heading[n](f32), vx[n](f32), vy[n](f32),
 *              length[n](f32), width[n](f32), height[n](f32), valid[n](i32),
 *              n_route(i32), route[n_route](i32), route_gt_len(i32), control_state(i32)
 *              (frames [first, first + n) of T; the others are zero / invalid)
 *   Roads[]:   id(i32), type(i32), S(i32),
 *              x[S](f32), y[S](f32), z[S](f32), heading[S](f32),
 *              [if lane (type 0-9): n_entry(i32), entry[](i32), n_exit(i32), exit[](i32), n_left(i32), left[](i32), n_right(i32), right[](i32), speed_limit(f32), length(f32), cum_length[S](f32), curvature[S](f32)]
 *   Traffic[]: id(i32), type(i32), stop_line(6xf32), heading(f32),
 *              n_states(i32), states[](i32), n_ctrl(i32), ctrl[](i32)
 *   Objects[]: id(i32), type(i32), T(i32), first(i32), n(i32),
 *              x[n](f32), y[n](f32), z[n](f32),
 *              heading[n](f32), vx[n](f32), vy[n](f32),
 *              length[n](f32), width[n](f32), height[n](f32), valid[n](i32)
 *   LaneGraph: n_lanes_graph(i32),
 *              [if n>0: lane_ids[n](i32), distances[n*n](f32)]
 *   Metadata:  id(char[128]), dataset(char[32]),
 *              scenario_length(i32), dt(f32),
 *              n_ooi(i32), ooi[](i32), n_ttp(i32), ttp[](i32)
 *   TrafficPhases (optional): tag "TLPHASE1",
 *              per traffic control: junction_id(i32), phase_idx(i32), both -1 if not in a light cycle
 */
'use strict';

window.parsePufferBinary = function parsePufferBinary(buffer) {
  try {
    const view = new DataView(buffer);
    let off = 0;

    const i32 = () => { const v = view.getInt32(off, true); off += 4; return v; };
    const f32 = () => { const v = view.getFloat32(off, true); off += 4; return v; };

    const i32arr = (n) => {
      const a = new Int32Array(buffer, off, n);
      off += n * 4;
      return a;
    };
    const f32arr = (n) => {
      const a = new Float32Array(buffer, off, n);
      off += n * 4;
      return a;
    };

    const intList = () => { const n = i32(); return n > 0 ? Array.from(i32arr(n)) : []; };

    const str = (len) => {
      const bytes = new Uint8Array(buffer, off, len);
      off += len;
      let end = bytes.indexOf(0);
      if (end === -1) end = len;
      return new TextDecoder().decode(bytes.subarray(0, end));
    };

    // Column-major f32 channels → row-major array of [c0, c1, ...] per row
    const colsToRows = (channels) => {
      const T = channels[0].length;
      const C = channels.length;
      const rows = new Array(T);
      for (let t = 0; t < T; t++) {
        const row = new Array(C);
        for (let c = 0; c < C; c++) row[c] = channels[c][t];
        rows[t] = row;
      }
      return rows;
    };

    // Stored frames [first, first + n) of T, zero-filled outside
    const readDynamicStateArrays = () => {
      const T = i32();
      const first = i32();
      const n = i32();
      const pad = (stored, Arr) => { const a = new Arr(T); a.set(stored, first); return a; };
      const xArr = pad(f32arr(n), Float32Array);
      const yArr = pad(f32arr(n), Float32Array);
      const zArr = pad(f32arr(n), Float32Array);
      const heading = pad(f32arr(n), Float32Array);
      const vxArr = pad(f32arr(n), Float32Array);
      const vyArr = pad(f32arr(n), Float32Array);
      const length = pad(f32arr(n), Float32Array);
      const width = pad(f32arr(n), Float32Array);
      const height = pad(f32arr(n), Float32Array);
      const valid = pad(i32arr(n), Int32Array);
      return {
        xyz: colsToRows([xArr, yArr, zArr]),
        heading: Array.from(heading),
        velocity: colsToRows([vxArr, vyArr]),
        length: Array.from(length),
        width: Array.from(width),
        height: Array.from(height),
        valid: Array.from(valid),
      };
    };

    const readDynamicEntity = (hasRoute) => {
      const id = i32();
      const type = i32();
      const states = readDynamicStateArrays();
      if (!hasRoute) return {id, type, ...states};
      const route = intList();
      const route_gt_len = i32();
      const control_state = i32();
      return {id, type, ...states, route, route_gt_len, control_state};
    };

    // --- Header ---
    const numAgents = i32();
    const numRoads = i32();
    const numTraffic = i32();
    const numObjects = i32();

    // --- Agents ---
    const agents = new Array(numAgents);
    for (let a = 0; a < numAgents; a++) agents[a] = readDynamicEntity(true);

    // --- Roads ---
    const road_map_elements = new Array(numRoads);
    for (let r = 0; r < numRoads; r++) {
      const id = i32();
      const type = i32();
      const S = i32();

      const xArr = f32arr(S);
      const yArr = f32arr(S);
      const zArr = f32arr(S);
      const heading = Array.from(f32arr(S));

      let entry_lanes = [], exit_lanes = [], left_neighbors = [], right_neighbors = [], speed_limit = 0, length = 0, cum_length = [], curvature = [];
      if (type >= TYPES.LANE_RANGE[0] && type <= TYPES.LANE_RANGE[1]) {
        entry_lanes = intList();
        exit_lanes = intList();
        left_neighbors = intList();
        right_neighbors = intList();
        speed_limit = f32();
        length = f32();
        cum_length = Array.from(f32arr(S));
        curvature = Array.from(f32arr(S));
      }

      road_map_elements[r] = { id, type, xyz: colsToRows([xArr, yArr, zArr]), heading, entry_lanes, exit_lanes, left_neighbors, right_neighbors, speed_limit, length, cum_length, curvature };
    }

    // --- Traffic ---
    const traffic_control_elements = new Array(numTraffic);
    for (let t = 0; t < numTraffic; t++) {
      const id = i32();
      const type = i32();
      const x1 = f32(), y1 = f32(), z1 = f32();
      const x2 = f32(), y2 = f32(), z2 = f32();
      const heading = f32();
      const states = intList();
      const controlled_lanes = intList();

      traffic_control_elements[t] = { id, type, stop_line: [[x1,y1,z1],[x2,y2,z2]], heading, states, controlled_lanes };
    }

    // --- Objects ---
    const objects = new Array(numObjects);
    for (let o = 0; o < numObjects; o++) objects[o] = readDynamicEntity(false);

    // --- Lane Graph Distances ---
    const nGraphLanes = i32();
    let lane_graph = null;
    if (nGraphLanes > 0) {
      const graphLaneIds = Array.from(i32arr(nGraphLanes));
      const distances = f32arr(nGraphLanes * nGraphLanes);
      lane_graph = { lane_ids: graphLaneIds, distances, n: nGraphLanes };
    }

    // --- Metadata ---
    const id = str(128);
    const dataset = str(32);
    const location = str(32);
    const scenario_length = i32();
    const dt = f32();
    const objects_of_interest = intList();
    const tracks_to_predict = intList();

    // --- Traffic light phases (optional tagged trailing section) ---
    const PHASE_TAG = 'TLPHASE1';
    if (off + PHASE_TAG.length <= buffer.byteLength
      && new TextDecoder().decode(new Uint8Array(buffer, off, PHASE_TAG.length)) === PHASE_TAG) {
      off += PHASE_TAG.length;
      for (let t = 0; t < numTraffic; t++) {
        traffic_control_elements[t].junction_id = i32();
        traffic_control_elements[t].phase_idx = i32();
      }
    }

    return {
      agents,
      road_map_elements,
      traffic_control_elements,
      objects,
      lane_graph,
      metadata: { id, dataset, location, scenario_length, dt, num_objects: numObjects, objects_of_interest, tracks_to_predict },
    };
  } catch (error) {
    if (error instanceof RangeError) {
      throw new Error('Unexpected end of binary payload');
    }
    throw error;
  };
};
