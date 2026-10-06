/* Parse PufferDrive .bin format into the scenario object expected by app.js.
 *
 * Binary layout (all little-endian):
 *   Header: num_agents(i32), num_roads(i32), num_traffic(i32), num_objects(i32)
 *   Agents[]:  id(i32), type(i32), T(i32),
 *              x[T](f32), y[T](f32), z[T](f32),
 *              heading[T](f32), vx[T](f32), vy[T](f32), yaw_rate[T](f32),
 *              length[T](f32), width[T](f32), height[T](f32), valid[T](i32),
 *              n_route(i32), route[n_route](i32), route_gt_len(i32),
 *              goal_x(f32), goal_y(f32), goal_z(f32), control_state(i32)
 *   Roads[]:   id(i32), type(i32), S(i32),
 *              x[S](f32), y[S](f32), z[S](f32), heading[S](f32),
 *              [if lane (type 0-9): n_entry(i32), entry[](i32), n_exit(i32), exit[](i32), n_left(i32), left[](i32), n_right(i32), right[](i32), speed_limit(f32), length(f32), cum_length[S](f32), curvature[S](f32)]
 *   Traffic[]: id(i32), type(i32), stop_line(6xf32), heading(f32),
 *              n_states(i32), states[](i32), n_ctrl(i32), ctrl[](i32)
 *   Objects[]: id(i32), type(i32), T(i32),
 *              x[T](f32), y[T](f32), z[T](f32),
 *              heading[T](f32), vx[T](f32), vy[T](f32),
 *              length[T](f32), width[T](f32), height[T](f32), valid[T](i32)
 *   LaneGraph: n_lanes_graph(i32),
 *              [if n>0: lane_ids[n](i32), distances[n*n](f32)]
 *   Metadata:  id(char[128]), dataset(char[32]),
 *              scenario_length(i32), dt(f32),
 *              n_ooi(i32), ooi[](i32), n_ttp(i32), ttp[](i32)
 *   TrafficGroups (optional): n(i32) == num_traffic,
 *              per traffic control: intersection_id(i32), signal_group_id(i32), signal_sequence(i32), -1 if unknown
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

    const readDynamicStateArrays = (T, hasYawRate) => {
      const xArr = f32arr(T);
      const yArr = f32arr(T);
      const zArr = f32arr(T);
      const heading = f32arr(T);
      const vxArr = f32arr(T);
      const vyArr = f32arr(T);
      const yawRate = hasYawRate ? Array.from(f32arr(T)) : null;
      const length = f32arr(T);
      const width = f32arr(T);
      const height = f32arr(T);
      const valid = i32arr(T);
      return {
        xyz: colsToRows([xArr, yArr, zArr]),
        heading: Array.from(heading),
        velocity: colsToRows([vxArr, vyArr]),
        ...(yawRate ? { yaw_rate: yawRate } : {}),
        length: Array.from(length),
        width: Array.from(width),
        height: Array.from(height),
        valid: Array.from(valid),
      };
    };

    const readDynamicEntity = (hasRoute) => {
      const id = i32();
      const type = i32();
      const states = readDynamicStateArrays(i32(), hasRoute); // only agents carry yaw rate
      if (!hasRoute) return {id, type, ...states};
      const route = intList();
      const route_gt_len = i32();
      f32();
      f32();
      f32();
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
    const scenario_length = i32();
    const dt = f32();
    const objects_of_interest = intList();
    const tracks_to_predict = intList();

    // --- Traffic control groups (optional trailing section) ---
    if (off < buffer.byteLength) {
      const nGroups = i32();
      for (let t = 0; t < nGroups && t < numTraffic; t++) {
        traffic_control_elements[t].intersection_id = i32();
        traffic_control_elements[t].signal_group_id = i32();
        traffic_control_elements[t].signal_sequence = i32();
      }
    }

    return {
      agents,
      road_map_elements,
      traffic_control_elements,
      objects,
      lane_graph,
      metadata: { id, dataset, scenario_length, dt, num_objects: numObjects, objects_of_interest, tracks_to_predict },
    };
  } catch (error) {
    if (error instanceof RangeError) {
      throw new Error('Unexpected end of binary payload');
    }
    throw error;
  };
};
