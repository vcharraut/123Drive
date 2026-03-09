/* Parse PufferDrive .bin format into the scenario object expected by app.js.
 *
 * Binary layout (all little-endian):
 *   Header: num_agents(i32), num_roads(i32), num_traffic(i32)
 *   Agents[]:  id(i32), type(i32), T(i32),
 *              x[T](f32), y[T](f32), z[T](f32),
 *              heading[T](f32), vx[T](f32), vy[T](f32),
 *              length[T](f32), width[T](f32), height[T](f32), valid[T](i32),
 *              n_route(i32), route[n_route](i32),
 *              goal_x(f32), goal_y(f32), goal_z(f32), mark_as_expert(i32)
 *   Roads[]:   id(i32), type(i32), S(i32),
 *              x[S](f32), y[S](f32), z[S](f32),
 *              [if lane (type 0-9): n_entry(i32), entry[](i32), n_exit(i32), exit[](i32), speed_limit(f32)]
 *   Traffic[]: id(i32), type(i32), x(f32), y(f32), z(f32),
 *              n_states(i32), states[](i32), n_ctrl(i32), ctrl[](i32)
 *   Metadata:  scenario_id(char[128]), map_id(i32), dataset_name(char[64]),
 *              scenario_length(i32), sdc_index(i32),
 *              n_ooi(i32), ooi[](i32), n_ttp(i32), ttp[](i32)
 */
'use strict';

window.parsePufferBinary = function parsePufferBinary(buffer) {
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

  // --- Header ---
  const numAgents = i32();
  const numRoads = i32();
  const numTraffic = i32();

  // --- Agents ---
  const agents = new Array(numAgents);
  for (let a = 0; a < numAgents; a++) {
    const id = i32();
    const type = i32();
    const T = i32();

    const xArr = f32arr(T);
    const yArr = f32arr(T);
    const zArr = f32arr(T);
    const heading = f32arr(T);
    const vxArr = f32arr(T);
    const vyArr = f32arr(T);
    const length = f32arr(T);
    const width = f32arr(T);
    const height = f32arr(T);
    const valid = i32arr(T);
    const route = intList();
    const goalX = f32(), goalY = f32(), goalZ = f32();
    i32(); // mark_as_expert (unused in viz)

    agents[a] = {
      id, type,
      xyz: colsToRows([xArr, yArr, zArr]),
      heading: Array.from(heading),
      velocity: colsToRows([vxArr, vyArr]),
      length: Array.from(length),
      width: Array.from(width),
      height: Array.from(height),
      valid: Array.from(valid),
      route,
      route_polyline: null,
    };
  }

  // --- Roads ---
  const road_map_elements = new Array(numRoads);
  for (let r = 0; r < numRoads; r++) {
    const id = i32();
    const type = i32();
    const S = i32();

    const xArr = f32arr(S);
    const yArr = f32arr(S);
    const zArr = f32arr(S);

    let entry_lanes = [], exit_lanes = [], speed_limit = 0;
    if (type >= 0 && type <= 9) {
      entry_lanes = intList();
      exit_lanes = intList();
      speed_limit = f32();
    }

    road_map_elements[r] = { id, type, xyz: colsToRows([xArr, yArr, zArr]), entry_lanes, exit_lanes, speed_limit };
  }

  // --- Traffic ---
  const traffic_control_elements = new Array(numTraffic);
  for (let t = 0; t < numTraffic; t++) {
    const id = i32();
    const type = i32();
    const x = f32(), y = f32(), z = f32();
    const states = intList();
    const controlled_lanes = intList();

    traffic_control_elements[t] = { id, type, xyz: [x, y, z], states, controlled_lanes };
  }

  // --- Metadata ---
  const scenario_id = str(128);
  const map_id = i32();
  const dataset_name = str(64);
  const scenario_length = i32();
  const sdc_index = i32();
  const objects_of_interest = intList();
  const tracks_to_predict = intList();

  return {
    scenario_id,
    agents,
    road_map_elements,
    traffic_control_elements,
    metadata: { map_id, dataset_name, scenario_length, sdc_index, objects_of_interest, tracks_to_predict },
  };
};
