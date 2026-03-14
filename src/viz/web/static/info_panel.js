/* 123Drive Viz — selection/info panel renderer */
'use strict';

(function initVizInfoPanel(globalScope) {
  function formatEntityState(data, t, escapeHtml) {
    const validAt = t < data.valid.length ? data.valid[t] : false;
    const x = validAt ? data.xyz[t][0].toFixed(2) : '—';
    const y = validAt ? data.xyz[t][1].toFixed(2) : '—';
    const z = validAt ? data.xyz[t][2].toFixed(2) : '—';
    const h = validAt ? ((data.heading[t] * 180 / Math.PI) % 360).toFixed(1) + '°' : '—';
    const vx = validAt && data.velocity[t] ? data.velocity[t][0].toFixed(2) : '—';
    const vy = validAt && data.velocity[t] ? data.velocity[t][1].toFixed(2) : '—';
    const vmag = validAt && data.velocity[t] ? Math.sqrt(data.velocity[t][0] ** 2 + data.velocity[t][1] ** 2).toFixed(2) : '—';
    const l = validAt ? (data.length[t] || 0).toFixed(2) : '—';
    const w = validAt ? (data.width[t] || 0).toFixed(2) : '—';
    const ht = validAt ? (data.height[t] || 0).toFixed(2) : '—';
    return {validAt, x, y, z, h, vx, vy, vmag, l, w, ht};
  }

  function buildTrajectoryRows(data, t, escapeHtml) {
    return data.xyz.slice(0, 50).map((pos, i) => {
      const cls = i === t ? 'class="current-row"' : '';
      const valid = data.valid[i] ? '✓' : '✗';
      return `<tr ${cls}><td>${i}</td><td>${escapeHtml(pos[0].toFixed(1))}</td><td>${escapeHtml(pos[1].toFixed(1))}</td><td>${valid}</td></tr>`;
    }).join('');
  }

  function renderAgentInfo(data, context) {
    const {t, scenario, AGENT_TYPE_NAMES, escapeHtml, safeIdList, getObjectsOfInterest} = context;

    const egoAgent = scenario.agents[0]; // By convention, the first agent is the ego (if any)
    const ttp = scenario.metadata.tracks_to_predict || [];
    const ooi = getObjectsOfInterest(scenario.metadata);
    const isEgo = egoAgent ? data.id === egoAgent.id : false;
    const isTtp = ttp.includes(data.id);
    const isOoi = ooi.includes(data.id);
    const typeName = AGENT_TYPE_NAMES[data.type] || `type_${data.type}`;
    const s = formatEntityState(data, t, escapeHtml);

    const badges = [
      `<span class="badge badge-${escapeHtml(typeName)}">${escapeHtml(typeName)}</span>`,
      isEgo ? '<span class="badge badge-ego">EGO</span>' : '',
      isTtp ? '<span class="badge badge-ttp">TTP</span>' : '',
      isOoi ? '<span class="badge badge-ooi">OOI</span>' : '',
    ].join('');

    const routeLanes = (data.route && data.route.length) ? safeIdList(data.route) : '—';
    const trajRows = buildTrajectoryRows(data, t, escapeHtml);

    return `
      ${badges}
      <div class="info-row"><span class="info-label">ID</span><span class="info-val">${escapeHtml(data.id)}</span></div>
      <div class="info-row"><span class="info-label">Valid</span><span class="info-val">${s.validAt ? '✓' : '✗'}</span></div>
      <div class="info-row"><span class="info-label">X,Y</span><span class="info-val">${escapeHtml(s.x)}, ${escapeHtml(s.y)}</span></div>
      <div class="info-row"><span class="info-label">Heading</span><span class="info-val">${escapeHtml(s.h)}</span></div>
      <div class="info-row"><span class="info-label">Speed</span><span class="info-val">${escapeHtml(s.vmag)} m/s</span></div>
      <div class="info-row"><span class="info-label">Vel XY</span><span class="info-val">${escapeHtml(s.vx)}, ${escapeHtml(s.vy)}</span></div>
      <div class="info-row"><span class="info-label">L×W×H</span><span class="info-val">${escapeHtml(s.l)}×${escapeHtml(s.w)}×${escapeHtml(s.ht)}</span></div>
      <div class="info-row"><span class="info-label">Route lanes</span><span class="info-val" style="font-size:9px">${routeLanes}</span></div>
      <details><summary>Trajectory</summary>
        <table class="traj-table"><thead><tr><th>#</th><th>X</th><th>Y</th><th>V</th></tr></thead>
        <tbody>${trajRows}</tbody></table>
      </details>`;
  }

  function renderObjectInfo(data, context) {
    const {t, OBJECT_TYPE_NAMES, escapeHtml} = context;
    const typeName = OBJECT_TYPE_NAMES[data.type] || `type_${data.type}`;
    const s = formatEntityState(data, t, escapeHtml);
    const trajRows = buildTrajectoryRows(data, t, escapeHtml);

    return `
      <span class="badge badge-${escapeHtml(typeName)}">${escapeHtml(typeName)}</span>
      <div class="info-row"><span class="info-label">ID</span><span class="info-val">${escapeHtml(data.id)}</span></div>
      <div class="info-row"><span class="info-label">Valid</span><span class="info-val">${s.validAt ? '✓' : '✗'}</span></div>
      <div class="info-row"><span class="info-label">X,Y,Z</span><span class="info-val">${escapeHtml(s.x)}, ${escapeHtml(s.y)}, ${escapeHtml(s.z)}</span></div>
      <div class="info-row"><span class="info-label">Heading</span><span class="info-val">${escapeHtml(s.h)}</span></div>
      <div class="info-row"><span class="info-label">Speed</span><span class="info-val">${escapeHtml(s.vmag)} m/s</span></div>
      <div class="info-row"><span class="info-label">Vel XY</span><span class="info-val">${escapeHtml(s.vx)}, ${escapeHtml(s.vy)}</span></div>
      <div class="info-row"><span class="info-label">L×W×H</span><span class="info-val">${escapeHtml(s.l)}×${escapeHtml(s.w)}×${escapeHtml(s.ht)}</span></div>
      <details><summary>Trajectory</summary>
        <table class="traj-table"><thead><tr><th>#</th><th>X</th><th>Y</th><th>V</th></tr></thead>
        <tbody>${trajRows}</tbody></table>
      </details>`;
  }

  function renderRoadInfo(data, context) {
    const {ROAD_TYPE_NAMES, escapeHtml, safeIdList} = context;
    const typeName = ROAD_TYPE_NAMES[data.type] || `type_${data.type}`;
    const npts = data.xyz.length;
    const isLane = data.type >= 0 && data.type <= 9;
    const isLine = data.type >= 10 && data.type <= 19;
    const isEdge = data.type >= 20 && data.type <= 29;

    const ptRows = data.xyz.slice(0, 30).map(([x, y], i) =>
      `<tr><td>${i}</td><td>${x.toFixed(2)}</td><td>${y.toFixed(2)}</td></tr>`).join('');

    let category = 'misc';
    if (isLane) category = 'lane';
    else if (isLine) category = 'road line';
    else if (isEdge) category = 'road edge';

    let html = `
      <div class="info-row"><span class="info-label">ID</span><span class="info-val">${escapeHtml(data.id)}</span></div>
      <div class="info-row"><span class="info-label">Category</span><span class="info-val">${escapeHtml(category)}</span></div>
      <div class="info-row"><span class="info-label">Type</span><span class="info-val">${escapeHtml(typeName)}</span></div>
      <div class="info-row"><span class="info-label">Points</span><span class="info-val">${escapeHtml(npts)}</span></div>`;

    if (isLane) {
      const entry = data.entry_lanes.length ? safeIdList(data.entry_lanes) : '—';
      const exit = data.exit_lanes.length ? safeIdList(data.exit_lanes) : '—';
      const sl = data.speed_limit ? data.speed_limit.toFixed(1) + ' m/s' : '—';
      html += `
      <div class="info-row"><span class="info-label">Entry</span><span class="info-val" style="font-size:9px">${entry}</span></div>
      <div class="info-row"><span class="info-label">Exit</span><span class="info-val" style="font-size:9px">${exit}</span></div>
      <div class="info-row"><span class="info-label">Speed lim</span><span class="info-val">${escapeHtml(sl)}</span></div>`;
    }

    html += `
      <details><summary>Polyline</summary>
        <table class="traj-table"><thead><tr><th>#</th><th>X</th><th>Y</th></tr></thead>
        <tbody>${ptRows}</tbody></table>
      </details>`;

    return html;
  }

  function renderTrafficControlInfo(data, context) {
    const {t, TC_TYPE_NAMES, TL_STATE_NAMES, TL_STATE_COLORS, escapeHtml, safeIdList} = context;
    const tcType = data.type || 1;
    const typeName = TC_TYPE_NAMES[tcType] || `type_${tcType}`;
    const controlled = data.controlled_lanes.length ? safeIdList(data.controlled_lanes) : '—';

    let html = `
      <div class="info-row"><span class="info-label">ID</span><span class="info-val">${escapeHtml(data.id)}</span></div>
      <div class="info-row"><span class="info-label">Type</span><span class="info-val">${escapeHtml(typeName)}</span></div>
      <div class="info-row"><span class="info-label">Stop line</span><span class="info-val">(${data.stop_line[0][0].toFixed(1)},${data.stop_line[0][1].toFixed(1)}) → (${data.stop_line[1][0].toFixed(1)},${data.stop_line[1][1].toFixed(1)})</span></div>
      <div class="info-row"><span class="info-label">Heading</span><span class="info-val">${(data.heading * 180 / Math.PI).toFixed(1)}°</span></div>
      <div class="info-row"><span class="info-label">Lanes</span><span class="info-val" style="font-size:9px">${controlled}</span></div>`;

    // Only show state timeline for traffic lights (type=1) with states
    if (tcType === 1 && data.states.length > 0) {
      const stateNow = t < data.states.length ? data.states[t] : 0;
      const stateName = TL_STATE_NAMES[stateNow] || 'unknown';
      const colStr = TL_STATE_COLORS[stateNow] || '#808080';

      let prev = -1;
      const transitions = data.states.reduce((acc, s, i) => {
        if (s !== prev) {
          acc.push({t: i, state: s});
          prev = s;
        }
        return acc;
      }, []);

      const transRows = transitions.map(tr =>
        `<tr><td>${tr.t}</td><td>${escapeHtml(TL_STATE_NAMES[tr.state] || tr.state)}</td></tr>`).join('');

      const fullRows = data.states.map((s, i) => {
        const cls = i === t ? 'class="current-row"' : '';
        return `<tr ${cls}><td>${i}</td><td>${escapeHtml(TL_STATE_NAMES[s] || s)}</td></tr>`;
      }).join('');

      html += `
      <div class="info-row"><span class="info-label">State @${t}</span><span class="info-val">
        <span class="tl-dot" style="background:${colStr}"></span>${escapeHtml(stateName)}
      </span></div>
      <div style="margin-top:6px;font-size:10px;color:#888">Transitions (${transitions.length})</div>
      <table class="traj-table"><thead><tr><th>@t</th><th>State</th></tr></thead>
        <tbody>${transRows}</tbody></table>
      <details><summary>Full timeline</summary>
        <table class="traj-table"><thead><tr><th>#</th><th>State</th></tr></thead>
        <tbody>${fullRows}</tbody></table>
      </details>`;
    }

    return html;
  }

  function renderElementInfoHtml(type, data, context) {
    if (type === 'agent') return renderAgentInfo(data, context);
    if (type === 'object') return renderObjectInfo(data, context);
    if (type === 'road') return renderRoadInfo(data, context);
    if (type === 'traffic_control') return renderTrafficControlInfo(data, context);
    return '<span class="empty-state">Click an element to inspect.</span>';
  }

  globalScope.VizInfoPanel = {renderElementInfoHtml};
})(window);
