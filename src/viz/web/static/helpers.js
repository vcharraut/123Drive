/* 123Drive Viz — shared pure helpers */
'use strict';

(function initVizHelpers(globalScope) {
  function escapeHtml(value) {
    return String(value ?? '')
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#39;');
  }

  function safeIdList(ids) {
    return (ids || []).map(v => escapeHtml(v)).join(', ') || '—';
  }

  function getObjectsOfInterest(metadata) {
    return metadata.objects_of_interest || [];
  }

  function setAppStatus(message = '', level = 'info') {
    const el = document.getElementById('app-status');
    if (!el) return;
    el.textContent = message;
    el.classList.remove('status-info', 'status-ok', 'status-error');
    if (!message) return;
    const cls = level === 'error' ? 'status-error' : (level === 'ok' ? 'status-ok' : 'status-info');
    el.classList.add(cls);
  }

  function getRoadChevrons(xyz, spacingM = 3, sizeM = 0.4) {
    const arrows = [];
    let dist = 0;
    let nextArrow = spacingM * 0.4;
    const hw = sizeM * 0.35;
    const useZ = xyz.some(pt => pt != null && pt.length > 2 && Number.isFinite(pt[2]));
    for (let i = 0; i < xyz.length - 1; i++) {
      const [x1, y1, z1 = 0] = xyz[i];
      const [x2, y2, z2 = z1] = xyz[i + 1];
      const dx = x2 - x1;
      const dy = y2 - y1;
      const len = Math.hypot(dx, dy);
      if (len < 0.01) continue;
      const ux = dx / len;
      const uy = dy / len;
      const nx = -uy;
      const ny = ux;
      while (dist + len >= nextArrow) {
        const t = (nextArrow - dist) / len;
        const cx = x1 + t * dx;
        const cy = y1 + t * dy;
        const cz = z1 + t * (z2 - z1);
        const hx = sizeM * 0.5;
        const left = [cx - ux * hx + nx * hw, cy - uy * hx + ny * hw];
        const tip = [cx + ux * hx, cy + uy * hx];
        const right = [cx - ux * hx - nx * hw, cy - uy * hx - ny * hw];
        if (useZ) {
          left.push(cz);
          tip.push(cz);
          right.push(cz);
        }
        arrows.push({
          path: [left, tip, right],
        });
        nextArrow += spacingM;
      }
      dist += len;
    }
    return arrows;
  }

  function getVehicleCorners(x, y, heading, length, width, z = null) {
    const cos = Math.cos(heading);
    const sin = Math.sin(heading);
    const hl = length / 2;
    const hw = width / 2;
    const local = [[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw], [-hl, -hw]];
    return local.map(([dx, dy]) => {
      const pt = [dx * cos - dy * sin + x, dx * sin + dy * cos + y];
      if (Number.isFinite(z)) pt.push(z);
      return pt;
    });
  }

  function getHeadingArrow(x, y, heading, length, z = null) {
    const al = length * 0.6;
    const start = [x, y];
    const end = [x + al * Math.cos(heading), y + al * Math.sin(heading)];
    if (Number.isFinite(z)) {
      start.push(z);
      end.push(z);
    }
    return [start, end];
  }

  function sceneBounds(scenario) {
    let xmin = Infinity;
    let xmax = -Infinity;
    let ymin = Infinity;
    let ymax = -Infinity;
    let zmin = Infinity;
    let zmax = -Infinity;
    let hasPoint = false;
    const addPoint = (pt) => {
      if (pt == null || pt.length < 2) return;
      const [x, y, z = 0] = pt;
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      hasPoint = true;
      if (x < xmin) xmin = x;
      if (x > xmax) xmax = x;
      if (y < ymin) ymin = y;
      if (y > ymax) ymax = y;
      if (Number.isFinite(z)) {
        if (z < zmin) zmin = z;
        if (z > zmax) zmax = z;
      }
    };

    for (const road of scenario.road_map_elements) {
      for (const pt of road.xyz) addPoint(pt);
    }
    for (const tc of scenario.traffic_control_elements || []) {
      for (const pt of tc.stop_line || []) addPoint(pt);
    }
    for (const agent of scenario.agents || []) {
      for (let i = 0; i < (agent.xyz || []).length; i++) {
        if (!agent.valid || agent.valid[i]) addPoint(agent.xyz[i]);
      }
    }
    for (const object of scenario.objects || []) {
      for (let i = 0; i < (object.xyz || []).length; i++) {
        if (!object.valid || object.valid[i]) addPoint(object.xyz[i]);
      }
    }

    if (!hasPoint) {
      return {xmin: 0, xmax: 0, ymin: 0, ymax: 0, zmin: 0, zmax: 0, cx: 0, cy: 0, cz: 0};
    }
    if (!Number.isFinite(zmin) || !Number.isFinite(zmax)) {
      zmin = 0;
      zmax = 0;
    }
    return {
      xmin,
      xmax,
      ymin,
      ymax,
      zmin,
      zmax,
      cx: (xmin + xmax) / 2,
      cy: (ymin + ymax) / 2,
      cz: (zmin + zmax) / 2,
    };
  }

  globalScope.VizHelpers = {
    escapeHtml,
    safeIdList,
    getObjectsOfInterest,
    setAppStatus,
    getRoadChevrons,
    getVehicleCorners,
    getHeadingArrow,
    sceneBounds,
  };
})(window);
