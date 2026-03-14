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
    for (let i = 0; i < xyz.length - 1; i++) {
      const [x1, y1] = [xyz[i][0], xyz[i][1]];
      const [x2, y2] = [xyz[i + 1][0], xyz[i + 1][1]];
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
        const hx = sizeM * 0.5;
        arrows.push({
          path: [
            [cx - ux * hx + nx * hw, cy - uy * hx + ny * hw],
            [cx + ux * hx, cy + uy * hx],
            [cx - ux * hx - nx * hw, cy - uy * hx - ny * hw],
          ],
        });
        nextArrow += spacingM;
      }
      dist += len;
    }
    return arrows;
  }

  function getVehicleCorners(x, y, heading, length, width) {
    const cos = Math.cos(heading);
    const sin = Math.sin(heading);
    const hl = length / 2;
    const hw = width / 2;
    const local = [[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw], [-hl, -hw]];
    return local.map(([dx, dy]) => [dx * cos - dy * sin + x, dx * sin + dy * cos + y]);
  }

  function getHeadingArrow(x, y, heading, length) {
    const al = length * 0.6;
    return [[x, y], [x + al * Math.cos(heading), y + al * Math.sin(heading)]];
  }

  function sceneBounds(scenario) {
    let xmin = Infinity;
    let xmax = -Infinity;
    let ymin = Infinity;
    let ymax = -Infinity;
    for (const road of scenario.road_map_elements) {
      for (const [x, y] of road.xyz) {
        if (x < xmin) xmin = x;
        if (x > xmax) xmax = x;
        if (y < ymin) ymin = y;
        if (y > ymax) ymax = y;
      }
    }
    return {xmin, xmax, ymin, ymax, cx: (xmin + xmax) / 2, cy: (ymin + ymax) / 2};
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
