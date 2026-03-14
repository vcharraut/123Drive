/* 123Drive Viz — deck.gl client-side renderer */
'use strict';

const {DeckGL, OrthographicView, OrbitView, PathLayer, PolygonLayer, ScatterplotLayer, TextLayer, PathStyleExtension, LinearInterpolator, COORDINATE_SYSTEM} = window.deck;

// ── Constants ────────────────────────────────────────────────────────────────
const SCROLL_ZOOM_SPEED = 0.012;
const WHEEL_ZOOM_SPEED = 0.002;
const MAX_PITCH_ANGLE = 80;
const FIT_VIEW_PADDING = 1.1;
const DEFAULT_3D_PITCH = 20;
const DEFAULT_3D_ROTATION_X = 30;

// ── Theme ───────────────────────────────────────────────────────────────────
const CLEAR_COLORS = {
  dark:  [0.031, 0.047, 0.071, 1],   // #080C12
  light: [0.941, 0.949, 0.961, 1],   // #F0F2F5
};

function getTheme() {
  return document.body.getAttribute('data-theme') || 'dark';
}

function setTheme(theme) {
  document.body.setAttribute('data-theme', theme);
  document.body.style.colorScheme = theme;
  localStorage.setItem('puffer-viz-theme', theme);
  const btn = document.getElementById('btn-theme');
  if (btn) btn.textContent = theme === 'dark' ? '\u2600' : '\u263E';
  if (typeof deckgl !== 'undefined') {
    deckgl.setProps({ parameters: { clearColor: CLEAR_COLORS[theme] } });
  }
}

// Restore saved theme or default to dark
const savedTheme = localStorage.getItem('puffer-viz-theme') || 'dark';
if (savedTheme === 'light') document.body.setAttribute('data-theme', 'light');
document.body.style.colorScheme = savedTheme;

// Flip Y axis: data uses Y-up (north) but OrthographicView uses Y-down (screen).
// Only applied in 2D; 3D uses identity to preserve chirality (driving side).
const FLIP_Y = [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const {
  escapeHtml,
  safeIdList,
  getObjectsOfInterest,
  setAppStatus,
  getRoadChevrons,
  getVehicleCorners,
  getHeadingArrow,
  sceneBounds,
} = window.VizHelpers;
const {renderElementInfoHtml} = window.VizInfoPanel;

// ── Constants ──────────────────────────────────────────────────────────────

const AGENT_COLORS = {
  0: [107, 114, 128],   // unset   – gray-500
  1: [37,  99, 235],    // vehicle – blue-600
  2: [5,  150, 105],    // pedestrian – emerald-600
  3: [217, 119,  6],    // cyclist – amber-600
  4: [107, 114, 128],   // other   – gray-500
};
const OBJECT_COLORS = {
  1: [220, 38, 38],
  2: [234, 88, 12],
  3: [22, 163, 74],
  4: [71, 85, 105],
  5: [8, 145, 178],
};
const EGO_COLOR = [220, 38, 38]; // red-600
const ROUTELESS_COLOR = [148, 163, 184]; // slate-400
const ROUTE_COLOR = [116, 136, 160, 88];
const EGO_ROUTE_COLOR = [220, 38, 38, 96];

// Type constants — loaded from /api/types, fallbacks until fetched
window.TYPES = {
  AGENT_TYPE_NAMES: {0:'unset', 1:'vehicle', 2:'pedestrian', 3:'cyclist', 4:'other'},
  ROAD_TYPE_NAMES: {},
  TC_TYPE_NAMES: {1:'traffic_light', 2:'stop_sign', 3:'yield_sign'},
  OBJECT_TYPE_NAMES: {1:'traffic_sign', 2:'traffic_cone', 3:'traffic_light', 4:'barrier', 5:'generic_object'},
  TL_STATE_NAMES: {0:'green', 1:'yellow', 2:'red', 3:'off', 4:'unknown'},
  TL_STATE_COLORS: {0:'#00FF00', 1:'#FFFF00', 2:'#FF0000', 3:'#808080', 4:'#808080'},
  LANE_RANGE: [0, 9],
  ROAD_LINE_RANGE: [10, 19],
  ROAD_EDGE_RANGE: [20, 29],
  ROAD_COLORS: {},
};
const TC_TYPE_COLORS = {2:[220,38,38], 3:[234,179,8]};

// RGB versions of TL_STATE_COLORS for deck.gl layers
function hexToRgb(hex) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return [r, g, b];
}
function getTlStateColorRgb(state) {
  const hex = window.TYPES.TL_STATE_COLORS[state] || '#808080';
  return hexToRgb(hex);
}


const SPEED_MS = {0.5: 200, 1: 100, 2: 50, 4: 25};

const EMPTY_DETAIL_HTML = '<span class="empty-state">Click an element to inspect.</span>';

// ── State ──────────────────────────────────────────────────────────────────

const state = {
  scenario: null,
  timestep: 0,
  playing: false,
  speed: 1,
  viewMode: '2d',
  followEgo: false,
  layers: {
    lanes: true, road_lines: true, road_edges: true, crosswalks: true,
    agents: true, objects: true, routes: true, trajectories: true, traffic_controls: true, agent_ids: true,
    unknowns: false,
    scatter_roads: false,
  },
  selected: null,
  pathFinder: { active: false, source: null, dest: null, path: null, distance: null },
  staticLayerCache: null,
  staticLayerCacheKey: null,
  playTimer: null,
  viewState: {target: [0, 0, 0], zoom: 0, rotationX: 0, rotationOrbit: 0},
  suppressNextCanvasClick: false,
};

const dragState = {
  pointerId: null,
  mode: null,
  startScreen: null,
  startGround: null,
  startTarget: null,
  startRotationX: 0,
  startRotationOrbit: 0,
  moved: false,
};

const DRAG_THRESHOLD_PX = 4;

// ── Geometry helpers ───────────────────────────────────────────────────────

const toXY = d => d.xyz.map(p => [p[0], p[1]]);

function getEgoState(scenario, t) {
  const agent = scenario.agents[0];
  if (!agent || !agent.valid[t]) return null;
  return {x: agent.xyz[t][0], y: agent.xyz[t][1], heading: agent.heading[t]};
}

function getEgoAgentId(scenario) {
  const agent = scenario.agents[0];
  return agent ? agent.id : null;
}

function agentHasRoute(agent) {
  return Boolean(agent && Array.isArray(agent.route) && agent.route.length);
}

function getAgentDisplayColor(agent, egoId) {
  if (agent.id === egoId) return EGO_COLOR;
  if (!agentHasRoute(agent)) return ROUTELESS_COLOR;
  return AGENT_COLORS[agent.type] || ROUTELESS_COLOR;
}

function getObjectDisplayColor(object) {
  return OBJECT_COLORS[object.type] || OBJECT_COLORS[5];
}

function buildValidSegments(xyz, valid, start, end, color) {
  const segs = [];
  let cur = [];
  for (let i = start; i < end; i++) {
    if (valid[i]) {
      cur.push([xyz[i][0], xyz[i][1]]);
      continue;
    }
    if (cur.length > 1) segs.push({path: cur, color});
    cur = [];
  }
  if (cur.length > 1) segs.push({path: cur, color});
  return segs;
}

function buildRoadMap(roadMapElements) {
  return new Map(roadMapElements.map(road => [road.id, road]));
}

function reconstructPath(scenario, srcId, dstId) {
  const lg = scenario.lane_graph;
  if (!lg) return null;
  const idToIdx = new Map(lg.lane_ids.map((id, i) => [id, i]));
  const srcIdx = idToIdx.get(srcId);
  const dstIdx = idToIdx.get(dstId);
  if (srcIdx === undefined || dstIdx === undefined) return null;

  const dist = (i, j) => lg.distances[i * lg.n + j];
  const totalDist = dist(srcIdx, dstIdx);
  if (!isFinite(totalDist)) return null;

  const roadMap = buildRoadMap(scenario.road_map_elements);
  const path = [srcId];
  let curIdx = srcIdx;
  let curId = srcId;
  const MAX_HOPS = lg.n;
  let hops = 0;
  while (curId !== dstId) {
    if (++hops > MAX_HOPS) return null;
    const lane = roadMap.get(curId);
    if (!lane) return null;
    const exits = (lane.exit_lanes || []).filter(eid => idToIdx.has(eid));
    const next = exits.find(eid => {
      const eidx = idToIdx.get(eid);
      return Math.abs(lg.lane_lengths[curIdx] + dist(eidx, dstIdx) - dist(curIdx, dstIdx)) < 0.01;
    });
    if (next === undefined) return null;
    path.push(next);
    curIdx = idToIdx.get(next);
    curId = next;
  }
  return { laneIds: path, distance: totalDist };
}

function getAgentRouteStart(agent) {
  const firstValid = agent.valid.findIndex(Boolean);
  return firstValid >= 0 ? agent.xyz[firstValid] : agent.xyz[0];
}

function cropPathToStart(path, startPos) {
  if (!Array.isArray(path) || path.length < 2 || !startPos) return path;
  const startIdx = path.reduce((bestIdx, point, idx, pts) => {
    const best = pts[bestIdx];
    const bestDist = (best[0] - startPos[0]) ** 2 + (best[1] - startPos[1]) ** 2;
    const curDist = (point[0] - startPos[0]) ** 2 + (point[1] - startPos[1]) ** 2;
    return curDist < bestDist ? idx : bestIdx;
  }, 0);
  return path.slice(startIdx);
}

function getAgentRouteSegments(agent, roadMap) {
  const startPos = getAgentRouteStart(agent);
  const laneSegments = (agent.route || [])
    .map(laneId => roadMap.get(laneId))
    .filter(road => road && Array.isArray(road.xyz) && road.xyz.length > 1)
    .map((road, idx) => {
      const path = road.xyz.map(([x, y]) => [x, y]);
      const croppedPath = idx === 0 ? cropPathToStart(path, startPos) : path;
      return {path: croppedPath, laneId: road.id, agentId: agent.id};
    })
    .filter(seg => seg.path.length > 1);

  if (laneSegments.length) return laneSegments;
  if (Array.isArray(agent.route_polyline) && agent.route_polyline.length > 1) {
    return [{path: agent.route_polyline.map(([x, y]) => [x, y]), agentId: agent.id}];
  }
  return [];
}

// ── deck.gl setup ──────────────────────────────────────────────────────────

const deckContainer = document.getElementById('deckgl-canvas');

const CONTROLLER_2D = {
  scrollZoom: { smooth: false, speed: SCROLL_ZOOM_SPEED },
  inertia: 400,
  keyboard: false,
  dragMode: 'pan',
};

const CONTROLLER_3D = false;

function instantViewState(vs) {
  const next = {...vs};
  delete next.transitionDuration;
  delete next.transitionInterpolator;
  delete next.transitionEasing;
  return next;
}

function convertTargetForViewMode(target, fromMode, toMode) {
  if (!Array.isArray(target) || target.length < 2 || fromMode === toMode) return target;
  return [target[0], -target[1], target[2] || 0];
}

function isPrimaryPointerEvent(event) {
  const srcEvent = event?.srcEvent || event;
  const button = srcEvent?.button;
  return button == null || button === 0;
}

function consumeSuppressedCanvasClick() {
  if (!state.suppressNextCanvasClick) return false;
  state.suppressNextCanvasClick = false;
  return true;
}

function updateFollowEgoUi() {
  document.getElementById('btn-follow').classList.toggle('active', state.followEgo);
}

function setFollowEgo(enabled) {
  state.followEgo = enabled;
  updateFollowEgoUi();
}

function disableFollowEgo() {
  if (!state.followEgo) return;
  setFollowEgo(false);
}

function syncCanvasCursor() {
  deckContainer.style.cursor = dragState.mode ? 'grabbing' : (state.viewMode === '3d' ? 'grab' : 'crosshair');
}

function getMainViewport() {
  const viewports = deckgl.getViewports();
  return viewports.find(v => v.id === 'main') || viewports[0] || null;
}

function getCanvasPosition(event) {
  const rect = deckContainer.getBoundingClientRect();
  return [event.clientX - rect.left, event.clientY - rect.top];
}

function reset3DDrag(event) {
  if (event && dragState.pointerId != null && deckContainer.hasPointerCapture?.(dragState.pointerId)) {
    deckContainer.releasePointerCapture(dragState.pointerId);
  }
  state.suppressNextCanvasClick = false;
  dragState.pointerId = null;
  dragState.mode = null;
  dragState.startScreen = null;
  dragState.startGround = null;
  dragState.startTarget = null;
  dragState.startRotationX = 0;
  dragState.startRotationOrbit = 0;
  dragState.moved = false;
  syncCanvasCursor();
}

function begin3DDrag(event) {
  if (state.viewMode !== '3d') return;
  if (event.button !== 0 && event.button !== 2) return;

  const mode = event.button === 0 ? 'pan' : 'rotate';
  const startScreen = getCanvasPosition(event);
  const viewport = getMainViewport();

  dragState.pointerId = event.pointerId;
  dragState.mode = mode;
  dragState.startScreen = startScreen;
  dragState.startGround = mode === 'pan' && viewport
    ? viewport.unproject(startScreen, {targetZ: 0})
    : null;
  dragState.startTarget = [...(state.viewState.target || [0, 0, 0])];
  dragState.startRotationX = state.viewState.rotationX || 0;
  dragState.startRotationOrbit = state.viewState.rotationOrbit || 0;
  dragState.moved = false;

  deckContainer.setPointerCapture?.(event.pointerId);
  syncCanvasCursor();
  event.preventDefault();
}

function update3DDrag(event) {
  if (state.viewMode !== '3d') return;
  if (dragState.pointerId == null || dragState.pointerId !== event.pointerId) return;
  if (!dragState.mode || !dragState.startScreen) return;

  const pos = getCanvasPosition(event);
  const dx = pos[0] - dragState.startScreen[0];
  const dy = pos[1] - dragState.startScreen[1];
  if (!dragState.moved && Math.hypot(dx, dy) < DRAG_THRESHOLD_PX) return;
  if (!dragState.moved) {
    dragState.moved = true;
    disableFollowEgo();
  }

  if (dragState.mode === 'pan') {
    const viewport = getMainViewport();
    if (!viewport || !dragState.startGround || !dragState.startTarget) return;
    const ground = viewport.unproject(pos, {targetZ: 0});
    if (!ground) return;

    setViewState(instantViewState({
      ...state.viewState,
      target: [
        dragState.startTarget[0] + dragState.startGround[0] - ground[0],
        dragState.startTarget[1] + dragState.startGround[1] - ground[1],
        dragState.startTarget[2] || 0,
      ],
    }));
  } else if (dragState.mode === 'rotate') {
    const width = Math.max(deckContainer.clientWidth, 1);
    const height = Math.max(deckContainer.clientHeight, 1);
    setViewState(instantViewState({
      ...state.viewState,
      rotationOrbit: dragState.startRotationOrbit + (dx / width) * 180,
      rotationX: Math.max(0, Math.min(MAX_PITCH_ANGLE, dragState.startRotationX + (dy / height) * 180)),
    }));
  }

  event.preventDefault();
}

function end3DDrag(event) {
  if (dragState.pointerId == null || dragState.pointerId !== event.pointerId) return;
  const shouldSuppressClick = dragState.moved && dragState.mode === 'pan';
  reset3DDrag(event);
  state.suppressNextCanvasClick = shouldSuppressClick;
}

function handle3DWheel(event) {
  if (state.viewMode !== '3d') return;
  event.preventDefault();
  disableFollowEgo();
  setViewState(instantViewState({
    ...state.viewState,
    zoom: (state.viewState.zoom || 0) - event.deltaY * WHEEL_ZOOM_SPEED,
  }));
}

function handleSelectableClick(event, cb) {
  if (consumeSuppressedCanvasClick()) return true;
  if (!isPrimaryPointerEvent(event)) return true;
  cb();
  return true;
}

const deckgl = new DeckGL({
  container: deckContainer,
  views: [new OrthographicView({id:'main'})],
  viewState: state.viewState,
  controller: CONTROLLER_2D,
  getCursor: ({isDragging}) => dragState.mode ? 'grabbing' : (state.viewMode === '3d' ? 'grab' : (isDragging ? 'grabbing' : 'crosshair')),
  pickingRadius: 8,
  layers: [],
  parameters: { clearColor: CLEAR_COLORS[getTheme()] },
  onViewStateChange: ({viewState}) => {
    // Clamp rotationX to avoid flipping upside down in 3D
    if (state.viewMode === '3d') {
      viewState.rotationX = Math.max(0, Math.min(80, viewState.rotationX));
    }
    if (state.followEgo) {
      // Follow-ego locks target; let user control zoom + rotation
      const keep = {zoom: viewState.zoom};
      if (state.viewMode === '3d') {
        keep.rotationX = viewState.rotationX;
        keep.rotationOrbit = viewState.rotationOrbit;
      }
      state.viewState = {...state.viewState, ...keep};
      deckgl.setProps({viewState: state.viewState});
    } else {
      state.viewState = viewState;
      deckgl.setProps({viewState});
    }
    updateZoomDisplay(viewState.zoom);
  },
  onClick: handleCanvasClick,
});

// Suppress right-click context menu on canvas
deckContainer.addEventListener('contextmenu', e => e.preventDefault());
deckContainer.addEventListener('pointerdown', begin3DDrag);
deckContainer.addEventListener('pointermove', update3DDrag);
deckContainer.addEventListener('pointerup', end3DDrag);
deckContainer.addEventListener('pointercancel', end3DDrag);
deckContainer.addEventListener('wheel', handle3DWheel, {passive: false});
syncCanvasCursor();

// ── Layer builders ─────────────────────────────────────────────────────────

function getStaticLayers(scenario, layerFlags) {
  const cacheKey = scenario.metadata.id + JSON.stringify(layerFlags);
  if (state.staticLayerCacheKey === cacheKey) return state.staticLayerCache;

  const roads = scenario.road_map_elements;
  const layers = [];
  const onClick = (cb) => ({
    pickable: true,
    onClick: ({object}, event) => object ? handleSelectableClick(event, () => cb(object)) : false,
  });

  // Helper: pixel-unit path layer
  const pxPath = (id, data, color, width, dashed = false) => new PathLayer({
    id, data,
    getPath: toXY,
    getColor: color,
    getWidth: width,
    widthUnits: 'pixels',
    jointRounded: true,
    capRounded: true,
    getDashArray: dashed ? [8, 5] : [0, 0],
    dashJustified: true,
    extensions: dashed ? [new PathStyleExtension({ dash: true })] : [],
    ...onClick(obj => selectElement('road', obj)),
  });

  // Helper: scatter points — one dot per polyline vertex
  const scatterRoads = (id, data, color, radius = 2) => {
    const pts = data.flatMap(road => road.xyz.map(p => ({pos: [p[0], p[1]], road})));
    return new ScatterplotLayer({
      id, data: pts,
      getPosition: d => d.pos,
      getFillColor: color,
      getRadius: radius, radiusUnits: 'pixels',
      pickable: true,
      onClick: ({object}, event) => object
        ? handleSelectableClick(event, () => selectElement('road', object.road))
        : false,
    });
  };

  // Dispatch to lines or scatter depending on flag.
  // Prefix id so deck.gl sees a new layer (not an in-place type change) when toggling.
  const roadLayer = (id, data, color, width, dashed = false) =>
    layerFlags.scatter_roads
      ? scatterRoads(`sc-${id}`, data, color, Math.max(2, width))
      : pxPath(id, data, color, width, dashed);

  // Lanes (types 0–9)
  if (layerFlags.lanes) {
    const knownLanes = roads.filter(r => r.type >= 1 && r.type <= 9);
    const unknownLanes = roads.filter(r => r.type === 0);
    if (knownLanes.length) layers.push(roadLayer('lanes-known', knownLanes, [205,210,215], 1));
    if (layerFlags.unknowns && unknownLanes.length) layers.push(roadLayer('lanes-unknown', unknownLanes, [124, 58, 237], 1, true));
  }

  // Road lines (types 10–19)
  if (layerFlags.road_lines) {
    const lines = roads.filter(r => r.type >= 10 && r.type <= 19);
    const whiteSolid  = lines.filter(r => [12,13].includes(r.type));
    const whiteDashed = lines.filter(r => r.type === 11);
    const yellowSolid = lines.filter(r => [15,16,17].includes(r.type));
    const yellowDash  = lines.filter(r => [14,18].includes(r.type));
    const unknown     = lines.filter(r => r.type === 10);
    const rest        = lines.filter(r => ![10,11,12,13,14,15,16,17,18].includes(r.type));

    if (whiteSolid.length)  layers.push(roadLayer('rl-white-solid',  whiteSolid,  [160,165,170], 1));
    if (whiteDashed.length) layers.push(roadLayer('rl-white-dashed', whiteDashed, [160,165,170], 1, true));
    if (yellowSolid.length) layers.push(roadLayer('rl-yellow-solid', yellowSolid, [180, 120,  0], 1));
    if (yellowDash.length)  layers.push(roadLayer('rl-yellow-dashed',yellowDash,  [180, 120,  0], 1, true));
    if (layerFlags.unknowns && unknown.length) layers.push(roadLayer('rl-unknown', unknown, [124, 58, 237], 1, true));
    if (rest.length)        layers.push(roadLayer('rl-rest',         rest,        [160,165,170], 1));
  }

  // Road edges (types 20–29)
  if (layerFlags.road_edges) {
    const knownEdges   = roads.filter(r => r.type >= 21 && r.type <= 29);
    const unknownEdges = roads.filter(r => r.type === 20);
    if (knownEdges.length)   layers.push(roadLayer('edges-known',   knownEdges,   [80, 90, 100], 1.5));
    if (layerFlags.unknowns && unknownEdges.length) layers.push(roadLayer('edges-unknown', unknownEdges, [124, 58, 237], 1.5));
  }

  // Crosswalks + speed bumps + stop signs
  if (layerFlags.crosswalks) {
    const cw = roads.filter(r => r.type === 31);
    const sb = roads.filter(r => r.type === 32);
    const ss = roads.filter(r => r.type === 33);
    if (cw.length) layers.push(roadLayer('crosswalks', cw, [217,119,6], 2));
    if (sb.length) layers.push(roadLayer('speed-bumps', sb, [219,39,119], 2));
    if (ss.length) layers.push(new ScatterplotLayer({
      id: 'stop-signs', data: ss,
      getPosition: d => [d.xyz[0], d.xyz[1]],
      getRadius: 3, radiusUnits: 'pixels',
      getFillColor: [220,38,38], getLineColor: [255,255,255],
      stroked: true, lineWidthUnits: 'pixels', getLineWidth: 1,
      ...onClick(obj => selectElement('road', obj)),
    }));
  }

  // Routes — render lane-by-lane to avoid artificial connector jumps.
  if (layerFlags.routes) {
    const egoId = getEgoAgentId(scenario);
    const roadMap = buildRoadMap(roads);
    const routeData = scenario.agents.flatMap(agent =>
      getAgentRouteSegments(agent, roadMap).map(seg => ({
        ...seg,
        color: agent.id === egoId ? EGO_ROUTE_COLOR : ROUTE_COLOR,
        width: agent.id === egoId ? 1.4 : 1,
      }))
    );

    if (routeData.length) layers.push(new PathLayer({
      id: 'routes', data: routeData,
      getPath: d => d.path,
      getColor: d => d.color,
      getWidth: d => d.width, widthUnits: 'pixels',
      getDashArray: [8, 5], dashJustified: true,
      extensions: [new PathStyleExtension({ dash: true })],
      jointRounded: true,
      capRounded: true,
    }));
  }

  state.staticLayerCache = layers;
  state.staticLayerCacheKey = cacheKey;
  return layers;
}

function getDynamicLayers(scenario, t, layerFlags, selected) {
  const egoId = getEgoAgentId(scenario);
  const ttp = new Set(scenario.metadata.tracks_to_predict || []);
  const objects = scenario.objects || [];
  const layers = [];

  if (layerFlags.objects && objects.length) {
    if (layerFlags.trajectories) {
      const objectHistData = objects.flatMap(o => {
        if (!o.xyz.length) return [];
        const color = [...getObjectDisplayColor(o), 180];
        return buildValidSegments(o.xyz, o.valid, 0, t + 1, color);
      });

      if (objectHistData.length) layers.push(new PathLayer({
        id: 'object-traj-history', data: objectHistData,
        getPath: d => d.path, getColor: d => d.color,
        getWidth: 1.25, widthUnits: 'pixels',
        jointRounded: true, capRounded: true,
      }));

      const objectFutureData = objects.flatMap(o => {
        if (!o.xyz.length || t >= o.xyz.length - 1) return [];
        const color = [...getObjectDisplayColor(o), 80];
        return buildValidSegments(o.xyz, o.valid, t, o.xyz.length, color);
      });

      if (objectFutureData.length) layers.push(new PathLayer({
        id: 'object-traj-future', data: objectFutureData,
        getPath: d => d.path, getColor: d => d.color,
        getWidth: 1, widthUnits: 'pixels',
        jointRounded: true, capRounded: true,
        getDashArray: [4, 4], dashJustified: true,
        extensions: [new PathStyleExtension({ dash: true })],
      }));
    }

    const validObjects = objects.filter(o => t < o.valid.length && o.valid[t]);

    if (state.viewMode === '2d') {
      const boxData = validObjects.map(o => ({
        corners: getVehicleCorners(o.xyz[t][0], o.xyz[t][1], o.heading[t], o.length[t] || 1.0, o.width[t] || 1.0),
        color: getObjectDisplayColor(o),
      }));
      if (boxData.length) layers.push(new PolygonLayer({
        id: 'objects-2d', data: boxData,
        getPolygon: d => d.corners,
        getFillColor: d => [...d.color, 170],
        getLineColor: d => [...d.color, 255],
        getLineWidth: 1, lineWidthUnits: 'pixels',
        stroked: true, filled: true, extruded: false,
        pickable: true, onClick: ({object, index}, event) => object
          ? handleSelectableClick(event, () => selectElement('object', validObjects[index]))
          : false,
      }));
    } else {
      const boxData3d = validObjects.map(o => ({
        corners: getVehicleCorners(o.xyz[t][0], o.xyz[t][1], o.heading[t], o.length[t] || 1.0, o.width[t] || 1.0),
        height: o.height[t] || 1.0,
        color: getObjectDisplayColor(o),
      }));
      if (boxData3d.length) layers.push(new PolygonLayer({
        id: 'objects-3d', data: boxData3d,
        getPolygon: d => d.corners,
        getFillColor: d => [...d.color, 170],
        getLineColor: d => [...d.color, 255],
        getElevation: d => d.height,
        stroked: true, filled: true, extruded: true,
        pickable: true, onClick: ({object, index}, event) => object
          ? handleSelectableClick(event, () => selectElement('object', validObjects[index]))
          : false,
      }));
    }

    const arrowData = validObjects.map(o => ({
      path: getHeadingArrow(o.xyz[t][0], o.xyz[t][1], o.heading[t], Math.max(o.length[t] || 1.0, 0.8)),
      color: [...getObjectDisplayColor(o), 220],
    }));
    if (arrowData.length) layers.push(new PathLayer({
      id: 'object-arrows', data: arrowData,
      getPath: d => d.path, getColor: d => d.color,
      getWidth: 1, widthUnits: 'pixels',
      capRounded: true,
    }));
  }

  if (layerFlags.agents) {
    // Trajectory history + future — split at validity gaps to avoid teleportation lines
    if (layerFlags.trajectories) {
      const HIST_WINDOW = 20;
      const FUT_WINDOW  = 30;

      const histData = scenario.agents.flatMap(a => {
        if (!a.xyz.length) return [];
        const color = [...getAgentDisplayColor(a, egoId), 200];
        return buildValidSegments(a.xyz, a.valid, Math.max(0, t - HIST_WINDOW), t + 1, color);
      });

      if (histData.length) layers.push(new PathLayer({
        id: 'traj-history', data: histData,
        getPath: d => d.path, getColor: d => d.color,
        getWidth: 1.5, widthUnits: 'pixels',
        jointRounded: true, capRounded: true,
      }));

      const futData = scenario.agents.flatMap(a => {
        if (!a.xyz.length || t >= a.xyz.length - 1) return [];
        const color = [...getAgentDisplayColor(a, egoId), 90];
        return buildValidSegments(a.xyz, a.valid, t, Math.min(a.xyz.length, t + FUT_WINDOW + 1), color);
      });

      if (futData.length) layers.push(new PathLayer({
        id: 'traj-future', data: futData,
        getPath: d => d.path, getColor: d => d.color,
        getWidth: 1, widthUnits: 'pixels',
        jointRounded: true, capRounded: true,
        getDashArray: [6, 4], dashJustified: true,
        extensions: [new PathStyleExtension({ dash: true })],
      }));
    }

    // Agent boxes
    const validAgents = scenario.agents.filter(a => t < a.valid.length && a.valid[t]);

    if (state.viewMode === '2d') {
      const boxData = validAgents.map(a => {
        const color = getAgentDisplayColor(a, egoId);
        const x = a.xyz[t][0], y = a.xyz[t][1];
        const h = a.heading[t], l = a.length[t] || 4.5, w = a.width[t] || 2;
        return {corners: getVehicleCorners(x, y, h, l, w), color, id: a.id};
      });
      if (boxData.length) layers.push(new PolygonLayer({
        id: 'agents-2d', data: boxData,
        getPolygon: d => d.corners,
        getFillColor: d => [...d.color, 200],
        getLineColor: [255,255,255,230],
        getLineWidth: 1, lineWidthUnits: 'pixels',
        stroked: true, filled: true, extruded: false,
        pickable: true, onClick: ({object, index}, event) => object
          ? handleSelectableClick(event, () => selectElement('agent', validAgents[index]))
          : false,
      }));
    } else {
      // 3D extruded boxes
      const boxData3d = validAgents.map(a => {
        const color = getAgentDisplayColor(a, egoId);
        const x = a.xyz[t][0], y = a.xyz[t][1], z = a.xyz[t][2] || 0;
        const h = a.heading[t], l = a.length[t] || 4.5, w = a.width[t] || 2, ht = a.height[t] || 1.5;
        return {corners: getVehicleCorners(x, y, h, l, w), height: ht, z, color, id: a.id, _agent: a};
      });
      if (boxData3d.length) layers.push(new PolygonLayer({
        id: 'agents-3d', data: boxData3d,
        getPolygon: d => d.corners,
        getFillColor: d => [...d.color, 200],
        getLineColor: [0,0,0,200],
        getElevation: d => d.height,
        stroked: true, filled: true, extruded: true,
        pickable: true, onClick: ({object, index}, event) => object
          ? handleSelectableClick(event, () => selectElement('agent', validAgents[index]))
          : false,
      }));
    }

    // Heading arrows
    const arrowData = validAgents.map(a => {
      const x = a.xyz[t][0], y = a.xyz[t][1];
      return {
        path: getHeadingArrow(x, y, a.heading[t], a.length[t] || 4.5),
        color: [...getAgentDisplayColor(a, egoId), 230],
      };
    });
    if (arrowData.length) layers.push(new PathLayer({
      id: 'agent-arrows', data: arrowData,
      getPath: d => d.path, getColor: d => d.color,
      getWidth: 1, widthUnits: 'pixels',
      capRounded: true,
    }));

    // Agent IDs
    if (layerFlags.agent_ids) {
      const labelData = validAgents.map(a => ({
        pos: [a.xyz[t][0], a.xyz[t][1]],
        text: String(a.id),
        color: a.id === egoId ? [255,255,255] : (agentHasRoute(a) ? [240,240,240] : [148,163,184]),
      }));
      if (labelData.length) layers.push(new TextLayer({
        id: 'agent-ids', data: labelData,
        getPosition: d => d.pos, getText: d => d.text,
        getSize: 9, getColor: d => d.color,
        getTextAnchor: 'middle', getAlignmentBaseline: 'center',
        fontFamily: 'JetBrains Mono, monospace', fontWeight: 600,
      }));
    }

    // TTP markers
    const ttpAgents = validAgents.filter(a => ttp.has(a.id));
    if (ttpAgents.length) layers.push(new ScatterplotLayer({
      id: 'ttp-markers', data: ttpAgents,
      getPosition: a => [a.xyz[t][0], a.xyz[t][1]],
      getRadius: a => (a.width[t] || 2) * 0.6,
      getFillColor: [0,0,0,0],
      getLineColor: [124,58,237,255],
      stroked: true, filled: true,
      lineWidthUnits: 'pixels', getLineWidth: 1.5,
    }));
  }

  // Traffic controls — rendered as stop lines
  if (layerFlags.traffic_controls && scenario.traffic_control_elements.length) {
    const tlElems = scenario.traffic_control_elements.filter(tc => (tc.type || 1) === 1);
    const signElems = scenario.traffic_control_elements.filter(tc => (tc.type || 1) !== 1);

    // Traffic lights — stop line with dynamic state color
    if (tlElems.length) {
      const tlData = tlElems.map(tl => {
        const s = tl.states.length && t < tl.states.length ? tl.states[t] : 4;
        return {path: [tl.stop_line[0].slice(0,2), tl.stop_line[1].slice(0,2)], color: getTlStateColorRgb(s), state: s, tl};
      });
      layers.push(new PathLayer({
        id: 'traffic-lights', data: tlData,
        getPath: d => d.path,
        getColor: d => [...d.color, 230],
        getWidth: 3, widthUnits: 'pixels',
        capRounded: true,
        pickable: true, onClick: ({object}, event) => object
          ? handleSelectableClick(event, () => selectElement('traffic_control', object.tl))
          : false,
      }));
    }

    // Stop/yield signs — striped stop line
    if (signElems.length) {
      const SIGN_COLORS = {2: [[220,38,38], [30,30,30]], 3: [[234,160,8], [30,30,30]]};
      const NUM_STRIPES = 8;
      const stripeSegments = signElems.flatMap(tc => {
        const [a, b] = [tc.stop_line[0].slice(0,2), tc.stop_line[1].slice(0,2)];
        const colors = SIGN_COLORS[tc.type] || [[128,128,128],[30,30,30]];
        return Array.from({length: NUM_STRIPES}, (_, i) => {
          const t0 = i / NUM_STRIPES, t1 = (i + 1) / NUM_STRIPES;
          const p0 = [a[0] + (b[0]-a[0])*t0, a[1] + (b[1]-a[1])*t0];
          const p1 = [a[0] + (b[0]-a[0])*t1, a[1] + (b[1]-a[1])*t1];
          return {path: [p0, p1], color: [...colors[i % 2], 230], tc};
        });
      });
      layers.push(new PathLayer({
        id: 'traffic-signs', data: stripeSegments,
        getPath: d => d.path,
        getColor: d => d.color,
        getWidth: 3, widthUnits: 'pixels',
        pickable: true, onClick: ({object}, event) => object
          ? handleSelectableClick(event, () => selectElement('traffic_control', object.tc))
          : false,
      }));
    }
  }

  // ── Selection highlights ──────────────────────────────────────────────────
  if (selected) {
    const BLUE  = [26, 115, 232];
    const GREEN = [16, 185, 129];
    const AMBER = [217, 119, 6];

    const roadMap = buildRoadMap(scenario.road_map_elements);

    const selPathLayer = (id, data, color, width) => new PathLayer({
      id, data, getPath: toXY,
      getColor: color, getWidth: width, widthUnits: 'pixels',
      jointRounded: true, capRounded: true,
    });

    if (selected.type === 'road') {
      const elem = selected.data;
      const entryElems = (elem.entry_lanes || []).map(id => roadMap.get(id)).filter(Boolean);
      const exitElems  = (elem.exit_lanes  || []).map(id => roadMap.get(id)).filter(Boolean);

      // Main segment — bright blue
      layers.push(selPathLayer('sel-main', [elem], [...BLUE, 255], 3));

      // Entry lanes — green
      if (entryElems.length)
        layers.push(selPathLayer('sel-entry', entryElems, [...GREEN, 200], 2));

      // Exit lanes — amber
      if (exitElems.length)
        layers.push(selPathLayer('sel-exit', exitElems, [...AMBER, 200], 2));

      // Heading chevrons on selected + connected lanes
      const chevronData = [elem, ...entryElems, ...exitElems]
        .flatMap(e => getRoadChevrons(e.xyz));
      if (chevronData.length) layers.push(new PathLayer({
        id: 'sel-chevrons', data: chevronData,
        getPath: d => d.path, getColor: [...BLUE, 220],
        getWidth: 1.5, widthUnits: 'pixels',
        jointRounded: true, capRounded: true,
      }));

    } else if (selected.type === 'agent') {
      const a = selected.data;
      const routeSegments = getAgentRouteSegments(a, roadMap);
      const historySegments = buildValidSegments(a.xyz, a.valid, 0, Math.min(t + 1, a.xyz.length), [...BLUE, 255]);
      const futureSegments = buildValidSegments(a.xyz, a.valid, t, a.xyz.length, [...BLUE, 180]);

      if (routeSegments.length) {
        layers.push(new PathLayer({
          id: 'sel-agent-route', data: routeSegments,
          getPath: d => d.path,
          getColor: [...BLUE, 255],
          getWidth: 3, widthUnits: 'pixels',
          jointRounded: true, capRounded: true,
        }));
      }

      if (historySegments.length) {
        layers.push(new PathLayer({
          id: 'sel-agent-history', data: historySegments,
          getPath: d => d.path,
          getColor: d => d.color,
          getWidth: 2.5, widthUnits: 'pixels',
          jointRounded: true, capRounded: true,
        }));
      }

      if (futureSegments.length) {
        layers.push(new PathLayer({
          id: 'sel-agent-future', data: futureSegments,
          getPath: d => d.path,
          getColor: d => d.color,
          getWidth: 2.5, widthUnits: 'pixels',
          getDashArray: [6, 4], dashJustified: true,
          extensions: [new PathStyleExtension({ dash: true })],
          jointRounded: true, capRounded: true,
        }));
      }

      if (t < a.xyz.length && a.valid[t]) {
        const corners = getVehicleCorners(
          a.xyz[t][0], a.xyz[t][1], a.heading[t],
          a.length[t] || 4.5, a.width[t] || 2
        );
        layers.push(new PolygonLayer({
          id: 'sel-agent',
          data: [{
            corners,
            height: a.height[t] || 1.5,
          }],
          getPolygon: d => d.corners,
          getElevation: d => d.height,
          getFillColor: [...BLUE, 55],
          getLineColor: [...BLUE, 255],
          getLineWidth: 3, lineWidthUnits: 'pixels',
          stroked: true, filled: true,
          extruded: state.viewMode === '3d',
        }));
      }

    } else if (selected.type === 'object') {
      const o = selected.data;
      const historySegments = buildValidSegments(o.xyz, o.valid, 0, Math.min(t + 1, o.xyz.length), [...BLUE, 255]);
      const futureSegments = buildValidSegments(o.xyz, o.valid, t, o.xyz.length, [...BLUE, 180]);

      if (historySegments.length) {
        layers.push(new PathLayer({
          id: 'sel-object-history', data: historySegments,
          getPath: d => d.path,
          getColor: d => d.color,
          getWidth: 2.5, widthUnits: 'pixels',
          jointRounded: true, capRounded: true,
        }));
      }

      if (futureSegments.length) {
        layers.push(new PathLayer({
          id: 'sel-object-future', data: futureSegments,
          getPath: d => d.path,
          getColor: d => d.color,
          getWidth: 2.5, widthUnits: 'pixels',
          getDashArray: [6, 4], dashJustified: true,
          extensions: [new PathStyleExtension({ dash: true })],
          jointRounded: true, capRounded: true,
        }));
      }

      if (t < o.xyz.length && o.valid[t]) {
        const corners = getVehicleCorners(
          o.xyz[t][0], o.xyz[t][1], o.heading[t],
          o.length[t] || 1.0, o.width[t] || 1.0
        );
        layers.push(new PolygonLayer({
          id: 'sel-object',
          data: [{corners, height: o.height[t] || 1.0}],
          getPolygon: d => d.corners,
          getElevation: d => d.height,
          getFillColor: [...BLUE, 55],
          getLineColor: [...BLUE, 255],
          getLineWidth: 3, lineWidthUnits: 'pixels',
          stroked: true, filled: true,
          extruded: state.viewMode === '3d',
        }));
      }

    } else if (selected.type === 'traffic_control') {
      const tl = selected.data;

      // Stop line highlight
      layers.push(new PathLayer({
        id: 'sel-tl', data: [tl],
        getPath: d => [d.stop_line[0].slice(0,2), d.stop_line[1].slice(0,2)],
        getColor: [...BLUE, 255],
        getWidth: 5, widthUnits: 'pixels',
        capRounded: true,
      }));

      // Controlled lanes
      const ctrlElems = (tl.controlled_lanes || []).map(id => roadMap.get(id)).filter(Boolean);
      if (ctrlElems.length) {
        layers.push(selPathLayer('sel-tl-lanes', ctrlElems, [...BLUE, 200], 2.5));
        // Chevrons on controlled lanes too
        const chevronData = ctrlElems.flatMap(e => getRoadChevrons(e.xyz));
        if (chevronData.length) layers.push(new PathLayer({
          id: 'sel-tl-chevrons', data: chevronData,
          getPath: d => d.path, getColor: [...BLUE, 200],
          getWidth: 1.5, widthUnits: 'pixels',
          jointRounded: true, capRounded: true,
        }));
      }
    }
  }

  // ── Path finder highlights ───────────────────────────────────────────────
  const pf = state.pathFinder;
  if (pf.active || pf.path) {
    const roadMap = (selected && selected.type === 'road')
      ? null  // already built above, but we need our own reference
      : null;
    const pfRoadMap = buildRoadMap(scenario.road_map_elements);
    const SRC_COLOR  = [16, 185, 129]; // green
    const DST_COLOR  = [239, 68, 68];  // red
    const PATH_COLOR = [6, 182, 212];  // cyan/teal

    if (pf.source) {
      const srcElem = pfRoadMap.get(pf.source);
      if (srcElem) layers.push(new PathLayer({
        id: 'pf-source', data: [srcElem], getPath: toXY,
        getColor: [...SRC_COLOR, 255], getWidth: 3, widthUnits: 'pixels',
        jointRounded: true, capRounded: true,
      }));
    }
    if (pf.dest) {
      const dstElem = pfRoadMap.get(pf.dest);
      if (dstElem) layers.push(new PathLayer({
        id: 'pf-dest', data: [dstElem], getPath: toXY,
        getColor: [...DST_COLOR, 255], getWidth: 3, widthUnits: 'pixels',
        jointRounded: true, capRounded: true,
      }));
    }
    if (pf.path) {
      const pathElems = pf.path.laneIds
        .filter(id => id !== pf.source && id !== pf.dest)
        .map(id => pfRoadMap.get(id))
        .filter(Boolean);
      if (pathElems.length) layers.push(new PathLayer({
        id: 'pf-path', data: pathElems, getPath: toXY,
        getColor: [...PATH_COLOR, 230], getWidth: 2.5, widthUnits: 'pixels',
        jointRounded: true, capRounded: true,
      }));

      // Connector lines between consecutive path lanes
      const connectors = [];
      const allIds = pf.path.laneIds;
      for (let i = 0; i < allIds.length - 1; i++) {
        const cur = pfRoadMap.get(allIds[i]);
        const nxt = pfRoadMap.get(allIds[i + 1]);
        if (cur && nxt && cur.xyz.length && nxt.xyz.length) {
          const lastPt = cur.xyz[cur.xyz.length - 1];
          const firstPt = nxt.xyz[0];
          connectors.push({ path: [[lastPt[0], lastPt[1]], [firstPt[0], firstPt[1]]] });
        }
      }
      if (connectors.length) layers.push(new PathLayer({
        id: 'pf-connectors', data: connectors,
        getPath: d => d.path,
        getColor: [...PATH_COLOR, 150], getWidth: 1.5, widthUnits: 'pixels',
        getDashArray: [4, 3], dashJustified: true,
        extensions: [new PathStyleExtension({ dash: true })],
      }));
    }
  }

  return layers;
}

// ── Render loop ────────────────────────────────────────────────────────────

function render() {
  if (!state.scenario) return;
  const s = state.scenario;
  const t = state.timestep;
  const staticL = getStaticLayers(s, state.layers);
  const dynL = getDynamicLayers(s, t, state.layers, state.selected);
  const mm = state.viewMode === '2d' ? FLIP_Y : IDENTITY;
  const allLayers = [...staticL, ...dynL].map(
    l => l.clone({modelMatrix: mm})
  );
  deckgl.setProps({layers: allLayers});

  if (state.followEgo) {
    const ego = getEgoState(s, t);
    if (ego) {
      let vs;
      if (state.viewMode === '2d') {
        vs = {...state.viewState, target: [ego.x, -ego.y, 0], transitionDuration: 80};
      } else {
        vs = {
          ...state.viewState,
          target: [ego.x, ego.y, 0],
          transitionDuration: 60,
        };
      }
      state.viewState = vs;
      deckgl.setProps({viewState: vs});
    }
  }
  updateZoomDisplay(state.viewState.zoom);

  document.getElementById('timestep-display').textContent =
    `${t} / ${s.metadata.scenario_length - 1}`;
  document.getElementById('timeline').value = t;
  document.getElementById('timeline').max = s.metadata.scenario_length - 1;

  // Update info panel if agent selected (position changes per timestep)
  if (state.selected && state.selected.type === 'agent') {
    renderElementInfo(state.selected.type, state.selected.data);
  }
  if (state.selected && state.selected.type === 'object') {
    renderElementInfo(state.selected.type, state.selected.data);
  }
  if (state.selected && state.selected.type === 'traffic_control') {
    renderElementInfo(state.selected.type, state.selected.data);
  }
}

// ── Selection & info panel ─────────────────────────────────────────────────

function isLaneType(roadType) {
  return roadType >= TYPES.LANE_RANGE[0] && roadType <= TYPES.LANE_RANGE[1];
}

function handlePathFinderClick(laneId) {
  const pf = state.pathFinder;
  const el = document.getElementById('element-detail');

  if (!pf.source) {
    // Set source
    pf.source = laneId;
    pf.dest = null;
    pf.path = null;
    pf.distance = null;
    el.innerHTML = `<div class="info-row"><span class="info-label">Path Finder</span><span class="info-val">Source: ${laneId}</span></div>
      <div class="info-row"><span class="info-label">Status</span><span class="info-val">Click destination lane</span></div>`;
    setAppStatus('Path Finder: click destination lane', 'info');
  } else if (!pf.dest) {
    // Set destination and compute
    pf.dest = laneId;
    const result = reconstructPath(state.scenario, pf.source, pf.dest);
    if (result) {
      pf.path = result;
      pf.distance = result.distance;
      el.innerHTML = `<div class="info-row"><span class="info-label">Path Finder</span><span class="info-val">${pf.source} → ${pf.dest}</span></div>
        <div class="info-row"><span class="info-label">Distance</span><span class="info-val">${result.distance.toFixed(1)} m</span></div>
        <div class="info-row"><span class="info-label">Lanes</span><span class="info-val">${result.laneIds.length} (${result.laneIds.join(' → ')})</span></div>`;
      setAppStatus(`Path: ${result.distance.toFixed(1)} m, ${result.laneIds.length} lanes`, 'ok');
    } else {
      pf.path = null;
      pf.distance = null;
      el.innerHTML = `<div class="info-row"><span class="info-label">Path Finder</span><span class="info-val">${pf.source} → ${pf.dest}</span></div>
        <div class="info-row"><span class="info-label">Status</span><span class="info-val" style="color:var(--red)">No path found</span></div>`;
      setAppStatus('Path Finder: no path between these lanes', 'error');
    }
  } else {
    // Reset — new source
    pf.source = laneId;
    pf.dest = null;
    pf.path = null;
    pf.distance = null;
    el.innerHTML = `<div class="info-row"><span class="info-label">Path Finder</span><span class="info-val">Source: ${laneId}</span></div>
      <div class="info-row"><span class="info-label">Status</span><span class="info-val">Click destination lane</span></div>`;
    setAppStatus('Path Finder: click destination lane', 'info');
  }
  render();
}

function selectElement(type, data) {
  // Intercept lane clicks in path finder mode
  if (state.pathFinder.active && type === 'road' && isLaneType(data.type)) {
    handlePathFinderClick(data.id);
    return;
  }
  state.selected = {type, data};
  renderElementInfo(type, data);
  render();
}

function handleCanvasClick({coordinate, layer, object}, event) {
  if (consumeSuppressedCanvasClick()) return;
  if (!isPrimaryPointerEvent(event)) return;
  if (!object && !layer) {
    state.selected = null;
    document.getElementById('element-detail').innerHTML = EMPTY_DETAIL_HTML;
    render();
  }
}

function renderElementInfo(type, data) {
  const el = document.getElementById('element-detail');
  const html = renderElementInfoHtml(type, data, {
    t: state.timestep,
    scenario: state.scenario,
    AGENT_TYPE_NAMES: window.TYPES.AGENT_TYPE_NAMES,
    OBJECT_TYPE_NAMES: window.TYPES.OBJECT_TYPE_NAMES,
    ROAD_TYPE_NAMES: window.TYPES.ROAD_TYPE_NAMES,
    TC_TYPE_NAMES: window.TYPES.TC_TYPE_NAMES,
    TL_STATE_NAMES: window.TYPES.TL_STATE_NAMES,
    TL_STATE_COLORS: window.TYPES.TL_STATE_COLORS,
    escapeHtml,
    safeIdList,
    getObjectsOfInterest,
  });
  el.innerHTML = html;
}

// ── Scenario loading ───────────────────────────────────────────────────────

async function loadScenario(filename) {
  document.getElementById('scenario-title').textContent = `Loading ${filename}…`;
  setAppStatus(`Loading ${filename}…`, 'info');
  try {
    reset3DDrag();
    const resp = await fetch(`/api/scenario/${encodeURIComponent(filename)}`);
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    const data = parsePufferBinary(await resp.arrayBuffer());
    state.scenario = data;
    state.timestep = 0;
    state.selected = null;
    state.pathFinder = { active: false, source: null, dest: null, path: null, distance: null };
    document.getElementById('btn-pathfinder').classList.remove('active');
    state.staticLayerCacheKey = null;
    setFollowEgo(false);

    document.getElementById('scenario-title').textContent =
      `${data.metadata.id} [${data.metadata.dataset}]`;

    renderScenarioMeta(data);
    document.getElementById('element-detail').innerHTML = EMPTY_DETAIL_HTML;
    setAppStatus(`Loaded ${filename}`, 'ok');

    fitView();
    // Center on Ego if available
    const ego = getEgoState(data, 0);
    if (ego) setViewState({
      ...state.viewState,
      target: state.viewMode === '3d' ? [ego.x, ego.y, 0] : [ego.x, -ego.y, 0],
    });
    render();
  } catch (err) {
    const errMsg = String(err?.message || err);
    state.scenario = null;
    state.selected = null;
    stopPlay();
    document.getElementById('scenario-title').textContent = `Failed to load ${filename}`;
    document.getElementById('scenario-meta').innerHTML = '<span class="empty-state">Failed to load scenario. Select another file and retry.</span>';
    document.getElementById('element-detail').innerHTML = EMPTY_DETAIL_HTML;
    setAppStatus(`Failed to load ${filename}: ${errMsg}`, 'error');
  }
}

function renderScenarioMeta(data) {
  const m = data.metadata;
  const ooi = getObjectsOfInterest(m);
  const el = document.getElementById('scenario-meta');
  el.innerHTML = `
    <div class="info-row"><span class="info-label">Dataset</span><span class="info-val">${escapeHtml(m.dataset)}</span></div>
    <div class="info-row"><span class="info-label">Length</span><span class="info-val">${escapeHtml(m.scenario_length)}</span></div>
    <div class="info-row"><span class="info-label">Agents</span><span class="info-val">${escapeHtml(data.agents.length)}</span></div>
    <div class="info-row"><span class="info-label">Roads</span><span class="info-val">${escapeHtml(data.road_map_elements.length)}</span></div>
    <div class="info-row"><span class="info-label">TCs</span><span class="info-val">${escapeHtml(data.traffic_control_elements.length)}</span></div>
    <div class="info-row"><span class="info-label">Objects</span><span class="info-val">${escapeHtml((data.objects || []).length)}</span></div>
    <div class="info-row"><span class="info-label">OOI</span><span class="info-val">${safeIdList(ooi)}</span></div>
    <div class="info-row"><span class="info-label">TTP</span><span class="info-val">${safeIdList(m.tracks_to_predict)}</span></div>`;
}

function updateZoomDisplay(zoom) {
  const size = Math.min(deckContainer.clientWidth, deckContainer.clientHeight);
  const radius = size / (2 ** (zoom + 1));
  const rounded = Math.round(radius / 10) * 10;
  document.getElementById('zoom-display').textContent = `${rounded} m`;
}

function setViewState(vs) {
  state.viewState = vs;
  deckgl.setProps({viewState: vs});
  updateZoomDisplay(vs.zoom);
}

function fitView() {
  if (!state.scenario) return;
  const b = sceneBounds(state.scenario);
  const w = b.xmax - b.xmin, h = b.ymax - b.ymin;
  const cw = deckContainer.clientWidth, ch = deckContainer.clientHeight;
  const zoom = Math.log2(Math.min(cw / (w * FIT_VIEW_PADDING), ch / (h * FIT_VIEW_PADDING)));
  const rotation = state.viewMode === '3d'
    ? {rotationX: state.viewState.rotationX || DEFAULT_3D_ROTATION_X, rotationOrbit: state.viewState.rotationOrbit || 0}
    : {};
  setViewState({
    target: [b.cx, state.viewMode === '2d' ? -b.cy : b.cy, 0], zoom, ...rotation,
    transitionDuration: 600,
    transitionInterpolator: LinearInterpolator && new LinearInterpolator(['target', 'zoom']),
  });
  deckgl.setProps({
    views: state.viewMode === '2d'
      ? [new OrthographicView({id:'main'})]
      : [new OrbitView({id:'main', fov:50})],
    controller: state.viewMode === '3d' ? CONTROLLER_3D : CONTROLLER_2D,
  });
}

// ── Scenario list ──────────────────────────────────────────────────────────

let allScenarios = [];
let activeScenario = null;

async function fetchTypes() {
  try {
    const resp = await fetch('/api/types');
    if (resp.ok) Object.assign(window.TYPES, await resp.json());
  } catch (_) { /* use fallbacks */ }
}

async function loadScenarioList() {
  try {
    await fetchTypes();
    const resp = await fetch('/api/scenarios');
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    allScenarios = await resp.json();
    renderScenarioList(allScenarios);
    if (!allScenarios.length) {
      setAppStatus('No scenarios found in selected directory', 'info');
    }
  } catch (err) {
    const errMsg = String(err?.message || err);
    allScenarios = [];
    renderScenarioList([]);
    setAppStatus(`Failed to load scenario list: ${errMsg}`, 'error');
  }
}

function renderScenarioList(items) {
  const container = document.getElementById('scenario-list');
  container.innerHTML = '';
  if (!items.length) {
    container.innerHTML = '<div class="empty-state">No scenarios.</div>';
    return;
  }
  items.forEach(name => {
    const el = document.createElement('div');
    el.className = `scenario-item${name === activeScenario ? ' active' : ''}`;
    el.dataset.name = String(name);
    el.textContent = String(name);
    el.addEventListener('click', () => {
      activeScenario = el.dataset.name;
      renderScenarioList(items);
      loadScenario(activeScenario);
    });
    container.appendChild(el);
  });
}

// ── Playback controls ──────────────────────────────────────────────────────

function setTimestep(t) {
  if (!state.scenario) return;
  state.timestep = Math.max(0, Math.min(t, state.scenario.metadata.scenario_length - 1));
  render();
}

function startPlay() {
  if (state.playTimer) return;
  state.playing = true;
  document.getElementById('btn-play').textContent = '⏸';
  const ms = SPEED_MS[state.speed] || 100;
  state.playTimer = setInterval(() => {
    if (!state.scenario) return;
    const next = state.timestep + 1;
    if (next >= state.scenario.metadata.scenario_length) {
      stopPlay();
      return;
    }
    setTimestep(next);
  }, ms);
}

function stopPlay() {
  if (state.playTimer) clearInterval(state.playTimer);
  state.playTimer = null;
  state.playing = false;
  document.getElementById('btn-play').textContent = '▶';
}

function togglePlay() {
  state.playing ? stopPlay() : startPlay();
}

// ── Event wiring ───────────────────────────────────────────────────────────

document.getElementById('btn-fit').addEventListener('click', fitView);

document.getElementById('btn-follow').addEventListener('click', () => {
  setFollowEgo(!state.followEgo);
  render();
});

document.getElementById('btn-pathfinder').addEventListener('click', () => {
  const pf = state.pathFinder;
  pf.active = !pf.active;
  const btn = document.getElementById('btn-pathfinder');
  btn.classList.toggle('active', pf.active);
  if (pf.active) {
    pf.source = null;
    pf.dest = null;
    pf.path = null;
    pf.distance = null;
    setAppStatus('Path Finder: click source lane', 'info');
    document.getElementById('element-detail').innerHTML =
      '<div class="info-row"><span class="info-label">Path Finder</span><span class="info-val">Click a lane to set source</span></div>';
  } else {
    pf.source = null;
    pf.dest = null;
    pf.path = null;
    pf.distance = null;
    setAppStatus('', 'ok');
    document.getElementById('element-detail').innerHTML = EMPTY_DETAIL_HTML;
  }
  render();
});

document.getElementById('btn-scatter').addEventListener('click', () => {
  state.layers.scatter_roads = !state.layers.scatter_roads;
  document.getElementById('btn-scatter').textContent = state.layers.scatter_roads ? '· Scatter' : '⬤ Lines';
  document.getElementById('btn-scatter').classList.toggle('active', state.layers.scatter_roads);
  state.staticLayerCacheKey = null;
  render();
});

document.getElementById('btn-3d').addEventListener('click', () => {
  reset3DDrag();
  const prevMode = state.viewMode;
  state.viewMode = state.viewMode === '2d' ? '3d' : '2d';
  document.getElementById('btn-3d').textContent = state.viewMode === '3d' ? '3D' : '2D';
  state.staticLayerCacheKey = null;
  if (state.viewMode === '3d') {
    const ego = state.scenario ? getEgoState(state.scenario, state.timestep) : null;
    if (ego) {
      const headingDeg = ego.heading * 180 / Math.PI;
      state.viewState = {
        ...instantViewState(state.viewState),
        target: [ego.x, ego.y, 0],
        rotationOrbit: -headingDeg + 90,
        rotationX: DEFAULT_3D_PITCH,
      };
    } else {
      state.viewState = {
        ...instantViewState(state.viewState),
        target: convertTargetForViewMode(state.viewState.target, prevMode, state.viewMode),
        rotationX: DEFAULT_3D_ROTATION_X,
        rotationOrbit: 0,
      };
    }
    deckgl.setProps({
      views: [new OrbitView({id:'main', fov:50})],
      viewState: state.viewState,
      controller: CONTROLLER_3D,
    });
  } else {
    state.viewState = {
      ...instantViewState(state.viewState),
      target: convertTargetForViewMode(state.viewState.target, prevMode, state.viewMode),
      rotationX: 0,
      rotationOrbit: 0,
    };
    deckgl.setProps({
      views: [new OrthographicView({id:'main'})],
      viewState: state.viewState,
      controller: CONTROLLER_2D,
    });
  }
  syncCanvasCursor();
  render();
});

document.getElementById('btn-play').addEventListener('click', togglePlay);
document.getElementById('btn-prev-first').addEventListener('click', () => { stopPlay(); setTimestep(0); });
document.getElementById('btn-prev').addEventListener('click', () => { stopPlay(); setTimestep(state.timestep - 1); });
document.getElementById('btn-next').addEventListener('click', () => { stopPlay(); setTimestep(state.timestep + 1); });
document.getElementById('btn-next-last').addEventListener('click', () => {
  stopPlay();
  if (state.scenario) setTimestep(state.scenario.metadata.scenario_length - 1);
});

document.getElementById('timeline').addEventListener('input', e => {
  stopPlay();
  setTimestep(parseInt(e.target.value));
});

document.getElementById('speed-select').addEventListener('change', e => {
  state.speed = parseFloat(e.target.value);
  if (state.playing) { stopPlay(); startPlay(); }
});

document.getElementById('scenario-search').addEventListener('input', e => {
  const q = e.target.value.toLowerCase();
  renderScenarioList(allScenarios.filter(s => s.toLowerCase().includes(q)));
});

document.querySelectorAll('#layer-toggles input[type=checkbox]').forEach(cb => {
  cb.addEventListener('change', () => {
    state.layers[cb.dataset.layer] = cb.checked;
    state.staticLayerCacheKey = null;
    render();
  });
});

document.getElementById('btn-search-go').addEventListener('click', () => {
  if (!state.scenario) return;
  const id = parseInt(document.getElementById('search-id').value, 10);
  const type = document.getElementById('search-type').value;
  if (isNaN(id)) return;

  if (type === 'agent') {
    const a = state.scenario.agents.find(a => a.id === id);
    if (a) selectElement('agent', a);
  } else if (type === 'object') {
    const o = (state.scenario.objects || []).find(o => o.id === id);
    if (o) selectElement('object', o);
  } else if (type === 'road') {
    const r = state.scenario.road_map_elements.find(r => r.id === id);
    if (r) selectElement('road', r);
  } else if (type === 'traffic_control') {
    const tc = state.scenario.traffic_control_elements.find(t => t.id === id);
    if (tc) selectElement('traffic_control', tc);
  }
});

document.getElementById('btn-search-clear').addEventListener('click', () => {
  state.selected = null;
  document.getElementById('element-detail').innerHTML = EMPTY_DETAIL_HTML;
  render();
});

// ── Keyboard navigation ────────────────────────────────────────────────────

document.addEventListener('keydown', e => {
  // Skip if typing in an input/select
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

  const ZOOM_STEP = 0.5;

  function zoomBy(delta) {
    if (!state.viewState) return;
    setViewState({
      ...state.viewState,
      zoom: (state.viewState.zoom || 0) + delta,
      transitionDuration: 150,
    });
  }

  switch (e.key) {
    case 'ArrowLeft':  e.preventDefault(); stopPlay(); setTimestep(state.timestep - 1); break;
    case 'ArrowRight': e.preventDefault(); stopPlay(); setTimestep(state.timestep + 1); break;
    case '+': case '=': zoomBy( ZOOM_STEP); break;
    case '-':           zoomBy(-ZOOM_STEP); break;
    case 'f': case 'F': fitView(); break;
    case ' ': e.preventDefault(); togglePlay(); break;
    case 'p': case 'P': document.getElementById('btn-pathfinder').click(); break;
    case 'Escape':
      if (state.pathFinder.active) { document.getElementById('btn-pathfinder').click(); }
      break;
  }
});

// ── Theme toggle ────────────────────────────────────────────────────────────

document.getElementById('btn-theme').addEventListener('click', () => {
  setTheme(getTheme() === 'dark' ? 'light' : 'dark');
});
setTheme(savedTheme);

// ── Init ───────────────────────────────────────────────────────────────────

loadScenarioList();
