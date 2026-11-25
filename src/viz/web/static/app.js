/* deck.gl viewer renderer. */
'use strict';

const {DeckGL, OrthographicView, OrbitView, PathLayer, PolygonLayer, ScatterplotLayer, TextLayer, PathStyleExtension} = window.deck;

// Constants
const SCROLL_ZOOM_SPEED = 0.012;
const WHEEL_ZOOM_SPEED = 0.002;
const MAX_PITCH_ANGLE = 80;
const FIT_VIEW_PADDING = 1.1;
const DEFAULT_3D_PITCH = 20;
const DEFAULT_3D_ROTATION_X = 30;
const CHASE_CAM_PITCH = 20;
const CHASE_CAM_ZOOM = 4.2;
const CHASE_CAM_OFFSET = 8;
const DRAG_THRESHOLD_PX = 4;

// Theme
const CLEAR_COLORS = {
  dark:  [0.031, 0.047, 0.071, 1],   // #080C12
  light: [0.941, 0.949, 0.961, 1],   // #F0F2F5
};
const savedTheme = localStorage.getItem('puffer-viz-theme') || 'dark';

function getTheme() {
  return document.body.getAttribute('data-theme') || 'dark';
}

function setTheme(theme) {
  document.body.setAttribute('data-theme', theme);
  document.body.style.colorScheme = theme;
  localStorage.setItem('puffer-viz-theme', theme);
  document.getElementById('btn-theme').textContent = theme === 'dark' ? '☀' : '☾';
  deckgl.setProps({parameters: {clearColor: CLEAR_COLORS[theme]}});
}

// Flip Y axis: data uses Y-up (north) but OrthographicView uses Y-down (screen).
// Only applied in 2D; 3D uses identity to preserve chirality (driving side).
const FLIP_Y = [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const {
  escapeHtml, safeIdList, getObjectsOfInterest, setAppStatus, getRoadChevrons, getVehicleCorners, getHeadingArrow,
  sceneBounds,
} = window.VizHelpers;
const {renderElementInfoHtml} = window.VizInfoPanel;

// Colors and types

const AGENT_COLORS = {
  0: [107, 114, 128],   // other, gray-500
  1: [37,  99, 235],    // vehicle, blue-600
  2: [5,  150, 105],    // pedestrian, emerald-600
  3: [217, 119,  6],    // cyclist, amber-600
};
const OBJECT_COLORS = {1: [220, 38, 38], 2: [234, 88, 12], 3: [22, 163, 74], 4: [71, 85, 105], 5: [8, 145, 178]};
const EGO_COLOR = [220, 38, 38]; // red-600
const ROUTELESS_COLOR = [148, 163, 184]; // slate-400
const BROWN = [139, 90, 43];
const BLUE = [26, 115, 232];
const GREEN = [16, 185, 129];
const RED = [239, 68, 68];
const AMBER = [217, 119, 6];
const PATH_COLOR = [6, 182, 212];  // cyan/teal
const UNKNOWN_COLOR = [124, 58, 237];

// Type-name and colour tables from bin_factory/puffer_types.py, loaded from /api/types before any scenario.
window.TYPES = {};
// py123d MapLayer values, as in bin_factory/puffer_types.py
const MAP_LAYER = {LANE: 0, LANE_GROUP: 1, INTERSECTION: 2, CROSSWALK: 3, WALKWAY: 4, CARPARK: 5,
                   GENERIC_DRIVABLE: 6, ROAD_EDGE: 8, ROAD_LINE: 9, SPEED_BUMP: 10, SHOULDER: 12};

// RGB version of TL_STATE_COLORS for deck.gl layers
function getTlStateColorRgb(state) {
  const hex = window.TYPES.TL_STATE_COLORS[state] || '#808080';
  return [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
}

const RECORDING_FPS = 30;
const RECORDING_GRID_STEP = 24;
const RECORDING_VIDEO_BITS_PER_SECOND = 16_000_000;
const RECORDING_MIME_TYPES = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];

const EMPTY_DETAIL_HTML = '<span class="empty-state">Click an element to inspect.</span>';

function infoRow(label, value) {
  return `<div class="info-row"><span class="info-label">${label}</span><span class="info-val">${value}</span></div>`;
}

// State

function newPathFinder(active = false) {
  return {active, source: null, dest: null, path: null};
}

function newRuler(active = false) {
  return {active, p1: null, p2: null, distance: null};
}

const state = {
  scenario: null,
  timestep: 0,
  playing: false,
  speed: 1,
  viewMode: '2d',
  followEgo: false,
  layers: { scatter_roads: false },  // checkbox layers are seeded from #layer-toggles
  selected: null,
  pathFinder: newPathFinder(),
  ruler: newRuler(),
  staticLayerCache: null,
  staticLayerCacheKey: null,
  playTimer: null,
  viewState: {target: [0, 0, 0], zoom: 0, rotationX: 0, rotationOrbit: 0},
  suppressNextCanvasClick: false,
  scenarioFilename: null,
  recording: {
    status: 'idle', recorder: null, chunks: [], mimeType: '', canvas: null, ctx: null, stream: null, tickId: null,
    widthCss: 0, heightCss: 0, exportName: null, capturePending: false,
    cachedBgColor: '#080C12', cachedGridColor: 'rgba(242,113,141,.04)',
  },
};

const IDLE_DRAG = {
  pointerId: null, mode: null, startScreen: null, startGround: null, startTarget: null,
  startRotationX: 0, startRotationOrbit: 0, moved: false,
};
const dragState = {...IDLE_DRAG};

// Recording

function getDeckCanvas() { return deckContainer.querySelector('canvas'); }
function supportsRecording() { return typeof window.MediaRecorder !== 'undefined'; }
function isRecording() { return state.recording.status === 'recording'; }
function isRecordingBusy() { return state.recording.status !== 'idle'; }

function getScenarioExportName(filename) {
  const base = String(filename || 'scenario')
    .replace(/\.[^.]+$/, '')
    .replace(/[\\/]+/g, '__')
    .replace(/[^A-Za-z0-9._-]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return base || 'scenario';
}

function getRecordingTitle() {
  if (!supportsRecording()) return 'Recording unsupported in this browser';
  if (!state.scenario) return 'Load a scenario to record';
  if (state.scenario.metadata.scenario_length <= 1) return 'Static maps cannot be recorded';
  if (state.recording.status === 'recording') return 'Stop recording and export mp4';
  if (state.recording.status === 'encoding') return 'Encoding mp4';
  return 'Start mp4 recording';
}

function updateRecordingUi() {
  btnRec.textContent = {recording: 'STOP', encoding: 'ENC'}[state.recording.status] || 'REC';
  btnRec.title = getRecordingTitle();
  btnRec.disabled = !supportsRecording() || !state.scenario || state.scenario.metadata.scenario_length <= 1
    || state.recording.status === 'encoding';
  btnRec.classList.toggle('active', state.recording.status === 'recording');
}

function resetRecordingRuntime() {
  const rec = state.recording;
  if (rec.tickId) clearInterval(rec.tickId);
  if (rec.stream) rec.stream.getTracks().forEach(track => track.stop());
  Object.assign(rec, {
    recorder: null, canvas: null, ctx: null, stream: null, tickId: null, widthCss: 0, heightCss: 0, capturePending: false,
  });
}

function resetRecordingState() {
  resetRecordingRuntime();
  Object.assign(state.recording, {status: 'idle', chunks: [], mimeType: '', exportName: null});
  updateRecordingUi();
}

function resizeRecordingCanvas() {
  const rec = state.recording;
  const rect = canvasContainer.getBoundingClientRect();
  const widthCss = Math.max(Math.round(rect.width), 1);
  const heightCss = Math.max(Math.round(rect.height), 1);
  if (rec.widthCss === widthCss && rec.heightCss === heightCss && rec.canvas) return;
  const dpr = Math.max(window.devicePixelRatio || 1, 1);
  rec.canvas.width = Math.max(Math.round(widthCss * dpr), 1);
  rec.canvas.height = Math.max(Math.round(heightCss * dpr), 1);
  rec.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  rec.widthCss = widthCss;
  rec.heightCss = heightCss;
}

function drawRecordingFrame() {
  if (!isRecording()) return;
  resizeRecordingCanvas();
  const deckCanvas = getDeckCanvas();
  if (!deckCanvas) return;

  const {ctx, widthCss: width, heightCss: height} = state.recording;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = state.recording.cachedBgColor;
  ctx.fillRect(0, 0, width, height);
  // Background grid, as the canvas container draws it behind the deck canvas
  ctx.strokeStyle = state.recording.cachedGridColor;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x <= width; x += RECORDING_GRID_STEP) {
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, height);
  }
  for (let y = 0; y <= height; y += RECORDING_GRID_STEP) {
    ctx.moveTo(0, y + 0.5);
    ctx.lineTo(width, y + 0.5);
  }
  ctx.stroke();
  ctx.drawImage(deckCanvas, 0, 0, width, height);
}

function requestRecordingFrame() {
  if (!isRecording()) return;
  state.recording.capturePending = true;
  deckgl.redraw(true);
}

function handleAfterRender() {
  if (!isRecording() || !state.recording.capturePending) return;
  state.recording.capturePending = false;
  drawRecordingFrame();
}

async function uploadRecording(blob, exportName) {
  const resp = await fetch(`/api/export/mp4?name=${encodeURIComponent(exportName)}`, {
    method: 'POST',
    headers: {'Content-Type': blob.type || 'application/octet-stream'},
    body: blob,
  });
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      detail = (await resp.json()).detail || detail;
    } catch (_) {
      const text = await resp.text();
      if (text) detail = text;
    }
    throw new Error(detail);
  }
  return await resp.blob();
}

async function startRecording() {
  if (isRecordingBusy() || !state.scenario || !supportsRecording()) return;
  if (!getDeckCanvas()) {
    setAppStatus('Deck canvas unavailable for recording', 'error');
    return;
  }

  const {isTypeSupported} = window.MediaRecorder;
  const mimeType = typeof isTypeSupported === 'function'
    ? RECORDING_MIME_TYPES.find(type => window.MediaRecorder.isTypeSupported(type)) || ''
    : '';
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    setAppStatus('2D canvas unavailable for recording', 'error');
    return;
  }

  Object.assign(state.recording, {
    status: 'recording', recorder: null, chunks: [], mimeType, canvas, ctx,
    exportName: getScenarioExportName(state.scenarioFilename),
    cachedGridColor: getComputedStyle(document.body).getPropertyValue('--grid-color').trim() || 'rgba(242,113,141,.04)',
    cachedBgColor: getComputedStyle(canvasContainer).backgroundColor || '#080C12',
  });
  resizeRecordingCanvas();
  try {
    const stream = canvas.captureStream(RECORDING_FPS);
    state.recording.stream = stream;
    const [videoTrack] = stream.getVideoTracks();
    if (videoTrack && 'contentHint' in videoTrack) videoTrack.contentHint = 'detail';
    const recorder = new MediaRecorder(stream, {
      ...(mimeType ? {mimeType} : {}), videoBitsPerSecond: RECORDING_VIDEO_BITS_PER_SECOND,
    });
    recorder.addEventListener('dataavailable', event => {
      if (event.data && event.data.size > 0) state.recording.chunks.push(event.data);
    });
    state.recording.recorder = recorder;
    recorder.start();
    state.recording.tickId = setInterval(requestRecordingFrame, 1000 / RECORDING_FPS);
    requestRecordingFrame();

    updateRecordingUi();
    setAppStatus(`Recording ${state.recording.exportName}.mp4`, 'info');
  } catch (err) {
    resetRecordingState();
    setAppStatus(`Recording failed: ${String(err?.message || err)}`, 'error');
  }
}

async function stopRecording(statusMessage = 'Encoding mp4…') {
  if (!isRecording()) return;
  const {recorder, exportName} = state.recording;
  if (!recorder) {
    resetRecordingState();
    setAppStatus('Recording failed: recorder unavailable', 'error');
    return;
  }
  const stopped = new Promise((resolve, reject) => {
    recorder.addEventListener('stop', resolve, {once: true});
    recorder.addEventListener('error', event => reject(event.error || new Error('Recording failed')), {once: true});
  });

  try {
    if (state.recording.tickId) clearInterval(state.recording.tickId);
    state.recording.tickId = null;
    requestRecordingFrame();
    recorder.stop();
    await stopped;
    state.recording.status = 'encoding';
    updateRecordingUi();
    setAppStatus(statusMessage, 'info');

    const blob = new Blob(state.recording.chunks, {type: state.recording.mimeType || 'video/webm'});
    if (!blob.size) throw new Error('Recording is empty');
    resetRecordingRuntime();
    const url = URL.createObjectURL(await uploadRecording(blob, exportName));
    const link = document.createElement('a');
    link.href = url;
    link.download = `${exportName}.mp4`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 0);
    resetRecordingState();
    setAppStatus(`Saved ${exportName}.mp4`, 'ok');
  } catch (err) {
    resetRecordingState();
    setAppStatus(`Recording failed: ${String(err?.message || err)}`, 'error');
  }
}

async function toggleRecording() {
  if (state.recording.status === 'recording') await stopRecording();
  else if (state.recording.status === 'idle') await startRecording();
}

// Geometry

function getPointZ(point) {
  return point != null && point.length > 2 && Number.isFinite(point[2]) ? point[2] : 0;
}

function toViewPoint(point, zOffset = 0) {
  if (state.viewMode === '3d') return [point[0], point[1], getPointZ(point) + zOffset];
  return [point[0], point[1]];
}

function toViewPath(points) {
  return points.map(point => toViewPoint(point));
}

const toXY = d => toViewPath(d.xyz);
const toPath = d => toViewPath(d.path);

function getEntityPoint(entity, t, zOffset = 0) {
  return toViewPoint(entity.xyz[t], zOffset);
}

function getEntityZ(entity, t) {
  return getPointZ(entity.xyz[t]);
}

function getVehiclePolygon(x, y, heading, length, width, z = 0) {
  return getVehicleCorners(x, y, heading, length, width, state.viewMode === '3d' ? z : null);
}

function getHeadingPath(x, y, heading, length, z = 0) {
  return getHeadingArrow(x, y, heading, length, state.viewMode === '3d' ? z : null);
}

function getSceneMetrics(scenario) {
  if (!scenario) return null;
  if (!scenario.__sceneMetrics) scenario.__sceneMetrics = sceneBounds(scenario);
  return scenario.__sceneMetrics;
}

function niceStep(value) {
  if (!Number.isFinite(value) || value <= 0) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(value));
  const normalized = value / magnitude;
  if (normalized <= 1) return magnitude;
  if (normalized <= 2) return 2 * magnitude;
  if (normalized <= 5) return 5 * magnitude;
  return 10 * magnitude;
}

function getEgoState(scenario, t) {
  const agent = scenario.agents[0];
  if (!agent || !agent.valid[t]) return null;
  return {x: agent.xyz[t][0], y: agent.xyz[t][1], z: getEntityZ(agent, t), heading: agent.heading[t]};
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
      cur.push([xyz[i][0], xyz[i][1], getPointZ(xyz[i])]);
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

  const roadMap = buildRoadMap(scenario.road_polylines);
  const path = [srcId];
  let curIdx = srcIdx;
  let curId = srcId;
  for (let hops = 1; curId !== dstId; hops++) {
    if (hops > lg.n) return null;
    const lane = roadMap.get(curId);
    if (!lane) return null;
    // The next lane is the exit that lies on a shortest path
    const next = (lane.exit_lanes || []).find(
      eid => idToIdx.has(eid) && Math.abs(lane.length + dist(idToIdx.get(eid), dstIdx) - dist(curIdx, dstIdx)) < 0.01,
    );
    if (next === undefined) return null;
    path.push(next);
    curIdx = idToIdx.get(next);
    curId = next;
  }
  return { laneIds: path, distance: totalDist };
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

// Keep the part of a lane polyline between arc lengths s0 and s1, interpolated at both ends.
function cropPathByS(path, cum, s0, s1) {
  if (!cum || cum.length !== path.length) return path;
  const at = s => {
    const found = cum.findIndex(c => c >= s);
    const i = found < 1 ? (found === 0 ? 1 : cum.length - 1) : found;
    const t = Math.min(Math.max((s - cum[i - 1]) / ((cum[i] - cum[i - 1]) || 1), 0), 1);
    return path[i - 1].map((v, k) => v + t * (path[i][k] - v));
  };
  return [at(s0), ...path.filter((_, i) => cum[i] > s0 && cum[i] < s1), at(s1)];
}

function getAgentRouteSegments(agent, roadMap) {
  const firstValid = agent.valid.findIndex(Boolean);
  const startPos = firstValid >= 0 ? agent.xyz[firstValid] : agent.xyz[0];
  const route = Array.isArray(agent.route) ? agent.route : [];
  return route
    .map((laneId, routeIdx) => ({road: roadMap.get(laneId), routeIdx}))
    .filter(({road}) => road && Array.isArray(road.xyz) && road.xyz.length > 1)
    .map(({road, routeIdx}, idx) => {
      const path = cropPathByS(
        road.xyz.map(([x, y, z = 0]) => [x, y, z]),
        road.cum_length,
        agent.route_entry_s[routeIdx],
        agent.route_exit_s[routeIdx],
      );
      return {path: idx === 0 ? cropPathToStart(path, startPos) : path, laneId: road.id, agentId: agent.id};
    })
    .filter(seg => seg.path.length > 1);
}

// deck.gl setup

const canvasContainer = document.getElementById('canvas-container');
const deckContainer = document.getElementById('deckgl-canvas');
const btnRec = document.getElementById('btn-rec');

const CONTROLLER_2D = {
  scrollZoom: { smooth: false, speed: SCROLL_ZOOM_SPEED },
  inertia: 400,
  keyboard: false,
  dragMode: 'pan',
};

const CONTROLLER_3D = false;

function instantViewState(vs) {
  const {transitionDuration, transitionInterpolator, transitionEasing, ...next} = vs;
  return next;
}

// The 3D chase view looking along the ego's heading
function egoOrbitViewState(ego, extra) {
  return {
    ...instantViewState(state.viewState),
    target: [ego.x, ego.y, ego.z],
    rotationOrbit: -(ego.heading * 180 / Math.PI) + 90,
    ...extra,
  };
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

function setFollowEgo(enabled) {
  state.followEgo = enabled;
  document.getElementById('btn-follow').classList.toggle('active', state.followEgo);
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
  Object.assign(dragState, IDLE_DRAG);
  syncCanvasCursor();
}

function begin3DDrag(event) {
  if (state.viewMode !== '3d' || (event.button !== 0 && event.button !== 2)) return;

  const mode = event.button === 0 ? 'pan' : 'rotate';
  const startScreen = getCanvasPosition(event);
  const viewport = getMainViewport();
  Object.assign(dragState, {
    pointerId: event.pointerId,
    mode,
    startScreen,
    startGround: mode === 'pan' && viewport
      ? viewport.unproject(startScreen, {targetZ: state.viewState.target?.[2] || 0})
      : null,
    startTarget: [...(state.viewState.target || [0, 0, 0])],
    startRotationX: state.viewState.rotationX || 0,
    startRotationOrbit: state.viewState.rotationOrbit || 0,
    moved: false,
  });

  deckContainer.setPointerCapture?.(event.pointerId);
  syncCanvasCursor();
  event.preventDefault();
}

function update3DDrag(event) {
  if (state.viewMode !== '3d' || dragState.pointerId == null || dragState.pointerId !== event.pointerId) return;
  if (!dragState.mode || !dragState.startScreen) return;

  const pos = getCanvasPosition(event);
  const dx = pos[0] - dragState.startScreen[0];
  const dy = pos[1] - dragState.startScreen[1];
  if (!dragState.moved) {
    if (Math.hypot(dx, dy) < DRAG_THRESHOLD_PX) return;
    dragState.moved = true;
    if (dragState.mode === 'pan') setFollowEgo(false);
  }

  if (dragState.mode === 'pan') {
    if (state.followEgo) return;
    const viewport = getMainViewport();
    if (!viewport || !dragState.startGround || !dragState.startTarget) return;
    const ground = viewport.unproject(pos, {targetZ: dragState.startTarget[2] || 0});
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
    const next = {
      ...state.viewState,
      rotationX: Math.max(0, Math.min(MAX_PITCH_ANGLE, dragState.startRotationX + (dy / height) * 180)),
    };
    if (!state.followEgo) next.rotationOrbit = dragState.startRotationOrbit + (dx / width) * 180;
    setViewState(instantViewState(next));
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
  setViewState(instantViewState({
    ...state.viewState,
    zoom: (state.viewState.zoom || 0) - event.deltaY * WHEEL_ZOOM_SPEED,
  }));
}

// Whether the click is spent: the click ending a pan, a non-primary button, or a ruler point.
function consumeClick(event, coordinate) {
  if (state.suppressNextCanvasClick) {
    state.suppressNextCanvasClick = false;
    return true;
  }
  if (!isPrimaryPointerEvent(event)) return true;
  if (state.ruler.active && coordinate) {
    handleRulerClick(coordinate);
    return true;
  }
  return false;
}

function handleSelectableClick(event, cb, coordinate) {
  if (!consumeClick(event, coordinate)) cb();
  return true;
}

// Layer props that select the picked element: getItem maps deck's pick info to the element.
function clickable(type, getItem) {
  return {
    pickable: true,
    onClick: (info, event) => (
      info.object ? handleSelectableClick(event, () => selectElement(type, getItem(info)), info.coordinate) : false
    ),
  };
}

const ROUNDED = {jointRounded: true, capRounded: true};

function dashed(dashArray) {
  return {getDashArray: dashArray, dashJustified: true, extensions: [new PathStyleExtension({ dash: true })]};
}

// Paths of width `width` in pixels, each datum in its own color
function colorPaths(id, data, width, props = {}) {
  return new PathLayer({id, data, getPath: toPath, getColor: d => d.color, getWidth: width, widthUnits: 'pixels', ...props});
}

// Whole road polylines in one color
function roadHighlight(id, data, color, width) {
  return new PathLayer({id, data, getPath: toXY, getColor: color, getWidth: width, widthUnits: 'pixels', ...ROUNDED});
}

function build3DAxisLayers(scenario) {
  if (state.viewMode !== '3d') return [];
  const bounds = getSceneMetrics(scenario);
  if (!bounds) return [];
  const zSpan = bounds.zmax - bounds.zmin;
  const baseSpan = Math.max(bounds.xmax - bounds.xmin, bounds.ymax - bounds.ymin, 10);
  const axisHeight = zSpan > 1e-3 ? zSpan : Math.max(10, baseSpan * 0.15);
  const margin = baseSpan * 0.06;
  const foot = Math.max(baseSpan * 0.04, 4);
  const tickStep = niceStep(axisHeight / 4);
  const [ox, oy, oz] = [bounds.xmin - margin, bounds.ymin - margin, bounds.zmin];
  const topZ = bounds.zmin + axisHeight;
  const zColor = [245, 158, 11, 230];
  const xColor = [96, 165, 250, 170];
  const yColor = [74, 222, 128, 170];

  const axisLines = [
    {path: [[ox, oy, oz], [ox, oy, topZ]], color: zColor},
    {path: [[ox, oy, oz], [ox + foot, oy, oz]], color: xColor},
    {path: [[ox, oy, oz], [ox, oy + foot, oz]], color: yColor},
  ];
  const tickLines = [];
  const labels = [
    {position: [ox, oy, topZ + foot * 0.25], text: 'Z', color: zColor},
    {position: [ox + foot * 1.15, oy, oz], text: 'X', color: xColor},
    {position: [ox, oy + foot * 1.15, oz], text: 'Y', color: yColor},
  ];
  for (let z = oz; z <= topZ + tickStep * 0.5; z += tickStep) {
    const tickZ = Math.min(z, topZ);
    tickLines.push({path: [[ox, oy, tickZ], [ox + foot * 0.45, oy, tickZ]], color: zColor});
    labels.push({position: [ox + foot * 0.65, oy, tickZ], text: tickZ.toFixed(tickStep < 1 ? 1 : 0), color: zColor});
    if (tickZ === topZ) break;
  }

  return [
    colorPaths('scene-axis-lines', axisLines, 2, {capRounded: true}),
    colorPaths('scene-axis-ticks', tickLines, 1.5, {capRounded: true}),
    new TextLayer({
      id: 'scene-axis-labels',
      data: labels,
      getPosition: d => d.position,
      getText: d => d.text,
      getSize: d => (d.text.length === 1 ? 12 : 10),
      sizeUnits: 'pixels',
      getColor: d => d.color,
      getTextAnchor: 'start',
      getAlignmentBaseline: 'center',
      fontFamily: 'JetBrains Mono, monospace',
      fontWeight: 600,
      billboard: true,
    }),
  ];
}

const deckgl = new DeckGL({
  container: deckContainer,
  views: [new OrthographicView({id:'main'})],
  viewState: state.viewState,
  controller: CONTROLLER_2D,
  glOptions: {preserveDrawingBuffer: true},
  getCursor: ({isDragging}) => dragState.mode ? 'grabbing' : (state.viewMode === '3d' ? 'grab' : (isDragging ? 'grabbing' : 'crosshair')),
  pickingRadius: 8,
  layers: [],
  parameters: { clearColor: CLEAR_COLORS[getTheme()] },
  onViewStateChange: ({viewState}) => {
    // Clamp rotationX to avoid flipping upside down in 3D
    if (state.viewMode === '3d') viewState.rotationX = Math.max(0, Math.min(MAX_PITCH_ANGLE, viewState.rotationX));
    // Follow-ego locks target; let user control zoom + pitch (not orbit in 3D chase cam)
    state.viewState = state.followEgo
      ? {...state.viewState, zoom: viewState.zoom, ...(state.viewMode === '3d' ? {rotationX: viewState.rotationX} : {})}
      : viewState;
    deckgl.setProps({viewState: state.viewState});
    updateZoomDisplay(viewState.zoom);
  },
  onAfterRender: handleAfterRender,
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

// Layer builders

function getStaticLayers(scenario, layerFlags) {
  const cacheKey = scenario.metadata.id + JSON.stringify(layerFlags);
  if (state.staticLayerCacheKey === cacheKey) return state.staticLayerCache;

  const roads = scenario.road_polylines;
  const layers = [];

  // Path widths use pixels.
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
    ...clickable('road', info => info.object),
  });

  // Draw one dot per vertex, plus heading arrows for polylines (surfaces store no heading).
  const scatterRoads = (id, data, color, radius = 2) => {
    const pts = data.flatMap(road => road.xyz.map((p, i) => ({pos: p, heading: road.heading?.[i], road})));
    const arrowData = pts
      .filter(d => Number.isFinite(d.heading))
      .map(d => ({path: getHeadingPath(d.pos[0], d.pos[1], d.heading, 1.0, d.pos[2] || 0)}));
    return [
      new ScatterplotLayer({
        id, data: pts,
        getPosition: d => toViewPoint(d.pos),
        getFillColor: color,
        getRadius: radius, radiusUnits: 'pixels',
        ...clickable('road', info => info.object.road),
      }),
      new PathLayer({
        id: `${id}-arrows`, data: arrowData,
        getPath: toPath,
        getColor: [...color.slice(0, 3), 160],
        getWidth: 1, widthUnits: 'pixels',
        capRounded: true,
      }),
    ];
  };

  // Draw lines or dots depending on the flag. Prefix the id so deck.gl sees a new layer, not an in-place type change.
  const addRoads = (id, data, color, width, dashed = false) => {
    if (!data.length) return;
    layers.push(layerFlags.scatter_roads
      ? scatterRoads(`sc-${id}`, data, color, Math.max(2, width))
      : pxPath(id, data, color, width, dashed));
  };

  // Lanes; type 0 is py123d LaneType.UNDEFINED
  if (layerFlags.lanes) {
    const lanes = roads.filter(r => r.layer === MAP_LAYER.LANE);
    addRoads('lanes-known', lanes.filter(r => r.type !== 0), [205,210,215], 1);
    if (layerFlags.unknowns) addRoads('lanes-unknown', lanes.filter(r => r.type === 0), UNKNOWN_COLOR, 1, true);
  }

  // Road lines, by py123d RoadLineType; a line that starts dashed is drawn dashed
  if (layerFlags.road_lines) {
    const lines = roads.filter(r => r.layer === MAP_LAYER.ROAD_LINE);
    const ofTypes = types => lines.filter(r => types.includes(r.type));
    addRoads('rl-white-solid', ofTypes([7, 11, 12]), [160,165,170], 1);
    addRoads('rl-white-dashed', ofTypes([3, 4, 9]), [160,165,170], 1, true);
    addRoads('rl-yellow-solid', ofTypes([6, 10]), [180, 120,  0], 1);
    addRoads('rl-yellow-dashed', ofTypes([2, 5, 8, 13]), [180, 120,  0], 1, true);
    if (layerFlags.unknowns) addRoads('rl-unknown', ofTypes([0, 1]), UNKNOWN_COLOR, 1, true);
    addRoads('rl-rest', lines.filter(r => !(r.type >= 0 && r.type <= 13)), [160,165,170], 1);
  }

  // Road edges; type 0 is py123d RoadEdgeType.UNKNOWN
  if (layerFlags.road_edges) {
    const edges = roads.filter(r => r.layer === MAP_LAYER.ROAD_EDGE);
    addRoads('edges-known', edges.filter(r => r.type !== 0), [80, 90, 100], 1.5);
    if (layerFlags.unknowns) addRoads('edges-unknown', edges.filter(r => r.type === 0), UNKNOWN_COLOR, 1.5);
  }

  // Surface outlines
  for (const [flag, layer, id, color] of [
    ['crosswalks',       MAP_LAYER.CROSSWALK,        'crosswalks',       [217,119,6]],
    ['speed_bumps',      MAP_LAYER.SPEED_BUMP,       'speed-bumps',      [219,39,119]],
    ['carparks',         MAP_LAYER.CARPARK,          'carparks',         [8,145,178]],
    ['lane_groups',      MAP_LAYER.LANE_GROUP,       'lane-groups',      [96,165,250]],
    ['intersections',    MAP_LAYER.INTERSECTION,     'intersections',    [147,51,234]],
    ['walkways',         MAP_LAYER.WALKWAY,          'walkways',         [22,163,74]],
    ['generic_drivable', MAP_LAYER.GENERIC_DRIVABLE, 'generic-drivable', [100,116,139]],
    ['shoulders',        MAP_LAYER.SHOULDER,         'shoulders',        [161,98,7]],
  ]) {
    if (layerFlags[flag]) addRoads(id, scenario.road_surfaces.filter(r => r.layer === layer), color, 2);
  }

  state.staticLayerCache = layers;
  state.staticLayerCacheKey = cacheKey;
  return layers;
}

function getDynamicLayers(scenario, t, layerFlags, selected) {
  const egoId = getEgoAgentId(scenario);
  const ttp = new Set(scenario.metadata.tracks_to_predict || []);
  const objects = scenario.objects || [];
  const is3d = state.viewMode === '3d';
  const layers = [];

  if (layerFlags.objects && objects.length) {
    if (layerFlags.trajectories) {
      const history = objects.flatMap(o => (
        o.xyz.length ? buildValidSegments(o.xyz, o.valid, 0, t + 1, [...getObjectDisplayColor(o), 180]) : []
      ));
      if (history.length) layers.push(colorPaths('object-traj-history', history, 1.25, ROUNDED));
      const future = objects.flatMap(o => (
        !o.xyz.length || t >= o.xyz.length - 1
          ? []
          : buildValidSegments(o.xyz, o.valid, t, o.xyz.length, [...getObjectDisplayColor(o), 80])
      ));
      if (future.length) layers.push(colorPaths('object-traj-future', future, 1, {...ROUNDED, ...dashed([4, 4])}));
    }

    const validObjects = objects.filter(o => t < o.valid.length && o.valid[t]);
    const boxes = validObjects.map(o => ({
      corners: getVehiclePolygon(
        o.xyz[t][0], o.xyz[t][1], o.heading[t], o.length[t] || 1.0, o.width[t] || 1.0, getEntityZ(o, t),
      ),
      height: o.height[t] || 1.0,
      color: getObjectDisplayColor(o),
    }));
    if (boxes.length) layers.push(new PolygonLayer({
      id: `objects-${is3d ? '3d' : '2d'}`, data: boxes,
      getPolygon: d => d.corners,
      getFillColor: d => [...d.color, 170],
      getLineColor: d => [...d.color, 255],
      ...(is3d ? {getElevation: d => d.height} : {getLineWidth: 1, lineWidthUnits: 'pixels'}),
      stroked: true, filled: true, extruded: is3d,
      ...clickable('object', info => validObjects[info.index]),
    }));

    const arrows = validObjects.map(o => ({
      path: getHeadingPath(o.xyz[t][0], o.xyz[t][1], o.heading[t], Math.max(o.length[t] || 1.0, 0.8), getEntityZ(o, t)),
      color: [...getObjectDisplayColor(o), 220],
    }));
    if (arrows.length) layers.push(colorPaths('object-arrows', arrows, 1, {capRounded: true}));
  }

  if (layerFlags.agents) {
    // Split trajectory history and future at validity gaps to avoid drawing across missing frames.
    if (layerFlags.trajectories) {
      const HIST_WINDOW = 20;
      const FUT_WINDOW  = 30;
      const history = scenario.agents.flatMap(a => (
        a.xyz.length
          ? buildValidSegments(a.xyz, a.valid, Math.max(0, t - HIST_WINDOW), t + 1, [...getAgentDisplayColor(a, egoId), 200])
          : []
      ));
      if (history.length) layers.push(colorPaths('traj-history', history, 1.5, ROUNDED));
      const future = scenario.agents.flatMap(a => (
        !a.xyz.length || t >= a.xyz.length - 1
          ? []
          : buildValidSegments(
            a.xyz, a.valid, t, Math.min(a.xyz.length, t + FUT_WINDOW + 1), [...getAgentDisplayColor(a, egoId), 90],
          )
      ));
      if (future.length) layers.push(colorPaths('traj-future', future, 1, {...ROUNDED, ...dashed([6, 4])}));
    }

    // Agent boxes, extruded in 3D
    const validAgents = scenario.agents.filter(a => t < a.valid.length && a.valid[t]);
    const boxes = validAgents.map(a => ({
      corners: getVehiclePolygon(
        a.xyz[t][0], a.xyz[t][1], a.heading[t], a.length[t] || 4.5, a.width[t] || 2, a.xyz[t][2] || 0,
      ),
      height: a.height[t] || 1.5,
      color: getAgentDisplayColor(a, egoId),
    }));
    if (boxes.length) layers.push(new PolygonLayer({
      id: `agents-${is3d ? '3d' : '2d'}`, data: boxes,
      getPolygon: d => d.corners,
      getFillColor: d => [...d.color, 200],
      ...(is3d
        ? {getLineColor: [0,0,0,200], getElevation: d => d.height}
        : {getLineColor: [255,255,255,230], getLineWidth: 1, lineWidthUnits: 'pixels'}),
      stroked: true, filled: true, extruded: is3d,
      ...clickable('agent', info => validAgents[info.index]),
    }));

    // Heading arrows
    const arrows = validAgents.map(a => ({
      path: getHeadingPath(a.xyz[t][0], a.xyz[t][1], a.heading[t], a.length[t] || 4.5, getEntityZ(a, t)),
      color: [...getAgentDisplayColor(a, egoId), 230],
    }));
    if (arrows.length) layers.push(colorPaths('agent-arrows', arrows, 1, {capRounded: true}));

    // Agent IDs
    if (layerFlags.agent_ids) {
      const labelData = validAgents.map(a => ({
        pos: getEntityPoint(a, t, Math.max((a.height[t] || 1.5) * 0.7, 0.8)),
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
      getPosition: a => getEntityPoint(a, t),
      getRadius: a => (a.width[t] || 2) * 0.6,
      getFillColor: [0,0,0,0],
      getLineColor: [124,58,237,255],
      stroked: true, filled: true,
      lineWidthUnits: 'pixels', getLineWidth: 1.5,
    }));
  }

  // Draw traffic controls as stop lines.
  if (layerFlags.traffic_controls && scenario.traffic_control_elements.length) {
    const stopLine = tc => tc.stop_line.map(p => [p[0], p[1], getPointZ(p)]);
    const lights = scenario.traffic_control_elements.filter(tc => (tc.type || 1) === 1);
    const signs = scenario.traffic_control_elements.filter(tc => (tc.type || 1) !== 1);

    // Traffic-light stop lines change color with their state.
    if (lights.length) {
      const data = lights.map(tl => {
        const s = tl.states.length && t < tl.states.length ? tl.states[t] : 4;
        return {path: stopLine(tl), color: [...getTlStateColorRgb(s), 230], tl};
      });
      layers.push(colorPaths('traffic-lights', data, 3, {
        capRounded: true, ...clickable('traffic_control', info => info.object.tl),
      }));
    }

    // Stop and yield signs use striped stop lines.
    if (signs.length) {
      const SIGN_COLORS = {2: [[220,38,38], [30,30,30]], 3: [[234,160,8], [30,30,30]]};
      const NUM_STRIPES = 8;
      const stripes = signs.flatMap(tc => {
        const [a, b] = stopLine(tc);
        const at = f => a.map((v, k) => v + (b[k] - v) * f);
        const colors = SIGN_COLORS[tc.type] || [[128,128,128],[30,30,30]];
        return Array.from({length: NUM_STRIPES}, (_, i) => ({
          path: [at(i / NUM_STRIPES), at((i + 1) / NUM_STRIPES)], color: [...colors[i % 2], 230], tc,
        }));
      });
      layers.push(colorPaths('traffic-signs', stripes, 3, clickable('traffic_control', info => info.object.tc)));
    }
  }

  // Selection and path finder highlights
  const pf = state.pathFinder;
  const roadMap = selected || pf.active || pf.path ? buildRoadMap(scenario.road_polylines) : null;
  const roadsById = ids => (ids || []).map(id => roadMap.get(id)).filter(Boolean);
  const addChevrons = (id, roads, color) => {
    const data = roads.flatMap(e => getRoadChevrons(e.xyz));
    if (data.length) layers.push(new PathLayer({
      id, data, getPath: toPath, getColor: color, getWidth: 1.5, widthUnits: 'pixels', ...ROUNDED,
    }));
  };
  if (selected?.type === 'road') {
    const elem = selected.data;
    const entryElems = roadsById(elem.entry_lanes);
    const exitElems = roadsById(elem.exit_lanes);
    layers.push(roadHighlight('sel-main', [elem], [...BLUE, 255], 3));
    if (entryElems.length) layers.push(roadHighlight('sel-entry', entryElems, [...GREEN, 200], 2));
    if (exitElems.length) layers.push(roadHighlight('sel-exit', exitElems, [...AMBER, 200], 2));
    addChevrons('sel-chevrons', [elem, ...entryElems, ...exitElems], [...BLUE, 220]);
  } else if (selected?.type === 'agent' || selected?.type === 'object') {
    const item = selected.data;
    const isAgent = selected.type === 'agent';
    const color = isAgent ? getAgentDisplayColor(item, egoId) : BLUE;
    if (isAgent) {
      const routeSegments = getAgentRouteSegments(item, roadMap);
      if (routeSegments.length) layers.push(new PathLayer({
        id: 'sel-agent-route', data: routeSegments,
        getPath: toPath, getColor: [...BROWN, 200], getWidth: 2, widthUnits: 'pixels', ...ROUNDED,
      }));
    }
    const history = buildValidSegments(item.xyz, item.valid, 0, Math.min(t + 1, item.xyz.length), [...color, 255]);
    const future = buildValidSegments(item.xyz, item.valid, t, item.xyz.length, [...color, 180]);
    if (history.length) layers.push(colorPaths(`sel-${selected.type}-history`, history, 2.5, ROUNDED));
    if (future.length) {
      layers.push(colorPaths(`sel-${selected.type}-future`, future, 2.5, {...dashed([6, 4]), ...ROUNDED}));
    }

    if (t < item.xyz.length && item.valid[t]) {
      const [length, width, height] = isAgent ? [4.5, 2, 1.5] : [1.0, 1.0, 1.0];
      const corners = getVehiclePolygon(
        item.xyz[t][0], item.xyz[t][1], item.heading[t],
        item.length[t] || length, item.width[t] || width,
        getEntityZ(item, t),
      );
      layers.push(new PolygonLayer({
        id: `sel-${selected.type}`,
        data: [{corners, height: item.height[t] || height}],
        getPolygon: d => d.corners,
        getElevation: d => d.height,
        getFillColor: [...color, 55],
        getLineColor: [...color, 255],
        getLineWidth: 3, lineWidthUnits: 'pixels',
        stroked: true, filled: true,
        extruded: is3d,
      }));
    }
  } else if (selected?.type === 'traffic_control') {
    const tl = selected.data;
    layers.push(new PathLayer({
      id: 'sel-tl', data: [tl],
      getPath: d => toViewPath(d.stop_line), getColor: [...BLUE, 255], getWidth: 5, widthUnits: 'pixels',
      capRounded: true,
    }));
    const ctrlElems = roadsById(tl.controlled_lanes);
    if (ctrlElems.length) {
      layers.push(roadHighlight('sel-tl-lanes', ctrlElems, [...BLUE, 200], 2.5));
      addChevrons('sel-tl-chevrons', ctrlElems, [...BLUE, 200]);
    }
  }

  if (pf.active || pf.path) {
    if (pf.source !== null && roadMap.has(pf.source)) {
      layers.push(roadHighlight('pf-source', [roadMap.get(pf.source)], [...GREEN, 255], 3));
    }
    if (pf.dest !== null && roadMap.has(pf.dest)) layers.push(roadHighlight('pf-dest', [roadMap.get(pf.dest)], [...RED, 255], 3));
    if (pf.path) {
      const pathElems = roadsById(pf.path.laneIds.filter(id => id !== pf.source && id !== pf.dest));
      if (pathElems.length) layers.push(roadHighlight('pf-path', pathElems, [...PATH_COLOR, 230], 2.5));

      // Connector lines between consecutive path lanes
      const connectors = [];
      const ids = pf.path.laneIds;
      for (let i = 0; i < ids.length - 1; i++) {
        const cur = roadMap.get(ids[i]);
        const nxt = roadMap.get(ids[i + 1]);
        if (cur && nxt && cur.xyz.length && nxt.xyz.length) {
          connectors.push({path: [cur.xyz[cur.xyz.length - 1], nxt.xyz[0]].map(p => [p[0], p[1], getPointZ(p)])});
        }
      }
      if (connectors.length) layers.push(new PathLayer({
        id: 'pf-connectors', data: connectors, getPath: toPath,
        getColor: [...PATH_COLOR, 150], getWidth: 1.5, widthUnits: 'pixels', ...dashed([4, 3]),
      }));
    }
  }

  // Ruler highlights
  const rl = state.ruler;
  if (rl.active || rl.p1) {
    const dots = [];
    if (rl.p1) dots.push({ position: rl.p1, color: [...GREEN, 255] });
    if (rl.p2) dots.push({ position: rl.p2, color: [...RED, 255] });
    if (dots.length) layers.push(new ScatterplotLayer({
      id: 'ruler-dots', data: dots,
      getPosition: d => d.position,
      getFillColor: d => d.color,
      getRadius: 4, radiusUnits: 'pixels',
    }));

    if (rl.p1 && rl.p2) {
      layers.push(new PathLayer({
        id: 'ruler-line', data: [{ path: [rl.p1, rl.p2] }],
        getPath: toPath,
        getColor: [245, 158, 11, 220], getWidth: 2, widthUnits: 'pixels', ...dashed([6, 4]),
      }));

      const mid = [(rl.p1[0] + rl.p2[0]) / 2, (rl.p1[1] + rl.p2[1]) / 2];
      if (is3d) mid.push((getPointZ(rl.p1) + getPointZ(rl.p2)) / 2);
      layers.push(new TextLayer({
        id: 'ruler-label', data: [{ position: mid, text: `${rl.distance.toFixed(2)} m` }],
        getPosition: d => d.position,
        getText: d => d.text,
        getSize: 14, sizeUnits: 'pixels',
        getColor: [255, 255, 255, 255],
        getBackgroundColor: [0, 0, 0, 180],
        background: true, backgroundPadding: [4, 2],
        fontFamily: 'monospace',
        getTextAnchor: 'middle', getAlignmentBaseline: 'center',
        getPixelOffset: [0, -16],
      }));
    }
  }

  return layers;
}

// Render loop

function render() {
  if (!state.scenario) return;
  const s = state.scenario;
  const t = state.timestep;
  const modelMatrix = state.viewMode === '2d' ? FLIP_Y : IDENTITY;
  const layers = [getStaticLayers(s, state.layers), build3DAxisLayers(s), getDynamicLayers(s, t, state.layers, state.selected)];
  deckgl.setProps({layers: layers.flat(2).map(l => l.clone({modelMatrix}))});

  const ego = state.followEgo ? getEgoState(s, t) : null;
  if (ego && state.viewMode === '2d') {
    setViewState({...state.viewState, target: [ego.x, -ego.y, 0], transitionDuration: 80});
  } else if (ego) {
    setViewState({
      ...state.viewState,
      target: [ego.x + CHASE_CAM_OFFSET * Math.cos(ego.heading), ego.y + CHASE_CAM_OFFSET * Math.sin(ego.heading), ego.z],
      rotationOrbit: -(ego.heading * 180 / Math.PI) + 90,
      transitionDuration: 60,
    });
  }
  updateZoomDisplay(state.viewState.zoom);

  const finalTimestep = Math.max(s.metadata.scenario_length - 1, 0);
  document.getElementById('timestep-display').textContent = `${t} / ${finalTimestep}`;
  document.getElementById('timeline').value = t;
  document.getElementById('timeline').max = finalTimestep;

  // Agent, object and traffic-control details change with the timestep
  if (state.selected && state.selected.type !== 'road') renderElementInfo(state.selected.type, state.selected.data);
}

// Selection and info panel

function handlePathFinderClick(laneId) {
  const pf = state.pathFinder;
  const el = document.getElementById('element-detail');

  if (pf.source === null || pf.dest !== null) {
    // Set the source, starting over after a completed search
    Object.assign(pf, {source: laneId, dest: null, path: null});
    el.innerHTML = infoRow('Path Finder', `Source: ${laneId}`) + infoRow('Status', 'Click destination lane');
    setAppStatus('Path Finder: click destination lane', 'info');
  } else {
    // Set destination and compute
    pf.dest = laneId;
    pf.path = reconstructPath(state.scenario, pf.source, pf.dest);
    const header = infoRow('Path Finder', `${pf.source} → ${pf.dest}`);
    if (pf.path) {
      const {distance, laneIds} = pf.path;
      el.innerHTML = header + infoRow('Distance', `${distance.toFixed(1)} m`)
        + infoRow('Lanes', `${laneIds.length} (${laneIds.join(' → ')})`);
      setAppStatus(`Path: ${distance.toFixed(1)} m, ${laneIds.length} lanes`, 'ok');
    } else {
      el.innerHTML = header + '<div class="info-row"><span class="info-label">Status</span>'
        + '<span class="info-val" style="color:var(--red)">No path found</span></div>';
      setAppStatus('Path Finder: no path between these lanes', 'error');
    }
  }
  render();
}

function selectElement(type, data) {
  // Intercept lane clicks in path finder mode
  if (state.pathFinder.active && type === 'road' && data.layer === MAP_LAYER.LANE) {
    handlePathFinderClick(data.id);
    return;
  }
  state.selected = {type, data};
  renderElementInfo(type, data);
  render();
}

function handleRulerClick(coordinate) {
  const r = state.ruler;
  const el = document.getElementById('element-detail');
  // Convert view-space coordinate to data-space (un-flip Y in 2D)
  const pt = state.viewMode === '2d'
    ? [coordinate[0], -coordinate[1]]
    : [coordinate[0], coordinate[1], coordinate[2] || 0];

  if (!r.p1 || r.p2) {
    // Set the first point, starting over after a completed measurement
    Object.assign(r, {p1: pt, p2: null, distance: null});
    el.innerHTML = infoRow('Ruler', `P1: (${pt[0].toFixed(1)}, ${pt[1].toFixed(1)})`) + infoRow('Status', 'Click second point');
    setAppStatus('Ruler: click second point', 'info');
  } else {
    r.p2 = pt;
    r.distance = Math.hypot(r.p2[0] - r.p1[0], r.p2[1] - r.p1[1]);
    el.innerHTML = infoRow('Ruler', 'P1 → P2') + infoRow('Distance', `${r.distance.toFixed(2)} m`);
    setAppStatus(`Ruler: ${r.distance.toFixed(2)} m`, 'ok');
  }
  render();
}

function handleCanvasClick({coordinate, layer, object}, event) {
  if (consumeClick(event, coordinate) || object || layer) return;
  state.selected = null;
  document.getElementById('element-detail').innerHTML = EMPTY_DETAIL_HTML;
  render();
}

function renderElementInfo(type, data) {
  document.getElementById('element-detail').innerHTML = renderElementInfoHtml(type, data, {
    t: state.timestep, scenario: state.scenario, ...window.TYPES, escapeHtml, safeIdList, getObjectsOfInterest,
  });
}

// Scenario loading

async function loadScenario(filename) {
  document.getElementById('scenario-title').textContent = `Loading ${filename}…`;
  setAppStatus(`Loading ${filename}…`, 'info');
  try {
    if (isRecording()) await stopRecording('Finalizing recording before loading scenario…');
    stopPlay();
    reset3DDrag();
    const resp = await fetch(`/api/scenario/${filename.split('/').map(encodeURIComponent).join('/')}`, {cache: 'no-store'});
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = parsePufferBinary(await resp.arrayBuffer());
    Object.assign(state, {
      scenario: data, scenarioFilename: filename, timestep: 0, selected: null,
      pathFinder: newPathFinder(), ruler: newRuler(), staticLayerCacheKey: null,
    });
    document.getElementById('btn-pathfinder').classList.remove('active');
    document.getElementById('btn-ruler').classList.remove('active');
    setFollowEgo(false);

    document.getElementById('scenario-title').textContent = `${data.metadata.id} [${data.metadata.dataset}]`;
    renderScenarioMeta(data);
    document.getElementById('element-detail').innerHTML = EMPTY_DETAIL_HTML;
    setAppStatus(`Loaded ${filename}`, 'ok');
    updatePlaybackUi();
    updateRecordingUi();

    fitView();
    // Center on Ego if available
    const ego = getEgoState(data, 0);
    if (ego) setViewState(instantViewState({
      ...state.viewState,
      target: state.viewMode === '3d' ? [ego.x, ego.y, ego.z] : [ego.x, -ego.y, 0],
    }));
    render();
  } catch (err) {
    Object.assign(state, {scenario: null, scenarioFilename: null, selected: null});
    stopPlay();
    document.getElementById('scenario-title').textContent = `Failed to load ${filename}`;
    document.getElementById('scenario-meta').innerHTML = '<span class="empty-state">Failed to load scenario. Select another file and retry.</span>';
    document.getElementById('element-detail').innerHTML = EMPTY_DETAIL_HTML;
    setAppStatus(`Failed to load ${filename}: ${String(err?.message || err)}`, 'error');
    updatePlaybackUi();
    updateRecordingUi();
  }
}

function renderScenarioMeta(data) {
  const m = data.metadata;
  document.getElementById('scenario-meta').innerHTML = [
    infoRow('Dataset', escapeHtml(m.dataset)),
    infoRow('Length', escapeHtml(m.scenario_length)),
    infoRow('Agents', escapeHtml(data.agents.length)),
    infoRow('Roads', escapeHtml(data.road_polylines.length + data.road_surfaces.length)),
    infoRow('TCs', escapeHtml(data.traffic_control_elements.length)),
    infoRow('Objects', escapeHtml((data.objects || []).length)),
    infoRow('OOI', safeIdList(getObjectsOfInterest(m))),
    infoRow('TTP', safeIdList(m.tracks_to_predict)),
  ].join('');
}

function updateZoomDisplay(zoom) {
  const size = Math.min(deckContainer.clientWidth, deckContainer.clientHeight);
  const radius = size / (2 ** (zoom + 1));
  document.getElementById('zoom-display').textContent = `${Math.round(radius / 10) * 10} m`;
}

function setViewState(vs) {
  state.viewState = vs;
  deckgl.setProps({viewState: vs});
  updateZoomDisplay(vs.zoom);
}

function fitView() {
  if (!state.scenario) return;
  const b = getSceneMetrics(state.scenario);
  const is3d = state.viewMode === '3d';
  const rotationX = is3d ? (state.viewState.rotationX || DEFAULT_3D_ROTATION_X) : 0;
  const projectedZ = is3d ? Math.max(b.zmax - b.zmin, 0) * Math.sin(rotationX * Math.PI / 180) : 0;
  const fitWidth = Math.max(b.xmax - b.xmin, 1);
  const fitHeight = Math.max(Math.max(b.ymax - b.ymin, 1) + projectedZ, 1);
  const zoom = Math.log2(Math.min(
    Math.max(deckContainer.clientWidth, 1) / (fitWidth * FIT_VIEW_PADDING),
    Math.max(deckContainer.clientHeight, 1) / (fitHeight * FIT_VIEW_PADDING),
  ));
  setViewState({
    target: [b.cx, is3d ? b.cy : -b.cy, is3d ? b.cz : 0], zoom,
    ...(is3d ? {rotationX, rotationOrbit: state.viewState.rotationOrbit || 0} : {}),
  });
  deckgl.setProps({
    views: is3d ? [new OrbitView({id:'main', fov:50})] : [new OrthographicView({id:'main'})],
    controller: is3d ? CONTROLLER_3D : CONTROLLER_2D,
  });
}

// Scenario list

let allScenarios = [];
let activeScenario = null;

async function loadScenarioList() {
  try {
    const types = await fetch('/api/types');
    if (!types.ok) throw new Error(`HTTP ${types.status}`);
    window.TYPES = await types.json();
    const resp = await fetch('/api/scenarios');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    allScenarios = await resp.json();
    renderScenarioList(allScenarios);
    if (!allScenarios.length) setAppStatus('No scenarios found in selected directory', 'info');
  } catch (err) {
    allScenarios = [];
    renderScenarioList([]);
    setAppStatus(`Failed to load scenario list: ${String(err?.message || err)}`, 'error');
  }
}

function renderScenarioList(items) {
  const container = document.getElementById('scenario-list');
  container.innerHTML = items.length ? '' : '<div class="empty-state">No scenarios.</div>';
  for (const name of items) {
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
  }
}

// Playback controls

function setTimestep(t) {
  if (!state.scenario) return;
  state.timestep = Math.max(0, Math.min(t, state.scenario.metadata.scenario_length - 1));
  render();
}

function updatePlaybackUi() {
  const disabled = !state.scenario || state.scenario.metadata.scenario_length <= 1;
  for (const id of ['btn-prev-first', 'btn-prev', 'btn-play', 'btn-next', 'btn-next-last', 'timeline', 'speed-select']) {
    document.getElementById(id).disabled = disabled;
  }
}

function startPlay() {
  if (state.playTimer || !state.scenario || state.scenario.metadata.scenario_length <= 1) return;
  state.playing = true;
  document.getElementById('btn-play').textContent = '⏸';
  const ms = 1000 * state.scenario.metadata.dt / state.speed;
  state.playTimer = setInterval(() => {
    if (!state.scenario) return;
    const next = state.timestep + 1;
    if (next >= state.scenario.metadata.scenario_length) {
      stopPlay();
      if (isRecording()) void stopRecording('Reached final timestep, encoding mp4…');
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

// Path finder and ruler exclude each other: switching one on or off resets both.
function toggleTool(tool) {
  const active = !state[tool].active;
  state.pathFinder = newPathFinder(tool === 'pathFinder' && active);
  state.ruler = newRuler(tool === 'ruler' && active);
  document.getElementById('btn-pathfinder').classList.toggle('active', state.pathFinder.active);
  document.getElementById('btn-ruler').classList.toggle('active', state.ruler.active);
  const [status, hint] = tool === 'ruler'
    ? ['Ruler: click first point', infoRow('Ruler', 'Click anywhere to set first point')]
    : ['Path Finder: click source lane', infoRow('Path Finder', 'Click a lane to set source')];
  setAppStatus(active ? status : '', active ? 'info' : 'ok');
  document.getElementById('element-detail').innerHTML = active ? hint : EMPTY_DETAIL_HTML;
  render();
}

// Event handlers

document.getElementById('btn-fit').addEventListener('click', fitView);

document.getElementById('btn-follow').addEventListener('click', () => {
  setFollowEgo(!state.followEgo);
  const ego = state.followEgo && state.viewMode === '3d' && state.scenario
    ? getEgoState(state.scenario, state.timestep)
    : null;
  if (ego) {
    state.viewState = egoOrbitViewState(ego, {rotationX: CHASE_CAM_PITCH, zoom: CHASE_CAM_ZOOM});
    deckgl.setProps({viewState: state.viewState});
  }
  render();
});

document.getElementById('btn-pathfinder').addEventListener('click', () => toggleTool('pathFinder'));
document.getElementById('btn-ruler').addEventListener('click', () => toggleTool('ruler'));

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
  const is3d = state.viewMode === '3d';
  document.getElementById('btn-3d').textContent = is3d ? '3D' : '2D';
  state.staticLayerCacheKey = null;
  const target = convertTargetForViewMode(state.viewState.target, prevMode, state.viewMode);
  const ego = is3d && state.scenario ? getEgoState(state.scenario, state.timestep) : null;
  if (ego) {
    state.viewState = egoOrbitViewState(ego, {rotationX: DEFAULT_3D_PITCH});
  } else if (is3d) {
    const z = state.scenario ? getSceneMetrics(state.scenario).cz : target[2];
    state.viewState = {
      ...instantViewState(state.viewState),
      target: [target[0], target[1], z || 0],
      rotationX: DEFAULT_3D_ROTATION_X,
      rotationOrbit: 0,
    };
  } else {
    state.viewState = {...instantViewState(state.viewState), target, rotationX: 0, rotationOrbit: 0};
  }
  deckgl.setProps({
    views: [is3d ? new OrbitView({id:'main', fov:50}) : new OrthographicView({id:'main'})],
    viewState: state.viewState,
    controller: is3d ? CONTROLLER_3D : CONTROLLER_2D,
  });
  syncCanvasCursor();
  render();
});

const stepTo = t => { stopPlay(); setTimestep(t); };
document.getElementById('btn-play').addEventListener('click', togglePlay);
document.getElementById('btn-prev-first').addEventListener('click', () => stepTo(0));
document.getElementById('btn-prev').addEventListener('click', () => stepTo(state.timestep - 1));
document.getElementById('btn-next').addEventListener('click', () => stepTo(state.timestep + 1));
document.getElementById('btn-next-last').addEventListener('click', () => {
  stopPlay();
  if (state.scenario) setTimestep(state.scenario.metadata.scenario_length - 1);
});
btnRec.addEventListener('click', () => { void toggleRecording(); });

document.getElementById('timeline').addEventListener('input', e => stepTo(parseInt(e.target.value)));

document.getElementById('speed-select').addEventListener('change', e => {
  state.speed = parseFloat(e.target.value);
  if (state.playing) { stopPlay(); startPlay(); }
});

document.getElementById('scenario-search').addEventListener('input', e => {
  const q = e.target.value.toLowerCase();
  renderScenarioList(allScenarios.filter(s => s.toLowerCase().includes(q)));
});

document.querySelectorAll('#layer-toggles input[type=checkbox]').forEach(cb => {
  state.layers[cb.dataset.layer] = cb.checked;
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

  const collections = {
    agent: state.scenario.agents,
    object: state.scenario.objects || [],
    road: [...state.scenario.road_polylines, ...state.scenario.road_surfaces],
    traffic_control: state.scenario.traffic_control_elements,
  };
  const item = collections[type]?.find(item => item.id === id);
  if (item) selectElement(type, item);
});

document.getElementById('btn-search-clear').addEventListener('click', () => {
  state.selected = null;
  document.getElementById('element-detail').innerHTML = EMPTY_DETAIL_HTML;
  render();
});

// Keyboard navigation

document.addEventListener('keydown', e => {
  // Skip if typing in an input/select
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

  const zoomBy = delta => setViewState({...state.viewState, zoom: (state.viewState.zoom || 0) + delta, transitionDuration: 150});
  switch (e.key) {
    case 'ArrowLeft':  e.preventDefault(); stepTo(state.timestep - 1); break;
    case 'ArrowRight': e.preventDefault(); stepTo(state.timestep + 1); break;
    case '+': case '=': zoomBy(0.5); break;
    case '-':           zoomBy(-0.5); break;
    case 'f': case 'F': fitView(); break;
    case ' ': e.preventDefault(); togglePlay(); break;
    case 'p': case 'P': document.getElementById('btn-pathfinder').click(); break;
    case 'r': case 'R': document.getElementById('btn-ruler').click(); break;
    case 'Escape':
      if (state.pathFinder.active) { document.getElementById('btn-pathfinder').click(); }
      else if (state.ruler.active) { document.getElementById('btn-ruler').click(); }
      break;
  }
});

// Theme toggle

document.getElementById('btn-theme').addEventListener('click', () => {
  setTheme(getTheme() === 'dark' ? 'light' : 'dark');
});
setTheme(savedTheme);
updatePlaybackUi();
updateRecordingUi();

window.addEventListener('beforeunload', event => {
  if (!isRecordingBusy()) return;
  event.preventDefault();
  event.returnValue = '';
});

// Initialization

loadScenarioList();
