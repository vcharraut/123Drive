/* Puffer Viz — deck.gl client-side renderer */
'use strict';

const {DeckGL, OrthographicView, OrbitView, PathLayer, PolygonLayer, ScatterplotLayer, TextLayer, PathStyleExtension, LinearInterpolator} = window.deck;

// ── Constants ──────────────────────────────────────────────────────────────

const AGENT_COLORS = {
  0: [107, 114, 128],   // unset   – gray-500
  1: [37,  99, 235],    // vehicle – blue-600
  2: [5,  150, 105],    // pedestrian – emerald-600
  3: [217, 119,  6],    // cyclist – amber-600
  4: [107, 114, 128],   // other   – gray-500
};
const EGO_COLOR = [220, 38, 38]; // red-600

const AGENT_TYPE_NAMES = {0:'unset', 1:'vehicle', 2:'pedestrian', 3:'cyclist', 4:'other'};

const ROAD_TYPE_NAMES = {
  0:'lane_unknown', 1:'lane_freeway', 2:'lane_surface_street', 3:'lane_bike',
  10:'road_line_unknown', 11:'road_line_broken_white', 12:'road_line_solid_white',
  13:'road_line_double_white', 14:'road_line_broken_yellow', 15:'road_line_double_yellow',
  16:'road_line_solid_yellow', 17:'road_line_solid_double_yellow', 18:'road_line_passing_yellow',
  20:'road_edge_unknown', 21:'road_edge_boundary', 22:'road_edge_median', 23:'road_edge_sidewalk',
  31:'crosswalk', 32:'speed_bump', 33:'stop_sign',
};

const TL_STATE_NAMES = {0:'unknown',1:'arrow_stop',2:'arrow_caution',3:'arrow_go',4:'stop',5:'caution',6:'go',7:'flashing_stop',8:'flashing_caution'};
const TL_STATE_COLORS = {
  0:[156,163,175], 1:[220,38,38],  2:[234,179,8],  3:[22,163,74],
  4:[220,38,38],   5:[234,179,8],  6:[22,163,74],  7:[234,88,12], 8:[234,179,8]
};

const SPEED_MS = {0.5: 200, 1: 100, 2: 50, 4: 25};

// ── State ──────────────────────────────────────────────────────────────────

const state = {
  scenario: null,
  timestep: 0,
  playing: false,
  speed: 1,
  viewMode: '2d',
  followEgo: false,
  layers: {
    lanes: true, road_lines: false, road_edges: true, crosswalks: true,
    agents: true, routes: false, trajectories: true, traffic_lights: true, agent_ids: true,
    unknowns: false,
  },
  selected: null,
  staticLayerCache: null,
  staticLayerCacheKey: null,
  playTimer: null,
  viewState: {target: [0, 0, 0], zoom: 0, rotationX: 0, rotationOrbit: 0},
};

// ── Geometry helpers ───────────────────────────────────────────────────────

const toXY = d => d.xyz.map(p => [p[0], p[1]]);

// Evenly-spaced chevron arrows along a polyline (spacingM, sizeM in world units)
function getRoadChevrons(xyz, spacingM = 18, sizeM = 2.5) {
  const arrows = [];
  let dist = 0, nextArrow = spacingM * 0.4;
  const hw = sizeM * 0.35;
  for (let i = 0; i < xyz.length - 1; i++) {
    const [x1, y1] = [xyz[i][0], xyz[i][1]];
    const [x2, y2] = [xyz[i+1][0], xyz[i+1][1]];
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.hypot(dx, dy);
    if (len < 0.01) continue;
    const ux = dx/len, uy = dy/len;
    const nx = -uy, ny = ux;
    while (dist + len >= nextArrow) {
      const t = (nextArrow - dist) / len;
      const cx = x1 + t * dx, cy = y1 + t * dy;
      const hx = sizeM * 0.5;
      arrows.push({path: [
        [cx - ux*hx + nx*hw, cy - uy*hx + ny*hw],
        [cx + ux*hx,         cy + uy*hx        ],
        [cx - ux*hx - nx*hw, cy - uy*hx - ny*hw],
      ]});
      nextArrow += spacingM;
    }
    dist += len;
  }
  return arrows;
}

function getVehicleCorners(x, y, heading, length, width) {
  const cos = Math.cos(heading), sin = Math.sin(heading);
  const hl = length / 2, hw = width / 2;
  const local = [[-hl,-hw],[hl,-hw],[hl,hw],[-hl,hw],[-hl,-hw]];
  return local.map(([dx,dy]) => [dx*cos - dy*sin + x, dx*sin + dy*cos + y]);
}

function getHeadingArrow(x, y, heading, length) {
  const al = length * 0.6;
  return [[x, y], [x + al * Math.cos(heading), y + al * Math.sin(heading)]];
}

function sceneBounds(scenario) {
  let xmin=Infinity, xmax=-Infinity, ymin=Infinity, ymax=-Infinity;
  for (const road of scenario.road_map_elements) {
    for (const [x,y] of road.xyz) {
      if (x < xmin) xmin = x; if (x > xmax) xmax = x;
      if (y < ymin) ymin = y; if (y > ymax) ymax = y;
    }
  }
  return {xmin, xmax, ymin, ymax, cx:(xmin+xmax)/2, cy:(ymin+ymax)/2};
}

function getEgoState(scenario, t) {
  const sdc = scenario.metadata.sdc_index;
  const agent = scenario.agents.find(a => a.id === sdc);
  if (!agent || !agent.valid[t]) return null;
  return {x: agent.xyz[t][0], y: agent.xyz[t][1], heading: agent.heading[t]};
}

// ── deck.gl setup ──────────────────────────────────────────────────────────

const deckContainer = document.getElementById('deckgl-canvas');

const deckgl = new DeckGL({
  container: deckContainer,
  views: [new OrthographicView({id:'main'})],
  viewState: state.viewState,
  controller: {
    dragPan: true,
    dragRotate: true,
    scrollZoom: { smooth: false, speed: 0.012 },
    inertia: 400,
    keyboard: false,
  },
  getCursor: ({isDragging}) => isDragging ? 'grabbing' : 'crosshair',
  layers: [],
  parameters: { clearColor: [1, 1, 1, 1] },
  onViewStateChange: ({viewState}) => {
    if (!state.followEgo) {
      state.viewState = viewState;
      deckgl.setProps({viewState});
    }
  },
  onClick: handleCanvasClick,
});

// Suppress right-click context menu on canvas
deckContainer.addEventListener('contextmenu', e => e.preventDefault());

// ── Layer builders ─────────────────────────────────────────────────────────

function getStaticLayers(scenario, layerFlags) {
  const cacheKey = scenario.scenario_id + JSON.stringify(layerFlags);
  if (state.staticLayerCacheKey === cacheKey) return state.staticLayerCache;

  const roads = scenario.road_map_elements;
  const layers = [];
  const onClick = (cb) => ({ pickable: true, onClick: ({object}) => cb(object) });

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

  // Lanes (types 1–9 only; type 0 = unknown, duplicates lane geometry)
  if (layerFlags.lanes) {
    const knownLanes = roads.filter(r => r.type >= 1 && r.type <= 9);
    if (knownLanes.length) layers.push(pxPath('lanes-known', knownLanes, [205,210,215], 1));
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

    if (whiteSolid.length)  layers.push(pxPath('rl-white-solid',  whiteSolid,  [160,165,170], 1));
    if (whiteDashed.length) layers.push(pxPath('rl-white-dashed', whiteDashed, [160,165,170], 1, true));
    if (yellowSolid.length) layers.push(pxPath('rl-yellow-solid', yellowSolid, [180, 120,  0], 1));
    if (yellowDash.length)  layers.push(pxPath('rl-yellow-dashed',yellowDash,  [180, 120,  0], 1, true));
    if (layerFlags.unknowns && unknown.length) layers.push(pxPath('rl-unknown', unknown, [124, 58, 237], 1, true));
    if (rest.length)        layers.push(pxPath('rl-rest',         rest,        [160,165,170], 1));
  }

  // Road edges (types 20–29)
  if (layerFlags.road_edges) {
    const knownEdges   = roads.filter(r => r.type >= 21 && r.type <= 29);
    const unknownEdges = roads.filter(r => r.type === 20);
    if (knownEdges.length)   layers.push(pxPath('edges-known',   knownEdges,   [80, 90, 100], 1.5));
    if (layerFlags.unknowns && unknownEdges.length) layers.push(pxPath('edges-unknown', unknownEdges, [124, 58, 237], 1.5));
  }

  // Crosswalks + speed bumps + stop signs
  if (layerFlags.crosswalks) {
    const cw = roads.filter(r => r.type === 31);
    const sb = roads.filter(r => r.type === 32);
    const ss = roads.filter(r => r.type === 33);
    if (cw.length) layers.push(pxPath('crosswalks', cw, [217,119,6], 2));
    if (sb.length) layers.push(pxPath('speed-bumps', sb, [219,39,119], 2));
    if (ss.length) layers.push(new ScatterplotLayer({
      id: 'stop-signs', data: ss,
      getPosition: d => [d.xyz[0], d.xyz[1]],
      getRadius: 3, radiusUnits: 'pixels',
      getFillColor: [220,38,38], getLineColor: [255,255,255],
      stroked: true, lineWidthUnits: 'pixels', getLineWidth: 1,
      ...onClick(obj => selectElement('road', obj)),
    }));
  }

  // Routes — disabled by default; compute_route_polyline produces bad polylines
  // when lane IDs in the route are not geometrically adjacent
  if (layerFlags.routes) {
    const agentsWithRoutes = scenario.agents.filter(a => a.route_polyline && a.route_polyline.length > 1);
    if (agentsWithRoutes.length) layers.push(new PathLayer({
      id: 'routes', data: agentsWithRoutes,
      getPath: d => d.route_polyline.map(p => [p[0], p[1]]),
      getColor: [180, 190, 200, 160],
      getWidth: 1, widthUnits: 'pixels',
      getDashArray: [8, 5], dashJustified: true,
      extensions: [new PathStyleExtension({ dash: true })],
    }));
  }

  state.staticLayerCache = layers;
  state.staticLayerCacheKey = cacheKey;
  return layers;
}

function getDynamicLayers(scenario, t, layerFlags, selected) {
  const sdc = scenario.metadata.sdc_index;
  const ttp = new Set(scenario.metadata.tracks_to_predict || []);
  const layers = [];

  if (layerFlags.agents) {
    // Trajectory history + future — split at validity gaps to avoid teleportation lines
    if (layerFlags.trajectories) {
      // Build continuous valid segments for a slice of xyz/valid arrays
      const validSegments = (xyz, valid, start, end, color) => {
        const segs = [];
        let cur = [];
        for (let i = start; i < end; i++) {
          if (valid[i]) {
            cur.push([xyz[i][0], xyz[i][1]]);
          } else if (cur.length > 1) {
            segs.push({path: cur, color});
            cur = [];
          } else {
            cur = [];
          }
        }
        if (cur.length > 1) segs.push({path: cur, color});
        return segs;
      };

      const HIST_WINDOW = 20;
      const FUT_WINDOW  = 30;

      const histData = scenario.agents.flatMap(a => {
        if (!a.xyz.length) return [];
        const isEgo = a.id === sdc;
        const color = [...(isEgo ? EGO_COLOR : (AGENT_COLORS[a.type] || [128,128,128])), 200];
        return validSegments(a.xyz, a.valid, Math.max(0, t - HIST_WINDOW), t + 1, color);
      });

      if (histData.length) layers.push(new PathLayer({
        id: 'traj-history', data: histData,
        getPath: d => d.path, getColor: d => d.color,
        getWidth: 1.5, widthUnits: 'pixels',
        jointRounded: true, capRounded: true,
      }));

      const futData = scenario.agents.flatMap(a => {
        if (!a.xyz.length || t >= a.xyz.length - 1) return [];
        const isEgo = a.id === sdc;
        const color = [...(isEgo ? EGO_COLOR : (AGENT_COLORS[a.type] || [128,128,128])), 90];
        return validSegments(a.xyz, a.valid, t, Math.min(a.xyz.length, t + FUT_WINDOW + 1), color);
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
        const isEgo = a.id === sdc;
        const color = isEgo ? EGO_COLOR : (AGENT_COLORS[a.type] || [128,128,128]);
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
        pickable: true, onClick: ({object, index}) => {
          selectElement('agent', validAgents[index]);
        },
      }));
    } else {
      // 3D extruded boxes
      const boxData3d = validAgents.map(a => {
        const isEgo = a.id === sdc;
        const color = isEgo ? EGO_COLOR : (AGENT_COLORS[a.type] || [128,128,128]);
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
        pickable: true, onClick: ({object, index}) => {
          selectElement('agent', validAgents[index]);
        },
      }));
    }

    // Heading arrows
    const arrowData = validAgents.map(a => {
      const x = a.xyz[t][0], y = a.xyz[t][1];
      return {path: getHeadingArrow(x, y, a.heading[t], a.length[t] || 4.5)};
    });
    if (arrowData.length) layers.push(new PathLayer({
      id: 'agent-arrows', data: arrowData,
      getPath: d => d.path, getColor: [255,255,255,220],
      getWidth: 1, widthUnits: 'pixels',
      capRounded: true,
    }));

    // Agent IDs
    if (layerFlags.agent_ids) {
      const labelData = validAgents.map(a => ({
        pos: [a.xyz[t][0], a.xyz[t][1]],
        text: String(a.id),
        isEgo: a.id === sdc,
      }));
      if (labelData.length) layers.push(new TextLayer({
        id: 'agent-ids', data: labelData,
        getPosition: d => d.pos, getText: d => d.text,
        getSize: 9, getColor: d => d.isEgo ? [255,255,255] : [240,240,240],
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

  // Traffic lights
  if (layerFlags.traffic_lights && scenario.traffic_control_elements.length) {
    const tlData = scenario.traffic_control_elements.map(tl => {
      const s = t < tl.states.length ? tl.states[t] : 0;
      return {pos: tl.xyz, color: TL_STATE_COLORS[s] || [128,128,128], state: s, tl};
    });
    layers.push(new ScatterplotLayer({
      id: 'traffic-lights', data: tlData,
      getPosition: d => [d.pos[0], d.pos[1]],
      getRadius: 4, radiusUnits: 'pixels',
      getFillColor: d => d.color,
      getLineColor: [255,255,255,200],
      stroked: true, lineWidthUnits: 'pixels', getLineWidth: 1,
      pickable: true, onClick: ({object}) => selectElement('traffic_light', object.tl),
    }));
  }

  // ── Selection highlights ──────────────────────────────────────────────────
  if (selected) {
    const BLUE  = [26, 115, 232];
    const GREEN = [16, 185, 129];
    const AMBER = [217, 119, 6];

    const roadMap = new Map(scenario.road_map_elements.map(r => [r.id, r]));

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
      if (t < a.xyz.length && a.valid[t]) {
        const corners = getVehicleCorners(
          a.xyz[t][0], a.xyz[t][1], a.heading[t],
          a.length[t] || 4.5, a.width[t] || 2
        );
        layers.push(new PolygonLayer({
          id: 'sel-agent', data: [corners],
          getPolygon: d => d,
          getFillColor: [...BLUE, 35],
          getLineColor: [...BLUE, 255],
          getLineWidth: 2, lineWidthUnits: 'pixels',
          stroked: true, filled: true,
        }));
      }

    } else if (selected.type === 'traffic_light') {
      const tl = selected.data;

      // TL dot ring
      layers.push(new ScatterplotLayer({
        id: 'sel-tl', data: [tl],
        getPosition: d => [d.xyz[0], d.xyz[1]],
        getRadius: 9, radiusUnits: 'pixels',
        getFillColor: [0,0,0,0], getLineColor: [...BLUE, 255],
        stroked: true, lineWidthUnits: 'pixels', getLineWidth: 2,
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

  return layers;
}

// ── Render loop ────────────────────────────────────────────────────────────

function render() {
  if (!state.scenario) return;
  const s = state.scenario;
  const t = state.timestep;
  const staticL = getStaticLayers(s, state.layers);
  const dynL = getDynamicLayers(s, t, state.layers, state.selected);
  deckgl.setProps({layers: [...staticL, ...dynL]});

  if (state.followEgo) {
    const ego = getEgoState(s, t);
    const r = parseInt(document.getElementById('follow-radius').value);
    if (ego) {
      const size = Math.min(deckContainer.clientWidth, deckContainer.clientHeight);
      const zoom = Math.log2(size / (2 * r));
      let vs;
      if (state.viewMode === '2d') {
        vs = {target: [ego.x, ego.y, 0], zoom, transitionDuration: 80};
      } else {
        // TPS: camera behind + slightly above ego.
        // OrbitView azimuth 0 = camera in -Y (north). heading 0 = east (+X).
        // Camera-behind = opposite of heading direction → rotationOrbit = -headingDeg - 90.
        const headingDeg = ego.heading * 180 / Math.PI;
        vs = {
          target: [ego.x, ego.y, 0],
          zoom: zoom - 1,
          rotationOrbit: -headingDeg - 90,
          rotationX: 20,
          transitionDuration: 60,
        };
      }
      state.viewState = vs;
      deckgl.setProps({viewState: vs});
    }
  }

  document.getElementById('timestep-display').textContent =
    `${t} / ${s.metadata.scenario_length - 1}`;
  document.getElementById('timeline').value = t;
  document.getElementById('timeline').max = s.metadata.scenario_length - 1;

  // Update info panel if agent selected (position changes per timestep)
  if (state.selected && state.selected.type === 'agent') {
    renderElementInfo(state.selected.type, state.selected.data);
  }
  if (state.selected && state.selected.type === 'traffic_light') {
    renderElementInfo(state.selected.type, state.selected.data);
  }
}

// ── Selection & info panel ─────────────────────────────────────────────────

function selectElement(type, data) {
  state.selected = {type, data};
  renderElementInfo(type, data);
  render();
}

function handleCanvasClick({coordinate, layer, object}) {
  if (!object && !layer) {
    state.selected = null;
    document.getElementById('element-detail').innerHTML = 'Click an element to inspect.';
    render();
  }
}

function renderElementInfo(type, data) {
  const el = document.getElementById('element-detail');
  const t = state.timestep;

  if (type === 'agent') {
    const a = data;
    const sdc = state.scenario.metadata.sdc_index;
    const ttp = state.scenario.metadata.tracks_to_predict || [];
    const ooi = state.scenario.metadata.objects_of_interests || [];
    const isEgo = a.id === sdc;
    const isTtp = ttp.includes(a.id);
    const isOoi = ooi.includes(a.id);
    const typeName = AGENT_TYPE_NAMES[a.type] || `type_${a.type}`;

    const validAt = t < a.valid.length ? a.valid[t] : false;
    const x = validAt ? a.xyz[t][0].toFixed(2) : '—';
    const y = validAt ? a.xyz[t][1].toFixed(2) : '—';
    const h = validAt ? (a.heading[t] * 180 / Math.PI % 360).toFixed(1) + '°' : '—';
    const vx = validAt && a.velocity[t] ? a.velocity[t][0].toFixed(2) : '—';
    const vy = validAt && a.velocity[t] ? a.velocity[t][1].toFixed(2) : '—';
    const vmag = validAt && a.velocity[t] ? Math.sqrt(a.velocity[t][0]**2 + a.velocity[t][1]**2).toFixed(2) : '—';
    const l = validAt ? (a.length[t]||0).toFixed(2) : '—';
    const w = validAt ? (a.width[t]||0).toFixed(2) : '—';
    const ht = validAt ? (a.height[t]||0).toFixed(2) : '—';

    const badges = [
      `<span class="badge badge-${typeName}">${typeName}</span>`,
      isEgo ? '<span class="badge badge-ego">EGO</span>' : '',
      isTtp ? '<span class="badge badge-ttp">TTP</span>' : '',
      isOoi ? '<span class="badge badge-ooi">OOI</span>' : '',
    ].join('');

    const routeLanes = (a.routes && a.routes[0]) ? a.routes[0].join(', ') : '—';

    // Trajectory table (up to 50 rows)
    const trajRows = a.xyz.slice(0, 50).map((pos, i) => {
      const cls = i === t ? 'class="current-row"' : '';
      const v = a.valid[i] ? '✓' : '✗';
      return `<tr ${cls}><td>${i}</td><td>${pos[0].toFixed(1)}</td><td>${pos[1].toFixed(1)}</td><td>${v}</td></tr>`;
    }).join('');

    el.innerHTML = `
      ${badges}
      <div class="info-row"><span class="info-label">ID</span><span class="info-val">${a.id}</span></div>
      <div class="info-row"><span class="info-label">Valid</span><span class="info-val">${validAt ? '✓' : '✗'}</span></div>
      <div class="info-row"><span class="info-label">X,Y</span><span class="info-val">${x}, ${y}</span></div>
      <div class="info-row"><span class="info-label">Heading</span><span class="info-val">${h}</span></div>
      <div class="info-row"><span class="info-label">Speed</span><span class="info-val">${vmag} m/s</span></div>
      <div class="info-row"><span class="info-label">Vel XY</span><span class="info-val">${vx}, ${vy}</span></div>
      <div class="info-row"><span class="info-label">L×W×H</span><span class="info-val">${l}×${w}×${ht}</span></div>
      <div class="info-row"><span class="info-label">Route lanes</span><span class="info-val" style="font-size:9px">${routeLanes}</span></div>
      <details><summary>Trajectory (50 steps)</summary>
        <table class="traj-table"><thead><tr><th>#</th><th>X</th><th>Y</th><th>V</th></tr></thead>
        <tbody>${trajRows}</tbody></table>
      </details>`;
  }

  else if (type === 'road') {
    const r = data;
    const typeName = ROAD_TYPE_NAMES[r.type] || `type_${r.type}`;
    const npts = r.xyz.length;
    const xs = r.xyz.map(p => p[0]), ys = r.xyz.map(p => p[1]);
    const bbox = `(${Math.min(...xs).toFixed(1)}, ${Math.min(...ys).toFixed(1)}) → (${Math.max(...xs).toFixed(1)}, ${Math.max(...ys).toFixed(1)})`;
    const entry = r.entry_lanes.length ? r.entry_lanes.join(', ') : '—';
    const exit = r.exit_lanes.length ? r.exit_lanes.join(', ') : '—';
    const sl = r.speed_limit ? r.speed_limit.toFixed(1) + ' m/s' : '—';

    const ptRows = r.xyz.slice(0, 30).map(([x,y,z], i) =>
      `<tr><td>${i}</td><td>${x.toFixed(2)}</td><td>${y.toFixed(2)}</td></tr>`).join('');

    el.innerHTML = `
      <div class="info-row"><span class="info-label">ID</span><span class="info-val">${r.id}</span></div>
      <div class="info-row"><span class="info-label">Type</span><span class="info-val">${typeName}</span></div>
      <div class="info-row"><span class="info-label">Points</span><span class="info-val">${npts}</span></div>
      <div class="info-row"><span class="info-label">BBox</span><span class="info-val" style="font-size:9px">${bbox}</span></div>
      <div class="info-row"><span class="info-label">Entry</span><span class="info-val" style="font-size:9px">${entry}</span></div>
      <div class="info-row"><span class="info-label">Exit</span><span class="info-val" style="font-size:9px">${exit}</span></div>
      <div class="info-row"><span class="info-label">Speed lim</span><span class="info-val">${sl}</span></div>
      <details><summary>Polyline (30 pts)</summary>
        <table class="traj-table"><thead><tr><th>#</th><th>X</th><th>Y</th></tr></thead>
        <tbody>${ptRows}</tbody></table>
      </details>`;
  }

  else if (type === 'traffic_light') {
    const tl = data;
    const stateNow = t < tl.states.length ? tl.states[t] : 0;
    const stateName = TL_STATE_NAMES[stateNow] || 'unknown';
    const col = TL_STATE_COLORS[stateNow] || [128,128,128];
    const colStr = `rgb(${col[0]},${col[1]},${col[2]})`;
    const controlled = tl.controlled_lanes.length ? tl.controlled_lanes.join(', ') : '—';

    // Find state transitions
    const transitions = [];
    let prev = -1;
    tl.states.forEach((s, i) => {
      if (s !== prev) { transitions.push({t:i, state:s}); prev = s; }
    });
    const transRows = transitions.map(tr =>
      `<tr><td>${tr.t}</td><td>${TL_STATE_NAMES[tr.state]||tr.state}</td></tr>`).join('');

    const fullRows = tl.states.map((s, i) => {
      const cls = i === t ? 'class="current-row"' : '';
      return `<tr ${cls}><td>${i}</td><td>${TL_STATE_NAMES[s]||s}</td></tr>`;
    }).join('');

    el.innerHTML = `
      <div class="info-row"><span class="info-label">ID</span><span class="info-val">${tl.id}</span></div>
      <div class="info-row"><span class="info-label">Pos</span><span class="info-val">${tl.xyz[0].toFixed(2)}, ${tl.xyz[1].toFixed(2)}</span></div>
      <div class="info-row"><span class="info-label">State @${t}</span><span class="info-val">
        <span class="tl-dot" style="background:${colStr}"></span>${stateName}
      </span></div>
      <div class="info-row"><span class="info-label">Lanes</span><span class="info-val" style="font-size:9px">${controlled}</span></div>
      <div style="margin-top:6px;font-size:10px;color:#888">Transitions (${transitions.length})</div>
      <table class="traj-table"><thead><tr><th>@t</th><th>State</th></tr></thead>
        <tbody>${transRows}</tbody></table>
      <details><summary>Full timeline</summary>
        <table class="traj-table"><thead><tr><th>#</th><th>State</th></tr></thead>
        <tbody>${fullRows}</tbody></table>
      </details>`;
  }
}

// ── Scenario loading ───────────────────────────────────────────────────────

async function loadScenario(filename) {
  document.getElementById('scenario-title').textContent = `Loading ${filename}…`;
  const resp = await fetch(`/api/scenario/${encodeURIComponent(filename)}`);
  if (!resp.ok) { alert('Failed to load scenario'); return; }
  const data = await resp.json();
  state.scenario = data;
  state.timestep = 0;
  state.selected = null;
  state.staticLayerCacheKey = null;
  state.followEgo = false;
  document.getElementById('btn-follow').classList.remove('active');

  document.getElementById('scenario-title').textContent =
    `${data.scenario_id} [${data.metadata.dataset_name}]`;

  renderScenarioMeta(data);
  document.getElementById('element-detail').innerHTML = 'Click an element to inspect.';

  // Fit view to scene
  fitView();
  render();
}

function renderScenarioMeta(data) {
  const m = data.metadata;
  const el = document.getElementById('scenario-meta');
  el.innerHTML = `
    <div class="info-row"><span class="info-label">Dataset</span><span class="info-val">${m.dataset_name}</span></div>
    <div class="info-row"><span class="info-label">Length</span><span class="info-val">${m.scenario_length}</span></div>
    <div class="info-row"><span class="info-label">Agents</span><span class="info-val">${data.agents.length}</span></div>
    <div class="info-row"><span class="info-label">Roads</span><span class="info-val">${data.road_map_elements.length}</span></div>
    <div class="info-row"><span class="info-label">TLs</span><span class="info-val">${data.traffic_control_elements.length}</span></div>
    <div class="info-row"><span class="info-label">SDC idx</span><span class="info-val">${m.sdc_index}</span></div>
    <div class="info-row"><span class="info-label">OOI</span><span class="info-val">${(m.objects_of_interests||[]).join(', ')||'—'}</span></div>
    <div class="info-row"><span class="info-label">TTP</span><span class="info-val">${(m.tracks_to_predict||[]).join(', ')||'—'}</span></div>`;
}

function setViewState(vs) {
  state.viewState = vs;
  deckgl.setProps({viewState: vs});
}

function fitView() {
  if (!state.scenario) return;
  const b = sceneBounds(state.scenario);
  const w = b.xmax - b.xmin, h = b.ymax - b.ymin;
  const cw = deckContainer.clientWidth, ch = deckContainer.clientHeight;
  const zoom = Math.log2(Math.min(cw / (w * 1.1), ch / (h * 1.1)));
  setViewState({
    target: [b.cx, b.cy, 0], zoom,
    transitionDuration: 600,
    transitionInterpolator: LinearInterpolator && new LinearInterpolator(['target', 'zoom']),
  });
  deckgl.setProps({
    views: state.viewMode === '2d'
      ? [new OrthographicView({id:'main'})]
      : [new OrbitView({id:'main', fov:50})],
  });
}

// ── Scenario list ──────────────────────────────────────────────────────────

let allScenarios = [];
let activeScenario = null;

async function loadScenarioList() {
  const resp = await fetch('/api/scenarios');
  allScenarios = await resp.json();
  renderScenarioList(allScenarios);
}

function renderScenarioList(items) {
  const container = document.getElementById('scenario-list');
  container.innerHTML = items.map(name =>
    `<div class="scenario-item${name === activeScenario ? ' active' : ''}" data-name="${name}">${name}</div>`
  ).join('');
  container.querySelectorAll('.scenario-item').forEach(el => {
    el.addEventListener('click', () => {
      activeScenario = el.dataset.name;
      renderScenarioList(items);
      loadScenario(activeScenario);
    });
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
  state.followEgo = !state.followEgo;
  document.getElementById('btn-follow').classList.toggle('active', state.followEgo);
  render();
});

document.getElementById('btn-3d').addEventListener('click', () => {
  state.viewMode = state.viewMode === '2d' ? '3d' : '2d';
  document.getElementById('btn-3d').textContent = state.viewMode === '3d' ? '3D' : '2D';
  state.staticLayerCacheKey = null;
  if (state.viewMode === '3d') {
    state.viewState = { ...state.viewState, rotationX: 30, rotationOrbit: 0 };
    deckgl.setProps({views: [new OrbitView({id:'main', fov:50})], viewState: state.viewState});
  } else {
    state.viewState = { ...state.viewState, rotationX: 0, rotationOrbit: 0 };
    deckgl.setProps({views: [new OrthographicView({id:'main'})], viewState: state.viewState});
  }
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
  const id = parseInt(document.getElementById('search-id').value);
  const type = document.getElementById('search-type').value;
  if (isNaN(id)) return;

  if (type === 'agent') {
    const a = state.scenario.agents.find(a => a.id === id);
    if (a) selectElement('agent', a);
  } else if (type === 'road') {
    const r = state.scenario.road_map_elements.find(r => r.id === id);
    if (r) selectElement('road', r);
  } else if (type === 'traffic_light') {
    const tl = state.scenario.traffic_control_elements.find(t => t.id === id);
    if (tl) selectElement('traffic_light', tl);
  }
});

document.getElementById('btn-search-clear').addEventListener('click', () => {
  state.selected = null;
  document.getElementById('element-detail').innerHTML = 'Click an element to inspect.';
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
  }
});

// ── Init ───────────────────────────────────────────────────────────────────

loadScenarioList();
