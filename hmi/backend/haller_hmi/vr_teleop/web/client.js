/**
 * Haller VR teleop — the WebXR client the headset loads.
 *
 * Ported from the reference stack's `relay/web/client.js`: same job, same
 * message schema, one structural difference that is not negotiable here.
 *
 * That client renders its settings panel as a DOM overlay. The Quest Browser
 * has NO `dom-overlay` support on device — this repo found that the hard way,
 * with the headset showing a black void — so everything the operator sees
 * inside XR is drawn to a 2D canvas, uploaded as a texture, and rendered on
 * world-locked WebGL quads. A render layer is mandatory too: without an
 * `XRWebGLLayer` there is simply nothing on screen.
 *
 * Layout: two quads hanging in front of the operator, camera tile above,
 * status/settings panel below it so the panel never covers the view.
 *
 * Controls (indices, not names — index is what the WebXR gamepad spec
 * guarantees):
 *   trigger (0)   analog, the gripper
 *   grip    (1)   the per-hand clutch / dead-man
 *   stick   (3)   click toggles the settings panel; push to navigate + adjust
 *   A / X   (4)   held = precision modifier (lower gains for fine work)
 *   B / Y   (5)   E-STOP. Chosen over A/X because the thumb rests on A/X
 *                 while gripping, and an E-STOP that fires by accident
 *                 teaches people to disable it.
 */

const BTN_TRIGGER = 0, BTN_GRIP = 1, BTN_STICK = 3, BTN_AX = 4, BTN_BY = 5;

// Where the panel and camera quads hang, metres in front of the operator.
const HUD_DIST = 1.15;
const CAM_W = 1.35, CAM_H = 1.0;      // camera quad, metres
const PANEL_W = 1.35, PANEL_H = 0.52;
const CAM_TEX_W = 1024, CAM_TEX_H = 768;
const PANEL_TEX_W = 1024, PANEL_TEX_H = 400;

// Distance back along the controller's grip axis to the operator's wrist
// pivot. The reference stack solves for this with a 5-second in-VR ritual
// (hold the wrist still, twist only, least-squares the offset). This is the
// same idea without the ritual: a single number, adjustable, that moves the
// read-out point off the palm and onto roughly where the wrist turns. It
// matters because a pure wrist twist swings the grip point through an arc,
// and an uncorrected mapping reads that arc as translation the operator
// never asked for. 0 disables.
const DEFAULT_WRIST_PIVOT_M = 0.09;

const api = (path) => new URL(path, new URL('./', location.href)).href;
const $ = (id) => document.getElementById(id);

const log = (msg) => {
  const el = $('log');
  const t = new Date().toTimeString().slice(0, 8);
  el.textContent = `${t}  ${msg}\n${el.textContent}`.split('\n').slice(0, 80).join('\n');
};

// ---------------------------------------------------------------- REST ----

async function jfetch(path, options) {
  const res = await fetch(api(path), {
    headers: { 'content-type': 'application/json' }, ...options,
  });
  if (!res.ok) throw new Error(`${res.status} ${(await res.text()).slice(0, 200)}`);
  return res.status === 204 ? null : res.json();
}

const state = {
  arms: [],
  session: null,
  ws: null,
  cfg: null,
  sides: {},
  stance: 'behind',
  wristPivotM: DEFAULT_WRIST_PIVOT_M,
  cameraId: null,
};

async function refreshStatus() {
  try {
    const st = await jfetch('../teleop/human');
    state.session = st;
    const running = !!st.running;
    const pill = $('running');
    pill.textContent = running ? `${st.state} · L:${st.left_arm ?? '—'} R:${st.right_arm ?? '—'}` : 'idle';
    pill.className = `pill ${running ? 'on' : ''}`;
    const col = st.collision || {};
    const g = $('guardState');
    g.textContent = col.available === false ? 'unavailable (no mounts)'
      : col.enabled ? `on · slack ${fmt(col.slack_m)} m` : `off · slack ${fmt(col.slack_m)} m`;
    g.className = `pill ${col.enabled ? 'on' : 'off'}`;
    if (col.available !== false) $('guard').value = col.enabled ? 'on' : 'off';
  } catch (e) { /* the panel is best-effort; the socket is the live path */ }
}

const fmt = (v) => (typeof v === 'number' && isFinite(v)) ? v.toFixed(3) : '—';

async function loadArms() {
  // /config, not a dedicated /arms: the backend advertises arms and cameras
  // together there, and it is the endpoint the cockpit already reads, so the
  // two pages can never disagree about which arms exist.
  const cfg = await jfetch('../config');
  state.arms = cfg.arms || [];
  const ids = state.arms.map((a) => a.id);
  for (const sel of [$('armRight'), $('armLeft')]) {
    sel.innerHTML = ids.map((id) => `<option value="${id}">${id}</option>`).join('');
  }
  if (ids.length > 1) $('armLeft').value = ids[1];
  // Prefer a live camera over a configured-but-absent one: `/cameras`
  // reports which actually opened, and a placeholder entry would give the
  // HUD a tile that never paints.
  try {
    const cams = await jfetch('../cameras');
    const arr = (Array.isArray(cams) ? cams : (cams.cameras || []))
      .filter((c) => c.active !== false);
    const base = arr.find((c) => c.role === 'base') || arr[0];
    if (base) state.cameraId = base.id;
  } catch { /* no cameras is survivable — the HUD just shows the panel */ }
  log(`arms: ${ids.join(', ') || 'none'}${state.cameraId ? `  camera: ${state.cameraId}` : ''}`);
}

// -------------------------------------------------------------- socket ----

function connect() {
  const url = api('ws').replace(/^http/, 'ws');
  const ws = new WebSocket(url);
  state.ws = ws;
  ws.onopen = () => { $('wsState').textContent = 'connected'; $('wsState').className = 'pill on'; };
  ws.onclose = () => {
    $('wsState').textContent = 'disconnected'; $('wsState').className = 'pill off';
    setTimeout(connect, 1000);
  };
  ws.onmessage = (ev) => {
    let msg; try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === 'ik_state') {
      if (msg.config && !state.cfg) buildSliders(msg.config);
      state.cfg = msg.config || state.cfg;
      state.sides = msg.sides || {};
    } else if (msg.type === 'config_applied') {
      for (const [k, v] of Object.entries(msg.config || {})) syncSlider(k, v);
    }
  };
}

function sendConfig(patch) {
  if (state.ws && state.ws.readyState === 1) {
    state.ws.send(JSON.stringify({ type: 'config_update', config: patch }));
  }
}

// The knobs the panel exposes, in the order they appear both on the page and
// on the in-VR settings list. Kept to the ones an operator actually reaches
// for mid-session; the rest stay in the config file where a considered
// change belongs.
const KNOBS = [
  ['scale_translation', 'translation gain', 0.1, 4, 0.05],
  ['scale_rotation', 'rotation gain', 0.1, 4, 0.05],
  ['precision_factor', 'precision factor', 0.05, 1, 0.05],
  ['pos_reach_limit', 'reach limit (m)', 0, 0.6, 0.01],
  ['rot_reach_limit', 'twist limit (rad)', 0, 2, 0.05],
  ['pose_filter_alpha', 'pose smoothing', 0.05, 1, 0.05],
  ['max_dq_deg_pos', 'step cap arm (°)', 0.25, 15, 0.25],
  ['max_dq_deg_rot', 'step cap wrist (°)', 0.25, 30, 0.25],
  ['lam_pos', 'IK damping', 0.001, 0.2, 0.001],
  ['w0', 'singularity ramp', 0.001, 0.1, 0.001],
];

function buildSliders(cfg) {
  const host = $('sliders');
  host.innerHTML = '';
  for (const [key, label, min, max, step] of KNOBS) {
    const row = document.createElement('div');
    row.className = 'slider';
    row.innerHTML = `<span>${label}</span>
      <input type="range" id="sl_${key}" min="${min}" max="${max}" step="${step}" value="${cfg[key]}">
      <output id="out_${key}">${(+cfg[key]).toFixed(3)}</output>`;
    host.appendChild(row);
    row.querySelector('input').addEventListener('input', (e) => {
      const v = parseFloat(e.target.value);
      $(`out_${key}`).textContent = v.toFixed(3);
      sendConfig({ [key]: v });
    });
  }
  // Client-side only: the read-out point lives in this file because only the
  // client has both the grip and the target-ray pose to build it from.
  const row = document.createElement('div');
  row.className = 'slider';
  row.innerHTML = `<span>wrist pivot (m)</span>
    <input type="range" id="sl_wrist" min="0" max="0.2" step="0.005" value="${state.wristPivotM}">
    <output id="out_wrist">${state.wristPivotM.toFixed(3)}</output>`;
  host.appendChild(row);
  row.querySelector('input').addEventListener('input', (e) => {
    state.wristPivotM = parseFloat(e.target.value);
    $('out_wrist').textContent = state.wristPivotM.toFixed(3);
  });
}

function syncSlider(key, value) {
  const el = $(`sl_${key}`);
  if (el && typeof value === 'number') {
    el.value = value;
    $(`out_${key}`).textContent = value.toFixed(3);
  }
}

// ------------------------------------------------------------ page wiring --

$('mode').addEventListener('change', () => {
  $('leftRow').style.display = $('mode').value === 'dual' ? '' : 'none';
});
$('mode').dispatchEvent(new Event('change'));

$('stance').addEventListener('change', () => { state.stance = $('stance').value; });

$('start').addEventListener('click', async () => {
  const dual = $('mode').value === 'dual';
  // Direct, with NO stance swap. The session's `left_arm` / `right_arm` mean
  // "the arm that HAND drives", and the selectors above are labelled by hand
  // — so the operator has already made the choice a stance swap would be
  // making for them, and swapping would invert it.
  //
  // The cockpit page does swap, and is right to: it never asks, it just takes
  // the two configured arms in order, so it has to decide which hand gets
  // which and the stance is what decides. That is the same reason the note
  // under these selectors says which arm sits under which hand in the behind
  // stance — it is guidance for PICKING, not something to apply twice.
  const body = {
    right_arm: $('armRight').value,
    left_arm: dual ? $('armLeft').value : null,
  };
  try {
    await jfetch('../teleop/human/start', {
      method: 'POST',
      body: JSON.stringify({ ...body, clutch_source: 'vr_grip', hz: 60 }),
    });
    log(`session started (${dual ? 'both arms' : 'single arm'})`);
  } catch (e) { log(`start failed: ${e.message}`); }
  refreshStatus();
});

$('stop').addEventListener('click', async () => {
  try { await jfetch('../teleop/human/stop', { method: 'POST' }); log('session stopped'); }
  catch (e) { log(`stop failed: ${e.message}`); }
  refreshStatus();
});

$('estop').addEventListener('click', () => triggerEstop('panel'));

$('guard').addEventListener('change', async () => {
  const enabled = $('guard').value === 'on';
  try {
    await jfetch('../teleop/human/collision', {
      method: 'POST', body: JSON.stringify({ enabled }),
    });
    log(`collision guard ${enabled ? 'ENABLED' : 'DISABLED'}`);
  } catch (e) { log(`guard toggle failed: ${e.message}`); }
  refreshStatus();
});

let estopAt = 0;
async function triggerEstop(source) {
  // Edge-limited rather than edge-detected: an E-STOP that is spammed by a
  // held button is harmless, but one that fires 90 times a second floods the
  // log the operator then has to read.
  if (Date.now() - estopAt < 1000) return;
  estopAt = Date.now();
  try { await jfetch('../estop', { method: 'POST' }); log(`E-STOP (${source})`); }
  catch (e) { log(`E-STOP failed: ${e.message}`); }
}

// ------------------------------------------------------------- WebXR -------

const gl_state = { gl: null, prog: null, quad: null, camTex: null, panelTex: null };

async function checkXR() {
  const btn = $('enter');
  if (!navigator.xr) {
    btn.textContent = 'WebXR unavailable — is this page on HTTPS?';
    log('navigator.xr is absent. WebXR only exists in a secure context: '
      + 'serve this over HTTPS, or over http://localhost via `adb reverse`.');
    return;
  }
  const ar = await navigator.xr.isSessionSupported('immersive-ar').catch(() => false);
  const vr = await navigator.xr.isSessionSupported('immersive-vr').catch(() => false);
  if (!ar && !vr) { btn.textContent = 'No immersive session supported'; return; }
  btn.disabled = false;
  btn.textContent = ar ? 'Enter VR (passthrough)' : 'Enter VR';
  btn.dataset.mode = ar ? 'immersive-ar' : 'immersive-vr';
}

$('enter').addEventListener('click', () => enterXR($('enter').dataset.mode));

async function enterXR(mode) {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2', { xrCompatible: true, alpha: true, antialias: true })
    || canvas.getContext('webgl', { xrCompatible: true, alpha: true, antialias: true });
  if (!gl) { log('no WebGL context'); return; }
  gl_state.gl = gl;

  let session;
  try {
    session = await navigator.xr.requestSession(mode, {
      optionalFeatures: ['local-floor', 'bounded-floor'],
    });
  } catch (e) { log(`requestSession failed: ${e.message}`); return; }

  await gl.makeXRCompatible();
  session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
  const refSpace = await session.requestReferenceSpace('local-floor')
    .catch(() => session.requestReferenceSpace('local'));

  initGL(gl);
  const cam = startCameraFeed();
  const hud = { placed: false, origin: [0, 1.4, -HUD_DIST], yaw: 0 };
  const ui = { open: false, index: 0, lastStick: 0, lastClick: false };
  const prev = { left: {}, right: {} };

  log(`XR session started (${mode})`);
  session.addEventListener('end', () => { log('XR session ended'); cam.stop(); });

  session.requestAnimationFrame(function onFrame(t, frame) {
    session.requestAnimationFrame(onFrame);
    const pose = frame.getViewerPose(refSpace);
    if (!pose) return;

    if (!hud.placed) { placeHud(hud, pose); hud.placed = true; }

    const xrFrame = sampleControllers(session, frame, refSpace, pose, prev, ui);
    if (state.ws && state.ws.readyState === 1) state.ws.send(JSON.stringify(xrFrame));
    applyHaptics(session);

    const layer = session.renderState.baseLayer;
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    uploadPanel(gl, ui);
    cam.upload(gl);

    for (const view of pose.views) {
      const vp = layer.getViewport(view);
      gl.viewport(vp.x, vp.y, vp.width, vp.height);
      drawQuad(gl, view, hud, gl_state.camTex, 0, CAM_H / 2 + PANEL_H / 2 + 0.03, CAM_W, CAM_H);
      drawQuad(gl, view, hud, gl_state.panelTex, 0, 0, PANEL_W, PANEL_H);
    }
  });
}

function placeHud(hud, pose) {
  const p = pose.transform.position, o = pose.transform.orientation;
  // Yaw only: the panel should hang level in front of the operator however
  // their head happened to be tilted when the session started.
  const yaw = Math.atan2(2 * (o.w * o.y + o.x * o.z), 1 - 2 * (o.y * o.y + o.z * o.z));
  hud.yaw = yaw;
  hud.origin = [p.x - Math.sin(yaw) * HUD_DIST, p.y - 0.15, p.z - Math.cos(yaw) * HUD_DIST];
}

// ------------------------------------------------------ controller sampling -

function sampleControllers(session, frame, refSpace, viewerPose, prev, ui) {
  const out = {
    type: 'vr_keypoints',
    ts_ms: Date.now(),
    stance: state.stance,
    vr_mode: 'ik',
    head: {
      position: [viewerPose.transform.position.x, viewerPose.transform.position.y,
                 viewerPose.transform.position.z],
      orientation: [viewerPose.transform.orientation.x, viewerPose.transform.orientation.y,
                    viewerPose.transform.orientation.z, viewerPose.transform.orientation.w],
    },
    left: null, right: null, dead_man: false,
  };

  for (const src of session.inputSources) {
    const hand = src.handedness;
    if (hand !== 'left' && hand !== 'right') continue;
    const gripPose = src.gripSpace ? frame.getPose(src.gripSpace, refSpace) : null;
    const rayPose = src.targetRaySpace ? frame.getPose(src.targetRaySpace, refSpace) : null;
    const btns = src.gamepad ? src.gamepad.buttons : [];
    const axes = src.gamepad ? src.gamepad.axes : [];
    const pressed = (i) => !!(btns[i] && btns[i].pressed);
    const value = (i) => (btns[i] ? (btns[i].value ?? (btns[i].pressed ? 1 : 0)) : 0);

    if (pressed(BTN_BY)) triggerEstop(`${hand} B/Y`);

    // Thumbstick click toggles the settings list; the stick itself walks it.
    const click = pressed(BTN_STICK);
    if (click && !prev[hand].click) ui.open = !ui.open;
    prev[hand].click = click;
    if (ui.open) stepSettings(ui, axes);

    if (!gripPose) {
      out[hand] = { tracked: false, position: [0, 0, 0], orientation: [0, 0, 0, 1],
                    trigger: 0, squeeze: false, precision: false };
      continue;
    }
    const gp = gripPose.transform.position, gq = gripPose.transform.orientation;
    // Read-out point: the grip position shifted back along the controller's
    // own +Z (which points toward the operator in WebXR's grip space) onto
    // roughly the wrist pivot. See DEFAULT_WRIST_PIVOT_M.
    const back = rotateVec([gq.x, gq.y, gq.z, gq.w], [0, 0, state.wristPivotM]);
    const ro = (rayPose || gripPose).transform.orientation;
    const squeeze = pressed(BTN_GRIP);
    if (squeeze) out.dead_man = true;
    out[hand] = {
      tracked: true,
      position: [gp.x + back[0], gp.y + back[1], gp.z + back[2]],
      // Orientation from the TARGET RAY, not the grip: grip frames on these
      // controllers sit tilted about 55° off the pointing direction, and the
      // operator's mental model of "which way the gripper faces" follows
      // where they are pointing.
      orientation: [ro.x, ro.y, ro.z, ro.w],
      trigger: value(BTN_TRIGGER),
      squeeze,
      precision: pressed(BTN_AX),
    };
  }
  return out;
}

function rotateVec(q, v) {
  const [x, y, z, w] = q, [vx, vy, vz] = v;
  const tx = 2 * (y * vz - z * vy), ty = 2 * (z * vx - x * vz), tz = 2 * (x * vy - y * vx);
  return [vx + w * tx + (y * tz - z * ty),
          vy + w * ty + (z * tx - x * tz),
          vz + w * tz + (x * ty - y * tx)];
}

function stepSettings(ui, axes) {
  const now = performance.now();
  if (now - ui.lastStick < 220) return;      // one step per deliberate push
  const x = axes[2] ?? axes[0] ?? 0, y = axes[3] ?? axes[1] ?? 0;
  if (Math.abs(y) > 0.6) {
    ui.index = (ui.index + (y > 0 ? 1 : -1) + KNOBS.length) % KNOBS.length;
    ui.lastStick = now;
  } else if (Math.abs(x) > 0.6 && state.cfg) {
    const [key, , min, max, step] = KNOBS[ui.index];
    const next = Math.min(max, Math.max(min, (state.cfg[key] ?? min) + (x > 0 ? step : -step)));
    state.cfg[key] = next;
    sendConfig({ [key]: next });
    syncSlider(key, next);
    ui.lastStick = now;
  }
}

function applyHaptics(session) {
  for (const src of session.inputSources) {
    const d = state.sides[src.handedness];
    if (!d || !d.haptic || d.haptic < 0.08) continue;
    try { src.gamepad?.hapticActuators?.[0]?.pulse?.(Math.min(1, d.haptic), 60); }
    catch { /* haptics are feedback, never a safety channel */ }
  }
}

// -------------------------------------------------------------- rendering --

const VERT = `
attribute vec2 aPos;
varying vec2 vUv;
uniform mat4 uProj, uView, uModel;
void main() {
  vUv = vec2(aPos.x + 0.5, 0.5 - aPos.y);
  gl_Position = uProj * uView * uModel * vec4(aPos, 0.0, 1.0);
}`;

const FRAG = `
precision mediump float;
varying vec2 vUv;
uniform sampler2D uTex;
void main() { gl_FragColor = texture2D(uTex, vUv); }`;

function initGL(gl) {
  const compile = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog);
  gl_state.prog = prog;
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5,
  ]), gl.STATIC_DRAW);
  gl_state.quad = buf;
  gl_state.camTex = makeTex(gl);
  gl_state.panelTex = makeTex(gl);
  gl_state.panelCanvas = Object.assign(document.createElement('canvas'),
    { width: PANEL_TEX_W, height: PANEL_TEX_H });
  gl_state.panelCtx = gl_state.panelCanvas.getContext('2d');
  gl_state.lastPanel = 0;
}

function makeTex(gl) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
    new Uint8Array([0, 0, 0, 200]));
  return tex;
}

function drawQuad(gl, view, hud, tex, dx, dy, w, h) {
  const prog = gl_state.prog;
  gl.useProgram(prog);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl_state.quad);
  const loc = gl.getAttribLocation(prog, 'aPos');
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uProj'), false, view.projectionMatrix);
  gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uView'), false, view.transform.inverse.matrix);
  const c = Math.cos(hud.yaw), s = Math.sin(hud.yaw);
  // Yaw-only model matrix: scale, then rotate about Y, then translate.
  gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uModel'), false, new Float32Array([
    w * c, 0, -w * s, 0,
    0, h, 0, 0,
    s, 0, c, 0,
    hud.origin[0] + dx * c, hud.origin[1] + dy, hud.origin[2] - dx * s, 1,
  ]));
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.uniform1i(gl.getUniformLocation(prog, 'uTex'), 0);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
}

// ----------------------------------------------------------- camera feed ---

function startCameraFeed() {
  if (!state.cameraId) return { upload() {}, stop() {} };
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.src = api(`../cameras/${state.cameraId}/stream`);
  const canvas = Object.assign(document.createElement('canvas'),
    { width: CAM_TEX_W, height: CAM_TEX_H });
  const ctx = canvas.getContext('2d');
  let last = 0;
  return {
    upload(gl) {
      // Throttled to ~30 Hz on purpose. Uploading a 1024x768 texture on every
      // display frame stalled the Quest's main thread badly enough to starve
      // the publish loop, which the backend then read as tracking loss and
      // turned into a spurious re-acquire.
      const now = performance.now();
      if (now - last < 33 || !img.naturalWidth) return;
      last = now;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      gl.bindTexture(gl.TEXTURE_2D, gl_state.camTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas);
    },
    stop() { img.src = ''; },
  };
}

// ------------------------------------------------------------- HUD panel ---

function uploadPanel(gl, ui) {
  const now = performance.now();
  if (now - gl_state.lastPanel < 100) return;    // 10 Hz is plenty for text
  gl_state.lastPanel = now;
  paintPanel(gl_state.panelCtx, ui);
  gl.bindTexture(gl.TEXTURE_2D, gl_state.panelTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, gl_state.panelCanvas);
}

function paintPanel(ctx, ui) {
  const W = PANEL_TEX_W, H = PANEL_TEX_H;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = 'rgba(10,13,16,0.88)';
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = 'rgba(90,169,230,0.35)';
  ctx.lineWidth = 3; ctx.strokeRect(1.5, 1.5, W - 3, H - 3);

  const st = state.session || {};
  const acq = st.acquire || {};
  const col = st.collision || {};

  // Left column: per-side authority. Hard column split, because long
  // acquisition strings used to run under the settings box on the right.
  ctx.textBaseline = 'top';
  let y = 18;
  ctx.font = '600 26px ui-sans-serif, system-ui, sans-serif';
  ctx.fillStyle = '#e8edf2';
  ctx.fillText(`${(st.state || 'idle').toUpperCase()}`, 20, y);
  y += 36;
  ctx.font = '20px ui-sans-serif, system-ui, sans-serif';

  for (const side of ['left', 'right']) {
    const armId = st[`${side}_arm`];
    if (!armId) continue;
    const a = acq[side] || {};
    const d = state.sides[side] || {};
    const auth = a.authority || '—';
    ctx.fillStyle = auth === 'driving' ? '#46d18a' : auth === 'acquiring' ? '#e8b046' : '#8b97a5';
    let line = `${side}/${armId}: ${auth}`;
    if (auth === 'acquiring' && a.remaining_ms != null) {
      line += `  ${(a.remaining_ms / 1000).toFixed(1)}s`;
    }
    if (a.reason && auth !== 'driving') line += `  (${a.reason})`;
    ctx.fillText(line, 20, y); y += 26;
    if (auth === 'driving' && d.orient_residual > 0.5) {
      ctx.fillStyle = '#e8b046';
      ctx.fillText('   wrist can\'t reach that twist — move your hand', 20, y);
      y += 26;
    }
    if (a.blocking && a.blocking.length) {
      ctx.fillStyle = '#8b97a5';
      ctx.fillText(`   waiting on: ${a.blocking.join(', ')}`, 20, y);
      y += 26;
    }
  }

  ctx.fillStyle = col.enabled ? '#8b97a5' : '#e2564a';
  ctx.fillText(col.available === false ? 'guard: unavailable'
    : `guard: ${col.enabled ? 'on' : 'OFF'}   slack ${fmt(col.slack_m)} m`, 20, H - 36);

  if (!ui.open) {
    ctx.fillStyle = '#5b6673';
    ctx.font = '18px ui-sans-serif, system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText('stick click = settings', W - 20, H - 34);
    ctx.textAlign = 'left';
    return;
  }

  // Right column: the settings list, opaque so nothing shows through it.
  const x0 = W * 0.52;
  ctx.fillStyle = 'rgba(22,27,33,0.98)';
  ctx.fillRect(x0, 10, W - x0 - 10, H - 20);
  ctx.strokeStyle = 'rgba(90,169,230,0.5)';
  ctx.lineWidth = 2; ctx.strokeRect(x0 + 0.5, 10.5, W - x0 - 11, H - 21);
  ctx.font = '19px ui-sans-serif, system-ui, sans-serif';
  const rows = Math.min(KNOBS.length, 9);
  const first = Math.max(0, Math.min(ui.index - 4, KNOBS.length - rows));
  for (let i = 0; i < rows; i++) {
    const k = first + i;
    const [key, label] = KNOBS[k];
    const val = state.cfg ? state.cfg[key] : null;
    const yy = 22 + i * 26;
    if (k === ui.index) {
      ctx.fillStyle = 'rgba(90,169,230,0.22)';
      ctx.fillRect(x0 + 6, yy - 3, W - x0 - 22, 25);
    }
    ctx.fillStyle = k === ui.index ? '#e8edf2' : '#8b97a5';
    ctx.fillText(label, x0 + 14, yy);
    ctx.textAlign = 'right';
    ctx.fillText(typeof val === 'number' ? val.toFixed(3) : '—', W - 20, yy);
    ctx.textAlign = 'left';
  }
}

// ------------------------------------------------------------------ boot ---

loadArms().catch((e) => log(`arm list failed: ${e.message}`));
connect();
checkXR();
setInterval(refreshStatus, 500);
