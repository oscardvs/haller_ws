/**
 * WebXR → the frame the backend's `QuestTeleoperator` converts.
 *
 * This file is deliberately thin: nothing here solves for joints. It samples
 * the headset and controller poses and ships them raw, and the clutch-relative
 * mapping and the 3+2 decoupled IK both live server-side in
 * `haller_hmi/vr_teleop/`, where they are tested against the real arm model
 * and cannot drift from its coordinate conventions.
 *
 * One geometric correction DOES belong here, because only the client has the
 * grip pose to build it from: the read-out point is shifted back along the
 * controller's own axis onto roughly where the operator's wrist turns
 * (`wristPivotM`). Without it a pure wrist twist swings the grip point
 * through an arc that the mapper can only read as translation nobody asked
 * for.
 *
 * Minimal WebXR types are declared locally rather than pulling in
 * `@types/webxr`: we touch about a dozen members of the API, and a dependency
 * whose only job is to describe them is not worth the supply chain.
 */

// ---- minimal WebXR surface -------------------------------------------------

type XRHandedness = "left" | "right" | "none";

type XRRigidTransform = {
  position: { x: number; y: number; z: number; w: number };
  orientation: { x: number; y: number; z: number; w: number };
};

type XRPose = { transform: XRRigidTransform; emulatedPosition: boolean };

export type XRGamepadLike = {
  buttons: readonly { pressed: boolean; value: number }[];
  /** `xr-standard` puts the thumbstick on 2/3 and a touchpad on 0/1; a
   *  controller with only a stick may report it on either pair. The only
   *  analog inputs this client has left, and therefore where the tuning
   *  menu and the precision modifier had to go — see `stickAxes`. */
  axes?: readonly number[];
  hapticActuators?: readonly {
    pulse?: (intensity: number, durationMs: number) => unknown;
  }[];
} | null;

export type XRInputSourceLike = {
  handedness: XRHandedness;
  gripSpace?: unknown;
  targetRaySpace?: unknown;
  gamepad?: XRGamepadLike;
};

export type XRFrameLike = {
  getViewerPose(space: unknown): XRPose | null | undefined;
  getPose(space: unknown, base: unknown): XRPose | null | undefined;
};

export type XRSessionLike = {
  inputSources: readonly XRInputSourceLike[];
  visibilityState?: "visible" | "visible-blurred" | "hidden";
  requestReferenceSpace(type: string): Promise<unknown>;
  requestAnimationFrame(cb: (t: number, frame: XRFrameLike) => void): number;
  updateRenderState?(state: Record<string, unknown>): void;
  renderState?: { baseLayer?: { framebuffer: unknown } | null };
  end(): Promise<void>;
  addEventListener(type: string, cb: () => void): void;
  removeEventListener(type: string, cb: () => void): void;
};

type NavigatorXR = {
  xr?: {
    isSessionSupported(mode: string): Promise<boolean>;
    requestSession(mode: string, init?: Record<string, unknown>): Promise<XRSessionLike>;
  };
};

/** `xr-standard` gamepad mapping. Index, not name, is what the spec guarantees. */
export const BUTTON_TRIGGER = 0;
export const BUTTON_SQUEEZE = 1;
/** A on the right controller, X on the left. */
export const BUTTON_AX = 4;
/** xr-standard thumbstick click. Free on both controllers — see
 *  `thumbstickPressed` for why the menu had to land here. */
export const BUTTON_THUMBSTICK = 3;
/** B on the right controller, Y on the left — the E-STOP. Chosen over A/X
 *  because the thumb rests ON A/X while gripping; B/Y takes a deliberate
 *  reach, and an E-STOP that fires by accident teaches people to disable it. */
export const BUTTON_BY = 5;

// ---- wire shape ------------------------------------------------------------

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number]; // (x, y, z, w)

export type ControllerSample = {
  position: Vec3;
  orientation: Quat;
  /** Analog trigger [0,1]. 1 = fully squeezed = gripper closed. */
  trigger: number;
  /** Grip button — this hand's dead-man. Each squeeze speaks only for its
   *  own arm on the backend; see the session's `dead_man_sides`. */
  squeeze: boolean;
  tracked: boolean;
  /** Fine-work modifier: the backend multiplies both mapping gains by
   *  `precision_factor` while this is set. Absent reads as false. */
  precision?: boolean;
};

export type VRFrame = {
  type: "vr_keypoints";
  ts_ms: number;
  /** Either grip — kept for the status chip. The per-side split itself rides
   *  on each controller's `squeeze`. */
  dead_man: boolean;
  /** How the operator's hand maps onto the gripper.
   *  "behind" (default): egocentric — the replica arm moves exactly like
   *  your own (goggles on, push forward = the arm extends INTO the default
   *  over-the-shoulder view). "mirror": face-to-face, the arm as your
   *  reflection. "front": face-to-face but screen-true against the
   *  threequarter tile. */
  stance?: "behind" | "mirror" | "front";
  head: { position: Vec3; orientation: Quat } | null;
  left: ControllerSample | null;
  right: ControllerSample | null;
};

export function xrSupported(): Promise<boolean> {
  const xr = (navigator as unknown as NavigatorXR).xr;
  if (!xr) return Promise.resolve(false);
  return xr
    .isSessionSupported("immersive-ar")
    .catch(() => false)
    .then((ar) => ar || xr.isSessionSupported("immersive-vr").catch(() => false));
}

/**
 * Why this can return false on a headset that plainly has WebXR: `navigator.xr`
 * is only exposed in a **secure context**. Over plain http on a LAN address the
 * property is simply absent, which reads as "unsupported" rather than as any
 * kind of permission error. The page has to be served over HTTPS (or from
 * localhost) for this to be true at all — see the VR panel's setup notes.
 */
export function xrAvailableAtAll(): boolean {
  return Boolean((navigator as unknown as NavigatorXR).xr);
}

export type TeleopXRSession = {
  session: XRSessionLike;
  /** "immersive-ar" = passthrough: the operator sees the REAL arms while
   *  driving them, plus the DOM-overlay HUD. "immersive-vr" is the fallback —
   *  functional but blind to the room, so the panel warns. */
  mode: "immersive-ar" | "immersive-vr";
};

/**
 * Passthrough AR first, VR second — teleoperating real arms means watching
 * the real arms, and on Quest an `immersive-ar` session is exactly the camera
 * passthrough with our DOM overlay composited on top.
 *
 * `local-floor` puts the origin on the physical floor with +Y up, which is
 * what `vr_input.shoulder_xr` assumes when it drops the shoulder line below
 * the headset. `local` would put the origin at the headset's start pose and
 * silently shift every shoulder estimate by the operator's height.
 *
 * `dom-overlay` is optional: where it is honoured (Quest browser, AR) the
 * `overlayRoot` element renders inside the headset and its buttons are
 * clickable with the controller ray; where it is not, the same element simply
 * stays on the 2D page — one code path either way.
 */
export async function requestTeleopSession(
  overlayRoot?: Element | null,
): Promise<TeleopXRSession> {
  const xr = (navigator as unknown as NavigatorXR).xr;
  if (!xr) throw new Error("WebXR unavailable (needs a secure context — HTTPS or localhost)");
  const init: Record<string, unknown> = {
    requiredFeatures: ["local-floor"],
    ...(overlayRoot
      ? { optionalFeatures: ["dom-overlay"], domOverlay: { root: overlayRoot } }
      : {}),
  };
  try {
    return { session: await xr.requestSession("immersive-ar", init), mode: "immersive-ar" };
  } catch {
    return { session: await xr.requestSession("immersive-vr", init), mode: "immersive-vr" };
  }
}

// ---- in-scene rendering ------------------------------------------------------
//
// Two jobs, one scene object. First: give the session something to composite,
// or the headset presents nothing at all — in AR a transparent clear leaves
// pure passthrough; in VR an intentional near-black replaces what would
// otherwise be an undefined void. Second: on browsers with no `dom-overlay`
// (the Meta Quest Browser rejects it on-device for immersive-ar — it only
// works in Meta's desktop emulator), the HUD has to live INSIDE the scene:
// a world-locked quad textured from a 2D canvas the panel repaints with the
// workspace camera and the status lines. Plain WebGL1, no dependencies.

/** Column-major 4x4 multiply, WebXR's matrix convention. Exported for tests. */
export function mat4Multiply(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] =
        a[0 * 4 + r] * b[c * 4 + 0] +
        a[1 * 4 + r] * b[c * 4 + 1] +
        a[2 * 4 + r] * b[c * 4 + 2] +
        a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

/** Where the HUD cluster spawns in local-floor space: eye-ish height, just
 *  over a metre out — near enough to read, far enough not to crowd the
 *  workspace. The operator can then grab it (point + trigger with the grip
 *  open) and put it wherever they like; the placement persists. */
export type HudAnchor = { pos: [number, number, number]; yawDeg: number };
export const DEFAULT_HUD_ANCHOR: HudAnchor = { pos: [0, 1.35, -1.15], yawDeg: 0 };

const CLUSTER_GAP_M = 0.05;   // vertical gap between camera tile and panel
const PANEL_W_FRAC = 0.82;    // panel width as a fraction of the tile width

const _rad = (d: number) => (d * Math.PI) / 180;

/** Yaw (degrees) that turns the cluster to face a head at `head`. The
 *  un-rotated quad faces +z, so we need RotY(yaw)·(0,0,1) ∥ horiz(head−pos). */
export function yawTowardHead(
  pos: readonly number[], head: readonly number[],
): number {
  return (Math.atan2(head[0] - pos[0], head[2] - pos[2]) * 180) / Math.PI;
}

/** The cluster's metric layout, shared by the renderer and the grab hit-test
 *  so they can never disagree about where a quad actually is. The camera tile
 *  is centred ON the anchor; the panel hangs below it. */
export function clusterLayout(
  camWidthM: number, camAspect: number, panelAspect: number, hasCam: boolean,
): { camH: number; panelW: number; panelH: number; panelYOff: number } {
  const camH = hasCam ? camWidthM * camAspect : 0;
  const panelW = camWidthM * PANEL_W_FRAC;
  const panelH = panelW * panelAspect;
  const panelYOff = hasCam ? -(camH / 2 + CLUSTER_GAP_M + panelH / 2) : 0;
  return { camH, panelW, panelH, panelYOff };
}

/** Ray vs one cluster quad (both lie in the anchor's rotated z=0 plane at a
 *  vertical offset). Returns the ray parameter t of the hit, or null. Pure —
 *  the grab interaction is pinned by tests, not by strapping on a headset. */
export function rayQuadHit(
  origin: readonly number[], dir: readonly number[],
  anchor: HudAnchor, yOffset: number, w: number, h: number,
): number | null {
  const c = Math.cos(_rad(anchor.yawDeg)), s = Math.sin(_rad(anchor.yawDeg));
  const n = [s, 0, c];                       // quad normal, RotY·(0,0,1)
  const cx = anchor.pos[0], cy = anchor.pos[1] + yOffset, cz = anchor.pos[2];
  const denom = dir[0] * n[0] + dir[1] * n[1] + dir[2] * n[2];
  if (Math.abs(denom) < 1e-6) return null;
  const t = ((cx - origin[0]) * n[0] + (cy - origin[1]) * n[1]
    + (cz - origin[2]) * n[2]) / denom;
  if (t < 0.05 || t > 8) return null;
  const px = origin[0] + dir[0] * t - cx;
  const py = origin[1] + dir[1] * t - cy;
  const pz = origin[2] + dir[2] * t - cz;
  const u = px * c - pz * s;                 // along the quad's x axis
  if (Math.abs(u) > w / 2 || Math.abs(py) > h / 2) return null;
  return t;
}

function _rotateByQuat(q: Quat, v: readonly number[]): [number, number, number] {
  const [x, y, z, w] = q;
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [
    v[0] + w * tx + (y * tz - z * ty),
    v[1] + w * ty + (z * tx - x * tz),
    v[2] + w * tz + (x * ty - y * tx),
  ];
}

export type ControllerRay = {
  origin: [number, number, number];
  dir: [number, number, number];
  trigger: boolean;
};

/** Target-ray origin/direction per hand, plus the trigger bit — everything
 *  the grab-to-move interaction needs from a frame. */
export function controllerRays(
  session: XRSessionLike, frame: XRFrameLike, refSpace: unknown,
): Partial<Record<"left" | "right", ControllerRay>> {
  const out: Partial<Record<"left" | "right", ControllerRay>> = {};
  for (const src of session.inputSources) {
    if (src.handedness !== "left" && src.handedness !== "right") continue;
    if (!src.targetRaySpace) continue;
    const pose = poseToPair(frame.getPose(src.targetRaySpace, refSpace));
    if (!pose) continue;
    out[src.handedness] = {
      origin: [...pose.position] as [number, number, number],
      dir: _rotateByQuat(pose.orientation, [0, 0, -1]),
      trigger: Boolean(src.gamepad?.buttons[BUTTON_TRIGGER]?.pressed),
    };
  }
  return out;
}

type XRViewLike = {
  projectionMatrix: Float32Array;
  transform: { inverse: { matrix: Float32Array } };
};

export type SceneDrawOpts = {
  /** Status/menu canvas — text only, repainted at ~10 Hz. */
  panel: HTMLCanvasElement | null;
  panelDirty: boolean;
  /** Workspace camera <img> (MJPEG). Textured EVERY frame — this is what
   *  makes the tile track at display rate instead of the panel's 10 Hz. */
  cam: HTMLImageElement | null;
  /** Mirror the tile horizontally (display only) — for cameras that face
   *  the operator. */
  camMirrored?: boolean;
  camWidthM?: number;
  anchor?: HudAnchor;
};

export type XRScene = {
  /** Clear both eyes; draw the camera tile and the status panel as two
   *  separate world quads at `anchor` — the panel BELOW the tile, never on
   *  top of the view it annotates. */
  render(frame: XRFrameLike, refSpace: unknown, opts: SceneDrawOpts): void;
};

const _NOOP_SCENE: XRScene = { render: () => {} };

export function attachRenderScene(
  session: XRSessionLike,
  mode: TeleopXRSession["mode"],
): XRScene {
  type LayerLike = {
    framebuffer: unknown;
    getViewport?: (v: unknown) =>
      { x: number; y: number; width: number; height: number } | null;
  };
  type LayerCtor = new (s: XRSessionLike, gl: unknown) => LayerLike;
  const XRWebGLLayerCtor = (globalThis as { XRWebGLLayer?: LayerCtor }).XRWebGLLayer;
  if (!XRWebGLLayerCtor || !session.updateRenderState) return _NOOP_SCENE;
  const canvas = document.createElement("canvas");
  const gl = (canvas.getContext("webgl", { xrCompatible: true, alpha: true })
    ?? canvas.getContext("webgl2", { xrCompatible: true, alpha: true })) as
    WebGLRenderingContext | null;
  if (!gl) return _NOOP_SCENE;
  const layer = new XRWebGLLayerCtor(session, gl);
  session.updateRenderState({ baseLayer: layer });
  const [r, g, b, a] = mode === "immersive-ar"
    ? [0, 0, 0, 0]           // transparent: passthrough shows through
    : [0.05, 0.05, 0.07, 1]; // VR fallback: deliberate near-black, not a void

  // Minimal textured-quad pipeline. Compiled lazily on the first HUD frame so
  // the overlay-capable path never pays for it.
  let prog: WebGLProgram | null = null;
  let uMVP: WebGLUniformLocation | null = null;
  let panelTex: WebGLTexture | null = null;
  let camTex: WebGLTexture | null = null;
  let camTexValid = false;
  let lastCamUploadMs = 0;

  /** Column-major T(pos+yOff) · RotY(yaw) · S(w, h, 1). `mirrorX` flips the
   *  quad for operator-facing cameras — display only, the source pixels are
   *  never touched. */
  function quadModel(
    anchor: HudAnchor, w: number, h: number, yOffset: number, mirrorX: boolean,
  ): Float32Array {
    const c = Math.cos(_rad(anchor.yawDeg)), s = Math.sin(_rad(anchor.yawDeg));
    const sx = mirrorX ? -w : w;
    return new Float32Array([
      c * sx, 0, -s * sx, 0,
      0, h, 0, 0,
      s, 0, c, 0,
      anchor.pos[0], anchor.pos[1] + yOffset, anchor.pos[2], 1,
    ]);
  }

  function ensurePipeline(): boolean {
    if (prog) return true;
    if (!gl) return false;
    const compile = (type: number, src: string) => {
      const s = gl.createShader(type);
      if (!s) return null;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      return gl.getShaderParameter(s, gl.COMPILE_STATUS) ? s : null;
    };
    const vs = compile(gl.VERTEX_SHADER,
      "attribute vec3 aPos; attribute vec2 aUV; uniform mat4 uMVP;" +
      "varying vec2 vUV;" +
      "void main(){ gl_Position = uMVP * vec4(aPos, 1.0); vUV = aUV; }");
    const fs = compile(gl.FRAGMENT_SHADER,
      "precision mediump float; varying vec2 vUV; uniform sampler2D uTex;" +
      "void main(){ gl_FragColor = texture2D(uTex, vUV); }");
    if (!vs || !fs) return false;
    const p = gl.createProgram();
    if (!p) return false;
    gl.attachShader(p, vs);
    gl.attachShader(p, fs);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) return false;
    // Interleaved x,y,z,u,v — unit quad centred on the origin, v flipped so
    // canvas row 0 lands at the TOP of the quad.
    const verts = new Float32Array([
      -0.5, -0.5, 0, 0, 1,
       0.5, -0.5, 0, 1, 1,
      -0.5,  0.5, 0, 0, 0,
       0.5,  0.5, 0, 1, 0,
    ]);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
    const aPos = gl.getAttribLocation(p, "aPos");
    const aUV = gl.getAttribLocation(p, "aUV");
    gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 20, 0);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aUV, 2, gl.FLOAT, false, 20, 12);
    gl.enableVertexAttribArray(aUV);
    uMVP = gl.getUniformLocation(p, "uMVP");
    const mkTex = () => {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return t;
    };
    panelTex = mkTex();
    camTex = mkTex();
    prog = p;
    return true;
  }

  return {
    render(frame, refSpace, opts) {
      const fb = (session.renderState?.baseLayer ?? layer).framebuffer;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb as WebGLFramebuffer | null);
      gl.clearColor(r, g, b, a);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      const panel = opts.panel;
      if (!panel || !ensurePipeline()) return;
      const pose = frame.getViewerPose(refSpace) as unknown as
        { views?: XRViewLike[] } | null | undefined;
      const views = pose?.views;
      if (!views?.length) return;
      gl.useProgram(prog);
      gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

      // Panel text: repainted at ~10 Hz, uploaded only when it changed.
      gl.bindTexture(gl.TEXTURE_2D, panelTex);
      if (opts.panelDirty) {
        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, 1);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, panel);
      }
      // Camera tile: textured straight from the MJPEG <img> — one
      // native-resolution sampling step, no canvas composite in between.
      // Uploads are THROTTLED to the stream's own 30 fps rather than done
      // per display frame: a 960x720 RGBA upload at 72-90 Hz measurably
      // stalls the Quest browser's main thread, and a stalled main thread
      // starves the 30 Hz publish loop — which the backend correctly reads
      // as tracking loss and answers with a re-acquire. The stream carries
      // no new pixels between its own frames anyway.
      const cam = opts.cam;
      const camReady = Boolean(cam && cam.complete && cam.naturalWidth > 0);
      const nowMs = performance.now();
      if (cam && camReady && nowMs - lastCamUploadMs >= 33) {
        gl.bindTexture(gl.TEXTURE_2D, camTex);
        try {
          gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, 0);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, cam);
          camTexValid = true;
          lastCamUploadMs = nowMs;
        } catch {
          /* a mid-frame MJPEG boundary can throw; keep the previous texture */
        }
      }

      const anchor = opts.anchor ?? DEFAULT_HUD_ANCHOR;
      const camW = opts.camWidthM && opts.camWidthM > 0 ? opts.camWidthM : 1.1;
      const camAspect = camReady && cam
        ? cam.naturalHeight / cam.naturalWidth : 0.75;
      const drawCam = camTexValid && cam !== null;
      const layout = clusterLayout(
        camW, camAspect, panel.height / panel.width, drawCam);

      const active = (session.renderState?.baseLayer ?? layer) as LayerLike;
      const camModel = quadModel(anchor, camW, layout.camH, 0,
                                 Boolean(opts.camMirrored));
      const panelModel = quadModel(anchor, layout.panelW, layout.panelH,
                                   layout.panelYOff, false);
      for (const view of views) {
        const vp = active.getViewport?.(view);
        if (!vp) continue;
        gl.viewport(vp.x, vp.y, vp.width, vp.height);
        const pv = mat4Multiply(view.projectionMatrix,
                                view.transform.inverse.matrix);
        if (drawCam) {
          gl.bindTexture(gl.TEXTURE_2D, camTex);
          gl.uniformMatrix4fv(uMVP, false, mat4Multiply(pv, camModel));
          gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        }
        gl.bindTexture(gl.TEXTURE_2D, panelTex);
        gl.uniformMatrix4fv(uMVP, false, mat4Multiply(pv, panelModel));
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      }
    },
  };
}

// ---- HUD canvas painter -------------------------------------------------------

export type HudStatusLike = {
  state?: string;
  last_error?: string | null;
  clutch?: { sides?: { left: boolean; right: boolean }; engaged?: boolean };
  collision?: { enabled: boolean; slack_m?: number; limited?: boolean };
  acquire?: Partial<Record<"left" | "right", {
    authority: string;
    remaining_ms: number | null;
    reason: string;
  }>>;
} | null;

/** The slice of recorder status the in-scene HUD shows. `takes` counts what
 *  this session has saved — the recorder reports frames, not an episode
 *  index, and an operator mid-run wants to know how many good ones are in
 *  the bag. */
export type RecorderHudLike = {
  recording: boolean;
  episode_frames: number;
  takes?: number;
} | null;

/**
 * Repaint the in-scene HUD. Layout: workspace camera on top (when a frame is
 * available), status strip below. Pure canvas 2D so it can run anywhere;
 * failures to draw the camera (stream warming up, no camera) degrade to the
 * text strip alone, never to an exception — this runs inside the XR loop.
 */
/** The view menu, as the HUD needs it. `views` is whatever the backend
 *  advertises — in sim that is the MuJoCo cameras, on the real rig it is the
 *  mast and the egocentric gripper cams, and the menu does not care which. */
export type VrMenuLike = {
  views: readonly { id: string; label: string }[];
  activeViewId: string | null;
  tileSize: string;
  /** Operator stance, display-only: it is chosen on the desktop panel before
   *  entering VR (every controller button in-session is spoken for), but the
   *  operator must be able to SEE which mapping their hands are wired to —
   *  a wrong stance reads as "the arm goes the opposite way". */
  stance?: "behind" | "mirror" | "front";
  /** The live-tuning list, when the operator has it open. */
  tuning?: {
    open: boolean;
    index: number;
    values: Readonly<Record<string, number>>;
  } | null;
  /** True while the precision modifier is held. Shown prominently: a
   *  modifier you cannot see is one you leave on. */
  precision?: boolean;
  /** The end-of-take decision is open — save or discard. Modal: it takes
   *  over the menu box and both stick clicks until the operator picks. */
  endPrompt?: boolean;
} | null;

/** Repaint the status/menu PANEL — text only. The workspace camera is not
 *  composited here any more: it renders as its own quad, textured at native
 *  resolution every display frame (see `attachRenderScene`), while this
 *  canvas repaints at ~10 Hz. Splitting them is what stopped the
 *  instructions from hovering on top of the view, and what let the tile
 *  track at display rate instead of the panel's cadence. */
export function paintHud(
  ctx: CanvasRenderingContext2D,
  status: HudStatusLike,
  rec: RecorderHudLike = null,
  menu: VrMenuLike = null,
  ik: IkSides = {},
): void {
  const W = ctx.canvas.width;
  const H = ctx.canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "rgba(8, 10, 14, 0.82)";
  ctx.fillRect(0, 0, W, H);

  // Two hard columns: status text on the left, the menu box on the right,
  // and a CLIP on the status column — an error string or a long side line is
  // arbitrarily wide, and letting one run under the menu is exactly the
  // overlapping-menus mess this layout replaces.
  const leftW = Math.round(W * 0.55) - 24;
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, leftW + 24, H);
  ctx.clip();

  ctx.textAlign = "left";
  ctx.font = "bold 30px monospace";
  let y = 46;
  const state = (status?.state ?? "—").toUpperCase();
  const col = status?.collision;
  ctx.fillStyle = "#e8eaed";
  ctx.fillText(state, 24, y);
  if (rec?.recording) {
    // Right-aligned to the column's edge so it cannot collide with the
    // state: the one line that must be readable at a glance while driving.
    ctx.fillStyle = "#f28b82";
    ctx.textAlign = "right";
    ctx.fillText(`● REC ${rec.episode_frames}`, leftW, y);
    ctx.textAlign = "left";
  }
  if (col?.enabled) {
    y += 38;
    const slack = col.slack_m;
    ctx.fillStyle = col.limited ? "#f28b82"
      : (slack ?? 1) < 0 ? "#fdd663" : "#9aa0a6";
    ctx.font = "28px monospace";
    ctx.fillText(
      col.limited
        ? `COLLISION HOLD ${slack !== undefined ? (slack * 1000).toFixed(0) : "—"} mm`
        : `clearance ${slack !== undefined ? (slack * 1000).toFixed(0) : "—"} mm`,
      24, y);
  }
  ctx.font = "28px monospace";
  // Per side: authority, then the solver's own reading of how the arm is
  // coping. There is no pose to match any more — the mapper re-anchors until
  // the side is driving, so acquisition is a countdown and nothing else —
  // and the space that list used to take is worth more spent on conditioning.
  for (const side of ["left", "right"] as const) {
    y += 38;
    const acq = status?.acquire?.[side];
    const grip = status?.clutch?.sides?.[side] ?? status?.clutch?.engaged;
    ctx.fillStyle = "#9aa0a6";
    ctx.fillText(`${side === "left" ? "L" : "R"} ${grip ? "●" : "○"}`, 24, y);
    if (!acq) continue;
    const d = ik[side];
    ctx.fillStyle = acq.authority === "driving" ? "#81c995"
      : acq.authority === "acquiring" ? "#fdd663" : "#9aa0a6";
    let line = acq.authority;
    if (acq.authority === "acquiring" && acq.remaining_ms !== null) {
      line += ` ${(acq.remaining_ms / 1000).toFixed(1)}s`;
    }
    if (acq.reason === "no_tracking") line += "  (no tracking)";
    if (acq.reason === "no_arm") line += "  (no arm this side)";
    // σ_min is in m/rad: how far the tool moves, in its worst direction, per
    // radian of joint travel. Falling toward zero IS the singular set.
    if (acq.authority === "driving" && typeof d?.sigma_min === "number") {
      line += `  σ ${d.sigma_min.toFixed(3)}`;
    }
    ctx.fillText(line, 120, y);
    // The 5-DoF wrist has one axis fewer than an arbitrary orientation asks
    // for. Say what to do about it — twisting harder at a deficit is exactly
    // the wrong response, and the controller is buzzing at the same moment.
    if (acq.authority === "driving"
        && (d?.orient_residual ?? 0) > ORIENT_DEFICIT) {
      y += 32;
      ctx.fillStyle = "#fdd663";
      ctx.fillText("wrist is out of twist — MOVE your hand", 120, y);
    }
  }
  y += 38;
  if (status?.last_error) {
    ctx.fillStyle = "#f28b82";
    ctx.fillText(status.last_error.slice(0, 60), 24, y);
  } else if (menu?.precision) {
    ctx.fillStyle = "#8ab4f8";
    ctx.fillText("◆ PRECISION — gains scaled down while held", 24, y);
  } else {
    ctx.fillStyle = "#9aa0a6";
    ctx.fillText("grips = drive · trigger = gripper · B/Y = E-STOP", 24, y);
  }
  ctx.restore();

  if (menu) paintMenu(ctx, menu, rec);
}

/** The menu box, bottom-right of the HUD.
 *
 *  It states the bindings rather than only reflecting state, because in a
 *  headset there is nowhere else to look them up: the record command in
 *  particular was previously documented only in a markdown file, which is no
 *  use with a headset on and both hands clutched.
 *
 *  Three faces, one box. The end-of-take decision is modal and takes it over
 *  entirely; the tuning list replaces the view list while it is open;
 *  otherwise it is the view menu it has always been.
 */
function paintMenu(
  ctx: CanvasRenderingContext2D,
  menu: NonNullable<VrMenuLike>,
  rec: RecorderHudLike,
): void {
  const W = ctx.canvas.width;
  const pad = 18;
  const lineH = 30;
  const tuningOpen = Boolean(menu.tuning?.open);
  const rows = menu.endPrompt ? 5
    : tuningOpen ? TUNING_WINDOW + 2
    // views (or the "no cameras" line), SIZE, REC, reset, grab, tune hint.
    : Math.max(1, menu.views.length) + 6 + (menu.stance ? 1 : 0);
  const boxW = Math.round(W * 0.42);
  const boxH = rows * lineH + pad * 2;
  const x = W - boxW - 24;
  const yTop = 16;

  // Fully opaque: the status column is clipped away from this box, and
  // nothing may ghost through from behind either.
  ctx.fillStyle = "rgb(10, 12, 16)";
  ctx.fillRect(x, yTop, boxW, boxH);
  ctx.strokeStyle = menu.endPrompt
    ? "rgba(242, 139, 130, 0.9)" : "rgba(154, 160, 166, 0.45)";
  ctx.lineWidth = 2;
  ctx.strokeRect(x, yTop, boxW, boxH);

  ctx.textAlign = "left";
  const y0 = yTop + pad + 22;
  if (menu.endPrompt) paintEndPrompt(ctx, rec, x + pad, y0, lineH);
  else if (tuningOpen) paintTuning(ctx, menu, x, x + pad, y0, boxW, lineH);
  else paintViewMenu(ctx, menu, rec, x + pad, y0, lineH);
}

/** How many knob rows the in-headset list shows at once. More than this and
 *  the box starts covering the view it hangs beside. */
const TUNING_WINDOW = 8;

function paintEndPrompt(
  ctx: CanvasRenderingContext2D, rec: RecorderHudLike,
  x: number, yStart: number, lineH: number,
): void {
  let y = yStart;
  ctx.font = "bold 24px monospace";
  ctx.fillStyle = "#f28b82";
  ctx.fillText(`TAKE ENDED · ${rec?.episode_frames ?? 0} frames`, x, y);
  y += lineH;
  ctx.font = "22px monospace";
  ctx.fillStyle = "#81c995";
  ctx.fillText("L stick click = SAVE", x, y);
  y += lineH;
  ctx.fillStyle = "#f28b82";
  ctx.fillText("R stick click = DISCARD", x, y);
  y += lineH;
  ctx.fillStyle = "#9aa0a6";
  ctx.fillText("hold A/X = keep rolling", x, y);
  y += lineH;
  // Said out loud because it is the one surprising part: `/record/stop`
  // takes the save decision AT stop time, so there is no way to end the
  // episode first and choose afterwards. The tail is a second of stillness.
  ctx.fillText("still rolling until you pick", x, y);
}

function paintTuning(
  ctx: CanvasRenderingContext2D, menu: NonNullable<VrMenuLike>,
  boxX: number, x: number, yStart: number, boxW: number, lineH: number,
): void {
  const tuning = menu.tuning!;
  const index = Math.min(Math.max(tuning.index, 0), TUNING_KNOBS.length - 1);
  let y = yStart;
  ctx.font = "bold 24px monospace";
  ctx.fillStyle = "#e8eaed";
  ctx.fillText("TUNE  (R stick walks / adjusts)", x, y);
  y += lineH;
  ctx.font = "22px monospace";
  const first = Math.max(
    0, Math.min(index - Math.floor(TUNING_WINDOW / 2),
                TUNING_KNOBS.length - TUNING_WINDOW));
  for (let i = 0; i < Math.min(TUNING_WINDOW, TUNING_KNOBS.length); i++) {
    const k = first + i;
    const knob = TUNING_KNOBS[k];
    const on = k === index;
    if (on) {
      ctx.fillStyle = "rgba(138, 180, 248, 0.22)";
      ctx.fillRect(boxX + 6, y - 20, boxW - 12, lineH - 2);
    }
    ctx.fillStyle = on ? "#e8eaed" : "#9aa0a6";
    ctx.fillText(`${on ? "▸" : " "} ${knob.label}`, x, y);
    ctx.textAlign = "right";
    ctx.fillText(formatKnob(tuning.values[knob.key]), boxX + boxW - 20, y);
    ctx.textAlign = "left";
    y += lineH;
  }
  ctx.fillStyle = "#9aa0a6";
  ctx.fillText("hold R stick = close", x, y);
}

function paintViewMenu(
  ctx: CanvasRenderingContext2D, menu: NonNullable<VrMenuLike>,
  rec: RecorderHudLike, x: number, yStart: number, lineH: number,
): void {
  let y = yStart;
  ctx.font = "bold 24px monospace";
  ctx.fillStyle = "#e8eaed";
  ctx.fillText(`VIEW  (L stick click = next)`, x, y);
  y += lineH;

  ctx.font = "22px monospace";
  for (const v of menu.views) {
    const on = v.id === menu.activeViewId;
    ctx.fillStyle = on ? "#8ab4f8" : "#9aa0a6";
    ctx.fillText(`${on ? "▸" : " "} ${v.label}`, x, y);
    y += lineH;
  }
  if (!menu.views.length) {
    ctx.fillStyle = "#9aa0a6";
    ctx.fillText("  (no cameras)", x, y);
    y += lineH;
  }

  ctx.fillStyle = "#9aa0a6";
  ctx.fillText(`SIZE  ${menu.tileSize}  (R stick = next)`, x, y);
  y += lineH;

  if (menu.stance) {
    const words = { behind: "egocentric", mirror: "mirror", front: "camera-true" };
    ctx.fillText(`STANCE  ${words[menu.stance]}  (set on panel)`, x, y);
    y += lineH;
  }

  const recording = Boolean(rec?.recording);
  const takes = rec?.takes ?? 0;
  ctx.fillStyle = recording ? "#f28b82" : "#9aa0a6";
  ctx.fillText(
    recording
      ? `● REC take ${takes + 1} · ${rec?.episode_frames ?? 0} fr — hold A/X to END`
      : takes
        ? `hold A/X to START a take  (${takes} saved)`
        : "hold A/X to START a take",
    x, y);
  y += lineH;

  ctx.fillStyle = menu.precision ? "#8ab4f8" : "#9aa0a6";
  ctx.fillText(
    menu.precision ? "◆ PRECISION (L stick pushed away)"
                   : "push L stick away = precision",
    x, y);
  y += lineH;

  ctx.fillStyle = "#9aa0a6";
  ctx.fillText("hold L stick = reset arms · hold R stick = tune", x, y);
  y += lineH;
  ctx.fillText("point + trigger (grip open) = move HUD", x, y);
}

function poseToPair(pose: XRPose | null | undefined) {
  if (!pose) return null;
  const p = pose.transform.position;
  const o = pose.transform.orientation;
  return {
    position: [p.x, p.y, p.z] as Vec3,
    orientation: [o.x, o.y, o.z, o.w] as Quat,
  };
}

/**
 * Sample one XR frame into the backend's wire shape.
 *
 * A controller that is present but whose pose cannot be resolved this frame is
 * reported `tracked: false` rather than omitted. That distinction matters: the
 * backend freezes a side that goes untracked and re-runs acquisition when it
 * returns, which is exactly the behaviour we want when a controller leaves the
 * headset's view. Dropping the key instead would look identical to "no
 * controller for this side", which is a different situation.
 *
 * `forceDisengaged` exists for the visibility failsafe: while the Quest system
 * menu is open the page keeps receiving the last input state, so a grip that
 * was held when the menu opened would otherwise stay held forever. The caller
 * forces every clutch bit false while the session is not fully visible.
 */
export function sampleVRFrame(
  session: XRSessionLike,
  frame: XRFrameLike,
  refSpace: unknown,
  opts: { tsMs: number; forceDisengaged?: boolean;
          stance?: "behind" | "mirror" | "front";
          precision?: boolean; wristPivotM?: number },
): VRFrame {
  const head = poseToPair(frame.getViewerPose(refSpace));

  let left: ControllerSample | null = null;
  let right: ControllerSample | null = null;
  let deadMan = false;

  for (const src of session.inputSources) {
    if (src.handedness !== "left" && src.handedness !== "right") continue;
    const buttons = src.gamepad?.buttons ?? [];
    const squeeze = !opts.forceDisengaged
      && Boolean(buttons[BUTTON_SQUEEZE]?.pressed);
    if (squeeze) deadMan = true;

    const pose = src.gripSpace ? poseToPair(frame.getPose(src.gripSpace, refSpace)) : null;
    // Read-out point: the grip position shifted back along the controller's
    // own +Z (which points toward the operator in WebXR grip space) onto
    // roughly the operator's wrist pivot. See DEFAULT_WRIST_PIVOT_M — a pure
    // twist about a palm-centred point is an ARC, and the mapper has no way
    // to tell that arc from translation the operator never asked for.
    const pivotM = opts.wristPivotM ?? 0;
    const back = pose && pivotM !== 0
      ? _rotateByQuat(pose.orientation, [0, 0, pivotM]) : null;
    // Orientation comes from the TARGET RAY, not the grip. On Quest Touch
    // controllers the grip frame is tilted ~50-60° from where a relaxed hand
    // actually points (it follows the handle, not the knuckles), and the
    // backend synthesizes the whole hand from this orientation — with the
    // grip frame, holding a controller naturally reads as a steeply
    // pitched-down wrist, which both drives wrist_flex oddly and makes the
    // acquisition gate nearly unmatchable. The ray frame is each vendor's
    // own answer to "where is this hand pointing"; position stays on the
    // grip (the ray origin sits forward of the palm).
    const ray = src.targetRaySpace
      ? poseToPair(frame.getPose(src.targetRaySpace, refSpace)) : null;
    const sample: ControllerSample = pose
      ? {
          position: back
            ? [pose.position[0] + back[0], pose.position[1] + back[1],
               pose.position[2] + back[2]]
            : pose.position,
          orientation: (ray ?? pose).orientation,
          trigger: buttons[BUTTON_TRIGGER]?.value ?? 0,
          squeeze,
          tracked: true,
          ...(opts.precision ? { precision: true } : {}),
        }
      : {
          position: [0, 0, 0],
          orientation: [0, 0, 0, 1],
          trigger: 0,
          squeeze,
          tracked: false,
        };
    if (src.handedness === "left") left = sample;
    else right = sample;
  }

  return {
    type: "vr_keypoints",
    ts_ms: opts.tsMs,
    dead_man: deadMan,
    ...(opts.stance ? { stance: opts.stance } : {}),
    head,
    left,
    right,
  };
}

/** A frame that asks for nothing: clutch open on both sides, no tracking.
 *  Sent as a parting shot on teardown so the backend releases immediately
 *  instead of waiting out the staleness budget. */
export function disengagedFrame(tsMs: number): VRFrame {
  return {
    type: "vr_keypoints",
    ts_ms: tsMs,
    dead_man: false,
    head: null,
    left: null,
    right: null,
  };
}

/** True while B (right) or Y (left) is down on any controller. The panel
 *  edge-detects this into exactly one POST /estop per press. */
export function estopPressed(session: XRSessionLike): boolean {
  for (const src of session.inputSources) {
    if (src.handedness !== "left" && src.handedness !== "right") continue;
    if (src.gamepad?.buttons[BUTTON_BY]?.pressed) return true;
  }
  return false;
}

/** True while A (right) or X (left) is down on any controller — the record
 *  toggle. Deliberately NOT edge-detected here: the panel runs it through
 *  `holdToggle` so a brush of the thumb cannot start or end a take. */
export function axPressed(session: XRSessionLike): boolean {
  for (const src of session.inputSources) {
    if (src.handedness !== "left" && src.handedness !== "right") continue;
    if (src.gamepad?.buttons[BUTTON_AX]?.pressed) return true;
  }
  return false;
}

/** True while ONE named controller's thumbstick is clicked in.
 *
 *  Per-hand, unlike the others: the two sticks drive different menu axes (left
 *  cycles the view, right cycles the tile size), which is what makes a menu
 *  possible at all without stealing an input that already means something.
 *  Trigger, grip, A/X and B/Y are all spoken for, and every one of them is
 *  either a dead-man or a safety action — none can be shared with a menu.
 */
export function thumbstickPressed(
  session: XRSessionLike, hand: "left" | "right",
): boolean {
  for (const src of session.inputSources) {
    if (src.handedness !== hand) continue;
    if (src.gamepad?.buttons[BUTTON_THUMBSTICK]?.pressed) return true;
  }
  return false;
}

/** Thumbstick deflection for one hand, as (x, y) in [-1, 1].
 *
 *  `xr-standard` puts the stick on axes 2/3 and a touchpad on 0/1; a
 *  controller carrying only one analog pair reports the stick on 0/1, so take
 *  whichever is present. y is POSITIVE toward the operator — pushing the
 *  stick away reads negative, which is the sign convention every caller here
 *  assumes.
 *
 *  These four axes are the only inputs this client had left. Every button is
 *  spoken for by a dead-man, a safety action or the recorder, so the tuning
 *  menu and the precision modifier both had to land here.
 */
export function stickAxes(
  session: XRSessionLike, hand: "left" | "right",
): [number, number] {
  for (const src of session.inputSources) {
    if (src.handedness !== hand) continue;
    const a = src.gamepad?.axes;
    if (!a) continue;
    return [a[2] ?? a[0] ?? 0, a[3] ?? a[1] ?? 0];
  }
  return [0, 0];
}

/** How far the LEFT stick must be pushed away from the operator before the
 *  precision modifier engages. Deliberately past the halfway point: the thumb
 *  rests on the stick while gripping, and gains that silently halve feel
 *  exactly like an arm that has started lagging. */
export const PRECISION_STICK_Y = -0.7;

/** The kit's fine-work modifier, moved off A/X.
 *
 *  A/X hold is the record toggle and stays that way — an accidental take
 *  boundary is corrupted data — so the modifier went to the one free input a
 *  driving hand can still reach: the left stick, pushed away and held. The
 *  HUD says PRECISION the whole time it is engaged, because a modifier you
 *  cannot see is a modifier you leave on by accident. */
export function precisionHeld(session: XRSessionLike): boolean {
  return stickAxes(session, "left")[1] <= PRECISION_STICK_Y;
}

/** Step an index around a ring, guarding against an empty list (which would
 *  otherwise produce NaN and paint an undefined selection). */
export function cycleIndex(len: number, i: number, dir = 1): number {
  if (len <= 0) return 0;
  return ((i + dir) % len + len) % len;
}

/** How wide the workspace-camera quad hangs, in metres, at HUD_DIST.
 *
 *  "Small" was the complaint, and the fix is mostly this number: the tile is a
 *  quad in the world, so apparent size is width/distance, not pixels. L spans
 *  roughly 76° of view at the HUD's 1.15 m — about as wide as you can put a
 *  panel before the edges leave the sweet spot of the optics.
 */
export const CAM_TILE_SIZES: readonly { name: string; widthM: number }[] = [
  { name: "S", widthM: 1.1 },
  { name: "M", widthM: 1.6 },
  { name: "L", widthM: 2.2 },
];

/** How long A/X must stay down before the record toggle fires. The thumb
 *  rests near A/X while gripping; a plain press would toggle takes by
 *  accident, and an accidental take boundary is corrupted data. */
export const RECORD_HOLD_MS = 500;

/** How long the LEFT stick must stay clicked before the arms reset home.
 *  Longer than a view-cycle click could ever be by accident, shorter than
 *  feels broken. The short-click action (next view) fires on release. */
export const RESET_HOLD_MS = 800;

export type HoldToggleState = {
  down: boolean;
  /** When the current press started (frame time, ms); null while released. */
  since: number | null;
  /** True for exactly one update: the moment the hold threshold is crossed. */
  toggled: boolean;
};

export function holdToggleInit(): HoldToggleState {
  return { down: false, since: null, toggled: false };
}

/**
 * Edge-AND-hold detection for a button that must fire once per deliberate
 * press, and only after `holdMs` of continuous hold. Pure: feed it the raw
 * button state and the frame timestamp, act when the returned state's
 * `toggled` is true. Release resets; re-pressing is required to fire again.
 */
export function holdToggle(
  prev: HoldToggleState,
  isDown: boolean,
  now: number,
  holdMs: number,
): HoldToggleState {
  if (!isDown) return { down: false, since: null, toggled: false };
  if (!prev.down) return { down: true, since: now, toggled: false };
  if (!prev.toggled && prev.since !== null && now - prev.since >= holdMs) {
    return { down: true, since: prev.since, toggled: true };
  }
  return { down: true, since: prev.since, toggled: false };
}

/** Fire a haptic pulse on one hand's controller, where supported. Best
 *  effort by design: haptics are feedback, never a safety channel. */
export function pulse(
  session: XRSessionLike,
  hand: "left" | "right",
  intensity: number,
  durationMs: number,
): void {
  for (const src of session.inputSources) {
    if (src.handedness !== hand) continue;
    try {
      src.gamepad?.hapticActuators?.[0]?.pulse?.(intensity, durationMs);
    } catch {
      /* haptics are decorative; a throwing actuator must not break sampling */
    }
  }
}

// ---- authority-transition haptics -------------------------------------------

export type SideAuthorityLike = "held" | "acquiring" | "driving";

export type HapticCue = {
  hand: "left" | "right";
  intensity: number;
  durationMs: number;
};

/**
 * What the operator's hands should feel when authority moves. Pure, so the
 * vocabulary is testable: a soft tick when a countdown starts, a firm buzz on
 * handover (the moment the arm goes live must never be ambiguous), a medium
 * tick on release. A hand whose authority did not change feels nothing.
 */
export function hapticCues(
  prev: Partial<Record<"left" | "right", SideAuthorityLike>>,
  next: Partial<Record<"left" | "right", SideAuthorityLike>>,
): HapticCue[] {
  const cues: HapticCue[] = [];
  for (const hand of ["left", "right"] as const) {
    const before = prev[hand];
    const after = next[hand];
    if (before === after || after === undefined) continue;
    if (after === "acquiring") cues.push({ hand, intensity: 0.25, durationMs: 60 });
    else if (after === "driving") cues.push({ hand, intensity: 0.8, durationMs: 180 });
    else if (before !== undefined) cues.push({ hand, intensity: 0.45, durationMs: 100 });
  }
  return cues;
}

// ---- the backend's ik_state push --------------------------------------------
//
// The socket answers every frame batch with the teleoperator's own view of
// what the solver is doing. It is the only channel that can tell the operator
// WHY an arm feels wrong — the status poll knows about authority and
// collisions, not about a wrist being asked for a twist two axes cannot
// deliver.

/** One side of the `ik_state` payload. Every field optional: the backend
 *  reports a short dict for an untracked or open-gripped hand, and a client
 *  that indexes into the long one unconditionally crashes inside the XR
 *  loop, which on a headset looks like the page dying. */
export type IkSideDiag = {
  tracked?: boolean;
  engaged?: boolean;
  driving?: boolean;
  /** 0..1 trouble mix the backend already computed — limit pressure, reach
   *  absorption, singularity proximity and orientation deficit, gated and
   *  maxed rather than averaged. */
  haptic?: number;
  limit_pressure_deg?: number;
  pos_err_m?: number;
  /** Smallest singular value of the position Jacobian, m/rad. The honest
   *  conditioning number on a 25 cm arm — see the port handover on why this
   *  and not |det J|. */
  sigma_min?: number;
  singularity?: number;
  /** The 1-DoF orientation deficit of a 5-DoF wrist, radians of demand the
   *  two wrist axes cannot serve. */
  orient_residual?: number;
  pos_absorbed?: number;
  rot_absorbed?: number;
};

export type IkSides = Partial<Record<"left" | "right", IkSideDiag>>;

/** Config values as they travel on the wire — numbers, with the two boolean
 *  knobs and the stance enum riding along. */
export type TuningValues = Record<string, number | boolean | string | null>;

export type VrSocketMessage =
  | { kind: "ik_state"; config: TuningValues | null; sides: IkSides }
  | { kind: "config_applied"; config: TuningValues };

/**
 * Read one server → client message off the teleop socket.
 *
 * Tolerates both names the contract uses for the same payload: the relay
 * answered `request_settings` with its `ik_state` dict, and the unified
 * socket answers with `settings`. Anything else — a frame echoed by a second
 * client, a keep-alive — returns null rather than throwing, because this runs
 * on the socket's message handler and a throw there is silent.
 */
export function parseVrSocketMessage(raw: unknown): VrSocketMessage | null {
  let msg: unknown = raw;
  if (typeof raw === "string") {
    try { msg = JSON.parse(raw); } catch { return null; }
  }
  if (!msg || typeof msg !== "object") return null;
  const m = msg as { type?: unknown; config?: unknown; sides?: unknown };
  let config = (m.config && typeof m.config === "object")
    ? m.config as TuningValues : null;
  if (m.type === "ik_state" || m.type === "settings") {
    const sides = (m.sides && typeof m.sides === "object")
      ? m.sides as IkSides : {};
    // A `settings` reply that spreads the config at the top level instead of
    // nesting it reads the same way here. Cheap, and the failure it prevents
    // is silent: sliders that simply never seed, showing defaults the robot
    // does not have.
    if (config === null && m.type === "settings") {
      const { type: _t, sides: _s, ...rest } = msg as Record<string, unknown>;
      void _t; void _s;
      if (Object.keys(rest).length) config = rest as TuningValues;
    }
    return { kind: "ik_state", config, sides };
  }
  if (m.type === "config_applied") {
    return { kind: "config_applied", config: config ?? {} };
  }
  return null;
}

/** Orientation residual (rad) above which the wrist is visibly short of the
 *  demand. The backend gates its own haptic mix at the same number; matching
 *  it keeps the buzz and the HUD line telling one story. */
export const ORIENT_DEFICIT = 0.5;

/** Below this the backend's mixed trouble signal is noise, not information.
 *  The kit's floor, kept. */
export const HAPTIC_FLOOR = 0.08;

/**
 * What a driving hand should feel from the solver's own diagnostics.
 *
 * Two cues, and the sharper one wins. The continuous one is the backend's
 * gated trouble mix, passed through as a light 60 ms buzz — the operator
 * feels the workspace edge, the joint stops and the singular set. The
 * discrete one fires on the EDGE of the orientation deficit: the wrist has
 * run out of axes, and the answer is to move the hand rather than twist
 * harder, so it gets a distinct hard pulse instead of blending into the hum
 * it would otherwise be part of.
 *
 * Pure, and edge-triggered off `prev`, so a sustained deficit buzzes once
 * rather than every 50 ms for as long as the operator holds the pose.
 */
export function ikHapticCues(prev: IkSides, next: IkSides): HapticCue[] {
  const cues: HapticCue[] = [];
  for (const hand of ["left", "right"] as const) {
    const d = next[hand];
    if (!d?.driving) continue;
    const wasShort = (prev[hand]?.orient_residual ?? 0) > ORIENT_DEFICIT;
    const isShort = (d.orient_residual ?? 0) > ORIENT_DEFICIT;
    if (isShort && !wasShort) {
      cues.push({ hand, intensity: 0.9, durationMs: 140 });
      continue;
    }
    const h = d.haptic ?? 0;
    if (h >= HAPTIC_FLOOR) {
      cues.push({ hand, intensity: Math.min(1, h), durationMs: 60 });
    }
  }
  return cues;
}

// ---- live tuning ------------------------------------------------------------

export type TuningKnob = {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  /** Client-side only: never sent as a `config_update`, persisted here. */
  local?: boolean;
};

/** The read-out point offset, in metres, back along the controller axis onto
 *  roughly where the operator's wrist turns. The reference stack solves for
 *  this with a five-second in-VR ritual; this is the same idea as one
 *  adjustable number. Client-side, because only the client has the grip pose
 *  to apply it to. */
export const WRIST_PIVOT_KEY = "wrist_pivot_m";
export const DEFAULT_WRIST_PIVOT_M = 0.09;

/**
 * The knobs the in-headset list exposes, in order.
 *
 * Kept to the ones an operator reaches for mid-session: a list you walk with
 * a thumbstick stops being usable somewhere around a dozen rows, and the rest
 * of `QuestTeleopConfig` belongs on the desktop panel where there is room for
 * it. Ranges mirror the backend's BOUNDS — it clamps regardless, this just
 * stops the stick asking for something that will be refused.
 */
export const TUNING_KNOBS: readonly TuningKnob[] = [
  { key: "scale_translation", label: "translation gain", min: 0.1, max: 4, step: 0.05 },
  { key: "scale_rotation", label: "rotation gain", min: 0.1, max: 4, step: 0.05 },
  { key: "precision_factor", label: "precision factor", min: 0.05, max: 1, step: 0.05 },
  { key: "pos_reach_limit", label: "reach limit (m)", min: 0, max: 0.6, step: 0.01 },
  { key: "rot_reach_limit", label: "twist limit (rad)", min: 0, max: 2, step: 0.05 },
  { key: "pose_filter_alpha", label: "pose smoothing", min: 0.05, max: 1, step: 0.05 },
  { key: "max_dq_deg_pos", label: "step cap arm (°)", min: 0.25, max: 15, step: 0.25 },
  { key: "max_dq_deg_rot", label: "step cap wrist (°)", min: 0.25, max: 30, step: 0.25 },
  { key: "lam_pos", label: "IK damping", min: 0.001, max: 0.2, step: 0.001 },
  { key: "w0", label: "singularity ramp", min: 0.001, max: 0.1, step: 0.001 },
  { key: WRIST_PIVOT_KEY, label: "wrist pivot (m)", min: 0, max: 0.2, step: 0.005, local: true },
];

/** The rest of what `QuestTeleopConfig` will take, desktop panel only —
 *  numbers you set once against a bench measurement, not mid-take. The two
 *  workspace floors are here rather than in the headset list on purpose:
 *  they bound the DEMAND and keep working when the collision guard is off,
 *  which is not a thing to nudge with a thumbstick while driving. */
export const DESK_ONLY_KNOBS: readonly TuningKnob[] = [
  { key: "lam0", label: "posture damping", min: 0, max: 0.5, step: 0.005 },
  { key: "mu", label: "posture bias", min: 0, max: 0.2, step: 0.005 },
  { key: "lam_rot", label: "wrist damping", min: 0.001, max: 1, step: 0.005 },
  { key: "min_tip_z", label: "tip floor (m)", min: -0.2, max: 0.4, step: 0.005 },
  { key: "min_wrist_z", label: "wrist floor (m)", min: -0.2, max: 0.4, step: 0.005 },
];

export const ALL_KNOBS: readonly TuningKnob[] = [...TUNING_KNOBS, ...DESK_ONLY_KNOBS];

export function clampKnob(knob: TuningKnob, value: number): number {
  if (!Number.isFinite(value)) return knob.min;
  return Math.min(knob.max, Math.max(knob.min, value));
}

/** Knob value for display. Three decimals across the board so the column
 *  does not jitter in width as a value crosses 1. */
export function formatKnob(value: number | undefined | null): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(3) : "—";
}

/** One deliberate stick push per this many ms. Without it a held stick walks
 *  the whole list in a frame. The kit's number. */
export const TUNING_REPEAT_MS = 220;

export type TuningNav = { index: number; lastStepMs: number };

export type TuningStep = {
  nav: TuningNav;
  /** The knob to write, or null when this push only moved the cursor. */
  patch: { key: string; value: number } | null;
};

/**
 * Walk and adjust the tuning list from one thumbstick sample. Pure.
 *
 * Pulling the stick back walks DOWN the list, pushing it away walks up; left
 * and right step the selected knob by its own step, clamped to its own range.
 * Vertical wins over horizontal so a diagonal push cannot both move the
 * cursor and change a value.
 */
export function stepTuning(
  nav: TuningNav,
  axes: readonly [number, number],
  now: number,
  values: Readonly<Record<string, number>>,
  knobs: readonly TuningKnob[] = TUNING_KNOBS,
): TuningStep {
  if (!knobs.length) return { nav, patch: null };
  const index = Math.min(Math.max(nav.index, 0), knobs.length - 1);
  if (now - nav.lastStepMs < TUNING_REPEAT_MS) return { nav: { ...nav, index }, patch: null };
  const [x, y] = axes;
  if (Math.abs(y) > 0.6) {
    return {
      nav: { index: cycleIndex(knobs.length, index, y > 0 ? 1 : -1), lastStepMs: now },
      patch: null,
    };
  }
  if (Math.abs(x) > 0.6) {
    const knob = knobs[index];
    const current = values[knob.key];
    const base = typeof current === "number" && Number.isFinite(current)
      ? current : knob.min;
    const next = clampKnob(knob, base + (x > 0 ? knob.step : -knob.step));
    return { nav: { index, lastStepMs: now }, patch: { key: knob.key, value: next } };
  }
  return { nav: { ...nav, index }, patch: null };
}

// ---- session launcher --------------------------------------------------------

export type Stance = "behind" | "mirror" | "front";

export type ArmPairing = { left_arm: string | null; right_arm: string | null };

/**
 * Which hand drives which arm, for `POST /teleop/human/start`.
 *
 * `left_arm` names the arm the operator's LEFT hand drives. In the "behind"
 * (egocentric) stance the operator faces the same way the arms reach, so the
 * arm under their right hand — the one on frame-right in the over-shoulder
 * view — is the one the config names "left". Facing the arms, the pairing is
 * direct. Getting this wrong reads as "the controls are inverted", not as a
 * subtle mapping issue.
 *
 * Arm identity comes from the ID, not from config order. `config.yaml` lists
 * the real rig as [right, left] while the sim config lists [left, right], so
 * a positional rule is right in one and inverted in the other — which is
 * exactly the failure the stance swap exists to prevent. Unrecognisable IDs
 * fall back to config order, first = the robot's left.
 *
 * A solo session keeps the arm on the side the same rule gives it and sends
 * null for the other, so "my right hand drives the arm I picked" holds
 * identically in a one-armed and a two-armed session.
 */
export function armPairing(
  armIds: readonly string[], stance: Stance, soloArm: string | null = null,
): ArmPairing {
  const named = (re: RegExp) => armIds.find((id) => re.test(id)) ?? null;
  let robotLeft = named(/left/i);
  let robotRight = named(/right/i);
  if (robotLeft !== null && robotLeft === robotRight) {
    robotLeft = null;
    robotRight = null;
  }
  if (robotLeft === null && robotRight === null) {
    robotLeft = armIds[0] ?? null;
    robotRight = armIds[1] ?? null;
  } else if (robotLeft === null) {
    robotLeft = armIds.find((id) => id !== robotRight) ?? null;
  } else if (robotRight === null) {
    robotRight = armIds.find((id) => id !== robotLeft) ?? null;
  }

  const behind = stance === "behind";
  const dual: ArmPairing = behind
    ? { left_arm: robotRight, right_arm: robotLeft }
    : { left_arm: robotLeft, right_arm: robotRight };
  if (!soloArm) return dual;
  if (dual.left_arm === soloArm) return { left_arm: soloArm, right_arm: null };
  if (dual.right_arm === soloArm) return { left_arm: null, right_arm: soloArm };
  // An arm the left/right resolution never placed (a third arm, an ID that
  // says neither). Treat it as the robot's left arm — the same convention
  // the positional fallback uses — so the answer is at least consistent.
  return behind
    ? { left_arm: null, right_arm: soloArm }
    : { left_arm: soloArm, right_arm: null };
}

export type SessionPreset = {
  id: string;
  label: string;
  /** null = both hands drive, one arm each. */
  soloArm: string | null;
};

/** The launcher's buttons, from the arms the backend actually enabled. A
 *  one-armed config offers exactly one preset rather than a "dual" button
 *  that can only fail. */
export function sessionPresets(armIds: readonly string[]): SessionPreset[] {
  const solo = armIds.map((id) => ({ id: `solo-${id}`, label: `solo ${id}`, soloArm: id }));
  if (armIds.length < 2) return solo;
  return [{ id: "dual", label: "dual", soloArm: null }, ...solo];
}
