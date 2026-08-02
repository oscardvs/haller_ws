/**
 * WebXR → the VR frame the backend's `vr_input.py` expects.
 *
 * This file is deliberately thin. Unlike `mediapipe.ts`, which fuses landmarks
 * into finished `SideFrame`s, nothing here computes geometry: it samples the
 * headset and controller poses and ships them raw. The shoulder estimate, the
 * two-link elbow IK and the synthetic hand all live server-side in
 * `vr_input.py`, where they can be tested against the real retargeter and
 * cannot drift from its coordinate conventions. Read that module's docstring
 * before changing the wire shape below.
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
  hapticActuators?: readonly {
    pulse?: (intensity: number, durationMs: number) => unknown;
  }[];
} | null;

export type XRInputSourceLike = {
  handedness: XRHandedness;
  gripSpace?: unknown;
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
   *  own arm on the backend; see vr_input.py's `dead_man_sides`. */
  squeeze: boolean;
  tracked: boolean;
};

export type VRFrame = {
  type: "vr_keypoints";
  ts_ms: number;
  /** Either grip — kept for the status chip and for backends that predate the
   *  per-side split. The split itself rides on each controller's `squeeze`. */
  dead_man: boolean;
  /** "none" (default): identical arms mounted side by side, no mirroring —
   *  egocentric frames already give opposite pan signs for symmetric
   *  gestures. "both": a rig whose two mounts are true mirror images. */
  mirror_mode?: "none" | "both";
  head: { position: Vec3; orientation: Quat } | null;
  left: ControllerSample | null;
  right: ControllerSample | null;
  /** Optional per-operator limb lengths, metres. Absent keys use BodyModel defaults. */
  body?: Partial<Record<
    "shoulder_drop" | "shoulder_back" | "shoulder_half_width" |
    "upper_arm" | "fore_arm" | "hand_len" | "hand_half_width", number
  >>;
};

export type BodyOverride = NonNullable<VRFrame["body"]>;

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

/** Where the HUD quad hangs in local-floor space: eye-ish height, just over a
 *  metre out — near enough to read, far enough not to crowd the workspace. */
const HUD_POS: [number, number, number] = [0, 1.35, -1.15];
const HUD_W = 1.1;   // metres; canvas is 4:3, height follows

type XRViewLike = {
  projectionMatrix: Float32Array;
  transform: { inverse: { matrix: Float32Array } };
};

export type XRScene = {
  /** Clear both eyes; when `hud` is given, draw it as a world-locked quad.
   *  `hudDirty` re-uploads the canvas texture (skip it on unchanged frames —
   *  a texture upload per display frame is the whole cost of this HUD). */
  render(frame: XRFrameLike, refSpace: unknown,
         hud: HTMLCanvasElement | null, hudDirty: boolean): void;
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
  let tex: WebGLTexture | null = null;
  let model: Float32Array | null = null;

  function ensurePipeline(hud: HTMLCanvasElement): boolean {
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
    tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const hudH = HUD_W * (hud.height / hud.width);
    // Column-major scale + translate: the quad's unit extents become metres.
    model = new Float32Array([
      HUD_W, 0, 0, 0,
      0, hudH, 0, 0,
      0, 0, 1, 0,
      HUD_POS[0], HUD_POS[1], HUD_POS[2], 1,
    ]);
    prog = p;
    return true;
  }

  return {
    render(frame, refSpace, hud, hudDirty) {
      const fb = (session.renderState?.baseLayer ?? layer).framebuffer;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb as WebGLFramebuffer | null);
      gl.clearColor(r, g, b, a);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      if (!hud || !ensurePipeline(hud)) return;
      const pose = frame.getViewerPose(refSpace) as unknown as
        { views?: XRViewLike[] } | null | undefined;
      const views = pose?.views;
      if (!views?.length) return;
      gl.useProgram(prog);
      gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      if (hudDirty) {
        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, 1);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, hud);
      }
      const active = (session.renderState?.baseLayer ?? layer) as LayerLike;
      for (const view of views) {
        const vp = active.getViewport?.(view);
        if (!vp) continue;
        gl.viewport(vp.x, vp.y, vp.width, vp.height);
        const mvp = mat4Multiply(
          mat4Multiply(view.projectionMatrix, view.transform.inverse.matrix),
          model as Float32Array,
        );
        gl.uniformMatrix4fv(uMVP, false, mvp);
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
    blocking: string[];
    reason: string;
  }>>;
} | null;

/**
 * Repaint the in-scene HUD. Layout: workspace camera on top (when a frame is
 * available), status strip below. Pure canvas 2D so it can run anywhere;
 * failures to draw the camera (stream warming up, no camera) degrade to the
 * text strip alone, never to an exception — this runs inside the XR loop.
 */
export function paintHud(
  ctx: CanvasRenderingContext2D,
  status: HudStatusLike,
  cam: HTMLImageElement | null,
): void {
  const W = ctx.canvas.width;
  const H = ctx.canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "rgba(8, 10, 14, 0.82)";
  ctx.fillRect(0, 0, W, H);

  const camH = Math.round(H * 0.72);
  if (cam && cam.complete && cam.naturalWidth > 0) {
    try {
      const scale = Math.min(W / cam.naturalWidth, camH / cam.naturalHeight);
      const dw = cam.naturalWidth * scale;
      const dh = cam.naturalHeight * scale;
      ctx.drawImage(cam, (W - dw) / 2, (camH - dh) / 2, dw, dh);
    } catch {
      /* a mid-frame MJPEG boundary can throw; next repaint recovers */
    }
  } else {
    ctx.fillStyle = "#9aa0a6";
    ctx.font = "28px monospace";
    ctx.textAlign = "center";
    ctx.fillText("no workspace camera", W / 2, camH / 2);
  }

  ctx.textAlign = "left";
  ctx.font = "bold 30px monospace";
  let y = camH + 42;
  const state = (status?.state ?? "—").toUpperCase();
  const col = status?.collision;
  ctx.fillStyle = "#e8eaed";
  ctx.fillText(state, 24, y);
  if (col?.enabled) {
    const slack = col.slack_m;
    ctx.fillStyle = col.limited ? "#f28b82"
      : (slack ?? 1) < 0 ? "#fdd663" : "#9aa0a6";
    ctx.fillText(
      col.limited
        ? `COLLISION HOLD ${slack !== undefined ? (slack * 1000).toFixed(0) : "—"} mm`
        : `clearance ${slack !== undefined ? (slack * 1000).toFixed(0) : "—"} mm`,
      300, y);
  }
  ctx.font = "28px monospace";
  for (const side of ["left", "right"] as const) {
    y += 40;
    const acq = status?.acquire?.[side];
    const grip = status?.clutch?.sides?.[side] ?? status?.clutch?.engaged;
    ctx.fillStyle = "#9aa0a6";
    ctx.fillText(`${side === "left" ? "L" : "R"} ${grip ? "●" : "○"}`, 24, y);
    if (!acq) continue;
    ctx.fillStyle = acq.authority === "driving" ? "#81c995"
      : acq.authority === "acquiring" ? "#fdd663" : "#9aa0a6";
    let line = acq.authority;
    if (acq.authority === "acquiring") {
      if (acq.remaining_ms !== null) line += ` ${(acq.remaining_ms / 1000).toFixed(1)}s`;
      if (acq.blocking.length) line += `  match: ${acq.blocking.join(", ")}`;
    }
    if (acq.reason === "no_tracking") line += "  (no tracking)";
    ctx.fillText(line, 140, y);
  }
  y += 40;
  if (status?.last_error) {
    ctx.fillStyle = "#f28b82";
    ctx.fillText(status.last_error.slice(0, 60), 24, y);
  } else {
    ctx.fillStyle = "#9aa0a6";
    ctx.fillText("grips = drive · trigger = gripper · B/Y = E-STOP", 24, y);
  }
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
  opts: { tsMs: number; body?: BodyOverride; forceDisengaged?: boolean;
          mirrorMode?: "none" | "both" },
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
    const sample: ControllerSample = pose
      ? {
          ...pose,
          trigger: buttons[BUTTON_TRIGGER]?.value ?? 0,
          squeeze,
          tracked: true,
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
    ...(opts.mirrorMode ? { mirror_mode: opts.mirrorMode } : {}),
    head,
    left,
    right,
    ...(opts.body ? { body: opts.body } : {}),
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
