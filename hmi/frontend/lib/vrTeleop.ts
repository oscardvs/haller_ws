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
/** A on the right controller, X on the left — the precision modifier, HELD.
 *  The kit's `PRECISION_BUTTON_INDEX = 4`, restored. A modifier belongs on
 *  the button the thumb already rests on while gripping; it is the take
 *  boundary that must be hard to hit by accident, and that now lives on B. */
export const BUTTON_AX = 4;
/** xr-standard thumbstick click — "go home". The kit's
 *  `REST_RAMP_BUTTON_INDEX = 3`: one click, one velocity-limited ramp to the
 *  rest pose. */
export const BUTTON_THUMBSTICK = 3;
/** B on the right controller, Y on the left — episode control, per hand.
 *  B ends the take and banks it; Y discards it and re-records. This is the
 *  kit's mapping, documented in `examples/record_so101.py`:
 *
 *      B (right controller)  end the current episode / end the reset phase
 *      Y (left controller)   discard + re-record the current episode
 *
 *  There is NO controller E-STOP. It lives on the desktop HMI and the bench
 *  cutoff — the kit never had one on the sticks, and binding a take boundary
 *  and a safety stop to the same thumb is what forced the old A/X juggle. */
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

export type XRScene = {
  /** Clear both eyes and draw nothing. The scene has no content: the
   *  operator is looking at their own bench through passthrough, which is the
   *  whole point of an AR session. */
  render(frame: XRFrameLike, refSpace: unknown): void;
};

const _NOOP_SCENE: XRScene = { render: () => {} };

/** Bare passthrough. Sets up the XR GL layer an immersive session requires,
 *  clears both eyes to transparent, and draws NOTHING.
 *
 *  This used to composite a status panel, a workspace-camera tile, a tuning
 *  list, a view menu and an end-of-take prompt onto quads in front of the
 *  operator. All of it is gone. The kit's in-headset client draws nothing at
 *  all — `examples/record_so101.py` puts it plainly, "in passthrough you see
 *  the real scene anyway" — and a panel hanging in front of a bench is
 *  something to look past while doing fine work with a real arm.
 *
 *  What replaced it as the feedback channel is what the kit already used:
 *  haptics. `recorderHapticCue` fires on every take transition and
 *  `ikHapticCues` on every limit, and both reach the operator without asking
 *  them to look away from their hands. Everything the panel used to say is on
 *  the desktop page, where reading it costs nothing.
 */
export function attachRenderScene(
  session: XRSessionLike,
  mode: TeleopXRSession["mode"],
): XRScene {
  type LayerLike = { framebuffer: unknown };
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
  // Transparent in AR so passthrough shows through untouched. The VR fallback
  // still needs SOMETHING behind the scene or the compositor shows a void.
  const [r, g, b, a] = mode === "immersive-ar"
    ? [0, 0, 0, 0]
    : [0.05, 0.05, 0.07, 1];

  return {
    render(frame: XRFrameLike, refSpace: unknown): void {
      const pose = frame.getViewerPose(refSpace);
      if (!pose) return;
      const active = (session.renderState?.baseLayer ?? layer) as LayerLike;
      gl.bindFramebuffer(gl.FRAMEBUFFER, active.framebuffer as WebGLFramebuffer | null);
      gl.clearColor(r, g, b, a);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
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
  /** Where the take is. Absent on a backend that predates the start gate, in
   *  which case `recording` alone still paints. */
  state?: TakeState;
  recording: boolean;
  episode_frames: number;
  /** Ticks seen and NOT written. A degraded read is a dropped frame, not a
   *  recorded one (port invariant 9) — so a take shedding rows has to say so
   *  while it is still being driven, not in review. */
  skipped_frames?: number;
  /** The single worst drop source, already reduced: "left_wrist", not a
   *  table. An operator needs a cable to go check. */
  worstDrop?: string | null;
  /** Measured against declared. `fps` in info.json is measured or the episode
   *  does not open (port invariant 10); a rate that has sagged is a take
   *  worth abandoning early. */
  fpsMeasured?: number | null;
  fpsDeclared?: number | null;
  /** The recorder's own verdict on whether the measured rate is FAITHFUL to
   *  the declared one — `recordRateFaithful`, not a threshold. The panel reads
   *  it; this painter holds no band of its own, so it cannot drift from the
   *  refusal the operator gets at arm time. `null` means NOT ANSWERABLE: the
   *  rate is unmeasured, or the backend publishes no tolerance. */
  rateFaithful?: boolean | null;
  /** Why an ARMED gate fell back to idle. Not an error — the gate saying why
   *  it dropped. Silent un-arming is the failure the gate exists to prevent. */
  invalidatedReason?: string | null;
  /** True while the page is holding ARMED itself because the backend has no
   *  /record/arm yet: nothing is written before ROLL, but the schema is not
   *  frozen, the 409s arrive late and the episode index is a guess. */
  localGate?: boolean;
  /** Takes this page has saved — the fallback count, and the floor under the
   *  dataset-wide one. */
  takes?: number;
  /** The gate's index for the take in hand — `episode_index` off
   *  `GET /record/status`, and nothing else. **Null whenever no take is
   *  armed:** the recorder sets it at ARM and clears it at STOP, so idle is
   *  null by construction, and a backend with no start gate never sends it at
   *  all. The take is then named `take N` off this page's own counter, which
   *  is true of the page-load, rather than `ep N` off a count that would be
   *  right only by coincidence.
   *
   *  This REPLACED a field called `episodes` that every reader below took as
   *  an index except the idle menu, which took it as a count. The two agreed
   *  for as long as `episode_index` did not exist. A different meaning got a
   *  different name so a stale reader breaks at the type rather than painting
   *  the wrong number — same reasoning as `record_rate_tolerance`. */
  episodeIndex?: number | null;
  /** How many episodes the DATASET holds. A count, and the idle menu is its
   *  only reader. **Not interchangeable with `episodeIndex`:** that one names
   *  the take in hand and renumbers across a prune, this one counts what is on
   *  disk and is still guessed past lerobot's RAM buffer (`episodesTotal`).
   *  They coincide on the happy path and are not the same fact. Null when no
   *  dataset has been read — a fresh repo, or the endpoint refused. */
  datasetCount?: number | null;
} | null;



/** Which hand drives which arm this session. Null on a hand with no arm: an
 *  absent side is a fact about the rig, not a fault, and the operator must be
 *  able to see the arm set from inside the headset — a session started on the
 *  wrong preset is otherwise only discovered 60 s into a take. */
export type ArmSetLike = { left: string | null; right: string | null } | null;


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
  /** Which hand drives which arm this session, when the pairing is known. */
  armSet?: ArmSetLike;
  /** The live-tuning list, when the operator has it open. */
  tuning?: {
    open: boolean;
    index: number;
    values: Readonly<Record<string, number>>;
    /** Knobs the operator moved this page-load — the ones the page re-asserts
     *  over the server's per-connection defaults. Marked, because a value that
     *  survives a reconnect while its neighbours revert is otherwise
     *  witchcraft. */
    dirty?: readonly string[];
  } | null;
  /** True while the precision modifier is held. Shown prominently: a
   *  modifier you cannot see is one you leave on. */
  precision?: boolean;
  /** The end-of-take decision is open. Modal: it takes over the menu box and
   *  both stick clicks until the operator picks. Four outcomes exist — keep,
   *  keep_stop, redo, drop — but the headset binds only the two that return
   *  to ARMED, because in a session whose point is banking takes the next one
   *  is always the expected next thing. Standing down is the desktop's, where
   *  a button costs nothing and collides with no trained gesture. */
  endPrompt?: boolean;
  /** True for a moment after the operator held the left stick during the
   *  prompt — the trained in-session home gesture, which must not fire
   *  through the tail of a take they may be about to keep. Answered rather
   *  than silently dropped: a gesture the operator has trained gets told no. */
  homeRefused?: boolean;
} | null;

/* Character budgets, and why they are numbers rather than a measureText call.
 *
 * The panel canvas is 1024 wide. The status column is clipped at 563 px and
 * the menu box has 392 px between its padding and its right-aligned value
 * column. Monospace advances at 0.6 em, so a row's width is (chars × 0.6 ×
 * fontPx) and the budgets below fall straight out of that:
 *
 *     status  22px from x=24 → 515 / 13.2 = 39 chars
 *     menu    18px body      → 392 / 10.8 = 36 chars
 *     menu    20px title     → 392 / 12.0 = 32 chars
 *
 * measureText is not used because this runs inside the XR loop at ~10 Hz and
 * the answer is a compile-time property of the copy, not of the frame. The
 * budgets are pinned by tests instead, which is where a too-long line should
 * be caught — not on someone's face in a headset, where the failure mode is
 * silent truncation of whatever sits at the end of the row.
 */
export const STATUS_MAX_CHARS = 39;
export const MENU_MAX_CHARS = 36;
export const MENU_TITLE_MAX_CHARS = 32;







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
          wristPivotM?: number },
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

    // ONE pose space for position AND orientation — the kit's form (its page
    // samples gripSpace || targetRaySpace for the whole pose). The constant
    // grip-vs-ray tilt cancels exactly in the mapper's clutch-relative delta
    // q_now · q_prev⁻¹, so a single space loses nothing; what the old
    // grip-position + ray-orientation mix bought was two phantom ~50-60°
    // rotation increments per ray dropout — one frame where getPose(ray)
    // returned null silently swapped the orientation source to the grip
    // frame and back, and the incremental mapper read each swap as real
    // hand rotation. The kit's 46 episodes were driven single-space.
    const space = src.gripSpace ?? src.targetRaySpace;
    const pose = space ? poseToPair(frame.getPose(space, refSpace)) : null;
    // Read-out point: the pose position shifted back along the controller's
    // own +Z (which points toward the operator in WebXR grip space) onto
    // roughly the operator's wrist pivot. See DEFAULT_WRIST_PIVOT_M — a pure
    // twist about a palm-centred point is an ARC, and the mapper has no way
    // to tell that arc from translation the operator never asked for.
    const pivotM = opts.wristPivotM ?? 0;
    const back = pose && pivotM !== 0
      ? _rotateByQuat(pose.orientation, [0, 0, pivotM]) : null;
    // Precision is PER HAND — the kit reads the driving hand's own A/X. A
    // single global flag re-anchored and re-scaled the arm the OTHER hand
    // was mid-reach with.
    const precision = Boolean(buttons[BUTTON_AX]?.pressed);
    const sample: ControllerSample = pose
      ? {
          position: back
            ? [pose.position[0] + back[0], pose.position[1] + back[1],
               pose.position[2] + back[2]]
            : pose.position,
          orientation: pose.orientation,
          trigger: buttons[BUTTON_TRIGGER]?.value ?? 0,
          squeeze,
          tracked: true,
          ...(precision ? { precision: true } : {}),
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
/** True while ONE named controller's episode button (B right / Y left) is
 *  down. Per-hand, because the two hands mean OPPOSITE things: B banks the
 *  take, Y throws it away. Raw — the panel edge-detects, so a held button
 *  fires once rather than once per frame. */
export function episodePressed(
  session: XRSessionLike, hand: "left" | "right",
): boolean {
  for (const src of session.inputSources) {
    if (src.handedness !== hand) continue;
    if (src.gamepad?.buttons[BUTTON_BY]?.pressed) return true;
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

/** The kit's fine-work modifier, back on A/X where the kit put it.
 *
 *  It moved to the left stick only because A/X had been taken by the record
 *  toggle. With the take boundary on B — a deliberate reach, which is what a
 *  take boundary wants — A/X is free again, and a modifier is exactly the
 *  binding that suits a button the thumb already rests on: hold it, it
 *  applies; let go, it stops. Either hand, so a one-handed driver keeps it. */
export function precisionHeld(session: XRSessionLike): boolean {
  for (const src of session.inputSources) {
    if (src.handedness !== "left" && src.handedness !== "right") continue;
    if (src.gamepad?.buttons[BUTTON_AX]?.pressed) return true;
  }
  return false;
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

// ---- the take machine -------------------------------------------------------

/** Where a take is in its life.
 *
 *  ARMED is the start gate: full-rate teleop, the dataset open and its schema
 *  frozen, and NOT ONE FRAME WRITTEN. Stock lerobot-record starts episode 0 the
 *  instant the process boots, which is unusable solo in a headset. */
export type TakeState = "idle" | "armed" | "rolling" | "prompt";

/** What the operator decided about a take that just ended. `redo` is a
 *  first-class outcome, not a failure: 11 of the kit's 46 episodes were
 *  rejected, a rate only visible because both outcomes exist.
 *
 *  The two that return to ARMED are the headset's, because the next take is
 *  always the expected next thing in a session whose point is banking them.
 *  The two that stand down to IDLE are desktop-only — there is no room on the
 *  controller for a gesture that does not collide with a trained one. */
export type EndChoice = "keep" | "keep_stop" | "redo" | "drop";

export type TakeEvent =
  /** B, right hand — the kit's "end the current episode / end the reset
   *  phase". One button walks the whole ladder: ARM, ROLL, then bank the
   *  take and re-arm for the next one. */
  | { kind: "end_episode" }
  /** Y, left hand — the kit's "discard + re-record the current episode".
   *  Only ever throws away a take that is still in progress; see `stepTake`
   *  for why it will not reach back for one already written. */
  | { kind: "rerecord" }
  | { kind: "choose"; choice: EndChoice }
  /** The recorder's own state, from the status poll. Truth outranks the
   *  client's guess — a take that ended some other way takes the prompt with
   *  it, and a gate that was invalidated must not keep saying ARMED. */
  | { kind: "recorder"; state: TakeState; invalidated?: boolean }
  /** Session teardown: whatever was on the HUD dies with it. */
  | { kind: "abort" };

/** The REST act a transition demands, or null. */
export type TakeAct =
  | { do: "arm" }
  | { do: "roll" }
  | { do: "stop"; save: boolean; rearm: boolean };

export type TakeTransition = { state: TakeState; act: TakeAct | null };

/**
 * Step the take machine. Pure: the caller fires `act` and commits `state`.
 *
 * `ev.invalidated` deliberately does not change the transition — an
 * invalidated gate lands in `idle` exactly as a stand-down does. It rides
 * along so the caller can tell the two apart and SAY which happened: silent
 * un-arming is the precise failure the gate exists to prevent.
 */
export function stepTake(state: TakeState, ev: TakeEvent): TakeTransition {
  if (ev.kind === "abort") return { state: "idle", act: null };
  if (ev.kind === "recorder") {
    // Reconcile, never act. The recorder has no concept of a prompt and must
    // never invent one.
    if (ev.state === "prompt") return { state, act: null };
    // The prompt is a client-side overlay on a recorder that is genuinely
    // still rolling — `/record/stop` takes the save decision AT stop time —
    // so a "rolling" report must not slam it shut four times a second.
    if (ev.state === "rolling" && state === "prompt") {
      return { state: "prompt", act: null };
    }
    return { state: ev.state, act: null };
  }
  if (ev.kind === "choose") {
    // Only the prompt takes a decision; anywhere else this is a stale click.
    if (state !== "prompt") return { state, act: null };
    const save = ev.choice === "keep" || ev.choice === "keep_stop";
    const rearm = ev.choice === "keep" || ev.choice === "redo";
    return { state: rearm ? "armed" : "idle", act: { do: "stop", save, rearm } };
  }
  if (ev.kind === "rerecord") {
    // Y — bin the take in progress and line the next one up. The kit's
    // re-record, with one deliberate limit: it reaches only for a take that
    // is still OPEN. In lerobot-record, Y during the reset phase re-records
    // the episode just written; here that episode is already on disk, and a
    // single button press in a headset is the wrong gesture for deleting
    // written data. Binning a live take costs nothing; the desk keeps
    // DELETE LAST EPISODE for the other case.
    if (state === "rolling" || state === "prompt") {
      return { state: "armed", act: { do: "stop", save: false, rearm: true } };
    }
    return { state, act: null };
  }
  // end_episode — B. The kit's ladder, and the prompt is deliberately not on
  // it: B banks the take and re-arms, because in a session whose point is
  // banking takes the next one is always the expected next thing. The prompt
  // stays for the desk, which has room for a question the controller does not.
  if (state === "idle") return { state: "armed", act: { do: "arm" } };
  if (state === "armed") return { state: "rolling", act: { do: "roll" } };
  return { state: "armed", act: { do: "stop", save: true, rearm: true } };
}

// ---- recorder haptics -------------------------------------------------------

/** What the hands feel when the recorder moves. Both hands get it: inside a
 *  headset the haptic is the fastest channel there is, and "did that take
 *  actually start" is a question the operator must never have to guess at.
 *  Pure, and keyed on the transition rather than the state, so a steady state
 *  buzzes nothing. Returns null when nothing changed. */
export function recorderHapticCue(
  prev: TakeState, next: TakeState, choice?: EndChoice | null,
): { intensity: number; durationMs: number } | null {
  if (prev === next) return null;
  // Loaded, nothing written.
  if (prev === "idle" && next === "armed") return { intensity: 0.35, durationMs: 90 };
  // Frames are landing. The firmest cue in the vocabulary, deliberately: this
  // is the only moment that costs data if it is missed.
  if (prev === "armed" && next === "rolling") return { intensity: 0.8, durationMs: 220 };
  if (prev === "rolling" && next === "prompt") return { intensity: 0.45, durationMs: 120 };
  if (prev === "prompt" && next === "rolling") return { intensity: 0.2, durationMs: 60 };
  if (prev === "prompt" && next === "armed") {
    // Banked and binned both land in ARMED, so the HUD alone cannot tell them
    // apart while the operator's eyes are on the workspace. The hands can.
    if (choice === "keep") return { intensity: 0.6, durationMs: 180 };
    if (choice === "redo") return { intensity: 0.3, durationMs: 90 };
    return null;
  }
  if (prev === "prompt" && next === "idle") {
    if (choice === "keep_stop") return { intensity: 0.6, durationMs: 180 };
    // `drop`, or the recorder ending it underneath us — same weight, because
    // from the operator's side it is the same outcome.
    return { intensity: 0.25, durationMs: 80 };
  }
  // The gate dropped: invalidated, or the session ended. Weak, but never
  // silent — un-arming without a word is the failure the gate prevents.
  if (prev === "armed" && next === "idle") return { intensity: 0.15, durationMs: 50 };
  return null;
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
  | { kind: "config_applied"; config: TuningValues }
  /** This connection's pose frame was taken, so the backend has named it the
   *  driver and handed over the session's token. Held across a reload so the
   *  page can prove, on the way back, that the session is its own.
   *
   *  `config?: undefined` is not filler: every other variant carries one, and
   *  without it `msg.config` stops type-checking across the union and every
   *  reader has to narrow first. Spelling out that this message HAS no config
   *  keeps the union readable and says the true thing. */
  | { kind: "session"; token: string; config?: undefined };

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
  if (m.type === "session") {
    const token = (msg as { token?: unknown }).token;
    if (typeof token === "string" && token) return { kind: "session", token };
    return null;
  }
  return null;
}

/** Orientation residual (rad) above which the wrist is visibly short of the
 *  demand. The backend gates its own haptic mix at the same number; matching
 *  it keeps the buzz and the HUD line telling one story. */
export const ORIENT_DEFICIT = 0.5;
/** Re-arm threshold for the deficit's one-shot buzz. Controller pose jitter
 *  makes the residual flutter around ORIENT_DEFICIT, and with a single
 *  threshold every flutter is a fresh "edge" — a 0.9-intensity pulse every
 *  50 ms, felt as a trembling controller. The cue re-arms only after the
 *  residual has genuinely receded below this. */
export const ORIENT_DEFICIT_CLEAR = 0.4;

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
    // Hysteresis: fire on crossing ORIENT_DEFICIT, re-arm only below
    // ORIENT_DEFICIT_CLEAR — a residual hovering at the threshold is one
    // sustained deficit, not a new edge every sample.
    const wasShort = (prev[hand]?.orient_residual ?? 0) > ORIENT_DEFICIT_CLEAR;
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
 *  to apply it to. 0.05 is the kit's shipped default — the old 0.09
 *  overshot an uncalibrated axis by 80%, and an overshot pivot REVERSES the
 *  ghost translation it exists to cancel: the tool wanders on every twist. */
export const WRIST_PIVOT_KEY = "wrist_pivot_m";
export const DEFAULT_WRIST_PIVOT_M = 0.05;

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

// ---- who owns a tuned value -------------------------------------------------

/** Merge the server's clamped echo (`config_applied`). Whatever the robot took
 *  IS the value — a box that snaps to a different number is the robot telling
 *  you what it accepted, and re-asserting the unclamped ask would fight it. */
export function applyServerConfig(
  local: Readonly<Record<string, number>>,
  server: TuningValues | null,
): Record<string, number> {
  const values: Record<string, number> = { ...local };
  if (!server) return values;
  for (const [key, v] of Object.entries(server)) {
    // The wire carries booleans and the stance enum alongside the numbers;
    // only the numbers are knobs.
    if (typeof v === "number" && Number.isFinite(v)) values[key] = v;
  }
  return values;
}

/**
 * Reconcile a fresh connection's `settings` against what the operator tuned.
 *
 * `QuestTeleopConfig` lives PER CONNECTION and HumanTeleopClient reconnects
 * after 50 ms, so a socket blip silently reverts every knob — and gains that
 * quietly halve feel exactly like an arm that has started lagging. The page is
 * therefore the source of truth for the knobs the operator moved, and only
 * those: the server owns everything untouched.
 *
 * `reassert` is what to send straight back as a `config_update`. Knobs marked
 * `local` in ALL_KNOBS never leave the client and never appear in it.
 */
export function reconcileConfig(
  local: Readonly<Record<string, number>>,
  dirty: readonly string[],
  server: TuningValues | null,
  knobs: readonly TuningKnob[] = ALL_KNOBS,
): { values: Record<string, number>; reassert: Record<string, number> } {
  const values: Record<string, number> = { ...local };
  const reassert: Record<string, number> = {};
  if (!server) return { values, reassert };
  const moved = new Set(dirty);
  const clientOnly = new Set(knobs.filter((k) => k.local).map((k) => k.key));
  for (const [key, v] of Object.entries(server)) {
    if (typeof v !== "number" || !Number.isFinite(v)) continue;
    if (!moved.has(key)) {
      values[key] = v;
      continue;
    }
    const mine = values[key];
    // Belt and braces: a `local` knob is never sent, so the server cannot be
    // reporting one — and must not be handed one back if it somehow is.
    if (!clientOnly.has(key) && typeof mine === "number" && Number.isFinite(mine)) {
      reassert[key] = mine;
    }
  }
  // A dirty key the server does not report is left alone and NOT re-asserted:
  // a knob the server has never heard of is not a knob it will accept.
  return { values, reassert };
}

// ---- dataset tally ----------------------------------------------------------

/** What one read of `GET /record/episodes` told us, plus where this page's own
 *  take counter stood the first time it read THIS dataset. */
export type DatasetTally = {
  repoId: string | null;
  /** Episodes the dataset meta reports on disk. */
  onDisk: number;
  baselineOnDisk: number;
  baselineTakes: number;
};

/**
 * Episodes in the dataset — the number the HUD counts from.
 *
 * `onDisk` alone is not enough: lerobot buffers ten episodes' metadata in RAM
 * before it writes `meta/episodes.jsonl`, so a disk read can sit that far
 * behind during a long run, and an episode counter that stalls at 7 while the
 * operator banks their tenth take is worse than none. The floor is what this
 * page has saved since it first read this dataset. Whichever is larger is the
 * true count; the two converge as soon as the buffer flushes.
 *
 * Returns null when nothing has been read — the caller falls back to its own
 * take counter rather than printing a confident zero.
 */
export function episodesTotal(
  tally: DatasetTally | null, takes: number,
): number | null {
  if (!tally) return null;
  const savedSince = Math.max(0, takes - tally.baselineTakes);
  return Math.max(tally.onDisk, tally.baselineOnDisk + savedSince);
}
