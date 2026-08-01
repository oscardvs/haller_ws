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
 * `@types/webxr`: we touch about six members of the API, and a dependency whose
 * only job is to describe them is not worth the supply chain.
 */

// ---- minimal WebXR surface -------------------------------------------------

type XRHandedness = "left" | "right" | "none";

type XRRigidTransform = {
  position: { x: number; y: number; z: number; w: number };
  orientation: { x: number; y: number; z: number; w: number };
};

type XRPose = { transform: XRRigidTransform; emulatedPosition: boolean };

export type XRInputSourceLike = {
  handedness: XRHandedness;
  gripSpace?: unknown;
  gamepad?: { buttons: readonly { pressed: boolean; value: number }[] } | null;
};

export type XRFrameLike = {
  getViewerPose(space: unknown): XRPose | null | undefined;
  getPose(space: unknown, base: unknown): XRPose | null | undefined;
};

export type XRSessionLike = {
  inputSources: readonly XRInputSourceLike[];
  requestReferenceSpace(type: string): Promise<unknown>;
  requestAnimationFrame(cb: (t: number, frame: XRFrameLike) => void): number;
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

// ---- wire shape ------------------------------------------------------------

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number]; // (x, y, z, w)

export type ControllerSample = {
  position: Vec3;
  orientation: Quat;
  /** Analog trigger [0,1]. 1 = fully squeezed = gripper closed. */
  trigger: number;
  tracked: boolean;
};

export type VRFrame = {
  type: "vr_keypoints";
  ts_ms: number;
  /** Grip/squeeze button — the dead-man. See ClutchSource "vr_grip". */
  dead_man: boolean;
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
  return xr.isSessionSupported("immersive-vr").catch(() => false);
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

export async function requestVRSession(): Promise<XRSessionLike> {
  const xr = (navigator as unknown as NavigatorXR).xr;
  if (!xr) throw new Error("WebXR unavailable (needs a secure context — HTTPS or localhost)");
  // `local-floor` puts the origin on the physical floor with +Y up, which is
  // what `vr_input.shoulder_xr` assumes when it drops the shoulder line below
  // the headset. `local` would put the origin at the headset's start pose and
  // silently shift every shoulder estimate by the operator's height.
  return xr.requestSession("immersive-vr", { requiredFeatures: ["local-floor"] });
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
 */
export function sampleVRFrame(
  session: XRSessionLike,
  frame: XRFrameLike,
  refSpace: unknown,
  opts: { tsMs: number; body?: BodyOverride },
): VRFrame {
  const head = poseToPair(frame.getViewerPose(refSpace));

  let left: ControllerSample | null = null;
  let right: ControllerSample | null = null;
  let deadMan = false;

  for (const src of session.inputSources) {
    if (src.handedness !== "left" && src.handedness !== "right") continue;
    const buttons = src.gamepad?.buttons ?? [];
    // Either hand's grip engages. One-handed driving of a single arm has to be
    // possible, and requiring both grips would make the other arm's controller
    // a dead-man for an arm it is not driving.
    if (buttons[BUTTON_SQUEEZE]?.pressed) deadMan = true;

    const pose = src.gripSpace ? poseToPair(frame.getPose(src.gripSpace, refSpace)) : null;
    const sample: ControllerSample = pose
      ? { ...pose, trigger: buttons[BUTTON_TRIGGER]?.value ?? 0, tracked: true }
      : {
          position: [0, 0, 0],
          orientation: [0, 0, 0, 1],
          trigger: 0,
          tracked: false,
        };
    if (src.handedness === "left") left = sample;
    else right = sample;
  }

  return {
    type: "vr_keypoints",
    ts_ms: opts.tsMs,
    dead_man: deadMan,
    head,
    left,
    right,
    ...(opts.body ? { body: opts.body } : {}),
  };
}
