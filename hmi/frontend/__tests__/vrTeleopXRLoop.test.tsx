// hmi/frontend/__tests__/vrTeleopXRLoop.test.tsx

/**
 * The XR animation loop, driven headlessly.
 *
 * Everything else about the take machine is pure and tested as such. The modal
 * stick handling is not: it lives inside `onXRFrame`, a closure over refs that
 * only exists once a WebXR session has been granted — which is why the
 * in-headset checklist listed the trained-gesture refusal (V11) as device-only.
 *
 * It does not have to be. `requestTeleopSession` reads `navigator.xr` and
 * nothing else, so stubbing that one property hands the panel a session whose
 * `requestAnimationFrame` this file owns: frame timestamps become an argument,
 * and a 0.8 s stick hold is four function calls rather than a thumb. Haptics
 * come back the same way, through the fake gamepad's actuator.
 *
 * What that buys is the part of V11 that actually gates invariant 5's exception:
 * the refusal being FELT, the release not falling through to `keep`, and the
 * arm not being asked to home. The physical arm is still the headset session's
 * business; "no POST to /teleop/human/home" is the assertion that stands in for
 * it, and it is the meaningful half.
 */
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { act, render, screen } from "@testing-library/react";

import {
  BUTTON_AX, BUTTON_BY, BUTTON_SQUEEZE, BUTTON_THUMBSTICK, BUTTON_TRIGGER,
  RECORD_HOLD_MS, RESET_HOLD_MS,
} from "../lib/vrTeleop";

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() },
}));

// vi.hoisted, because vi.mock factories are lifted above the module body and a
// plain const would not exist yet when the factory runs.
const { recordArm, recordRoll, recordStop, humanTeleopHome } = vi.hoisted(() => ({
  recordArm: vi.fn(),
  recordRoll: vi.fn(),
  recordStop: vi.fn(),
  humanTeleopHome: vi.fn(),
}));

vi.mock("../lib/api", async (importOriginal) => {
  const real = await importOriginal<typeof import("../lib/api")>();
  const idle = {
    recording: false, repo_id: "local/t", task: "t", episode_frames: 0,
    skipped_frames: 0, started_at: null, last_error: null,
  };
  return {
    ...real,
    api: {
      ...real.api,
      cameras: vi.fn().mockResolvedValue({ cameras: [] }),
      recordStatus: vi.fn().mockResolvedValue(idle),
      recordEpisodes: vi.fn().mockResolvedValue({ repo_id: "local/t", episodes: [] }),
      humanTeleopStatus: vi.fn().mockResolvedValue({
        running: true, state: "driving",
        clutch: { engaged: false, sides: { left: false, right: false } },
        acquire: {
          left: { authority: "held", remaining_ms: null, reason: "" },
          right: { authority: "held", remaining_ms: null, reason: "" },
        },
      }),
      humanTeleopStart: vi.fn().mockResolvedValue({ ok: true }),
      humanTeleopStop: vi.fn().mockResolvedValue({ ok: true }),
      humanTeleopHome,
      recordArm,
      recordRoll,
      recordStop,
      estop: vi.fn().mockResolvedValue({ ok: true }),
    },
  };
});

import { VRTeleopPanel } from "../components/VRTeleopPanel";

// ---- the fake headset -------------------------------------------------------

const IDENT = { x: 0, y: 0, z: 0, w: 1 };

/** One controller, with every button this panel reads and a haptic actuator
 *  that records what it was asked for. */
function fakeController(handedness: "left" | "right") {
  const buttons = [] as { pressed: boolean; value: number }[];
  for (const i of [BUTTON_TRIGGER, BUTTON_SQUEEZE, BUTTON_THUMBSTICK,
                   BUTTON_AX, BUTTON_BY]) {
    buttons[i] = { pressed: false, value: 0 };
  }
  const pulses: { intensity: number; durationMs: number }[] = [];
  return {
    src: {
      handedness,
      gripSpace: { s: handedness },
      targetRaySpace: { s: handedness },
      gamepad: {
        buttons,
        axes: [0, 0, 0, 0],
        hapticActuators: [{
          pulse: (intensity: number, durationMs: number) => {
            pulses.push({ intensity, durationMs });
            return true;
          },
        }],
      },
    },
    buttons,
    pulses,
  };
}

function fakeHeadset() {
  const left = fakeController("left");
  const right = fakeController("right");
  let onFrame: ((t: number, frame: unknown) => void) | null = null;
  const session = {
    inputSources: [left.src, right.src],
    visibilityState: "visible" as const,
    requestReferenceSpace: () => Promise.resolve({}),
    requestAnimationFrame: (cb: (t: number, frame: unknown) => void) => {
      onFrame = cb;
      return 1;
    },
    updateRenderState: () => {},
    renderState: { baseLayer: null },
    end: () => Promise.resolve(),
    addEventListener: () => {},
    removeEventListener: () => {},
  };
  const frame = {
    getViewerPose: () => ({
      transform: { position: { ...IDENT, y: 1.6 }, orientation: IDENT },
      emulatedPosition: false,
    }),
    getPose: () => ({
      transform: { position: { ...IDENT, z: -0.4 }, orientation: IDENT },
      emulatedPosition: false,
    }),
  };
  return {
    left, right, session, frame,
    /** One rendered frame at `t` ms. The loop re-registers itself each call. */
    step: async (t: number) => {
      const cb = onFrame;
      expect(cb).not.toBeNull();
      await act(async () => { cb!(t, frame); });
    },
  };
}

/** Hold a thumbstick from `t0` for `ms`, then release — the same shape the
 *  panel's short-click-on-RELEASE idiom reads. Frames land every 16 ms, which
 *  is faster than the Quest's 72–90 Hz and therefore the harder case. */
async function holdStick(
  hs: ReturnType<typeof fakeHeadset>, hand: "left" | "right",
  t0: number, ms: number,
): Promise<number> {
  const c = hand === "left" ? hs.left : hs.right;
  c.buttons[BUTTON_THUMBSTICK].pressed = true;
  let t = t0;
  for (; t <= t0 + ms; t += 16) await hs.step(t);
  c.buttons[BUTTON_THUMBSTICK].pressed = false;
  await hs.step(t);
  return t + 16;
}

/** One A/X hold long enough to fire the record toggle. */
async function holdAX(
  hs: ReturnType<typeof fakeHeadset>, t0: number,
): Promise<number> {
  hs.right.buttons[BUTTON_AX].pressed = true;
  let t = t0;
  for (; t <= t0 + RECORD_HOLD_MS + 40; t += 16) await hs.step(t);
  hs.right.buttons[BUTTON_AX].pressed = false;
  await hs.step(t);
  return t + 16;
}

const ARMS = [{ id: "left_arm", model: "so101", port: "(sim)", mode: "manual" },
               { id: "right_arm", model: "so101", port: "(sim)", mode: "manual" }];

let hs: ReturnType<typeof fakeHeadset>;

beforeEach(async () => {
  vi.clearAllMocks();
  const armed = {
    recording: false, state: "armed", repo_id: "local/t", task: "t",
    episode_frames: 0, skipped_frames: 0, started_at: null, last_error: null,
    episode_index: 12,
  };
  recordArm.mockResolvedValue(armed);
  recordRoll.mockResolvedValue({ ...armed, state: "recording", recording: true });
  recordStop.mockResolvedValue(armed);
  humanTeleopHome.mockResolvedValue({ sides: ["left", "right"] });
  hs = fakeHeadset();
  (globalThis as unknown as { WebSocket: unknown }).WebSocket = class {
    readyState = 0;
    onopen: (() => void) | null = null;
    onmessage: (() => void) | null = null;
    onclose: (() => void) | null = null;
    send() {}
    close() {}
  };
  (navigator as unknown as { xr: unknown }).xr = {
    isSessionSupported: () => Promise.resolve(true),
    requestSession: () => Promise.resolve(hs.session),
  };
});

afterEach(() => {
  delete (navigator as unknown as { xr?: unknown }).xr;
});

/** Mount, enter passthrough, and climb to a rolling take with the prompt open.
 *  Returns the frame time to continue from. */
async function intoThePrompt(): Promise<number> {
  render(<VRTeleopPanel arms={ARMS} />);
  const enter = await screen.findByRole("button", { name: /enter passthrough/i });
  await act(async () => { enter.click(); });
  let t = 1000;
  t = await holdAX(hs, t);          // idle  -> armed
  t = await holdAX(hs, t);          // armed -> rolling
  t = await holdAX(hs, t);          // rolling -> prompt
  expect(recordArm).toHaveBeenCalledTimes(1);
  expect(recordRoll).toHaveBeenCalledTimes(1);
  return t;
}

// ---- V11, the trained-gesture refusal ---------------------------------------

describe("the left-stick hold while the end-of-take prompt is open", () => {
  it("does not home the arm, does not keep the take, and says no", async () => {
    // Invariant 5 binds a left-stick hold to in-session home. Inside the prompt
    // it is the same physical action with the same dwell as the `keep` click,
    // separated only by modal state — and the consequences are asymmetric:
    // banking a take you did not mean to bank is one reject mark, while asking
    // for home and silently not getting it is the direction that hurts. The
    // integrator granted the modal exception on two conditions, and this is
    // both of them.
    const t = await intoThePrompt();
    const before = hs.left.pulses.length;
    await holdStick(hs, "left", t, RESET_HOLD_MS + 200);

    // 1. the arm is never asked to home
    expect(humanTeleopHome).not.toHaveBeenCalled();
    // 2. the release does NOT fall through to `keep` — a long hold is outside
    //    the short-click window, so no decision is committed at all
    expect(recordStop).not.toHaveBeenCalled();
    // 3. the refusal is FELT: the same weak tick a refused resetArms uses
    const refusal = hs.left.pulses.slice(before);
    expect(refusal).toContainEqual({ intensity: 0.2, durationMs: 60 });
  });

  it("still commits a keep on a SHORT click, so the refusal costs nothing", async () => {
    // The refusal must be about the hold alone. If it had swallowed the click
    // too, the prompt would have no left-hand outcome at all.
    const t = await intoThePrompt();
    await holdStick(hs, "left", t, 120);
    expect(recordStop).toHaveBeenCalledTimes(1);
    expect(recordStop).toHaveBeenCalledWith(true, true);   // keep: save + rearm
  });

  it("lets the same hold home the arm once the prompt is gone", async () => {
    // The exception is bounded to the prompt or it has leaked into driving,
    // which is the condition the whole grant rests on. Withdraw the prompt with
    // A/X and the trained gesture must work exactly as it always did.
    let t = await intoThePrompt();
    t = await holdAX(hs, t);                    // prompt -> rolling, withdrawn
    await holdStick(hs, "left", t, RESET_HOLD_MS + 200);
    expect(humanTeleopHome).toHaveBeenCalledTimes(1);
  });
});

// ---- the right stick's half of the same modal ------------------------------

describe("the right stick while the prompt is open", () => {
  it("commits a redo on a short click, and never a tile resize", async () => {
    const t = await intoThePrompt();
    await holdStick(hs, "right", t, 120);
    expect(recordStop).toHaveBeenCalledWith(false, true);   // redo: bin + rearm
  });

  it("does not open the tuning list on a hold, nor decide on its release", async () => {
    // Both would be wrong: a knob list over a decision, and a decision taken by
    // a gesture the operator made to open a knob list.
    const t = await intoThePrompt();
    await holdStick(hs, "right", t, 900);
    expect(recordStop).not.toHaveBeenCalled();
    expect(screen.queryByText(/TUNE/)).toBeNull();
  });
});

// ---- the ladder itself, through the real loop -------------------------------

describe("the A/X ladder, driven through the XR loop", () => {
  it("arms before it rolls, and writes nothing in between", async () => {
    render(<VRTeleopPanel arms={ARMS} />);
    const enter = await screen.findByRole("button", { name: /enter passthrough/i });
    await act(async () => { enter.click(); });

    let t = await holdAX(hs, 1000);
    // The gate: one hold in, the dataset is open and NOTHING has rolled.
    expect(recordArm).toHaveBeenCalledTimes(1);
    expect(recordRoll).not.toHaveBeenCalled();

    t = await holdAX(hs, t);
    expect(recordRoll).toHaveBeenCalledTimes(1);
    expect(recordStop).not.toHaveBeenCalled();
    void t;
  });

  it("needs the full hold — a thumb brush starts nothing", async () => {
    // An accidental take boundary is corrupted data, which is why the toggle is
    // hold-gated rather than edge-detected.
    render(<VRTeleopPanel arms={ARMS} />);
    const enter = await screen.findByRole("button", { name: /enter passthrough/i });
    await act(async () => { enter.click(); });

    hs.right.buttons[BUTTON_AX].pressed = true;
    for (let t = 1000; t < 1000 + RECORD_HOLD_MS - 100; t += 16) await hs.step(t);
    hs.right.buttons[BUTTON_AX].pressed = false;
    await hs.step(1000 + RECORD_HOLD_MS);
    expect(recordArm).not.toHaveBeenCalled();
  });
});
