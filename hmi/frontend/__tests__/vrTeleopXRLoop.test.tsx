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
import { toast } from "sonner";
import { act, render, screen, waitFor } from "@testing-library/react";

import {
  BUTTON_AX, BUTTON_BY, BUTTON_SQUEEZE, BUTTON_THUMBSTICK, BUTTON_TRIGGER,
  RECORD_HOLD_MS, RESET_HOLD_MS,
} from "../lib/vrTeleop";

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() },
}));

// vi.hoisted, because vi.mock factories are lifted above the module body and a
// plain const would not exist yet when the factory runs.
// The recorder's state lives here, and the mocks READ it rather than each
// returning a fixed literal. The panel polls /record/status every 250 ms and
// reconciles the take machine against whatever it says — so a status mock that
// answered "idle" while the test had driven the page to ARMED would race the
// assertions and knock the state back. It did: roughly two runs in five failed,
// on a different test each time. A fake backend has to be as consistent as a
// real one or the test is measuring the scheduler.
const gate = vi.hoisted(() => ({
  status: {
    recording: false, state: "idle", repo_id: "local/t", task: "t",
    episode_frames: 0, skipped_frames: 0, started_at: null,
    last_error: null, episode_index: 12,
  } as Record<string, unknown>,
}));

const { recordArm, recordRoll, recordStop, humanTeleopHome, estop,
        humanTeleopStop, recordStart } = vi.hoisted(() => ({
  recordArm: vi.fn(),
  recordRoll: vi.fn(),
  recordStop: vi.fn(),
  humanTeleopHome: vi.fn(),
  estop: vi.fn(),
  humanTeleopStop: vi.fn(),
  recordStart: vi.fn(),
}));

vi.mock("../lib/api", async (importOriginal) => {
  const real = await importOriginal<typeof import("../lib/api")>();
  return {
    ...real,
    api: {
      ...real.api,
      cameras: vi.fn().mockResolvedValue({ cameras: [] }),
      recordStatus: vi.fn(() => Promise.resolve({ ...gate.status })),
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
      humanTeleopStop,
      humanTeleopHome,
      recordArm,
      recordStart,
      recordRoll,
      recordStop,
      estop,
    },
  };
});

import { api, ApiError } from "../lib/api";
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
    /** One rendered frame at `t` ms. The loop re-registers itself each call.
     *  A null callback here means the session was never entered — see
     *  `clickEnterPassthrough`. */
    step: async (t: number) => {
      const cb = onFrame;
      expect(cb).not.toBeNull();
      await act(async () => { cb!(t, frame); });
    },
  };
}

/** Click `Enter Passthrough` once it is actually clickable.
 *
 *  The button renders DISABLED until `xrSupported()` resolves
 *  (`VRTeleopPanel.tsx:1428`; `supported` starts `null` at `:168` and is set
 *  from a promise at `:338`). `findByRole` matches a disabled button perfectly
 *  well, and **clicking a disabled button is a no-op** — so a click landing in
 *  that window silently fails to enter the session. The loop never registers a
 *  frame callback, and the first `step` after entry reads null.
 *
 *  That is the whole residual flake: measured 4 of 15 full-suite runs red
 *  before this wait, every failure the first step after an entry. It is a
 *  property of the harness, not of the panel — an operator cannot click a
 *  button that is not there yet. */
async function clickEnterPassthrough(): Promise<void> {
  const enter = await screen.findByRole("button", { name: /enter passthrough/i });
  await waitFor(() => expect((enter as HTMLButtonElement).disabled).toBe(false));
  await act(async () => { enter.click(); });
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
  gate.status = {
    recording: false, state: "idle", repo_id: "local/t", task: "t",
    episode_frames: 0, skipped_frames: 0, started_at: null,
    last_error: null, episode_index: 12,
  };
  /** Move the fake recorder, and answer with where it now is. */
  const settle = (state: string) => {
    gate.status = { ...gate.status, state, recording: state === "recording" };
    return { ...gate.status };
  };
  recordArm.mockImplementation(async () => settle("armed"));
  recordRoll.mockImplementation(async () => settle("recording"));
  recordStop.mockImplementation(async (_save: boolean, rearm?: boolean) =>
    settle(rearm ? "armed" : "idle"));
  humanTeleopHome.mockResolvedValue({ sides: ["left", "right"] });
  estop.mockResolvedValue({ ok: true });
  humanTeleopStop.mockResolvedValue({ ok: true });
  // The local-gate ROLL path: /record/start, not /record/roll.
  recordStart.mockImplementation(async () => settle("recording"));
  // `clearAllMocks` clears CALLS, not implementations, and `recordStatus`'s
  // lives in the vi.mock factory rather than here — so a test that overrides it
  // would leak that override into every test after it. Re-established per test.
  vi.mocked(api.recordStatus).mockImplementation(
    async () => ({ ...gate.status }) as Awaited<ReturnType<typeof api.recordStatus>>);
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
  await clickEnterPassthrough();
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

// ---- the re-arm that was asked for and refused -------------------------------

describe("a re-arm the recorder refused", () => {
  /** The contract that makes this reachable: a `{save, rearm}` whose save
   *  lands and whose re-arm does not is a 200 carrying `state:"idle"` and a
   *  reason — NOT a 409, because a 409 would report a banked take as a
   *  failure. So a 200 is no longer evidence that the ask was honoured. */
  const refuseRearm = () =>
    recordStop.mockImplementation(async () => ({
      ok: true, state: "idle", recording: false, episode_frames: 412,
      episode_index: null, repo_id: "u/haller_pick", task: "pick",
      invalidated_reason: "re-arm refused: bus timeout on right_arm",
    }));

  it("says NOT re-armed, rather than announcing a gate that is down", async () => {
    refuseRearm();
    const t = await intoThePrompt();
    await holdStick(hs, "left", t, 120);              // keep: {save, rearm}

    const said = (toast.error as ReturnType<typeof vi.fn>).mock.calls
      .map((c) => String(c[0]));
    expect(said.some((m) => m.includes("NOT re-armed"))).toBe(true);
    expect(said.some((m) => m.includes("bus timeout on right_arm"))).toBe(true);
    // The claim that would have been made from the ask instead of the outcome.
    const claimed = (toast.success as ReturnType<typeof vi.fn>).mock.calls
      .map((c) => String(c[0]));
    expect(claimed.some((m) => m.includes("armed for the next"))).toBe(false);
  });

  it("still reports the take as SAVED, because it was", async () => {
    // A refused re-arm must not read as a lost take. The frames are on disk;
    // only the next gate is not open. Leading with the failure would send an
    // operator hunting for a take that is sitting safely in the dataset.
    refuseRearm();
    const t = await intoThePrompt();
    await holdStick(hs, "left", t, 120);

    const said = (toast.error as ReturnType<typeof vi.fn>).mock.calls
      .map((c) => String(c[0]));
    expect(said.some((m) => m.includes("saved") && m.includes("412"))).toBe(true);
  });

  it("still says armed for the next when the re-arm actually landed", async () => {
    // The other side of the same read: the default harness re-arms, and the
    // happy path must keep the words it has always had. Without this, dropping
    // the announcement entirely would pass the two tests above.
    const t = await intoThePrompt();
    await holdStick(hs, "left", t, 120);

    const claimed = (toast.success as ReturnType<typeof vi.fn>).mock.calls
      .map((c) => String(c[0]));
    expect(claimed.some((m) => m.includes("armed for the next"))).toBe(true);
  });
});

describe("the A/X ladder, driven through the XR loop", () => {
  it("arms before it rolls, and writes nothing in between", async () => {
    render(<VRTeleopPanel arms={ARMS} />);
    await clickEnterPassthrough();

    const t = await holdAX(hs, 1000);
    // The gate: one hold in, the dataset is open and NOTHING has rolled.
    expect(recordArm).toHaveBeenCalledTimes(1);
    expect(recordRoll).not.toHaveBeenCalled();

    await holdAX(hs, t);
    expect(recordRoll).toHaveBeenCalledTimes(1);
    expect(recordStop).not.toHaveBeenCalled();
  });

  it("needs the full hold — a thumb brush starts nothing", async () => {
    // An accidental take boundary is corrupted data, which is why the toggle is
    // hold-gated rather than edge-detected.
    render(<VRTeleopPanel arms={ARMS} />);
    await clickEnterPassthrough();

    hs.right.buttons[BUTTON_AX].pressed = true;
    for (let t = 1000; t < 1000 + RECORD_HOLD_MS - 100; t += 16) await hs.step(t);
    hs.right.buttons[BUTTON_AX].pressed = false;
    await hs.step(1000 + RECORD_HOLD_MS);
    expect(recordArm).not.toHaveBeenCalled();
  });
});

// ---- the E-STOP, the one input a modal may never own -----------------------

/** Press B/Y on one controller for `frames` frames from `t0`, then release. */
async function pressEstop(
  hand: "left" | "right", t0: number, frames = 1,
): Promise<number> {
  const c = hand === "left" ? hs.left : hs.right;
  c.buttons[BUTTON_BY].pressed = true;
  let t = t0;
  for (let i = 0; i < frames; i++, t += 16) await hs.step(t);
  c.buttons[BUTTON_BY].pressed = false;
  await hs.step(t);
  return t + 16;
}

async function enterSession(): Promise<void> {
  render(<VRTeleopPanel arms={ARMS} />);
  await clickEnterPassthrough();
}

describe("the E-STOP", () => {
  it("fires once per press, however long the button is held", async () => {
    // One press, one POST. A per-frame scan would post at display rate — 72-90
    // requests a second at the exact moment the rig is in trouble and the
    // backend is walking every motor in-process.
    //
    // The request is made to HANG deliberately. `fireEstop` tears the session
    // down once it resolves, which stops the loop — so against a fast /estop
    // this assertion holds no matter what the loop does, and a test that cannot
    // fail is not evidence. With the request in flight the loop keeps running
    // under a held button, which is the state that can actually post twice.
    //
    // It pins the PROPERTY, not one mechanism: the rising-edge check and the
    // in-flight guard each prevent the second post on their own. Removing both
    // fails this test; removing either alone does not, and that is the honest
    // description of what it covers.
    estop.mockReturnValueOnce(new Promise(() => {}));
    await enterSession();
    await pressEstop("right", 1000, 30);
    expect(estop).toHaveBeenCalledTimes(1);
  });

  it("fires from either controller", async () => {
    // B on the right, Y on the left. Whichever hand is free is the one that
    // reaches it, and which hand that is depends on what went wrong.
    await enterSession();
    await pressEstop("left", 1000, 3);
    expect(estop).toHaveBeenCalledTimes(1);
  });

  it("fires while the end-of-take prompt owns both sticks", async () => {
    // THE ONE THAT MATTERS. The prompt is modal and takes both thumbsticks for
    // its duration — and the E-STOP is the single input it must never take. A
    // modal that can swallow the stop is the same class of defect as a stop
    // that cannot be read, and this page has now had one of those.
    const t = await intoThePrompt();
    expect(estop).not.toHaveBeenCalled();
    await pressEstop("right", t, 3);
    expect(estop).toHaveBeenCalledTimes(1);
    // And it did not quietly commit the take on its way out.
    expect(recordStop).not.toHaveBeenCalled();
  });

  it("fires while the tuning list is open", async () => {
    // The other modal. Same rule, less consequence, but a stop that depends on
    // which menu happens to be up is not a stop.
    await enterSession();
    const t = await holdStick(hs, "right", 1000, 700);  // hold = open the list
    await pressEstop("right", t, 3);
    expect(estop).toHaveBeenCalledTimes(1);
  });

  it("is scanned at display rate, not at the 33 ms publish rate", async () => {
    // A press that begins and ends BETWEEN two publish ticks must still fire.
    // 33 ms of extra latency on a stop button is 33 ms too many, so the scan
    // sits above the publish throttle in the loop rather than below it.
    //
    // The first frame is what makes this falsifiable: it publishes and sets the
    // throttle clock. Pressing 10 ms later lands inside the throttle window, so
    // a scan that had drifted below it would see nothing at all.
    await enterSession();
    await hs.step(1000);                  // publishes; throttle clock now 1000
    await pressEstop("right", 1010, 1);   // 10 ms later — inside the window
    expect(estop).toHaveBeenCalledTimes(1);
  });

  it("buzzes both hands hard, because a stop must be felt to have happened", async () => {
    await enterSession();
    const l = hs.left.pulses.length;
    const r = hs.right.pulses.length;
    await pressEstop("right", 1000, 2);
    expect(hs.left.pulses.slice(l)).toContainEqual({ intensity: 1.0, durationMs: 300 });
    expect(hs.right.pulses.slice(r)).toContainEqual({ intensity: 1.0, durationMs: 300 });
  });

  it("leaves the headset but does NOT stop the backend session", async () => {
    // After a stop the operator deals with the rig, and re-arming lives on the
    // 2D panel — so the session is left for them to find rather than torn down
    // underneath them.
    await enterSession();
    await pressEstop("right", 1000, 2);
    expect(estop).toHaveBeenCalledTimes(1);
    expect(humanTeleopStop).not.toHaveBeenCalled();
  });

  it("keeps firing on a fresh press after a failed request", async () => {
    // The in-flight guard must not latch. A refused or dropped /estop that left
    // the guard set would make the SECOND press — the one the operator makes
    // because they saw nothing happen — do nothing at all.
    //
    // The panel is deliberately NOT remounted between the two presses: a fresh
    // mount gets a fresh ref, so it would pass with the guard latched and prove
    // nothing. Re-entering the session on the same instance is what keeps the
    // ref under test.
    estop.mockRejectedValueOnce(new Error("network"));
    render(<VRTeleopPanel arms={ARMS} />);
    await clickEnterPassthrough();
    await pressEstop("right", 1000, 2);
    expect(estop).toHaveBeenCalledTimes(1);

    // fireEstop tears the session down, so the second press needs a second
    // session — on the same component, and therefore the same guard.
    hs = fakeHeadset();
    (navigator as unknown as { xr: unknown }).xr = {
      isSessionSupported: () => Promise.resolve(true),
      requestSession: () => Promise.resolve(hs.session),
    };
    // Re-enter on the SAME instance — `clickEnterPassthrough` does not render,
    // so the component (and therefore the in-flight guard) is the one under test.
    await clickEnterPassthrough();
    await pressEstop("right", 2000, 2);
    expect(estop).toHaveBeenCalledTimes(2);
  });
});

// ---- V13: the fallback against a backend that has no start gate ------------

describe("the local gate, on a backend without /record/arm", () => {
  /** What an unmounted FastAPI route actually answers. */
  const notMounted = () =>
    recordArm.mockRejectedValue(new ApiError(404, "Not Found"));

  it("holds ARMED itself, and still writes nothing before ROLL", async () => {
    // The operator-facing half of the gate is the only half the fallback can
    // give: nothing is written while you get ready. It is also the half that
    // cost Oscar two 60 s episodes of himself putting a headset on, so it is
    // the half worth having before the routes land.
    notMounted();
    await enterSession();
    const t = await holdAX(hs, 1000);            // idle -> armed, locally

    expect(recordArm).toHaveBeenCalledTimes(1);
    expect(recordRoll).not.toHaveBeenCalled();
    expect(recordStart).not.toHaveBeenCalled(); // NOTHING has been opened yet

    await holdAX(hs, t);                         // armed -> rolling
    // ROLL goes through the plain /record/start the page was holding back.
    expect(recordStart).toHaveBeenCalledTimes(1);
    expect(recordRoll).not.toHaveBeenCalled();
  });

  it("says so once per session, not once per take", async () => {
    // A caveat repeated every take is a caveat the operator learns to dismiss.
    notMounted();
    await enterSession();
    let t = await holdAX(hs, 1000);
    t = await holdAX(hs, t);                     // roll
    t = await holdAX(hs, t);                     // prompt
    await holdStick(hs, "left", t, 120);         // keep -> re-arms, probes again

    expect(recordArm).toHaveBeenCalledTimes(2);  // it re-probes every arm...
    const noted = (toast.info as ReturnType<typeof vi.fn>).mock.calls
      .filter((c) => String(c[0]).includes("no start gate"));
    expect(noted).toHaveLength(1);               // ...and says it once
  });

  it("shows the caveat on the desktop card, not only in a toast", async () => {
    // The toast is gone in seconds. The HUD carries "(local gate)" for as long
    // as it is true and the desktop card now does too — a caveat the operator
    // can only have seen once is a caveat they do not have.
    notMounted();
    await enterSession();
    await holdAX(hs, 1000);
    expect(await screen.findByText(/local gate/)).toBeInTheDocument();
  });

  it("upgrades silently the first time /record/arm answers", async () => {
    // No toast, no caveat, no code path the operator has to know about. The
    // fallback existing at all is a fact about the backend, not about them.
    await enterSession();                        // recordArm resolves by default
    const t = await holdAX(hs, 1000);
    const noted = (toast.info as ReturnType<typeof vi.fn>).mock.calls
      .filter((c) => String(c[0]).includes("no start gate"));
    expect(noted).toHaveLength(0);
    expect(screen.queryByText(/local gate/)).toBeNull();

    await holdAX(hs, t);
    expect(recordRoll).toHaveBeenCalledTimes(1);
    expect(recordStart).not.toHaveBeenCalled();  // the server gate owns ROLL
  });

  it("treats a 409 as a refusal, never as a missing route", async () => {
    // The distinction the fallback turns on. 404 is "this backend has no gate";
    // 409 is the gate WORKING — a colliding camera key, a measured rate under
    // the floor — and swallowing one as the other would arm locally against a
    // recorder that had just refused, which is the worst of both.
    recordArm.mockRejectedValue(new ApiError(409, "camera key collision: top"));
    await enterSession();
    await holdAX(hs, 1000);
    expect(screen.queryByText(/local gate/)).toBeNull();
    const refused = (toast.error as ReturnType<typeof vi.fn>).mock.calls
      .filter((c) => String(c[0]).includes("camera key collision"));
    expect(refused).toHaveLength(1);
    // And it did not silently roll on anyway.
    expect(recordRoll).not.toHaveBeenCalled();
    expect(recordStart).not.toHaveBeenCalled();
  });
});

// ---- the reconcile race: a status read that predates the transition --------

describe("a /record/status read that resolves across a transition", () => {
  it("does not walk the take machine back to where the read was taken", async () => {
    // Written from the CLAIM rather than from the guard it is aimed at, because
    // the guard is what is in question: `armAct` settles the machine to ARMED
    // from /record/arm's own answer, and the 250 ms poll then reconciles
    // against whatever /record/status says. A read ISSUED BEFORE the hold still
    // describes IDLE. If it lands AFTER the hold it is stale by exactly one
    // transition, and reconciling to it walks the machine backwards.
    //
    // Asserted in the operator's terms, not the machine's: from ARMED the next
    // A/X hold must ROLL. A machine that walked back to IDLE arms a second time
    // instead, and no frames ever land — the operator holds A/X twice and is
    // still not recording.
    await enterSession();

    // Freeze the poll. Every read from here is captured and left pending, so
    // the one released below is provably a read taken BEFORE the arm.
    const inFlight: Array<(v: unknown) => void> = [];
    const asOfNow = { ...gate.status };            // idle, right now
    vi.mocked(api.recordStatus).mockImplementation(
      () => new Promise((res) => { inFlight.push(res as (v: unknown) => void); }));
    await waitFor(() => expect(inFlight.length).toBeGreaterThan(0));

    const t = await holdAX(hs, 1000);              // idle -> armed
    expect(recordArm).toHaveBeenCalledTimes(1);

    // ...and only now does the pre-arm read come back, still saying idle.
    await act(async () => { inFlight[0](asOfNow); });

    await holdAX(hs, t);                           // armed -> rolling
    expect(recordRoll).toHaveBeenCalledTimes(1);
    expect(recordArm).toHaveBeenCalledTimes(1);    // and NOT armed a second time
  });

  it("CONTROL: rolls when that same read resolves BEFORE the transition", async () => {
    // The control for the test above, and the reason its red means anything.
    // Identical harness — frozen poll, one released read — except the read is
    // released BEFORE the arm, so it is not stale. If this did not roll either,
    // the failure above would be a fact about the frozen poll and not about the
    // race, and no amount of staring at the guard would have said so.
    await enterSession();

    const inFlight: Array<(v: unknown) => void> = [];
    const asOfNow = { ...gate.status };
    vi.mocked(api.recordStatus).mockImplementation(
      () => new Promise((res) => { inFlight.push(res as (v: unknown) => void); }));
    await waitFor(() => expect(inFlight.length).toBeGreaterThan(0));

    await act(async () => { inFlight[0](asOfNow); });   // BEFORE the arm
    const t = await holdAX(hs, 1000);                  // idle -> armed
    expect(recordArm).toHaveBeenCalledTimes(1);

    await holdAX(hs, t);                               // armed -> rolling
    expect(recordRoll).toHaveBeenCalledTimes(1);
  });
});
