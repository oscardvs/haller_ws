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

/** One press of B — the kit's episode button — held for `frames` frames.
 *  Edge-detected rather than hold-gated: B takes a deliberate reach, so it
 *  does not need a dwell to protect it the way a thumb-rest button did. */
async function pressB(
  hs: ReturnType<typeof fakeHeadset>, t0: number, frames = 2,
): Promise<number> {
  hs.right.buttons[BUTTON_BY].pressed = true;
  let t = t0;
  for (let i = 0; i < frames; i++, t += 16) await hs.step(t);
  hs.right.buttons[BUTTON_BY].pressed = false;
  await hs.step(t);
  return t + 16;
}

const ARMS = [{ id: "left_arm", model: "so101", port: "(sim)", mode: "manual" },
               { id: "right_arm", model: "so101", port: "(sim)", mode: "manual" }];

/** A full, honest `/teleop/human` answer — the fields the panel reads plus
 *  the ones the type requires, consistent for either running state. */
function teleopStatus(running: boolean):
    Awaited<ReturnType<typeof api.humanTeleopStatus>> {
  const side = {
    authority: "held", reason: "clutch_open",
    remaining_ms: null, ramp: null,
  } as const;
  return {
    running,
    state: running ? "driving" : "idle",
    left_arm: running ? "left_arm" : null,
    right_arm: running ? "right_arm" : null,
    started_at: running ? 0 : null,
    last_error: null,
    tracking: {
      left: { age_ms: null, lost: false },
      right: { age_ms: null, lost: false },
    },
    acquire: { acquire_ms: 0, match_dwell_ms: 0, left: side, right: side },
    clutch: {
      engaged: false,
      sides: { left: false, right: false },
      reason: "vr_grip_mode",
    },
  };
}

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
  // Same leak rule for the status read `enterVR`'s adopt-or-start branch
  // makes: a test that overrides it must not decide the next test's entry
  // path. The default backend has a session RUNNING, so every ordinary test
  // enters by ADOPTING it — no start call on entry.
  vi.mocked(api.humanTeleopStatus).mockImplementation(
    async () => teleopStatus(true));
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

/** Mount, enter passthrough, and climb to a rolling take. Returns the frame
 *  time to continue from; one more press of B banks it. */
async function intoARollingTake(): Promise<number> {
  render(<VRTeleopPanel arms={ARMS} />);
  await clickEnterPassthrough();
  let t = 1000;
  t = await pressB(hs, t);          // idle  -> armed
  t = await pressB(hs, t);          // armed -> rolling
  expect(recordArm).toHaveBeenCalledTimes(1);
  expect(recordRoll).toHaveBeenCalledTimes(1);
  return t;
}

// ---- entry: adopt a running session, start one otherwise --------------------

describe("entering against the backend session", () => {
  /** The cockpit flow: position the arm, pick the rate, click start on the
   *  desktop, THEN put the headset on. Refusing here (the old behaviour) tore
   *  the XR session down with "already running; stop it first". */
  it("joins a running session instead of starting a second one", async () => {
    render(<VRTeleopPanel arms={ARMS} />);
    await clickEnterPassthrough();
    await hs.step(1000);
    expect(api.humanTeleopStart).not.toHaveBeenCalled();
  });

  it("starts its own session when none is running", async () => {
    // A consistent fake backend: stopped until start() lands, running after.
    let started = false;
    vi.mocked(api.humanTeleopStatus).mockImplementation(
      async () => teleopStatus(started));
    vi.mocked(api.humanTeleopStart).mockImplementation(async () => {
      started = true;
      return { ok: true as const, ...teleopStatus(true) };
    });
    render(<VRTeleopPanel arms={ARMS} />);
    await clickEnterPassthrough();
    await hs.step(1000);
    expect(api.humanTeleopStart).toHaveBeenCalledTimes(1);
    const body = vi.mocked(api.humanTeleopStart).mock.calls[0][0];
    expect(body.left_arm || body.right_arm).toBeTruthy();
  });
});

// ---- V11, the trained-gesture refusal ---------------------------------------

// ---- the right stick's half of the same modal ------------------------------

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
    const t = await intoARollingTake();
    await pressB(hs, t);                              // end: {save, rearm}

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
    const t = await intoARollingTake();
    await pressB(hs, t);

    const said = (toast.error as ReturnType<typeof vi.fn>).mock.calls
      .map((c) => String(c[0]));
    expect(said.some((m) => m.includes("saved") && m.includes("412"))).toBe(true);
  });

  it("still says armed for the next when the re-arm actually landed", async () => {
    // The other side of the same read: the default harness re-arms, and the
    // happy path must keep the words it has always had. Without this, dropping
    // the announcement entirely would pass the two tests above.
    const t = await intoARollingTake();
    await pressB(hs, t);

    const claimed = (toast.success as ReturnType<typeof vi.fn>).mock.calls
      .map((c) => String(c[0]));
    expect(claimed.some((m) => m.includes("armed for the next"))).toBe(true);
  });
});

describe("the B ladder, driven through the XR loop", () => {
  it("arms before it rolls, and writes nothing in between", async () => {
    render(<VRTeleopPanel arms={ARMS} />);
    await clickEnterPassthrough();

    const t = await pressB(hs, 1000);
    // The gate: one hold in, the dataset is open and NOTHING has rolled.
    expect(recordArm).toHaveBeenCalledTimes(1);
    expect(recordRoll).not.toHaveBeenCalled();

    await pressB(hs, t);
    expect(recordRoll).toHaveBeenCalledTimes(1);
    expect(recordStop).not.toHaveBeenCalled();
  });

  it("steps once per press, however long B is held", async () => {
    // Edge-detected, not level-scanned: a B held through a whole demo must
    // not walk the ladder at display rate. The dwell that used to protect
    // A/X is gone with the binding that needed it.
    render(<VRTeleopPanel arms={ARMS} />);
    await clickEnterPassthrough();

    hs.right.buttons[BUTTON_BY].pressed = true;
    for (let t = 1000; t < 2000; t += 16) await hs.step(t);
    hs.right.buttons[BUTTON_BY].pressed = false;
    await hs.step(2000);
    expect(recordArm).toHaveBeenCalledTimes(1);
    expect(recordRoll).not.toHaveBeenCalled();
  });

  it("keeps A/X off the recorder entirely — it is the precision modifier", async () => {
    // The binding swap, pinned. A/X held is a gain change and nothing else;
    // if it still reached the take machine, a fine-work grab would open a
    // dataset.
    render(<VRTeleopPanel arms={ARMS} />);
    await clickEnterPassthrough();

    hs.right.buttons[BUTTON_AX].pressed = true;
    for (let t = 1000; t < 2000; t += 16) await hs.step(t);
    hs.right.buttons[BUTTON_AX].pressed = false;
    await hs.step(2000);
    expect(recordArm).not.toHaveBeenCalled();
    expect(recordRoll).not.toHaveBeenCalled();
  });
});

// ---- the E-STOP that is no longer on the sticks ----------------------------

describe("the controller E-STOP, removed", () => {
  it("never fires /estop, from either hand", async () => {
    // B and Y are episode control now, which is the kit's mapping and the
    // whole point of the swap. The stop lives on the desktop HMI card and the
    // bench cutoff. This is the guard that the two never share a button
    // again: a B that both banked a take and dropped torque would end every
    // episode by killing the arm.
    render(<VRTeleopPanel arms={ARMS} />);
    await clickEnterPassthrough();
    let t = 1000;
    for (const c of [hs.left, hs.right]) {
      c.buttons[BUTTON_BY].pressed = true;
      for (let i = 0; i < 4; i++, t += 16) await hs.step(t);
      c.buttons[BUTTON_BY].pressed = false;
      await hs.step(t); t += 16;
    }
    expect(estop).not.toHaveBeenCalled();
  });
});

async function enterSession(): Promise<void> {
  render(<VRTeleopPanel arms={ARMS} />);
  await clickEnterPassthrough();
}

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
    const t = await pressB(hs, 1000);            // idle -> armed, locally

    expect(recordArm).toHaveBeenCalledTimes(1);
    expect(recordRoll).not.toHaveBeenCalled();
    expect(recordStart).not.toHaveBeenCalled(); // NOTHING has been opened yet

    await pressB(hs, t);                         // armed -> rolling
    // ROLL goes through the plain /record/start the page was holding back.
    expect(recordStart).toHaveBeenCalledTimes(1);
    expect(recordRoll).not.toHaveBeenCalled();
  });

  it("says so once per session, not once per take", async () => {
    // A caveat repeated every take is a caveat the operator learns to dismiss.
    notMounted();
    await enterSession();
    let t = await pressB(hs, 1000);
    t = await pressB(hs, t);                     // roll
    await pressB(hs, t);                         // end -> re-arms, probes again

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
    await pressB(hs, 1000);
    expect(await screen.findByText(/local gate/)).toBeInTheDocument();
  });

  it("upgrades silently the first time /record/arm answers", async () => {
    // No toast, no caveat, no code path the operator has to know about. The
    // fallback existing at all is a fact about the backend, not about them.
    await enterSession();                        // recordArm resolves by default
    const t = await pressB(hs, 1000);
    const noted = (toast.info as ReturnType<typeof vi.fn>).mock.calls
      .filter((c) => String(c[0]).includes("no start gate"));
    expect(noted).toHaveLength(0);
    expect(screen.queryByText(/local gate/)).toBeNull();

    await pressB(hs, t);
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
    await pressB(hs, 1000);
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

    const t = await pressB(hs, 1000);              // idle -> armed
    expect(recordArm).toHaveBeenCalledTimes(1);

    // ...and only now does the pre-arm read come back, still saying idle.
    await act(async () => { inFlight[0](asOfNow); });

    await pressB(hs, t);                           // armed -> rolling
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
    const t = await pressB(hs, 1000);                  // idle -> armed
    expect(recordArm).toHaveBeenCalledTimes(1);

    await pressB(hs, t);                               // armed -> rolling
    expect(recordRoll).toHaveBeenCalledTimes(1);
  });
});
