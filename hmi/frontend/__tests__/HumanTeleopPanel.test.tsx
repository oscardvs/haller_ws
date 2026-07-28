/**
 * Panel-level tests that drive the REAL requestAnimationFrame loop.
 *
 * The face model is decimated: it runs on one tracking tick in FACE_EVERY_N,
 * and `jaw_open` only carries a value on the ticks that actually ran it. That
 * split is the whole reason the backend's staleness budget sits above the
 * decimation gap, so it is worth testing against the shipping loop rather than
 * against a restatement of the tick arithmetic.
 */
import { render, screen, act, waitFor, cleanup, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

const hoisted = vi.hoisted(() => ({
  detectCalls: [] as { t: number; face: boolean }[],
  queued: [] as { jaw_open: number | null; clutch_source: string }[],
  runners: [] as unknown[],
  started: [] as unknown[],
  calibrated: [] as unknown[],
  /** jawOpen the fake FaceLandmarker reports whenever it is actually run. */
  jaw: 0.62,
}));

vi.mock("@/lib/mediapipe", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/mediapipe")>();
  const EMPTY_HANDS = { worldLandmarks: [], landmarks: [], handednesses: [] };
  const EMPTY_POSE = { worldLandmarks: [], landmarks: [] };
  const FACE = {
    faceBlendshapes: [{ categories: [{ categoryName: "jawOpen", score: hoisted.jaw }] }],
  };
  class FakeRunner {
    constructor() { hoisted.runners.push(this); }
    async load() { /* no WASM in jsdom */ }
    detect(_video: unknown, t: number, opts?: { face?: boolean }) {
      hoisted.detectCalls.push({ t, face: !!opts?.face });
      return { hands: EMPTY_HANDS, pose: EMPTY_POSE, face: opts?.face ? FACE : null };
    }
    close() { /* noop */ }
  }
  return { ...actual, MediaPipeRunner: FakeRunner };
});

vi.mock("@/lib/humanTeleopClient", () => {
  class FakeClient {
    connect() { /* noop */ }
    close() { /* noop */ }
    queueFrame(frame: { jaw_open: number | null; clutch_source: string }) {
      hoisted.queued.push(frame);
    }
    tick() { /* noop */ }
  }
  return { HumanTeleopClient: FakeClient };
});

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      humanTeleopStart: async (body: unknown) => { hoisted.started.push(body); return {}; },
      humanTeleopCalibrate: async (body: unknown) => { hoisted.calibrated.push(body); return {}; },
      humanTeleopSwap: async () => ({}),
      humanTeleopStop: async () => ({}),
    },
  };
});

vi.mock("sonner", () => ({
  toast: { error: vi.fn(), success: vi.fn() },
}));

import { HumanTeleopPanel } from "@/components/HumanTeleopPanel";
import { useTelemetry } from "@/lib/telemetry";
import type { HumanTeleopStatus } from "@/lib/api";

let rafCb: FrameRequestCallback | null = null;

beforeEach(() => {
  hoisted.detectCalls.length = 0;
  hoisted.queued.length = 0;
  hoisted.runners.length = 0;
  hoisted.started.length = 0;
  hoisted.calibrated.length = 0;
  rafCb = null;
  localStorage.clear();
  useTelemetry.setState({ lastFrame: null });

  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => { rafCb = cb; return 1; });
  vi.stubGlobal("cancelAnimationFrame", () => { /* noop */ });

  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia: async () => ({ getTracks: () => [] }) },
  });
  // jsdom implements neither of these; the loop needs readyState >= 2 to run.
  vi.spyOn(HTMLMediaElement.prototype, "readyState", "get").mockReturnValue(4);
  vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
  // Returning null makes CameraOverlay.draw bail after sizing the canvas,
  // which keeps jsdom's "not implemented" noise out of the test output.
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

/** Render the panel and wait for the camera/model init effect to settle. */
async function bootPanel() {
  render(<HumanTeleopPanel armIds={["arm_left", "arm_right"]} />);
  await waitFor(() => expect(hoisted.runners.length).toBe(1));
}

/** Fire one animation frame at `t` ms. */
function frameAt(t: number) {
  act(() => { rafCb?.(t); });
}

function telemetry(human: Partial<HumanTeleopStatus>) {
  act(() => {
    useTelemetry.setState({
      lastFrame: {
        t: 0,
        base: { linear: 0, angular: 0, odom: { x: 0, y: 0, yaw: 0 }, scan_min_range: null },
        arms: {},
        alerts: [],
        human_teleop: human as HumanTeleopStatus,
      },
    });
  });
}

describe("HumanTeleopPanel face decimation", () => {
  it("runs the face model on one tracking tick in three and nulls jaw_open on the rest", async () => {
    await bootPanel();
    // 100 ms apart so every callback clears the loop's ~30 Hz (33 ms) cap.
    for (let i = 0; i < 10; i++) frameAt(100 + i * 100);

    expect(hoisted.detectCalls.map((c) => c.face)).toEqual([
      true, false, false,
      true, false, false,
      true, false, false,
      true,
    ]);
    expect(hoisted.queued.map((f) => f.jaw_open)).toEqual([
      hoisted.jaw, null, null,
      hoisted.jaw, null, null,
      hoisted.jaw, null, null,
      hoisted.jaw,
    ]);
  });

  it("counts processed ticks, not raw animation frames", async () => {
    await bootPanel();
    frameAt(100);                       // processed — face tick 0
    frameAt(110);                       // dropped by the 30 Hz cap
    frameAt(120);                       // dropped by the 30 Hz cap
    frameAt(200);                       // processed — face tick 1

    // Four callbacks, two processed ticks. If the counter advanced per raw
    // callback the second processed tick would have landed on a face tick.
    expect(hoisted.detectCalls.map((c) => c.face)).toEqual([true, false]);
    expect(hoisted.queued.map((f) => f.jaw_open)).toEqual([hoisted.jaw, null]);
  });
});

describe("HumanTeleopPanel clutch source", () => {
  it("cannot change authority while a session is running", async () => {
    await bootPanel();
    const toggle = screen.getByRole("button", { name: /clutch:/i });
    expect(toggle).toBeEnabled();

    telemetry({ running: true, state: "tracking" });
    expect(screen.getByRole("button", { name: /clutch:/i })).toBeDisabled();
  });

  it("publishes the newly selected source on the very next frame", async () => {
    await bootPanel();
    frameAt(100);
    expect(hoisted.queued.at(-1)!.clutch_source).toBe("spacebar");

    fireEvent.click(screen.getByRole("button", { name: /clutch:/i }));
    frameAt(200);
    // The frame literal closes over `clutchSource`, so the loop has to be
    // rebuilt on a source change or every subsequent frame lies about who
    // holds authority.
    expect(hoisted.queued.at(-1)!.clutch_source).toBe("mouth");
  });

  it("refuses to start in mouth mode without an adequate calibration", async () => {
    await bootPanel();
    fireEvent.click(screen.getByRole("button", { name: /clutch:/i }));
    expect(screen.getByRole("button", { name: /clutch: mouth/i })).toBeInTheDocument();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^start$/i }));
    });
    expect(hoisted.started).toEqual([]);
    expect(hoisted.calibrated).toEqual([]);
  });
});
