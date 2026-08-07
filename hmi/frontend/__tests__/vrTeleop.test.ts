import { describe, expect, it, vi } from "vitest";

import {
  BUTTON_AX, BUTTON_BY, BUTTON_SQUEEZE, BUTTON_TRIGGER,
  axPressed, disengagedFrame, estopPressed, hapticCues, holdToggle,
  holdToggleInit, pulse, RECORD_HOLD_MS, sampleVRFrame,
  type XRFrameLike, type XRInputSourceLike, type XRSessionLike,
} from "../lib/vrTeleop";

const IDENT = { x: 0, y: 0, z: 0, w: 1 };

function controller(
  handedness: "left" | "right",
  opts: { squeeze?: boolean; trigger?: number; estop?: boolean; ax?: boolean;
          pose?: boolean;
          pulse?: (intensity: number, durationMs: number) => unknown } = {},
): XRInputSourceLike {
  const buttons: { pressed: boolean; value: number }[] = [];
  buttons[BUTTON_TRIGGER] = { pressed: false, value: opts.trigger ?? 0 };
  buttons[BUTTON_SQUEEZE] = { pressed: opts.squeeze ?? false, value: 0 };
  buttons[BUTTON_BY] = { pressed: opts.estop ?? false, value: 0 };
  buttons[BUTTON_AX] = { pressed: opts.ax ?? false, value: 0 };
  return {
    handedness,
    gripSpace: opts.pose === false ? undefined : { space: handedness },
    gamepad: {
      buttons,
      ...(opts.pulse ? { hapticActuators: [{ pulse: opts.pulse }] } : {}),
    },
  };
}

function session(sources: XRInputSourceLike[]): XRSessionLike {
  return {
    inputSources: sources,
    requestReferenceSpace: () => Promise.resolve({}),
    requestAnimationFrame: () => 0,
    end: () => Promise.resolve(),
    addEventListener: () => {},
    removeEventListener: () => {},
  };
}

const frame: XRFrameLike = {
  getViewerPose: () => ({
    transform: { position: { ...IDENT, y: 1.6 }, orientation: IDENT },
    emulatedPosition: false,
  }),
  getPose: () => ({
    transform: { position: { ...IDENT, z: -0.4 }, orientation: IDENT },
    emulatedPosition: false,
  }),
};

describe("sampleVRFrame", () => {
  it("ships each hand's squeeze separately and ORs them into dead_man", () => {
    const s = session([
      controller("left", { squeeze: true }),
      controller("right", { squeeze: false }),
    ]);
    const out = sampleVRFrame(s, frame, {}, { tsMs: 42 });
    expect(out.dead_man).toBe(true);
    expect(out.left?.squeeze).toBe(true);
    expect(out.right?.squeeze).toBe(false);
  });

  it("keeps squeeze on an untracked controller so the clutch is honest", () => {
    // A controller that lost pose tracking still has a held grip; the backend
    // freezes the side on tracking, not on a fabricated clutch release.
    const s = session([controller("left", { squeeze: true, pose: false })]);
    const out = sampleVRFrame(s, frame, {}, { tsMs: 1 });
    expect(out.left?.tracked).toBe(false);
    expect(out.left?.squeeze).toBe(true);
  });

  it("forceDisengaged zeroes every clutch bit while poses keep flowing", () => {
    // The Quest system menu freezes input state; a grip held when it opened
    // must not stay held. Poses may keep shipping — releasing authority is
    // the clutch's job.
    const s = session([
      controller("left", { squeeze: true }),
      controller("right", { squeeze: true }),
    ]);
    const out = sampleVRFrame(s, frame, {}, { tsMs: 1, forceDisengaged: true });
    expect(out.dead_man).toBe(false);
    expect(out.left?.squeeze).toBe(false);
    expect(out.right?.squeeze).toBe(false);
    expect(out.left?.tracked).toBe(true);
  });

  it("reports the analog trigger through untouched", () => {
    const s = session([controller("right", { trigger: 0.62 })]);
    const out = sampleVRFrame(s, frame, {}, { tsMs: 1 });
    expect(out.right?.trigger).toBeCloseTo(0.62);
  });
});

describe("estopPressed", () => {
  it("fires on B/Y from either controller", () => {
    expect(estopPressed(session([controller("left", { estop: true })]))).toBe(true);
    expect(estopPressed(session([controller("right", { estop: true })]))).toBe(true);
  });

  it("stays quiet for grips and triggers", () => {
    const s = session([
      controller("left", { squeeze: true, trigger: 1 }),
      controller("right", { squeeze: true, trigger: 1 }),
    ]);
    expect(estopPressed(s)).toBe(false);
  });
});

describe("axPressed", () => {
  it("fires on A/X from either controller", () => {
    expect(axPressed(session([controller("left", { ax: true })]))).toBe(true);
    expect(axPressed(session([controller("right", { ax: true })]))).toBe(true);
  });

  it("stays quiet for grips, triggers and the E-STOP button", () => {
    const s = session([
      controller("left", { squeeze: true, trigger: 1, estop: true }),
      controller("right", { squeeze: true, trigger: 1 }),
    ]);
    expect(axPressed(s)).toBe(false);
  });
});

describe("holdToggle", () => {
  const HOLD = RECORD_HOLD_MS;

  it("does not fire until the hold threshold is crossed", () => {
    let st = holdToggleInit();
    st = holdToggle(st, true, 0, HOLD);       // press starts
    expect(st.toggled).toBe(false);
    st = holdToggle(st, true, HOLD - 1, HOLD); // still short
    expect(st.toggled).toBe(false);
  });

  it("fires exactly once at the threshold while held", () => {
    let st = holdToggleInit();
    st = holdToggle(st, true, 0, HOLD);
    st = holdToggle(st, true, HOLD, HOLD);
    expect(st.toggled).toBe(true);
    // Holding longer must not re-fire.
    st = holdToggle(st, true, HOLD + 500, HOLD);
    expect(st.toggled).toBe(false);
  });

  it("releasing before the threshold never fires, and re-pressing restarts", () => {
    let st = holdToggleInit();
    st = holdToggle(st, true, 0, HOLD);
    st = holdToggle(st, false, 100, HOLD);    // let go early
    expect(st.toggled).toBe(false);
    // A fresh press gets a fresh clock: the earlier partial hold doesn't count.
    st = holdToggle(st, true, 1000, HOLD);
    expect(st.toggled).toBe(false);
    st = holdToggle(st, true, 1000 + HOLD, HOLD);
    expect(st.toggled).toBe(true);
  });
});

describe("hapticCues", () => {
  it("buzzes hard exactly when an arm goes live", () => {
    const cues = hapticCues(
      { left: "acquiring", right: "held" },
      { left: "driving", right: "held" },
    );
    expect(cues).toHaveLength(1);
    expect(cues[0].hand).toBe("left");
    expect(cues[0].intensity).toBeGreaterThan(0.5);
  });

  it("ticks softly on countdown start and medium on release", () => {
    const start = hapticCues({ left: "held" }, { left: "acquiring" });
    expect(start[0].intensity).toBeLessThan(0.5);
    const release = hapticCues({ left: "driving" }, { left: "held" });
    expect(release[0].intensity).toBeGreaterThan(start[0].intensity);
  });

  it("says nothing when nothing changed, or on first sight", () => {
    expect(hapticCues({ left: "driving" }, { left: "driving" })).toHaveLength(0);
    // First status after entering a session: no prior state, no phantom buzz.
    expect(hapticCues({}, { left: "held", right: "held" })).toHaveLength(0);
  });
});

describe("pulse", () => {
  it("reaches only the named hand and survives a throwing actuator", () => {
    const leftPulse = vi.fn<(i: number, ms: number) => unknown>();
    const rightPulse = vi.fn<(i: number, ms: number) => unknown>(() => {
      throw new Error("nope");
    });
    const s = session([
      controller("left", { pulse: leftPulse }),
      controller("right", { pulse: rightPulse }),
    ]);
    pulse(s, "left", 0.5, 100);
    expect(leftPulse).toHaveBeenCalledWith(0.5, 100);
    expect(rightPulse).not.toHaveBeenCalled();
    expect(() => pulse(s, "right", 1, 10)).not.toThrow();
  });
});

describe("disengagedFrame", () => {
  it("asks for nothing at all", () => {
    const f = disengagedFrame(123);
    expect(f.dead_man).toBe(false);
    expect(f.head).toBeNull();
    expect(f.left).toBeNull();
    expect(f.right).toBeNull();
    expect(f.ts_ms).toBe(123);
  });
});

import { mat4Multiply, paintHud, type HudStatusLike } from "../lib/vrTeleop";

describe("mat4Multiply", () => {
  it("multiplies column-major like WebXR expects", () => {
    const I = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
    const T = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 2,3,4,1]); // translate
    const S = new Float32Array([2,0,0,0, 0,2,0,0, 0,0,2,0, 0,0,0,1]); // scale
    expect(Array.from(mat4Multiply(I, T))).toEqual(Array.from(T));
    // T * S: scale first, then translate — translation must survive unscaled.
    const TS = mat4Multiply(T, S);
    expect(TS[0]).toBe(2);
    expect([TS[12], TS[13], TS[14]]).toEqual([2, 3, 4]);
    // S * T: translation gets scaled.
    const ST = mat4Multiply(S, T);
    expect([ST[12], ST[13], ST[14]]).toEqual([4, 6, 8]);
  });
});

describe("paintHud", () => {
  function stubCtx() {
    const texts: string[] = [];
    const ctx = {
      canvas: { width: 1024, height: 768 },
      clearRect: () => {},
      fillRect: () => {},
      drawImage: () => {},
      fillText: (t: string) => { texts.push(t); },
      set fillStyle(_v: string) {},
      set font(_v: string) {},
      set textAlign(_v: string) {},
    };
    return { ctx: ctx as unknown as CanvasRenderingContext2D, texts };
  }

  it("shows the collision hold and the blocking joints", () => {
    const { ctx, texts } = stubCtx();
    const status: HudStatusLike = {
      state: "acquiring",
      collision: { enabled: true, slack_m: -0.004, limited: true },
      clutch: { sides: { left: true, right: false } },
      acquire: {
        left: { authority: "acquiring", remaining_ms: 1200,
                blocking: ["gripper"], reason: "matching" },
        right: { authority: "held", remaining_ms: null,
                 blocking: [], reason: "clutch_open" },
      },
    };
    paintHud(ctx, status, null);
    const all = texts.join(" | ");
    expect(all).toContain("ACQUIRING");
    expect(all).toContain("COLLISION HOLD -4 mm");
    expect(all).toContain("match: gripper");
    expect(all).toContain("no workspace camera");
  });

  it("never throws on an empty status", () => {
    const { ctx, texts } = stubCtx();
    expect(() => paintHud(ctx, null, null)).not.toThrow();
    expect(texts.join(" ")).toContain("E-STOP");
  });

  it("shows the REC badge and frame count while recording, and hides it otherwise", () => {
    const on = stubCtx();
    paintHud(on.ctx, { state: "driving" }, null, { recording: true, episode_frames: 612 });
    expect(on.texts.join(" | ")).toContain("● REC 612");

    const off = stubCtx();
    paintHud(off.ctx, { state: "driving" }, null, null);
    expect(off.texts.join(" | ")).not.toContain("● REC");
  });
});

describe("sampleVRFrame orientation source", () => {
  it("takes orientation from the target ray, position from the grip", () => {
    // Quest grip frames tilt ~50-60° from where the hand points; the backend
    // synthesizes the whole hand from this orientation, so shipping the grip
    // frame made a naturally held controller read as a pitched-down wrist.
    const src: XRInputSourceLike = {
      handedness: "right",
      gripSpace: { tag: "grip" },
      targetRaySpace: { tag: "ray" },
      gamepad: { buttons: [] },
    };
    const s = session([src]);
    const f: XRFrameLike = {
      getViewerPose: frame.getViewerPose,
      getPose: (space: unknown) => ({
        transform: {
          position: { x: 1, y: 2, z: 3, w: 1 },
          orientation: (space as { tag: string }).tag === "ray"
            ? { x: 0.5, y: 0, z: 0, w: 0.866 }
            : { x: 0, y: 0, z: 0, w: 1 },
        },
        emulatedPosition: false,
      }),
    };
    const out = sampleVRFrame(s, f, {}, { tsMs: 1 });
    expect(out.right?.position).toEqual([1, 2, 3]);
    expect(out.right?.orientation[0]).toBeCloseTo(0.5);
  });

  it("falls back to the grip orientation when no ray space exists", () => {
    const src: XRInputSourceLike = {
      handedness: "left",
      gripSpace: { tag: "grip" },
      gamepad: { buttons: [] },
    };
    const out = sampleVRFrame(session([src]), frame, {}, { tsMs: 1 });
    expect(out.left?.orientation).toEqual([0, 0, 0, 1]);
  });
});
