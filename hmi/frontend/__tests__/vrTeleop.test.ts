import { describe, expect, it, vi } from "vitest";

import {
  BUTTON_AX, BUTTON_BY, BUTTON_SQUEEZE, BUTTON_TRIGGER,
  BUTTON_THUMBSTICK, CAM_TILE_SIZES, cycleIndex, thumbstickPressed,
  disengagedFrame, episodePressed, hapticCues, holdToggle,
  holdToggleInit, precisionHeld, pulse, RECORD_HOLD_MS, sampleVRFrame,
  type XRFrameLike, type XRInputSourceLike, type XRSessionLike,
} from "../lib/vrTeleop";

const IDENT = { x: 0, y: 0, z: 0, w: 1 };

function controller(
  handedness: "left" | "right",
  opts: { squeeze?: boolean; trigger?: number; episode?: boolean; ax?: boolean;
          pose?: boolean;
          pulse?: (intensity: number, durationMs: number) => unknown } = {},
): XRInputSourceLike {
  const buttons: { pressed: boolean; value: number }[] = [];
  buttons[BUTTON_TRIGGER] = { pressed: false, value: opts.trigger ?? 0 };
  buttons[BUTTON_SQUEEZE] = { pressed: opts.squeeze ?? false, value: 0 };
  buttons[BUTTON_BY] = { pressed: opts.episode ?? false, value: 0 };
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

  it("ships the stance so the backend picks the matching hand rotation", () => {
    const s = session([controller("right", {})]);
    const out = sampleVRFrame(s, frame, {}, { tsMs: 1, stance: "front" });
    expect(out.stance).toBe("front");
    // No stance option → no key on the wire; the backend's default (behind,
    // the passthrough stance) must own that case, not a frontend fallback.
    const bare = sampleVRFrame(s, frame, {}, { tsMs: 1 });
    expect(bare.stance).toBeUndefined();
  });
});

describe("episodePressed", () => {
  // Per hand, and that is the whole point: B banks the take, Y bins it.
  // A helper that answered "either" would make the two indistinguishable.
  it("reads B and Y apart", () => {
    const bOnly = session([
      controller("left"), controller("right", { episode: true }),
    ]);
    expect(episodePressed(bOnly, "right")).toBe(true);
    expect(episodePressed(bOnly, "left")).toBe(false);

    const yOnly = session([
      controller("left", { episode: true }), controller("right"),
    ]);
    expect(episodePressed(yOnly, "left")).toBe(true);
    expect(episodePressed(yOnly, "right")).toBe(false);
  });

  it("stays quiet for grips and triggers", () => {
    const s = session([
      controller("left", { squeeze: true, trigger: 1 }),
      controller("right", { squeeze: true, trigger: 1 }),
    ]);
    expect(episodePressed(s, "left")).toBe(false);
    expect(episodePressed(s, "right")).toBe(false);
  });
});

describe("precisionHeld", () => {
  it("fires on A/X from either controller", () => {
    expect(precisionHeld(session([controller("left", { ax: true })]))).toBe(true);
    expect(precisionHeld(session([controller("right", { ax: true })]))).toBe(true);
  });

  it("stays quiet for grips, triggers and the episode button", () => {
    const s = session([
      controller("left", { squeeze: true, trigger: 1, episode: true }),
      controller("right", { squeeze: true, trigger: 1 }),
    ]);
    expect(precisionHeld(s)).toBe(false);
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

import { mat4Multiply } from "../lib/vrTeleop";

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

describe("sampleVRFrame pose source", () => {
  const mixedFrame: XRFrameLike = {
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

  it("takes position AND orientation from one space — grip preferred", () => {
    // The kit samples gripSpace || targetRaySpace for the WHOLE pose, and
    // the constant grip-vs-ray tilt cancels in the mapper's clutch-relative
    // delta. The old grip-position + ray-orientation mix injected a phantom
    // ~50-60° rotation increment whenever one frame's ray pose came back
    // null — the source swap read as real hand rotation to the incremental
    // mapper, twice per dropout.
    const src: XRInputSourceLike = {
      handedness: "right",
      gripSpace: { tag: "grip" },
      targetRaySpace: { tag: "ray" },
      gamepad: { buttons: [] },
    };
    const out = sampleVRFrame(session([src]), mixedFrame, {}, { tsMs: 1 });
    expect(out.right?.position).toEqual([1, 2, 3]);
    // Grip orientation, NOT the ray's 0.5-x quat.
    expect(out.right?.orientation[0]).toBe(0);
    expect(out.right?.orientation[3]).toBe(1);
  });

  it("falls back to the target ray only when no grip space exists", () => {
    const src: XRInputSourceLike = {
      handedness: "left",
      targetRaySpace: { tag: "ray" },
      gamepad: { buttons: [] },
    };
    const out = sampleVRFrame(session([src]), mixedFrame, {}, { tsMs: 1 });
    expect(out.left?.position).toEqual([1, 2, 3]);
    expect(out.left?.orientation[0]).toBeCloseTo(0.5);
  });
});

describe("sampleVRFrame precision", () => {
  it("stamps precision per hand from that hand's own A/X", () => {
    // Kit semantics: the driving hand's own modifier. A global flag
    // re-anchored and re-scaled the arm the OTHER hand was mid-reach with.
    const s = session([
      controller("left", { ax: true }),
      controller("right"),
    ]);
    const out = sampleVRFrame(s, frame, {}, { tsMs: 1 });
    expect(out.left?.precision).toBe(true);
    expect(out.right?.precision).toBeUndefined();
  });
});

// ---- view menu ---------------------------------------------------------------

describe("thumbstickPressed", () => {
  function stick(hand: "left" | "right", down: boolean): XRInputSourceLike {
    const buttons: { pressed: boolean; value: number }[] = [];
    buttons[BUTTON_THUMBSTICK] = { pressed: down, value: 0 };
    return { handedness: hand, gripSpace: { space: hand }, gamepad: { buttons } };
  }

  it("is per-hand, so the two sticks can mean different things", () => {
    const s = session([stick("left", true), stick("right", false)]);
    expect(thumbstickPressed(s, "left")).toBe(true);
    expect(thumbstickPressed(s, "right")).toBe(false);
  });

  it("is false when a controller reports no gamepad at all", () => {
    const s = session([{ handedness: "left", gripSpace: {}, gamepad: null }]);
    expect(thumbstickPressed(s, "left")).toBe(false);
  });

  it("does not collide with the precision modifier", () => {
    // A/X held must not read as a stick click: they are different actions and
    // the thumb rests near both.
    const s = session([controller("right", { ax: true })]);
    expect(precisionHeld(s)).toBe(true);
    expect(thumbstickPressed(s, "right")).toBe(false);
  });
});

describe("cycleIndex", () => {
  it("wraps forward and backward", () => {
    expect(cycleIndex(3, 0, 1)).toBe(1);
    expect(cycleIndex(3, 2, 1)).toBe(0);
    expect(cycleIndex(3, 0, -1)).toBe(2);
  });

  it("returns 0 for an empty list instead of NaN", () => {
    // A NaN index paints an undefined selection rather than failing loudly.
    expect(cycleIndex(0, 0, 1)).toBe(0);
    expect(Number.isNaN(cycleIndex(0, 5, 1))).toBe(false);
  });
});
