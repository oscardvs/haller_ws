import { describe, expect, it } from "vitest";

import {
  armPairing, clampKnob, DEFAULT_WRIST_PIVOT_M, formatKnob, HAPTIC_FLOOR,
  ikHapticCues, ORIENT_DEFICIT, parseVrSocketMessage, precisionHeld,
  PRECISION_STICK_Y, sampleVRFrame, sessionPresets, stepTuning, stickAxes,
  TUNING_KNOBS, TUNING_REPEAT_MS, WRIST_PIVOT_KEY,
  BUTTON_SQUEEZE, BUTTON_TRIGGER,
  type IkSides, type XRFrameLike, type XRInputSourceLike, type XRSessionLike,
} from "../lib/vrTeleop";

// ---- fixtures ---------------------------------------------------------------

const IDENT = { x: 0, y: 0, z: 0, w: 1 };

function controller(
  handedness: "left" | "right",
  opts: { axes?: number[]; squeeze?: boolean } = {},
): XRInputSourceLike {
  const buttons: { pressed: boolean; value: number }[] = [];
  buttons[BUTTON_TRIGGER] = { pressed: false, value: 0 };
  buttons[BUTTON_SQUEEZE] = { pressed: opts.squeeze ?? false, value: 0 };
  return {
    handedness,
    gripSpace: { space: handedness },
    gamepad: { buttons, ...(opts.axes ? { axes: opts.axes } : {}) },
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

/** A frame whose grip pose is the identity rotation at the origin, so a
 *  wrist-pivot shift lands exactly on +z. */
const frameAtOrigin: XRFrameLike = {
  getViewerPose: () => ({
    transform: { position: { ...IDENT, y: 1.6 }, orientation: IDENT },
    emulatedPosition: false,
  }),
  getPose: () => ({
    transform: { position: { ...IDENT }, orientation: IDENT },
    emulatedPosition: false,
  }),
};

// ---- the read-out point ------------------------------------------------------

describe("wrist pivot", () => {
  it("shifts the read-out point back along the controller's own axis", () => {
    // Grip orientation is the identity here, so +z of the grip frame is world
    // +z and the whole offset lands on that axis. A pure twist about a
    // palm-centred point is an arc the mapper can only read as translation.
    const s = session([controller("right")]);
    const out = sampleVRFrame(s, frameAtOrigin, {},
                              { tsMs: 1, wristPivotM: DEFAULT_WRIST_PIVOT_M });
    expect(out.right?.position[0]).toBeCloseTo(0);
    expect(out.right?.position[1]).toBeCloseTo(0);
    expect(out.right?.position[2]).toBeCloseTo(DEFAULT_WRIST_PIVOT_M);
  });

  it("is exactly off at zero, and off when unset", () => {
    const s = session([controller("right")]);
    for (const opts of [{ tsMs: 1 }, { tsMs: 1, wristPivotM: 0 }]) {
      const out = sampleVRFrame(s, frameAtOrigin, {}, opts);
      expect(out.right?.position).toEqual([0, 0, 0]);
    }
  });

  it("never moves an untracked hand off the origin sentinel", () => {
    const src = controller("right");
    const s = session([{ ...src, gripSpace: undefined }]);
    const out = sampleVRFrame(s, frameAtOrigin, {},
                              { tsMs: 1, wristPivotM: 0.09 });
    expect(out.right?.tracked).toBe(false);
    expect(out.right?.position).toEqual([0, 0, 0]);
  });
});

describe("precision on the wire", () => {
  it("rides on every tracked hand, and is absent when not held", () => {
    const s = session([controller("left"), controller("right")]);
    const on = sampleVRFrame(s, frameAtOrigin, {}, { tsMs: 1, precision: true });
    expect(on.left?.precision).toBe(true);
    expect(on.right?.precision).toBe(true);
    const off = sampleVRFrame(s, frameAtOrigin, {}, { tsMs: 1 });
    expect(off.left?.precision).toBeUndefined();
  });

  it("never appears on a legacy field the backend no longer reads", () => {
    const s = session([controller("right")]);
    const out = sampleVRFrame(s, frameAtOrigin, {}, { tsMs: 1, stance: "behind" });
    // Every frame takes the ik path now — there is no mode to dispatch on,
    // no arm-mounting convention and no limb-length override on the wire.
    expect(out).not.toHaveProperty("vr_mode");
    expect(out).not.toHaveProperty("mirror_mode");
    expect(out).not.toHaveProperty("body");
  });
});

// ---- sticks ------------------------------------------------------------------

describe("stickAxes", () => {
  it("prefers the xr-standard thumbstick pair", () => {
    const s = session([controller("right", { axes: [0.1, 0.2, -0.8, 0.4] })]);
    expect(stickAxes(s, "right")).toEqual([-0.8, 0.4]);
  });

  it("falls back to a controller that reports only one analog pair", () => {
    const s = session([controller("left", { axes: [0.5, -0.5] })]);
    expect(stickAxes(s, "left")).toEqual([0.5, -0.5]);
  });

  it("reads (0, 0) for a hand with no gamepad rather than throwing", () => {
    const s = session([{ handedness: "left", gripSpace: {} }]);
    expect(stickAxes(s, "left")).toEqual([0, 0]);
    expect(stickAxes(s, "right")).toEqual([0, 0]);
  });
});

describe("precisionHeld", () => {
  it("engages on the LEFT stick pushed away, past the halfway point", () => {
    expect(precisionHeld(session([controller("left", { axes: [0, 0, 0, -1] })])))
      .toBe(true);
    expect(precisionHeld(session([controller("left", { axes: [0, 0, 0, -0.5] })])))
      .toBe(false);
    // Exactly on the threshold counts: the constant is the engage point.
    expect(precisionHeld(session([
      controller("left", { axes: [0, 0, 0, PRECISION_STICK_Y] })])))
      .toBe(true);
  });

  it("ignores the right stick, which belongs to the tuning list", () => {
    const s = session([controller("right", { axes: [0, 0, 0, -1] })]);
    expect(precisionHeld(s)).toBe(false);
  });
});

// ---- the socket's control channel -------------------------------------------

describe("parseVrSocketMessage", () => {
  it("reads the ik_state push, config and per-side diagnostics together", () => {
    const msg = parseVrSocketMessage(JSON.stringify({
      type: "ik_state",
      config: { scale_rotation: 1.6 },
      sides: { left: { driving: true, sigma_min: 0.04 } },
    }));
    expect(msg?.kind).toBe("ik_state");
    expect(msg?.config?.scale_rotation).toBe(1.6);
    expect(msg && msg.kind === "ik_state" && msg.sides.left?.sigma_min).toBe(0.04);
  });

  it("accepts `settings` as the same payload under the contract's name", () => {
    // The relay answered request_settings with its ik_state dict; the unified
    // socket answers with `settings`. One reader, both spellings — a client
    // that understands only one of them silently never seeds its sliders.
    const msg = parseVrSocketMessage({ type: "settings", config: { lam_pos: 0.01 } });
    expect(msg?.kind).toBe("ik_state");
    expect(msg?.config?.lam_pos).toBe(0.01);
  });

  it("reads a settings reply that spreads the config at the top level", () => {
    // Contract-tolerance, not speculation: the failure it prevents is silent
    // — sliders that never seed and show defaults the robot does not have.
    const msg = parseVrSocketMessage({ type: "settings", scale_rotation: 1.6,
                                       stance: "behind" });
    expect(msg?.kind).toBe("ik_state");
    expect(msg?.config?.scale_rotation).toBe(1.6);
    expect(msg?.config).not.toHaveProperty("type");
  });

  it("reads the clamped echo back off config_applied", () => {
    const msg = parseVrSocketMessage({ type: "config_applied",
                                       config: { scale_translation: 4 } });
    expect(msg).toEqual({ kind: "config_applied", config: { scale_translation: 4 } });
  });

  it("returns null for anything else instead of throwing on the handler", () => {
    for (const raw of ["not json", "", null, 42, { type: "vr_keypoints" }, {}]) {
      expect(parseVrSocketMessage(raw)).toBeNull();
    }
  });

  it("survives an ik_state with no sides at all", () => {
    const msg = parseVrSocketMessage({ type: "ik_state" });
    expect(msg).toEqual({ kind: "ik_state", config: null, sides: {} });
  });
});

describe("ikHapticCues", () => {
  const driving = (extra: object) => ({ left: { driving: true, ...extra } });

  it("passes the backend's trouble mix through as a light buzz", () => {
    const cues = ikHapticCues({}, driving({ haptic: 0.5 }));
    expect(cues).toEqual([{ hand: "left", intensity: 0.5, durationMs: 60 }]);
  });

  it("stays quiet below the floor, where the mix is noise", () => {
    expect(ikHapticCues({}, driving({ haptic: HAPTIC_FLOOR - 0.01 }))).toEqual([]);
  });

  it("hits hard on the EDGE of an orientation deficit, then stops nagging", () => {
    const short = driving({ haptic: 0.3, orient_residual: ORIENT_DEFICIT + 0.2 });
    const first = ikHapticCues({}, short);
    expect(first).toEqual([{ hand: "left", intensity: 0.9, durationMs: 140 }]);
    // Still short on the next push, but the operator has been told: back to
    // the ordinary buzz rather than a hard pulse every 50 ms.
    const again = ikHapticCues(short, short);
    expect(again).toEqual([{ hand: "left", intensity: 0.3, durationMs: 60 }]);
  });

  it("says nothing about a hand that is not driving", () => {
    const idle: IkSides = { right: { driving: false, haptic: 1, orient_residual: 2 } };
    expect(ikHapticCues({}, idle)).toEqual([]);
  });

  it("never asks for more than full intensity", () => {
    const cues = ikHapticCues({}, driving({ haptic: 3 }));
    expect(cues[0].intensity).toBe(1);
  });
});

// ---- the tuning list ---------------------------------------------------------

describe("stepTuning", () => {
  const values = { scale_translation: 1, scale_rotation: 1.6 };
  const nav = { index: 0, lastStepMs: 0 };

  it("walks down the list on a pull back, and wraps", () => {
    const down = stepTuning(nav, [0, 1], 1000, values);
    expect(down.nav.index).toBe(1);
    expect(down.patch).toBeNull();
    const up = stepTuning(nav, [0, -1], 1000, values);
    expect(up.nav.index).toBe(TUNING_KNOBS.length - 1);
  });

  it("steps the selected knob by its own step, and clamps to its own range", () => {
    const up = stepTuning(nav, [1, 0], 1000, values);
    expect(up.patch).toEqual({ key: "scale_translation", value: 1.05 });
    // scale_translation bottoms out at 0.1; a stick held left cannot go under.
    const floored = stepTuning(nav, [-1, 0], 1000, { scale_translation: 0.1 });
    expect(floored.patch).toEqual({ key: "scale_translation", value: 0.1 });
  });

  it("rate-limits to one step per deliberate push", () => {
    const first = stepTuning(nav, [0, 1], 1000, values);
    const tooSoon = stepTuning(first.nav, [0, 1], 1000 + TUNING_REPEAT_MS - 1, values);
    expect(tooSoon.nav.index).toBe(first.nav.index);
    expect(tooSoon.patch).toBeNull();
    const later = stepTuning(first.nav, [0, 1], 1000 + TUNING_REPEAT_MS, values);
    expect(later.nav.index).toBe(2);
  });

  it("ignores a resting stick and a diagonal's horizontal half", () => {
    expect(stepTuning(nav, [0.3, 0.3], 1000, values).patch).toBeNull();
    // Vertical wins: a diagonal push must not both move the cursor and write.
    const diag = stepTuning(nav, [0.9, 0.9], 1000, values);
    expect(diag.patch).toBeNull();
    expect(diag.nav.index).toBe(1);
  });

  it("starts a knob the server has not reported at its own minimum", () => {
    const onLamPos = { index: TUNING_KNOBS.findIndex((k) => k.key === "lam_pos"),
                       lastStepMs: 0 };
    const step = stepTuning(onLamPos, [1, 0], 1000, {});
    expect(step.patch?.key).toBe("lam_pos");
    expect(step.patch?.value).toBeCloseTo(0.002);
  });

  it("carries the wrist pivot, which never leaves the client", () => {
    const knob = TUNING_KNOBS.find((k) => k.key === WRIST_PIVOT_KEY);
    expect(knob?.local).toBe(true);
    expect(TUNING_KNOBS.filter((k) => k.local)).toHaveLength(1);
  });
});

describe("clampKnob / formatKnob", () => {
  const knob = { key: "k", label: "k", min: 0.1, max: 4, step: 0.05 };
  it("clamps both ends and refuses NaN", () => {
    expect(clampKnob(knob, 9)).toBe(4);
    expect(clampKnob(knob, -1)).toBe(0.1);
    expect(clampKnob(knob, Number.NaN)).toBe(0.1);
  });
  it("prints a fixed width, and says nothing it does not know", () => {
    expect(formatKnob(1)).toBe("1.000");
    expect(formatKnob(undefined)).toBe("—");
    expect(formatKnob(Number.NaN)).toBe("—");
  });
});

// ---- the session launcher ------------------------------------------------------

describe("armPairing", () => {
  // The real rig's config lists [right, left]; the sim config lists
  // [left, right]. A positional rule is therefore correct in one and inverted
  // in the other, which is exactly the crossed-arms failure the stance swap
  // exists to prevent — so identity comes from the ID.
  const real = ["right", "left"];
  const sim = ["left", "right"];

  it("swaps hands against arms in the behind stance, whatever the config order", () => {
    for (const ids of [real, sim]) {
      expect(armPairing(ids, "behind")).toEqual({ left_arm: "right", right_arm: "left" });
    }
  });

  it("pairs directly when the operator faces the arms", () => {
    for (const ids of [real, sim]) {
      for (const stance of ["mirror", "front"] as const) {
        expect(armPairing(ids, stance)).toEqual({ left_arm: "left", right_arm: "right" });
      }
    }
  });

  it("falls back to config order for IDs that say neither", () => {
    expect(armPairing(["alpha", "beta"], "front"))
      .toEqual({ left_arm: "alpha", right_arm: "beta" });
    expect(armPairing(["alpha", "beta"], "behind"))
      .toEqual({ left_arm: "beta", right_arm: "alpha" });
  });

  it("puts a solo arm on the hand the same rule would have given it", () => {
    expect(armPairing(real, "behind", "left")).toEqual({ left_arm: null, right_arm: "left" });
    expect(armPairing(real, "behind", "right")).toEqual({ left_arm: "right", right_arm: null });
    expect(armPairing(real, "front", "left")).toEqual({ left_arm: "left", right_arm: null });
    expect(armPairing(real, "front", "right")).toEqual({ left_arm: null, right_arm: "right" });
  });

  it("handles a rig with only one arm enabled at all", () => {
    // config.solo-real.yaml: one arm, named "left". Behind the bench it sits
    // under the operator's right hand, exactly as it would bimanually.
    expect(armPairing(["left"], "behind")).toEqual({ left_arm: null, right_arm: "left" });
    expect(armPairing(["left"], "front")).toEqual({ left_arm: "left", right_arm: null });
    expect(armPairing(["right"], "behind")).toEqual({ left_arm: "right", right_arm: null });
  });

  it("never sends the same arm to both hands", () => {
    for (const ids of [real, sim, ["left"], ["right"], ["alpha", "beta"]]) {
      for (const stance of ["behind", "mirror", "front"] as const) {
        const p = armPairing(ids, stance);
        expect(p.left_arm === null || p.left_arm !== p.right_arm).toBe(true);
      }
    }
  });

  it("resolves nothing out of an empty arm list rather than inventing a side", () => {
    expect(armPairing([], "behind")).toEqual({ left_arm: null, right_arm: null });
  });
});

describe("sessionPresets", () => {
  it("offers dual plus one solo button per arm", () => {
    expect(sessionPresets(["right", "left"])).toEqual([
      { id: "dual", label: "dual", soloArm: null },
      { id: "solo-right", label: "solo right", soloArm: "right" },
      { id: "solo-left", label: "solo left", soloArm: "left" },
    ]);
  });

  it("offers no dual button on a one-armed rig — it could only fail", () => {
    expect(sessionPresets(["left"])).toEqual([
      { id: "solo-left", label: "solo left", soloArm: "left" },
    ]);
  });

  it("offers nothing at all when no arm is enabled", () => {
    expect(sessionPresets([])).toEqual([]);
  });
});
