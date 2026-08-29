import { describe, expect, it } from "vitest";

import {
  applyServerConfig, BUTTON_AX, BUTTON_BY,
  reconcileConfig, recorderHapticCue, stepTake,
  TUNING_KNOBS, WRIST_PIVOT_KEY,
  type EndChoice, type TakeState,
} from "../lib/vrTeleop";
import { worstDropSource } from "../components/VRTeleopPanel";

// ---- fixtures ---------------------------------------------------------------

/** A recording 2D context. Every fact the HUD carries is a string, and the
 *  text is the only half a headless run can check — so assertions read the
 *  captured `fillText` calls and never a pixel. `x`/`y` are captured too,
 *  which is what lets a row-count pin exist at all. */
function stubCtx() {
  const calls: {
    text: string; x: number; y: number; font: string; align: string;
  }[] = [];
  // font and textAlign are RECORDED, not swallowed: the panel canvas is a
  // fixed 1024 px and monospace advances at 0.6 em, so a row's width is a
  // property of its copy and its font — which makes overflow a thing a test
  // can catch. It is not a thing a headset catches: the canvas simply clips,
  // and whatever sat at the end of the row is silently gone.
  let font = "";
  let align = "left";
  const ctx = {
    canvas: { width: 1024, height: 768 },
    clearRect: () => {}, fillRect: () => {}, strokeRect: () => {},
    drawImage: () => {},
    save: () => {}, restore: () => {},
    beginPath: () => {}, rect: () => {}, clip: () => {},
    measureText: (t: string) => ({ width: t.length * 17 }),
    fillText: (text: string, x: number, y: number) => {
      calls.push({ text, x, y, font, align });
    },
    set fillStyle(_v: string) {},
    set font(v: string) { font = v; },
    set textAlign(v: string) { align = v; },
    set strokeStyle(_v: string) {},
    set lineWidth(_v: number) {},
  };
  return {
    ctx: ctx as unknown as CanvasRenderingContext2D,
    calls,
    text: () => calls.map((c) => c.text).join(" | "),
  };
}

const menu = {
  views: [{ id: "mast", label: "mast" }, { id: "left_wrist", label: "left wrist" }],
  activeViewId: "mast",
  tileSize: "M",
};

/** A live session with both sides driving and the collision guard measuring —
 *  the widest the status column ever gets, and therefore the one to size it
 *  against. */
const driving = {
  state: "driving",
  collision: { enabled: true, slack_m: 0.042 },
  clutch: { sides: { left: true, right: true } },
  acquire: {
    left: { authority: "driving", remaining_ms: null, reason: "" },
    right: { authority: "driving", remaining_ms: null, reason: "" },
  },
};

/** Every cue the §1b table names, as the transition that produces it. Kept as
 *  data so a new cue cannot be added without the ordering pins seeing it. */
const CUES: readonly (readonly [TakeState, TakeState, EndChoice | null])[] = [
  ["idle", "armed", null],
  ["armed", "rolling", null],
  ["rolling", "prompt", null],
  ["prompt", "rolling", null],
  ["prompt", "armed", "keep"],
  ["prompt", "armed", "redo"],
  ["prompt", "idle", "keep_stop"],
  ["prompt", "idle", "drop"],
  ["prompt", "idle", null],
  ["armed", "idle", null],
];

const STATES: readonly TakeState[] = ["idle", "armed", "rolling", "prompt"];

// ---- the take machine -------------------------------------------------------

describe("stepTake", () => {
  it("climbs the ladder on B, and banks the take at the top", () => {
    // The kit's episode button: one press per rung, and the last one ENDS the
    // episode rather than asking about it. `examples/record_so101.py` binds B
    // to "end the current episode / end the reset phase" and there is no
    // question in between.
    const arm = stepTake("idle", { kind: "end_episode" });
    expect(arm).toEqual({ state: "armed", act: { do: "arm" } });
    const roll = stepTake(arm.state, { kind: "end_episode" });
    expect(roll).toEqual({ state: "rolling", act: { do: "roll" } });
    const banked = stepTake(roll.state, { kind: "end_episode" });
    expect(banked).toEqual({
      state: "armed", act: { do: "stop", save: true, rearm: true } });
  });

  it("bins a live take on Y, and reaches for nothing already written", () => {
    // Y is the kit's "discard + re-record". It throws away the take in
    // progress and lines the next one up.
    expect(stepTake("rolling", { kind: "rerecord" }))
      .toEqual({ state: "armed", act: { do: "stop", save: false, rearm: true } });
    // A take that is already on disk is NOT its business: one press of a
    // thumb button in a headset must not delete written data. IDLE and ARMED
    // hold nothing open, so Y is a no-op there rather than a deletion.
    expect(stepTake("idle", { kind: "rerecord" }))
      .toEqual({ state: "idle", act: null });
    expect(stepTake("armed", { kind: "rerecord" }))
      .toEqual({ state: "armed", act: null });
  });

  it("gives each decision its exact {save, rearm} pair", () => {
    expect(stepTake("prompt", { kind: "choose", choice: "keep" }))
      .toEqual({ state: "armed", act: { do: "stop", save: true, rearm: true } });
    expect(stepTake("prompt", { kind: "choose", choice: "redo" }))
      .toEqual({ state: "armed", act: { do: "stop", save: false, rearm: true } });
    expect(stepTake("prompt", { kind: "choose", choice: "keep_stop" }))
      .toEqual({ state: "idle", act: { do: "stop", save: true, rearm: false } });
    expect(stepTake("prompt", { kind: "choose", choice: "drop" }))
      .toEqual({ state: "idle", act: { do: "stop", save: false, rearm: false } });
  });

  it("lands both headset decisions back in ARMED, never idle", () => {
    // ARMED is the resting state of a recording session, not a step on the way
    // to one: it writes nothing, so sitting in it costs nothing, and a decision
    // that dropped the operator to idle would make banking 46 takes a
    // three-rung ladder climbed 46 times.
    for (const choice of ["keep", "redo"] as const) {
      expect(stepTake("prompt", { kind: "choose", choice }).state).toBe("armed");
    }
  });

  it("ignores a decision taken outside the prompt", () => {
    // A stick click that lands a frame after the prompt closed is stale, and
    // must not stop a take that is already rolling again.
    for (const state of ["idle", "armed", "rolling"] as const) {
      expect(stepTake(state, { kind: "choose", choice: "keep" }))
        .toEqual({ state, act: null });
    }
  });

  it("never lets the recorder invent a prompt, or act on its own report", () => {
    // The prompt is a client-side overlay; the recorder has no concept of it.
    // And the poll reconciles — a status read that fired REST would turn one
    // fault into a loop of them at 4 Hz.
    for (const state of STATES) {
      for (const reported of STATES) {
        const tr = stepTake(state, { kind: "recorder", state: reported });
        expect(tr.act).toBeNull();
        if (state !== "prompt") expect(tr.state).not.toBe("prompt");
      }
    }
  });

  it("holds the prompt open against a recorder that is still rolling", () => {
    // The one that matters. /record/stop takes the save decision AT stop time,
    // so the recorder is genuinely still recording while the prompt is open and
    // says exactly that every 250 ms. A reconcile that believed the report
    // would slam the prompt shut four times a second and the decision could
    // never be taken. Forty polls is ten seconds of deliberation.
    let state: TakeState = "prompt";
    for (let i = 0; i < 40; i++) {
      const tr = stepTake(state, { kind: "recorder", state: "rolling" });
      expect(tr.act).toBeNull();
      state = tr.state;
    }
    expect(state).toBe("prompt");
    // And the decision still lands when it finally comes.
    expect(stepTake(state, { kind: "choose", choice: "keep" }))
      .toEqual({ state: "armed", act: { do: "stop", save: true, rearm: true } });
  });

  it("banks ten takes without ever passing through idle", () => {
    // The workflow claim, run as a workflow. ARMED is the resting state of a
    // recording session, so a solo operator banking 46 episodes climbs the
    // ladder ONCE and then cycles roll → end → keep. If any decision dropped
    // through idle, that is a three-rung climb 46 times over, and the whole
    // reason the gate returns to ARMED is gone.
    const seen: TakeState[] = [];
    const stops: { save: boolean; rearm: boolean }[] = [];
    let state = stepTake("idle", { kind: "end_episode" }).state;   // the one climb
    for (let take = 0; take < 10; take++) {
      const roll = stepTake(state, { kind: "end_episode" });
      expect(roll).toEqual({ state: "rolling", act: { do: "roll" } });
      // Two presses per take, not three and a decision: B rolls, B ends.
      const done = stepTake(roll.state, { kind: "end_episode" });
      expect(done.act).toEqual({ do: "stop", save: true, rearm: true });
      stops.push({ save: true, rearm: true });
      state = done.state;
      seen.push(roll.state, state);
      // The 250 ms poll runs throughout; a reconcile mid-cycle must not move
      // the operator anywhere they did not ask to go.
      state = stepTake(state, { kind: "recorder", state: "armed" }).state;
      seen.push(state);
    }
    expect(state).toBe("armed");
    expect(seen).not.toContain("idle");
    expect(stops).toHaveLength(10);
  });

  it("closes the prompt when the recorder ends the take underneath it", () => {
    // A fault, or the cockpit stopping it from the other surface: there is no
    // take left to decide about, so the HUD must stop offering the decision.
    expect(stepTake("prompt", { kind: "recorder", state: "idle" }))
      .toEqual({ state: "idle", act: null });
    expect(stepTake("prompt", { kind: "recorder", state: "armed" }))
      .toEqual({ state: "armed", act: null });
  });

  it("lands an invalidated gate exactly where a stand-down lands", () => {
    // The flag rides along so the caller can SAY which of the two happened;
    // it must not move the machine, or the gate would need a second ladder.
    expect(stepTake("armed", { kind: "recorder", state: "idle", invalidated: true }))
      .toEqual(stepTake("armed", { kind: "recorder", state: "idle" }));
  });

  it("aborts to idle from anywhere, without a REST call", () => {
    // Teardown: the session is going away, so there is nobody left to answer a
    // /record/stop and nothing on the HUD worth keeping.
    for (const state of STATES) {
      expect(stepTake(state, { kind: "abort" })).toEqual({ state: "idle", act: null });
    }
  });
});

describe("the gestures the take machine rides on", () => {
  it("puts the take boundary on a reach, and the modifier under the thumb", () => {
    // The kit's split, restored. B is a deliberate reach and carries the take
    // boundary; A/X is where the thumb already rests while gripping and
    // carries precision, which is harmless to trigger by accident and
    // self-cancelling on release. The old arrangement was the other way
    // round, which is why precision had to be exiled to the left stick.
    expect(stepTake("idle", { kind: "end_episode" }).act).toEqual({ do: "arm" });
    // No hold gate on B: it takes a reach, so it does not need one.
    expect(BUTTON_BY).toBe(5);
    expect(BUTTON_AX).toBe(4);
  });

});

// ---- recorder haptics -------------------------------------------------------

describe("recorderHapticCue", () => {
  it("makes the roll the firmest cue in the vocabulary", () => {
    // The only moment that costs data if it is missed: frames are landing now.
    // "Did that take actually start" must never be a question the operator has
    // to look away from the workspace to answer.
    const roll = recorderHapticCue("armed", "rolling");
    expect(roll).not.toBeNull();
    for (const [prev, next, choice] of CUES) {
      if (prev === "armed" && next === "rolling") continue;
      const cue = recorderHapticCue(prev, next, choice);
      expect(cue).not.toBeNull();
      expect(cue!.intensity).toBeLessThan(roll!.intensity);
    }
  });

  it("keeps banked and binned distinguishable, though both land in ARMED", () => {
    // keep and redo end in the same state, so the HUD alone cannot separate
    // them — and the operator's eyes are on the workspace, not the HUD.
    const keep = recorderHapticCue("prompt", "armed", "keep");
    const redo = recorderHapticCue("prompt", "armed", "redo");
    expect(keep!.intensity).not.toBe(redo!.intensity);
    expect(stepTake("prompt", { kind: "choose", choice: "keep" }).state)
      .toBe(stepTake("prompt", { kind: "choose", choice: "redo" }).state);
  });

  it("says nothing while a state simply persists", () => {
    // Keyed on the transition, not the state: the status poll reconciles four
    // times a second, and a state-keyed cue would buzz through a whole take.
    for (const state of STATES) {
      expect(recorderHapticCue(state, state)).toBeNull();
      expect(recorderHapticCue(state, state, "keep")).toBeNull();
    }
  });

  it("ticks rather than falling silent when the gate drops", () => {
    // Un-arming without a word is the precise failure the gate exists to
    // prevent. Weakest of the lot, because nothing was lost — but never null.
    const dropped = recorderHapticCue("armed", "idle");
    expect(dropped).not.toBeNull();
    for (const [prev, next, choice] of CUES) {
      if (prev === "armed" && next === "idle") continue;
      expect(recorderHapticCue(prev, next, choice)!.intensity)
        .toBeGreaterThan(dropped!.intensity);
    }
  });

  it("returns null for a transition the table does not name", () => {
    for (const [prev, next] of [["idle", "rolling"], ["idle", "prompt"],
                                ["armed", "prompt"], ["rolling", "armed"]] as const) {
      expect(recorderHapticCue(prev, next)).toBeNull();
    }
  });
});

// ---- who owns a tuned value -------------------------------------------------

describe("reconcileConfig", () => {
  const local = {
    scale_translation: 2, scale_rotation: 1.6, [WRIST_PIVOT_KEY]: 0.11,
  };

  it("lets the server own every knob the operator has not moved", () => {
    const { values, reassert } = reconcileConfig(
      local, [], { scale_translation: 1, scale_rotation: 1 });
    expect(values.scale_translation).toBe(1);
    expect(values.scale_rotation).toBe(1);
    expect(reassert).toEqual({});
  });

  it("keeps the operator's knob and sends that one straight back", () => {
    // QuestTeleopConfig lives per connection and HumanTeleopClient reconnects
    // after 50 ms, so without the re-assert a socket blip halves a gain
    // mid-take — which feels exactly like an arm that has started lagging.
    // Narrow on purpose: the neighbour still reverts to what the robot says.
    const { values, reassert } = reconcileConfig(
      local, ["scale_translation"], { scale_translation: 1, scale_rotation: 1 });
    expect(values.scale_translation).toBe(2);
    expect(values.scale_rotation).toBe(1);
    expect(reassert).toEqual({ scale_translation: 2 });
  });

  it("never re-asserts a knob that never left the client", () => {
    // The wrist pivot is applied to the grip pose here and persisted in
    // localStorage; the server has no field to accept it into.
    expect(TUNING_KNOBS.find((k) => k.key === WRIST_PIVOT_KEY)?.local).toBe(true);
    const { values, reassert } = reconcileConfig(
      local, [WRIST_PIVOT_KEY], { [WRIST_PIVOT_KEY]: 0.09 });
    expect(values[WRIST_PIVOT_KEY]).toBe(0.11);
    expect(reassert).toEqual({});
  });

  it("leaves a dirty key the server has never heard of alone", () => {
    // A knob the server does not report is not a knob it will accept, so
    // re-asserting it is a config_update the far end drops on the floor.
    const { values, reassert } = reconcileConfig(
      { ...local, made_up: 3 }, ["made_up"], { scale_rotation: 1 });
    expect(values.made_up).toBe(3);
    expect(reassert).toEqual({});
  });

  it("changes nothing and re-asserts nothing without a config", () => {
    expect(reconcileConfig(local, ["scale_translation"], null))
      .toEqual({ values: { ...local }, reassert: {} });
  });

  it("ignores what is not a number — the wire carries the stance too", () => {
    const server = { stance: "behind", collision: true, lam_pos: null,
                     scale_rotation: Number.NaN, scale_translation: 1 };
    const { values } = reconcileConfig(local, [], server);
    expect(values.scale_translation).toBe(1);
    expect(values.scale_rotation).toBe(local.scale_rotation);
    expect(values).not.toHaveProperty("stance");
    expect(values).not.toHaveProperty("collision");
  });
});

describe("applyServerConfig", () => {
  const local = { scale_translation: 9, scale_rotation: 1.6 };

  it("lets the clamped echo win, even over a knob the operator moved", () => {
    // Whatever the robot took IS the value: a box that snaps to a different
    // number is the robot saying what it accepted, and re-asserting the
    // unclamped ask would fight it on every reconnect.
    expect(applyServerConfig(local, { scale_translation: 4 }).scale_translation)
      .toBe(4);
  });

  it("never touches a key the echo does not carry", () => {
    expect(applyServerConfig(local, { scale_translation: 4 }).scale_rotation)
      .toBe(1.6);
    expect(applyServerConfig(local, null)).toEqual(local);
    expect(applyServerConfig(local, { stance: "behind", collision: true }))
      .toEqual(local);
  });
});

// ---- the arm set ------------------------------------------------------------

// ---- the HUD, through a recording context ------------------------------------

describe("worstDropSource", () => {
  // Invariant 9 makes a degraded read a dropped frame, so a take that is
  // shedding rows has to say so while it is still being driven. "30% dropped"
  // is unactionable; "30% dropped, all of them the left wrist camera" sends
  // the operator to a cable.
  it("names the single heaviest source across cameras and arms", () => {
    expect(worstDropSource({
      cameras: { top: 3, left_wrist: 128 },
      arms: { left_arm: 12 },
    })).toBe("cam left_wrist");
    expect(worstDropSource({
      cameras: { top: 3 },
      arms: { left_arm: 240 },
    })).toBe("arm left_arm");
  });

  it("says which KIND, because a camera and an arm can share a name", () => {
    // The reason the recorder reports two maps rather than one: arms are
    // `left`/`right` and nothing stops a camera being named for a side. A bare
    // "left" would send the operator to the wrong end of the rig, confidently.
    expect(worstDropSource({ cameras: { left: 9 }, arms: { left: 4 } }))
      .toBe("cam left");
    expect(worstDropSource({ cameras: { left: 4 }, arms: { left: 9 } }))
      .toBe("arm left");
  });

  it("stays silent when nothing has dropped, which is the common case", () => {
    // A line painted every frame for a rig that is fine is a line the operator
    // stops reading.
    expect(worstDropSource(undefined)).toBeNull();
    expect(worstDropSource({})).toBeNull();
    expect(worstDropSource({ cameras: {}, arms: {} })).toBeNull();
    expect(worstDropSource({ cameras: { top: 0 } })).toBeNull();
  });

  it("reads a half-filled report rather than needing both maps", () => {
    expect(worstDropSource({ arms: { right_arm: 5 } })).toBe("arm right_arm");
    expect(worstDropSource({ cameras: { top: 5 } })).toBe("cam top");
  });
});
