import { describe, expect, it } from "vitest";

import {
  applyServerConfig, describeArmSet, holdToggle, holdToggleInit, paintHud,
  MENU_MAX_CHARS, reconcileConfig, recorderHapticCue, RECORD_HOLD_MS,
  RESET_HOLD_MS, STATUS_MAX_CHARS, stepTake,
  TUNING_KNOBS, WRIST_PIVOT_KEY,
  type EndChoice, type RecorderHudLike, type TakeState,
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

/** Loaded and writing nothing — the resting state of a recording session. */
const armedRec: NonNullable<RecorderHudLike> = {
  state: "armed", recording: false, episode_frames: 0, episodes: 12,
};

/** Mid-take, and the same object the prompt paints from: the recorder is
 *  still rolling while the operator decides. */
const rollingRec: NonNullable<RecorderHudLike> = {
  state: "rolling", recording: true, episode_frames: 412, episodes: 12,
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
  it("climbs the ladder, and asks for exactly one REST call per rung", () => {
    const arm = stepTake("idle", { kind: "ax_hold" });
    expect(arm).toEqual({ state: "armed", act: { do: "arm" } });
    const roll = stepTake(arm.state, { kind: "ax_hold" });
    expect(roll).toEqual({ state: "rolling", act: { do: "roll" } });
    // Ending a take calls nothing: /record/stop carries the save decision, so
    // it cannot be sent until the operator has taken one.
    const prompt = stepTake(roll.state, { kind: "ax_hold" });
    expect(prompt).toEqual({ state: "prompt", act: null });
    // Withdrawing costs nothing either — the recorder never stopped.
    expect(stepTake(prompt.state, { kind: "ax_hold" }))
      .toEqual({ state: "rolling", act: null });
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
    let state = stepTake("idle", { kind: "ax_hold" }).state;   // the one climb
    for (let take = 0; take < 10; take++) {
      const roll = stepTake(state, { kind: "ax_hold" });
      expect(roll).toEqual({ state: "rolling", act: { do: "roll" } });
      const prompt = stepTake(roll.state, { kind: "ax_hold" });
      const done = stepTake(prompt.state, { kind: "choose", choice: "keep" });
      expect(done.act).toEqual({ do: "stop", save: true, rearm: true });
      stops.push({ save: true, rearm: true });
      state = done.state;
      seen.push(roll.state, prompt.state, state);
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
  it("keeps the whole ladder on the record hold the operator already has", () => {
    // A/X hold is the record command and stays it (port invariant 6). The gate
    // added a rung to that ladder rather than a second button, because there
    // is no spare one: every control on the pair is already spoken for.
    let hold = holdToggleInit();
    hold = holdToggle(hold, true, 0, RECORD_HOLD_MS);
    expect(holdToggle(hold, true, RECORD_HOLD_MS - 1, RECORD_HOLD_MS).toggled)
      .toBe(false);
    hold = holdToggle(hold, true, RECORD_HOLD_MS, RECORD_HOLD_MS);
    expect(hold.toggled).toBe(true);
    expect(stepTake("idle", { kind: "ax_hold" }).act).toEqual({ do: "arm" });
  });

  it("refuses home at the dwell it already means, rather than teaching a second", () => {
    // The left-stick hold during the prompt is the SAME physical action as
    // in-session home, at the same dwell — refused, because homing through the
    // tail of a take would corrupt one the operator may be about to keep. It
    // is answered rather than silently dropped, and it is not a decision.
    let hold = holdToggleInit();
    hold = holdToggle(hold, true, 0, RESET_HOLD_MS);
    expect(holdToggle(hold, true, RESET_HOLD_MS - 1, RESET_HOLD_MS).toggled)
      .toBe(false);
    expect(holdToggle(hold, true, RESET_HOLD_MS, RESET_HOLD_MS).toggled).toBe(true);
    const { ctx, text } = stubCtx();
    paintHud(ctx, {}, rollingRec, { ...menu, endPrompt: true, homeRefused: true });
    expect(text()).toContain("home refused mid-take");
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

describe("describeArmSet", () => {
  it("names both arms in the operator's terms", () => {
    // Behind the bench the hands cross: the left hand drives the arm named
    // "right". Reading that back is how a wrong stance becomes visible before
    // an arm moves rather than after.
    expect(describeArmSet({ left: "right", right: "left" }))
      .toBe("L→right · R→left");
  });

  it("omits a side that has no arm", () => {
    expect(describeArmSet({ left: "left", right: null })).toBe("L→left");
    expect(describeArmSet({ left: null, right: "right" })).toBe("R→right");
  });

  it("says none rather than an empty string when there is nothing paired", () => {
    expect(describeArmSet({ left: null, right: null })).toBe("none");
    expect(describeArmSet(null)).toBe("none");
  });
});

// ---- the HUD, through a recording context ------------------------------------

describe("paintHud and the start gate", () => {
  it("paints ARMED as its own thing, never as REC", () => {
    // The gate's whole benefit is knowing that nothing is being written; an
    // operator who cannot tell the two apart at a glance has its cost only.
    const { ctx, text } = stubCtx();
    paintHud(ctx, {}, armedRec, menu);
    expect(text()).toContain("◆ ARMED ep 12");
    expect(text()).not.toContain("● REC");
    expect(text()).toContain("nothing written yet");
  });

  it("says when the page is holding the gate itself", () => {
    // Against a backend with no /record/arm nothing is written before ROLL,
    // but the schema is not frozen, the 409s arrive late and the episode index
    // is a guess. Which of the two gates is in force has to be visible.
    const { ctx, text } = stubCtx();
    paintHud(ctx, {}, { ...armedRec, localGate: true }, menu);
    expect(text()).toContain("(local gate)");
  });

  it("paints the frame count once frames are landing", () => {
    const { ctx, text } = stubCtx();
    paintHud(ctx, {}, rollingRec, menu);
    expect(text()).toContain("● REC ep 12 · 412");
    expect(text()).toContain("A/X to END");
  });

  it("takes the whole box for the decision, and names both gestures", () => {
    const { ctx, text } = stubCtx();
    paintHud(ctx, {}, rollingRec, { ...menu, endPrompt: true });
    expect(text()).toContain("TAKE ENDED · 412 frames");
    expect(text()).toContain("L click = KEEP");
    expect(text()).toContain("R click = REDO");
    expect(text()).toContain("L = keep · R = redo");
    // The surprising part, said out loud: the recorder has not stopped, and
    // the tail of the episode is the operator holding still while they pick.
    expect(text()).toContain("still rolling until you pick");
    expect(text()).not.toContain("SIZE");
  });

  it("swaps the mnemonic for the refusal without moving a row", () => {
    // The refusal takes the mnemonic's row rather than adding one: a box that
    // changes height under the operator reads as a fault of its own.
    const plain = stubCtx();
    paintHud(plain.ctx, {}, rollingRec, { ...menu, endPrompt: true });
    const refused = stubCtx();
    paintHud(refused.ctx, {}, rollingRec,
             { ...menu, endPrompt: true, homeRefused: true });
    expect(refused.text()).toContain("home refused mid-take — pick first");
    expect(refused.text()).not.toContain("L = keep · R = redo");
    expect(refused.calls.map((c) => [c.x, c.y]))
      .toEqual(plain.calls.map((c) => [c.x, c.y]));
  });

  it("puts a sagging rate above every other complaint", () => {
    // One line, worst first: a health strip that lists everything is a strip
    // nobody reads while driving. 90% of declared is the backend's own refusal
    // threshold, so the HUD and the 409 tell one story.
    const { ctx, text } = stubCtx();
    paintHud(ctx, {}, { ...rollingRec, fpsMeasured: 21, fpsDeclared: 30,
                        worstDrop: "left_wrist", skipped_frames: 40 }, menu);
    expect(text()).toContain("RATE 21/30 fps");
    expect(text()).not.toContain("dropping: left_wrist");
    expect(text()).not.toContain("40 frames dropped");
  });

  it("names the one cable to go and check when the rate is fine", () => {
    const { ctx, text } = stubCtx();
    paintHud(ctx, {}, { ...rollingRec, fpsMeasured: 29, fpsDeclared: 30,
                        worstDrop: "left_wrist", skipped_frames: 40 }, menu);
    expect(text()).toContain("dropping: left_wrist");
    expect(text()).not.toContain("RATE");
  });

  it("counts shed rows while the take is still being driven", () => {
    // A degraded read is a dropped frame, not a recorded one (port invariant
    // 9), so a take shedding rows says so now — not in review.
    const { ctx, text } = stubCtx();
    paintHud(ctx, {}, { ...rollingRec, skipped_frames: 40 }, menu);
    expect(text()).toContain("40 frames dropped");
  });

  it("says why the gate dropped, and says it while idle", () => {
    // Not an error: it is the gate telling the operator why it un-armed. Idle
    // is exactly when they are standing there wondering why nothing is armed.
    const { ctx, text } = stubCtx();
    paintHud(ctx, {}, { state: "idle", recording: false, episode_frames: 0,
                        invalidatedReason: "arm set changed" }, menu);
    expect(text()).toContain("GATE DROPPED — arm set changed");
  });

  it("shows the arm set, and nothing at all when there is none to show", () => {
    // A session started on the wrong preset is otherwise discovered 60 s into
    // a take, when the arm that moves is not the one the hand meant.
    const { ctx, text } = stubCtx();
    paintHud(ctx, {}, armedRec, { ...menu, armSet: { left: "right", right: "left" } });
    expect(text()).toContain("ARMS  L→right · R→left");

    const unknown = stubCtx();
    paintHud(unknown.ctx, {}, armedRec, menu);
    expect(unknown.text()).not.toContain("ARMS");
  });

  it("marks the knobs the operator owns in the tuning list", () => {
    // A value that survives a reconnect while its neighbours revert is
    // otherwise witchcraft, and the marker is the only place it is explained.
    const { ctx, text } = stubCtx();
    paintHud(ctx, {}, null, { ...menu, tuning: {
      open: true, index: 0,
      values: { scale_translation: 2, scale_rotation: 1.6 },
      dirty: ["scale_translation"],
    } });
    expect(text()).toContain("◆ 2.000");
    expect(text()).toContain("1.600");
    expect(text()).not.toContain("◆ 1.600");
    expect(text()).toContain("◆ = yours");
  });

  it("never throws on a status the poll has not answered yet", () => {
    // This runs inside the XR loop, where a throw looks like the page dying
    // with a headset on and there is no console to read it from.
    const { ctx } = stubCtx();
    expect(() => paintHud(ctx, null, null, null)).not.toThrow();
    expect(() => paintHud(ctx, null, null, { ...menu, endPrompt: true }))
      .not.toThrow();
    expect(() => paintHud(ctx, {}, { recording: false, episode_frames: 0 }, menu))
      .not.toThrow();
    expect(() => paintHud(ctx, {}, {
      ...rollingRec, episodes: null, worstDrop: null, fpsMeasured: null,
      fpsDeclared: null, invalidatedReason: null,
    }, { ...menu, armSet: null, tuning: null })).not.toThrow();
  });
});

describe("every painted row fits the box it is painted in", () => {
  // The panel canvas is a fixed 1024 px wide, the status column is clipped at
  // 563 px and the menu box has 392 px of usable width. A row that exceeds its
  // column is not a layout wobble — the canvas clips it, so the END of the row
  // silently disappears, and it is the end of a row that carries the payload:
  // `... · B/Y = E-STOP` was being cut off the HUD entirely before this test
  // existed. Monospace advances at 0.6 em, so the width is computable and this
  // is checkable here rather than on someone's face in a headset.
  const ADVANCE = 0.6;
  const W = 1024;
  const leftW = Math.round(W * 0.55) - 24;
  const boxW = Math.round(W * 0.42);
  const boxX = W - boxW - 24;

  /** Every row paintHud drew, with the room it actually had. */
  function overflows(calls: ReturnType<typeof stubCtx>["calls"]) {
    const bad: string[] = [];
    for (const c of calls) {
      const px = Number(/(\d+)px/.exec(c.font)?.[1] ?? 22);
      const inMenu = c.x >= boxX;
      const left = inMenu ? boxX + 18 : 24;
      const right = inMenu ? boxX + boxW - 18 : leftW;
      const room = c.align === "right" ? c.x - left : right - c.x;
      if (c.text.length * px * ADVANCE > room) {
        bad.push(`${c.text}  (${c.text.length} chars @ ${px}px, ${room}px room)`);
      }
    }
    return bad;
  }

  const paints: [string, () => ReturnType<typeof stubCtx>][] = [
    ["idle", () => {
      const s = stubCtx();
      paintHud(s.ctx, driving, { recording: false, episode_frames: 0, episodes: 12 },
               { ...menu, stance: "behind", armSet: { left: "left_arm", right: "right_arm" } });
      return s;
    }],
    ["armed", () => {
      const s = stubCtx();
      paintHud(s.ctx, driving, armedRec,
               { ...menu, stance: "behind", armSet: { left: "left_arm", right: "right_arm" } });
      return s;
    }],
    ["armed, local gate", () => {
      const s = stubCtx();
      paintHud(s.ctx, driving, { ...armedRec, localGate: true }, menu);
      return s;
    }],
    ["rolling", () => {
      const s = stubCtx();
      paintHud(s.ctx, driving, rollingRec, { ...menu, stance: "front" });
      return s;
    }],
    ["prompt", () => {
      const s = stubCtx();
      paintHud(s.ctx, driving, { ...rollingRec, state: "prompt" },
               { ...menu, endPrompt: true });
      return s;
    }],
    ["prompt, home refused", () => {
      const s = stubCtx();
      paintHud(s.ctx, driving, { ...rollingRec, state: "prompt" },
               { ...menu, endPrompt: true, homeRefused: true });
      return s;
    }],
    ["tuning", () => {
      const s = stubCtx();
      paintHud(s.ctx, driving, null, {
        ...menu,
        tuning: { open: true, index: 2, values: { scale_translation: 1.6 },
                  dirty: ["scale_translation"] },
      });
      return s;
    }],
    ["precision", () => {
      const s = stubCtx();
      paintHud(s.ctx, driving, null, { ...menu, precision: true });
      return s;
    }],
    ["a wrist out of twist", () => {
      const s = stubCtx();
      paintHud(s.ctx, driving, null, menu, { left: { driving: true, orient_residual: 0.9 } });
      return s;
    }],
    ["a collision hold", () => {
      const s = stubCtx();
      paintHud(s.ctx, { ...driving, collision: { enabled: true, slack_m: -0.004, limited: true } },
               rollingRec, menu);
      return s;
    }],
    ["a sagging rate", () => {
      const s = stubCtx();
      paintHud(s.ctx, driving,
               { ...rollingRec, fpsMeasured: 21.7, fpsDeclared: 30 }, menu);
      return s;
    }],
    ["a drop source", () => {
      const s = stubCtx();
      paintHud(s.ctx, driving,
               { ...rollingRec, worstDrop: "left_wrist", skipped_frames: 128 }, menu);
      return s;
    }],
    ["skipped frames", () => {
      const s = stubCtx();
      paintHud(s.ctx, driving, { ...rollingRec, skipped_frames: 1284 }, menu);
      return s;
    }],
    ["a hand that stopped tracking mid-countdown", () => {
      // The widest this row ever gets. At the column's 28px it needed 487px
      // against 419px of room, so the operator was being told a hand had
      // stopped tracking in a sentence that ran off the panel.
      const s = stubCtx();
      paintHud(s.ctx, {
        state: "acquiring",
        acquire: {
          left: { authority: "acquiring", remaining_ms: 1200, reason: "no_tracking" },
          right: { authority: "acquiring", remaining_ms: 1200, reason: "no_tracking" },
        },
      }, null, menu);
      return s;
    }],
    ["a solo session's absent side", () => {
      const s = stubCtx();
      paintHud(s.ctx, {
        state: "driving",
        acquire: {
          left: { authority: "held", remaining_ms: null, reason: "no_arm" },
          right: { authority: "driving", remaining_ms: null, reason: "" },
        },
      }, null, { ...menu, armSet: { left: null, right: "so101_right" } },
      { right: { driving: true, sigma_min: 0.0312 } });
      return s;
    }],
    ["a long error the backend handed up", () => {
      const s = stubCtx();
      paintHud(s.ctx, {
        state: "fault",
        last_error: "shoulder_pan overloaded and dropped torque mid-sweep; "
          + "four joints left energised, see /estop",
      }, null, menu);
      return s;
    }],
    ["no cameras at all", () => {
      const s = stubCtx();
      paintHud(s.ctx, driving, null, { ...menu, views: [], activeViewId: null });
      return s;
    }],
    ["camera labels and arm ids longer than the box", () => {
      const s = stubCtx();
      paintHud(s.ctx, driving, null, {
        ...menu,
        views: [{ id: "a", label: "over the shoulder threequarter (right_arm)" },
                { id: "b", label: "left wrist gripper close-up (left_arm)" }],
        activeViewId: "a",
        armSet: { left: "so101_left_arm_lower_bench",
                  right: "so101_right_arm_upper_bench" },
      });
      return s;
    }],
    ["a gate that dropped", () => {
      const s = stubCtx();
      paintHud(s.ctx, driving, {
        recording: false, episode_frames: 0,
        invalidatedReason: "arm set changed",
      }, menu);
      return s;
    }],
  ];

  for (const [name, paint] of paints) {
    it(`fits: ${name}`, () => {
      expect(overflows(paint().calls)).toEqual([]);
    });
  }

  it("keeps the E-STOP binding inside the status column", () => {
    // The specific regression: this row is the only place in the headset the
    // operator can read what B/Y does, and E-STOP leads it so that even a
    // clip cannot eat the part that matters.
    const { ctx, calls } = stubCtx();
    paintHud(ctx, {}, null, menu);
    const hint = calls.find((c) => c.text.includes("E-STOP"));
    expect(hint).toBeDefined();
    expect(hint!.text.startsWith("B/Y = E-STOP")).toBe(true);
    expect(hint!.text.length).toBeLessThanOrEqual(STATUS_MAX_CHARS);
  });

  it("states the budgets it is checking, so a change to one is deliberate", () => {
    expect(MENU_MAX_CHARS).toBe(36);
    expect(STATUS_MAX_CHARS).toBe(39);
  });
});

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

describe("the rate gate the HUD warns at", () => {
  const at = (measured: number, rateGate?: number) => {
    const { ctx, text } = stubCtx();
    paintHud(ctx, driving, {
      ...rollingRec, fpsMeasured: measured, fpsDeclared: 30, rateGate,
    }, menu);
    return text();
  };

  it("warns at the threshold the RECORDER published, not a copy of it", () => {
    // The recorder refuses a take below its own gate. A HUD holding a second
    // copy drifts from the 409 and the two tell the operator different
    // stories — so a backend running a stricter gate must move this line too.
    expect(at(28, 0.95)).toContain("RATE 28/30");
    expect(at(28, 0.9)).not.toContain("RATE");
  });

  it("falls back to 0.9 for a backend that publishes no gate", () => {
    expect(at(26)).toContain("RATE 26/30");
    expect(at(28)).not.toContain("RATE");
  });
});
