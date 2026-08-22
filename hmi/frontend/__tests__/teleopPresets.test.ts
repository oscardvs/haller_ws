// hmi/frontend/__tests__/teleopPresets.test.ts
//
// Which hand drives which arm. This is the one decision in the launcher whose
// mistakes are invisible on screen and obvious the moment an arm moves, so it
// is pinned here rather than read off a rendered button.
import { describe, it, expect, beforeEach } from "vitest";

import {
  pairingFor, sidesOf, readStance, writeStance, isStance, STANCE_LS_KEY,
} from "../lib/stance";
import {
  describePairing, isSimArm, presetsFor, simLeaderFor,
} from "../components/cockpit/teleopPresets";
import { jointRows } from "../components/cockpit/TeleopTab";
import type { TelemetryFrame } from "../lib/telemetry";

/** The sim configs declare left first. */
const ARMS = ["left", "right"];
/** config.yaml declares the SAME two arms the other way round. The pairing
 *  must not notice — that is the whole point of the identity rule. */
const ARMS_REVERSED = ["right", "left"];

describe("sidesOf", () => {
  it("reads the side off the id, not off the declaration order", () => {
    expect(sidesOf(ARMS)).toEqual({ robotLeft: "left", robotRight: "right" });
    expect(sidesOf(ARMS_REVERSED)).toEqual({
      robotLeft: "left", robotRight: "right",
    });
  });

  it("places ids that name no side by declaration order, left slot first", () => {
    expect(sidesOf(["armA", "armB"])).toEqual({
      robotLeft: "armA", robotRight: "armB",
    });
  });

  it("leaves the missing side null on a one-arm rig", () => {
    // config.solo-real.yaml is `left` alone; config.solo-sim.yaml is `right`
    // alone. Neither may be silently promoted into the other's slot.
    expect(sidesOf(["left"])).toEqual({ robotLeft: "left", robotRight: null });
    expect(sidesOf(["right"])).toEqual({ robotLeft: null, robotRight: "right" });
  });
});

describe("pairingFor", () => {
  it("crosses the sides when the operator stands behind the arms", () => {
    // Behind, the operator faces the way the arms reach, so the robot's LEFT
    // arm is the one on their right.
    expect(pairingFor("behind", ARMS)).toEqual({
      left_arm: "right", right_arm: "left",
    });
  });

  it("lines the sides up directly when facing the arms", () => {
    for (const stance of ["mirror", "front"] as const) {
      expect(pairingFor(stance, ARMS)).toEqual({
        left_arm: "left", right_arm: "right",
      });
    }
  });

  it("gives the same answer whichever order the config declares the arms", () => {
    // The bug this rule replaces: config.yaml declares [right, left] and the
    // sim configs declare [left, right], so a positional rule made "behind"
    // mean opposite things on the two — the controls read as inverted on
    // exactly one of them.
    for (const stance of ["behind", "mirror", "front"] as const) {
      expect(pairingFor(stance, ARMS_REVERSED)).toEqual(
        pairingFor(stance, ARMS),
      );
    }
  });

  it("puts a solo arm on the hand the dual session would have used", () => {
    // The property: "my right hand drives the arm I picked" means the same
    // thing with one arm in the session as with two.
    for (const stance of ["behind", "mirror", "front"] as const) {
      const dual = pairingFor(stance, ARMS);
      for (const arm of ARMS) {
        expect(pairingFor(stance, ARMS, arm)).toEqual({
          left_arm: dual.left_arm === arm ? arm : null,
          right_arm: dual.right_arm === arm ? arm : null,
        });
      }
    }
  });

  it("drives the robot's left arm with the RIGHT hand in the behind stance", () => {
    // Spelled out because this is the case that flipped: a solo session on
    // `left` used to land on the left hand regardless of which arm it was.
    expect(pairingFor("behind", ["left"], "left")).toEqual({
      left_arm: null, right_arm: "left",
    });
    expect(pairingFor("front", ["left"], "left")).toEqual({
      left_arm: "left", right_arm: null,
    });
  });

  it("falls back to declaration order for ids that name no side", () => {
    expect(pairingFor("behind", ["armA", "armB"])).toEqual({
      left_arm: "armB", right_arm: "armA",
    });
    expect(pairingFor("front", ["armA", "armB"])).toEqual({
      left_arm: "armA", right_arm: "armB",
    });
  });

  it("never sends two nulls, which the backend refuses", () => {
    // An arm outside the resolved pair still has to produce a startable body.
    const odd = pairingFor("behind", ["left", "right"], "spare");
    expect([odd.left_arm, odd.right_arm].filter(Boolean)).toEqual(["spare"]);
  });

  it("never invents an arm the rig does not have", () => {
    expect(pairingFor("behind", ["left"])).toEqual({
      left_arm: null, right_arm: "left",
    });
    expect(pairingFor("behind", [])).toEqual({
      left_arm: null, right_arm: null,
    });
  });
});

describe("stance persistence", () => {
  beforeEach(() => localStorage.clear());

  it("round-trips a chosen stance", () => {
    writeStance("mirror");
    expect(readStance()).toBe("mirror");
  });

  it("falls back to behind, which is also the backend's default", () => {
    // A frame that carries no stance is read as "behind" server-side. If the
    // two disagreed, an unset preference would mean one thing here and
    // another on the robot.
    expect(readStance()).toBe("behind");
    localStorage.setItem(STANCE_LS_KEY, "sideways");
    expect(readStance()).toBe("behind");
  });

  it("recognises exactly the three stances", () => {
    expect(["behind", "mirror", "front"].every(isStance)).toBe(true);
    expect(isStance("sideways")).toBe(false);
    expect(isStance(null)).toBe(false);
  });
});

describe("presetsFor", () => {
  it("offers dual plus one solo per configured arm", () => {
    const ids = presetsFor(ARMS, "behind").map((p) => p.id);
    expect(ids).toEqual(["dual", "solo-left", "solo-right"]);
  });

  it("offers dual disabled, with the reason, on a one-arm rig", () => {
    // Shown rather than hidden: "there is no second arm" is a fact about the
    // robot, and a picker that quietly drops the option makes the operator
    // doubt their memory.
    const [dual, ...solos] = presetsFor(["left"], "behind");
    expect(dual.unavailable).toMatch(/2 enabled arms/);
    expect(solos.map((p) => p.id)).toEqual(["solo-left"]);
    expect(solos[0].unavailable).toBeNull();
  });

  it("re-derives the pairing when the stance changes", () => {
    const behind = presetsFor(ARMS, "behind")[0];
    const front = presetsFor(ARMS, "front")[0];
    expect(behind.pairing).not.toEqual(front.pairing);
    expect(behind.detail).toBe("L hand → right · R hand → left");
    expect(front.detail).toBe("L hand → left · R hand → right");
  });

  it("describes exactly the pairing it would post", () => {
    // The button text and the start body must come from one object: a preset
    // that says one mapping and posts another is invisible until an arm moves.
    for (const p of presetsFor(ARMS_REVERSED, "behind")) {
      expect(p.detail).toBe(describePairing(p.pairing));
    }
  });

  it("offers the same sessions whichever order the config declares the arms", () => {
    const byId = (ps: ReturnType<typeof presetsFor>) =>
      Object.fromEntries(ps.map((p) => [p.id, p.pairing]));
    expect(byId(presetsFor(ARMS_REVERSED, "behind")))
      .toEqual(byId(presetsFor(ARMS, "behind")));
  });

  it("names only the hand that drives something on a solo preset", () => {
    expect(describePairing({ left_arm: "left", right_arm: null }))
      .toBe("L hand → left");
    expect(describePairing({ left_arm: null, right_arm: null })).toBe("no arm");
  });
});

describe("simLeaderFor", () => {
  const real = (id: string) => ({ id, port: `/dev/haller_arm_${id}` });
  const sim = (id: string) => ({ id, port: "(sim)" });

  it("is null on an all-real rig — there is no viewer to drag", () => {
    expect(simLeaderFor([real("left"), real("right")])).toBeNull();
  });

  it("leads with the sim arm on a hybrid rig", () => {
    // config.hybrid-real-sim.yaml is exactly this shape: one real arm, one
    // sim arm. Only the sim one can be dragged in the MuJoCo viewer.
    expect(simLeaderFor([real("left"), sim("right")]))
      .toEqual({ leader: "right", follower: "left" });
  });

  it("is null with a single arm — nothing left to follow", () => {
    expect(simLeaderFor([sim("right")])).toBeNull();
  });

  it("reads sim-ness off the port /config actually reports", () => {
    expect(isSimArm(sim("left"))).toBe(true);
    expect(isSimArm(real("left"))).toBe(false);
  });
});

describe("jointRows", () => {
  const frame = (
    goals: Record<string, number>,
    measured: Record<string, number>,
  ): TelemetryFrame => ({
    t: 1,
    base: { linear: 0, angular: 0, odom: { x: 0, y: 0, yaw: 0 }, scan_min_range: null },
    arms: {
      left: {
        mode: "auto",
        joints: Object.fromEntries(
          Object.entries(measured).map(([k, pos]) => [
            k, { pos, min: -180, max: 180, torque: true },
          ]),
        ),
      },
    },
    alerts: [],
    human_teleop: {
      running: true, state: "driving", left_arm: "left", right_arm: null,
      started_at: 0, last_error: null,
      tracking: { left: { age_ms: 10, lost: false }, right: { age_ms: null, lost: true } },
      goal_deg: { left: goals },
    },
  });

  it("pairs each goal with its measured angle and the gap between them", () => {
    const rows = jointRows(frame({ elbow_flex: 12.5 }, { elbow_flex: 10 }), "left", "left");
    expect(rows).toBe("elbow_flex\t12.5\t10.0\t2.5");
  });

  it("is empty for a side that is not in the session", () => {
    // A solo session's absent side must not render a table of em-dashes; the
    // panel says "not in session" instead.
    expect(jointRows(frame({}, { elbow_flex: 10 }), null, "right")).toBe("");
    expect(jointRows(null, "left", "left")).toBe("");
  });

  it("shows a goal for a joint the arm does not report", () => {
    // The two disagreeing about what this arm has is worth seeing, not
    // silently dropping.
    const rows = jointRows(frame({ gripper: 40 }, { elbow_flex: 10 }), "left", "left");
    expect(rows.split("\n")).toEqual([
      "elbow_flex\t—\t10.0\t—",
      "gripper\t40.0\t—\t—",
    ]);
  });
});
