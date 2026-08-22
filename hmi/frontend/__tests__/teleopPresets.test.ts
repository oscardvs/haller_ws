// hmi/frontend/__tests__/teleopPresets.test.ts
//
// Which hand drives which arm. This is the one decision in the launcher whose
// mistakes are invisible on screen and obvious the moment an arm moves, so it
// is pinned here rather than read off a rendered button.
import { describe, it, expect, beforeEach } from "vitest";

import {
  pairingFor, readStance, writeStance, isStance, STANCE_LS_KEY,
} from "../lib/stance";
import {
  describePairing, isSimArm, presetsFor, simLeaderFor,
} from "../components/cockpit/teleopPresets";
import { jointRows } from "../components/cockpit/TeleopTab";
import type { TelemetryFrame } from "../lib/telemetry";

/** Declaration order as /config reports it. */
const ARMS = ["left", "right"];

describe("pairingFor", () => {
  it("takes the declared pair in reverse when the operator stands behind", () => {
    // Standing behind, the operator faces the way the arms reach: the arm
    // under their right hand is the one on frame-right of the over-shoulder
    // view, which is the first-declared arm.
    expect(pairingFor("behind", ARMS)).toEqual({
      left_arm: "right", right_arm: "left",
    });
  });

  it("takes it as declared face to face", () => {
    for (const stance of ["mirror", "front"] as const) {
      expect(pairingFor(stance, ARMS)).toEqual({
        left_arm: "left", right_arm: "right",
      });
    }
  });

  it("puts a solo arm on the hand the same rule picks, and null on the other", () => {
    expect(pairingFor("behind", ARMS, "right")).toEqual({
      left_arm: "right", right_arm: null,
    });
    expect(pairingFor("front", ARMS, "right")).toEqual({
      left_arm: null, right_arm: "right",
    });
  });

  it("never invents an arm the rig does not have", () => {
    // A one-arm rig has no second side. `undefined` on the wire would be an
    // absent key, which is not the same message as "this side drives nothing".
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
