// hmi/frontend/__tests__/cockpitLib.test.ts
//
// The three cockpit rules that are safety- or data-relevant rather than
// cosmetic, pinned as pure functions so they can be tested without standing up
// a webcam, a websocket and six tabs.
import { describe, it, expect } from "vitest";

import { isEditableTarget } from "../lib/keys";
import { shouldKeepTeleopMounted } from "../components/cockpit/lib";
import { gridPlan } from "../components/cockpit/cameraGrid";
import type { CameraInfo } from "../lib/api";

function cam(id: string, role: "base" | "wrist"): CameraInfo {
  return {
    id, role, source: "opencv", active: true, width: 640, height: 480, fps: 30,
  };
}

describe("isEditableTarget — WASD must not leak into text fields", () => {
  const editable = (tag: string) => isEditableTarget(document.createElement(tag));

  it("treats text-entry elements as editable", () => {
    expect(editable("input")).toBe(true);
    expect(editable("textarea")).toBe(true);
    // A focused <select> consumes letter keys for type-ahead — the teleop
    // tab's arm-assignment dropdowns are exactly that.
    expect(editable("select")).toBe(true);
  });

  it("leaves ordinary elements drivable", () => {
    expect(editable("button")).toBe(false);
    expect(editable("div")).toBe(false);
  });

  it("treats contenteditable as editable", () => {
    const el = document.createElement("div");
    el.contentEditable = "true";
    // jsdom does not derive isContentEditable from the attribute.
    Object.defineProperty(el, "isContentEditable", { value: true });
    expect(isEditableTarget(el)).toBe(true);
  });

  it("is false for a null target, so a key with no target still drives", () => {
    // Key-UP is dispatched without a target in some paths; treating that as
    // "editable" would swallow the release and latch the drive on.
    expect(isEditableTarget(null)).toBe(false);
  });
});

describe("shouldKeepTeleopMounted", () => {
  it("does not mount the panel before the operator has ever opened the tab", () => {
    // The webcam is not opened speculatively.
    expect(
      shouldKeepTeleopMounted({ opened: false, onTeleopTab: false, sessionRunning: true }),
    ).toBe(false);
  });

  it("stays mounted off-tab while a session is running", () => {
    // The publish loop IS the teleop input; unmounting stops the robot
    // receiving poses while the backend still thinks a session is live.
    expect(
      shouldKeepTeleopMounted({ opened: true, onTeleopTab: false, sessionRunning: true }),
    ).toBe(true);
  });

  it("releases the camera when leaving the tab with nothing running", () => {
    expect(
      shouldKeepTeleopMounted({ opened: true, onTeleopTab: false, sessionRunning: false }),
    ).toBe(false);
  });

  it("is mounted on the tab even with no session", () => {
    expect(
      shouldKeepTeleopMounted({ opened: true, onTeleopTab: true, sessionRunning: false }),
    ).toBe(true);
  });
});

describe("gridPlan", () => {
  it("gives a lone camera the full width", () => {
    const plan = gridPlan([cam("base_front", "base")]);
    expect(plan.columns).toBe(1);
    expect(plan.span(cam("base_front", "base"))).toBe(1);
  });

  it("splits two cameras evenly rather than widening the base one", () => {
    // Both sim configs report two role:base cameras. The design's fixed
    // 3-column grid with base spanning 2 left a third of the screen empty and
    // stacked them vertically.
    const cams = [cam("threequarter_sim", "base"), cam("overhead_sim", "base")];
    const plan = gridPlan(cams);
    expect(plan.columns).toBe(2);
    expect(cams.map(plan.span)).toEqual([1, 1]);
  });

  it("widens base cameras once there are enough tiles to spare the space", () => {
    const cams = [
      cam("base_front", "base"),
      cam("wrist_right", "wrist"),
      cam("wrist_left", "wrist"),
    ];
    const plan = gridPlan(cams);
    expect(plan.columns).toBe(3);
    expect(cams.map(plan.span)).toEqual([2, 1, 1]);
  });

  it("never asks for more than three columns", () => {
    const cams = Array.from({ length: 7 }, (_, i) => cam(`c${i}`, "wrist"));
    expect(gridPlan(cams).columns).toBe(3);
  });
});
