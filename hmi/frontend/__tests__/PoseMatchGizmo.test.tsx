import { render } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { PoseMatchGizmo } from "@/components/PoseMatchGizmo";

/** The dashed polyline is the robot; the solid one is the operator. */
function armPoints(container: HTMLElement, dashed: boolean): [number, number][] {
  const lines = Array.from(container.querySelectorAll("polyline"));
  const el = lines.find((l) =>
    dashed ? l.getAttribute("stroke-dasharray") : !l.getAttribute("stroke-dasharray"));
  if (!el) return [];
  return (el.getAttribute("points") ?? "").trim().split(/\s+/)
    .map((p) => p.split(",").map(Number) as [number, number]);
}

const spread = (pts: [number, number][]) =>
  Math.max(...pts.map(([x, y], _i, all) =>
    Math.max(...all.map(([x2, y2]) => Math.hypot(x - x2, y - y2)))));

describe("PoseMatchGizmo", () => {
  it("draws a visible arm for a pose pointing straight away from the lens", () => {
    // The regression this widget exists for. shoulder_pan/lift/elbow all zero
    // is the SO-101's rest pose and retargets to an arm along +Z — straight
    // away from the camera. Projected onto a front-facing mirror that is a
    // POINT, so the on-body ghost drew a dot on the shoulder and the operator
    // had nothing to match. An oblique view has to keep it visible.
    const { container } = render(
      <PoseMatchGizmo side="right" matched={false} operator={null}
                      robot={{ upper: [0, 0, 1], fore: [0, 0, 1] }} />,
    );
    expect(spread(armPoints(container, true))).toBeGreaterThan(20);
  });

  it("separates 'away from the lens' from 'out to the side'", () => {
    // If those two collapsed onto each other the view would be no better than
    // the orthographic one it replaces — the operator could not tell which
    // direction they were being asked for.
    const away = render(
      <PoseMatchGizmo side="right" matched={false} operator={null}
                      robot={{ upper: [0, 0, 1], fore: [0, 0, 1] }} />,
    );
    const side = render(
      <PoseMatchGizmo side="right" matched={false} operator={null}
                      robot={{ upper: [1, 0, 0], fore: [1, 0, 0] }} />,
    );
    const [, awayElbow] = armPoints(away.container, true);
    const [, sideElbow] = armPoints(side.container, true);
    expect(Math.hypot(awayElbow[0] - sideElbow[0], awayElbow[1] - sideElbow[1]))
      .toBeGreaterThan(15);
  });

  it("draws both arms through the same projection so they are comparable", () => {
    // Same direction in, same figure out — otherwise overlaying one on the
    // other would not mean the poses match.
    const arm = { upper: [0.3, 0.2, 0.93], fore: [0, 0.5, 0.87] };
    const { container } = render(
      <PoseMatchGizmo side="left" matched robot={arm} operator={arm} />,
    );
    expect(armPoints(container, true)).toEqual(armPoints(container, false));
  });

  it("omits an arm it has no pose for", () => {
    const { container } = render(
      <PoseMatchGizmo side="left" matched={false} robot={null} operator={null} />,
    );
    expect(container.querySelectorAll("polyline")).toHaveLength(0);
  });
});
