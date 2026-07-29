"use client";

/**
 * Two arms in one synthetic view: where the robot is, and where you are.
 *
 * WHY THIS EXISTS ALONGSIDE THE ON-BODY GHOST
 * -------------------------------------------
 * The camera overlay draws the robot's pose orthographically onto a
 * front-facing mirror, which is exactly right for a pose with a sideways
 * component and useless for one without. An arm pointing straight away from
 * the lens projects to a POINT — and `shoulder_pan 0, shoulder_lift 0,
 * elbow_flex 0`, the SO-101's rest pose and the one an operator meets first,
 * is precisely that arm. The overlay was drawing a correct dot on the
 * shoulder and the operator had nothing to match.
 *
 * The fix is not a better overlay, it is a different camera. Both arms are
 * rendered through one fixed three-quarter view, so:
 *   - no pose the operator is likely to rest in collapses, and
 *   - the two figures are directly comparable, because whatever the
 *     projection distorts it distorts identically for both.
 *
 * The operator turns their arm until the solid figure sits on the dashed one.
 * Depth stops being ambiguous because the view is oblique: pointing away from
 * the lens and pointing sideways are different directions here.
 *
 * Only shoulder_pan / shoulder_lift / elbow_flex are geometry, so only those
 * three are drawn. wrist_flex, wrist_roll and the gripper have no limb to
 * show; the per-joint error column beside the scope bars carries those.
 */

/** Yaw about the vertical, then pitch. Chosen so that "straight at the lens",
 *  "straight away" and "straight out to the side" are three visibly different
 *  directions rather than two aliases and a dot. */
const YAW = (35 * Math.PI) / 180;
const PITCH = (25 * Math.PI) / 180;

const UPPER_PX = 30;
const FORE_PX = 26;

/** World is MediaPipe pose-world: +X image-right, +Y DOWN, +Z away from the
 *  lens. SVG y also runs down, so the vertical axis needs no flip. */
function project(v: number[] | undefined): [number, number] {
  const [x, y, z] = [v?.[0] ?? 0, v?.[1] ?? 0, v?.[2] ?? 0];
  const x1 = x * Math.cos(YAW) + z * Math.sin(YAW);
  const z1 = -x * Math.sin(YAW) + z * Math.cos(YAW);
  const y2 = y * Math.cos(PITCH) - z1 * Math.sin(PITCH);
  return [x1, y2];
}

export type ArmDirections = { upper: number[]; fore: number[] };

function points(arm: ArmDirections, cx: number, cy: number): string {
  const [ux, uy] = project(arm.upper);
  const [fx, fy] = project(arm.fore);
  const ex = cx + ux * UPPER_PX;
  const ey = cy + uy * UPPER_PX;
  return `${cx},${cy} ${ex},${ey} ${ex + fx * FORE_PX},${ey + fy * FORE_PX}`;
}

export function PoseMatchGizmo({
  side, robot, operator, matched, authority,
}: {
  side: "left" | "right";
  /** The arm's current pose, from the backend. */
  robot: ArmDirections | null;
  /** The operator's own arm, from this frame's world landmarks. */
  operator: ArmDirections | null;
  matched: boolean;
  authority?: string;
}) {
  const W = 108, H = 96;
  const cx = W / 2, cy = 30;
  const live = "var(--haller-live, oklch(80% 0.18 142))";
  const warn = "var(--haller-warn, oklch(75% 0.16 70))";
  const ghostColour = matched ? live : "oklch(72% 0.02 250)";

  return (
    <div
      className="rounded-sm px-1 pb-0.5"
      style={{ background: "color-mix(in oklab, var(--card) 88%, transparent)" }}
    >
      <div className="flex items-center justify-between px-1 pt-0.5 font-mono text-[9px] text-muted-foreground">
        <span>{side}</span>
        <span style={{ color: matched ? live : warn }}>
          {authority === "driving" ? "live" : matched ? "matched" : "match"}
        </span>
      </div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden>
        {/* Ground plane and vertical, so "away from the lens" and "downward"
            are readable as directions rather than as arbitrary angles. */}
        <g stroke="var(--border)" strokeWidth="1" opacity="0.7">
          <line x1={cx} y1={cy} x2={cx + Math.cos(YAW) * 26}
                y2={cy + Math.sin(YAW) * Math.sin(PITCH) * 26} />
          <line x1={cx} y1={cy} x2={cx + Math.sin(YAW) * 26}
                y2={cy - Math.cos(YAW) * Math.sin(PITCH) * 26} />
          <line x1={cx} y1={cy} x2={cx} y2={cy + 26} />
        </g>
        {robot ? (
          <polyline
            points={points(robot, cx, cy)}
            fill="none" stroke={ghostColour} strokeWidth="4"
            strokeDasharray="5 4" strokeLinecap="round" strokeLinejoin="round"
            opacity="0.85"
          />
        ) : null}
        {operator ? (
          <polyline
            points={points(operator, cx, cy)}
            fill="none" stroke={matched ? live : warn} strokeWidth="2"
            strokeLinecap="round" strokeLinejoin="round"
          />
        ) : null}
        <circle cx={cx} cy={cy} r="2.5" fill="var(--muted-foreground)" />
      </svg>
    </div>
  );
}
