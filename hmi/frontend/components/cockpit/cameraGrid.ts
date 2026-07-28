"use client";

/**
 * How to lay out N camera tiles in the space available.
 *
 * The v3 design drew a fixed 3×2 grid with base-role cameras spanning two
 * columns, which is right for the hardware config it was drawn against
 * (one base camera, one wrist, one reserved slot). It is wrong for the sim
 * configs, where BOTH cameras are role: base — every tile spans two of three
 * columns, so the grid stacks vertically and leaves a third of the screen
 * empty. It is also wrong for a single-camera rig.
 *
 * So the column count follows the camera count, and the base-camera double
 * width only applies once there are enough tiles for it to buy anything.
 */
import type { CameraInfo } from "@/lib/api";

export type CameraGridPlan = {
  columns: number;
  /** Column span for one camera. */
  span: (c: CameraInfo) => number;
};

export function gridPlan(cameras: CameraInfo[]): CameraGridPlan {
  const columns = Math.min(Math.max(cameras.length, 1), 3);
  // Below three tiles, everything already gets at least half the width;
  // widening one would only starve the other.
  const wideBase = cameras.length >= 3;
  return {
    columns,
    span: (c) => (wideBase && c.role === "base" ? 2 : 1),
  };
}
