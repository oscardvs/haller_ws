import { describe, it, expect } from "vitest";
import { fuseLandmarkResults, buildOverlaySides, type SideFrame } from "../lib/mediapipe";

const sample_pose_left_shoulder = { x: 0.5, y: 0.4, z: 0.0, visibility: 0.95 };
const sample_pose_left_elbow    = { x: 0.5, y: 0.5, z: 0.0, visibility: 0.93 };
const sample_pose_left_wrist    = { x: 0.5, y: 0.6, z: 0.0, visibility: 0.91 };

const sample_hand_landmarks = Array.from({ length: 21 }, (_, i) => ({
  x: i * 0.01, y: i * 0.01, z: 0.0, visibility: 0,
}));

describe("fuseLandmarkResults", () => {
  it("returns null sides when nothing is detected", () => {
    const out = fuseLandmarkResults(
      { worldLandmarks: [] },
      { worldLandmarks: [], handednesses: [] },
    );
    expect(out.left).toBeNull();
    expect(out.right).toBeNull();
  });

  it("builds a side from pose + hand for the left arm only", () => {
    // MediaPipe pose world-landmarks: an array of 33 points. We feed 33 entries
    // and the helper indexes by enum constants.
    const pose = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0, visibility: 0 }));
    pose[11] = sample_pose_left_shoulder;   // LEFT_SHOULDER
    pose[13] = sample_pose_left_elbow;      // LEFT_ELBOW
    pose[15] = sample_pose_left_wrist;      // LEFT_WRIST

    const out = fuseLandmarkResults(
      { worldLandmarks: [pose] },
      {
        worldLandmarks: [sample_hand_landmarks],
        handednesses: [[{ categoryName: "Left", score: 0.95, index: 0, displayName: "" }]],
      },
    );
    expect(out.left).not.toBeNull();
    expect(out.right).toBeNull();
    const left = out.left as SideFrame;
    expect(left.pose.shoulder).toEqual([
      sample_pose_left_shoulder.x,
      sample_pose_left_shoulder.y,
      sample_pose_left_shoulder.z,
    ]);
    expect(left.confidence).toBeGreaterThan(0);
  });
});

const norm_hand = Array.from({ length: 21 }, (_, i) => ({
  x: 0.10 + i * 0.01, y: 0.20 + i * 0.01, z: 0.0, visibility: 0,
}));

function norm_pose() {
  const p = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0, visibility: 0 }));
  p[11] = { x: 0.50, y: 0.40, z: 0, visibility: 0.95 };  // LEFT_SHOULDER
  p[13] = { x: 0.55, y: 0.50, z: 0, visibility: 0.93 };  // LEFT_ELBOW
  p[15] = { x: 0.60, y: 0.60, z: 0, visibility: 0.91 };  // LEFT_WRIST
  return p;
}

const NO_OPTS = { leftLost: false, rightLost: false, leftPinch01: 0.5, rightPinch01: 0.5 };

describe("buildOverlaySides", () => {
  it("returns null sides when nothing is detected", () => {
    const out = buildOverlaySides({ landmarks: [] }, { landmarks: [], handednesses: [] }, NO_OPTS);
    expect(out.left).toBeNull();
    expect(out.right).toBeNull();
  });

  it("returns null for a side with a pose but no matching hand", () => {
    const out = buildOverlaySides(
      { landmarks: [norm_pose()] },
      { landmarks: [], handednesses: [] },
      NO_OPTS,
    );
    expect(out.left).toBeNull();
    expect(out.right).toBeNull();
  });

  it("builds the left side with pose as shoulder,elbow,wrist in order", () => {
    const out = buildOverlaySides(
      { landmarks: [norm_pose()] },
      { landmarks: [norm_hand], handednesses: [[{ categoryName: "Left", score: 0.95, index: 0, displayName: "" }]] },
      NO_OPTS,
    );
    expect(out.right).toBeNull();
    expect(out.left!.pose).toEqual([[0.50, 0.40], [0.55, 0.50], [0.60, 0.60]]);
  });

  it("emits all six hand landmarks in the documented order, thumb_tip and index_tip first", () => {
    const out = buildOverlaySides(
      { landmarks: [norm_pose()] },
      { landmarks: [norm_hand], handednesses: [[{ categoryName: "Left", score: 0.95, index: 0, displayName: "" }]] },
      NO_OPTS,
    );
    // CameraOverlay draws the pinch line between hand[0] and hand[1].
    expect(out.left!.hand[0]).toEqual([norm_hand[4].x, norm_hand[4].y]);   // THUMB_TIP
    expect(out.left!.hand[1]).toEqual([norm_hand[8].x, norm_hand[8].y]);   // INDEX_TIP
    // Full order matters too: a later diagnostics task may rely on it positionally.
    expect(out.left!.hand).toEqual([
      [norm_hand[4].x,  norm_hand[4].y],   // thumb_tip
      [norm_hand[8].x,  norm_hand[8].y],   // index_tip
      [norm_hand[5].x,  norm_hand[5].y],   // index_mcp
      [norm_hand[9].x,  norm_hand[9].y],   // middle_mcp
      [norm_hand[17].x, norm_hand[17].y],  // pinky_mcp
      [norm_hand[0].x,  norm_hand[0].y],   // wrist
    ]);
  });

  it("passes the lost flag and pinch01 through per side", () => {
    const out = buildOverlaySides(
      { landmarks: [norm_pose()] },
      { landmarks: [norm_hand], handednesses: [[{ categoryName: "Left", score: 0.95, index: 0, displayName: "" }]] },
      { leftLost: true, rightLost: false, leftPinch01: 0.12, rightPinch01: 0.9 },
    );
    expect(out.left!.lost).toBe(true);
    expect(out.left!.pinch01).toBeCloseTo(0.12);
  });
});
