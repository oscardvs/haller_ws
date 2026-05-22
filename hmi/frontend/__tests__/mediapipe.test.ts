import { describe, it, expect } from "vitest";
import { fuseLandmarkResults, type SideFrame } from "../lib/mediapipe";

const sample_pose_left_shoulder = { x: 0.5, y: 0.4, z: 0.0, visibility: 0.95 };
const sample_pose_left_elbow    = { x: 0.5, y: 0.5, z: 0.0, visibility: 0.93 };
const sample_pose_left_wrist    = { x: 0.5, y: 0.6, z: 0.0, visibility: 0.91 };

const sample_hand_landmarks = Array.from({ length: 21 }, (_, i) => ({
  x: i * 0.01, y: i * 0.01, z: 0.0,
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
        handednesses: [[{ categoryName: "Left", score: 0.95 }]],
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
