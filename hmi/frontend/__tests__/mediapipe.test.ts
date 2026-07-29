import { describe, it, expect, vi } from "vitest";
import { fuseLandmarkResults, buildOverlaySides, buildGhostSides, extractJawOpen, JawTraceRecorder, FACE_EVERY_N, MediaPipeRunner, type SideFrame } from "@/lib/mediapipe";
import { FaceLandmarker } from "@mediapipe/tasks-vision";

// Real HandLandmarker/PoseLandmarker, a FaceLandmarker that always rejects its
// GPU delegate — exactly the failure mode that took down the pre-existing
// hand/pose spacebar path when the face model was loaded eagerly and
// unguarded inside MediaPipeRunner.load(). See lib/mediapipe.ts's class doc.
vi.mock("@mediapipe/tasks-vision", () => {
  class FakeHandLandmarker {
    detectForVideo() { return { worldLandmarks: [], landmarks: [], handednesses: [] }; }
    close() { /* noop */ }
  }
  class FakePoseLandmarker {
    detectForVideo() { return { worldLandmarks: [], landmarks: [] }; }
    close() { /* noop */ }
  }
  return {
    FilesetResolver: { forVisionTasks: vi.fn().mockResolvedValue({}) },
    HandLandmarker: { createFromOptions: vi.fn().mockResolvedValue(new FakeHandLandmarker()) },
    PoseLandmarker: { createFromOptions: vi.fn().mockResolvedValue(new FakePoseLandmarker()) },
    FaceLandmarker: {
      createFromOptions: vi.fn().mockRejectedValue(new Error("gpu delegate init failed")),
    },
  };
});

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

describe("extractJawOpen", () => {
  it("returns the jawOpen score from the blendshape categories", () => {
    const result = {
      faceBlendshapes: [{
        categories: [
          { categoryName: "eyeBlinkLeft", score: 0.02 },
          { categoryName: "jawOpen", score: 0.73 },
          { categoryName: "mouthSmile", score: 0.11 },
        ],
      }],
    };
    expect(extractJawOpen(result as never)).toBeCloseTo(0.73);
  });

  it("returns null when no face was detected", () => {
    expect(extractJawOpen(null)).toBeNull();
    expect(extractJawOpen(undefined)).toBeNull();
    expect(extractJawOpen({ faceBlendshapes: [] } as never)).toBeNull();
  });

  it("returns null when jawOpen is absent from the categories", () => {
    const result = {
      faceBlendshapes: [{ categories: [{ categoryName: "mouthSmile", score: 0.4 }] }],
    };
    expect(extractJawOpen(result as never)).toBeNull();
  });

  it("pins FACE_EVERY_N to 3", () => {
    // Task 6's panel tick math and the backend's 250ms staleness budget
    // both depend on this exact value — see lib/mediapipe.ts's doc comment.
    expect(FACE_EVERY_N).toBe(3);
  });
});

describe("MediaPipeRunner: a failing face model must not take hand/pose down with it", () => {
  it("load() resolves, and detect() stays usable, when the face model rejects", async () => {
    const runner = new MediaPipeRunner();
    // Regression guard: this used to be `Promise.all`-equivalent (three
    // unconditional awaits including FaceLandmarker), so a GPU delegate
    // failure on the face model rejected load() itself — which killed the
    // keypoint WebSocket (created only after load() resolved) and, with it,
    // hand/pose tracking that had already succeeded.
    await expect(runner.load()).resolves.toBeUndefined();

    const ok = await runner.loadFace();
    expect(ok).toBe(false);

    // detect() must not throw even when face:true is requested — it degrades
    // to face: null, exactly like a decimated tick, rather than crashing the
    // tracking loop.
    const result = runner.detect({} as HTMLVideoElement, 0, { face: true });
    expect(result.face).toBeNull();
    expect(result.hands).toBeDefined();
    expect(result.pose).toBeDefined();
  });

  it("loadFace() is idempotent after a failure — it does not retry every tick", async () => {
    const createFromOptions = vi.mocked(FaceLandmarker.createFromOptions);
    createFromOptions.mockClear();
    const runner = new MediaPipeRunner();
    await runner.load();
    const first = await runner.loadFace();
    const second = await runner.loadFace();
    expect(first).toBe(false);
    expect(second).toBe(false);
    // The panel calls loadFace() on every face tick (~every 100ms) while
    // mouth mode is selected; a real retry-every-call would hammer the GPU
    // delegate constructor at that rate for a model that already failed.
    expect(createFromOptions).toHaveBeenCalledTimes(1);
  });
});

// ---- ghost projection --------------------------------------------------

const shoulders = (aspectShoulderY = 0.4) => {
  const lm: { x: number; y: number; z: number; visibility: number }[] = [];
  for (let i = 0; i < 33; i++) lm.push({ x: 0, y: 0, z: 0, visibility: 0 });
  lm[11] = { x: 0.6, y: aspectShoulderY, z: 0, visibility: 0.9 };  // left
  lm[12] = { x: 0.4, y: aspectShoulderY, z: 0, visibility: 0.9 };  // right
  return { landmarks: [lm] };
};

const GHOST_DOWN = { upper: [0, 1, 0], fore: [0, 1, 0] };

describe("buildGhostSides", () => {
  it("anchors each ghost arm at that shoulder", () => {
    const out = buildGhostSides(
      shoulders(),
      { left: GHOST_DOWN, right: GHOST_DOWN },
      { aspect: 1, leftMatched: false, rightMatched: false },
    );
    expect(out.left!.pose[0]).toEqual([0.6, 0.4]);
    expect(out.right!.pose[0]).toEqual([0.4, 0.4]);
    expect(out.left!.pose).toHaveLength(3);   // shoulder, elbow, wrist
  });

  it("keeps the ghost's angle true in PIXEL space, not landmark space", () => {
    // Landmarks are normalized to [0,1] on both axes, so a direction written in
    // them is skewed by the frame's aspect ratio. A 45-degree ghost drawn
    // without correcting for that lands at 30 degrees on a 16:9 canvas, and an
    // operator who lines their arm up with it is 15 degrees out — the overlay
    // would be actively lying about the pose it is asking them to match.
    const aspect = 16 / 9;
    const d = Math.SQRT1_2;
    const out = buildGhostSides(
      shoulders(),
      { left: { upper: [d, d, 0], fore: [d, d, 0] }, right: null },
      { aspect, leftMatched: false, rightMatched: false },
    );
    const [sx, sy] = out.left!.pose[0];
    const [ex, ey] = out.left!.pose[1];
    // Pixels: x scales with width, y with height, and width/height = aspect.
    const dxPx = (ex - sx) * aspect;
    const dyPx = ey - sy;
    expect(dxPx).toBeCloseTo(dyPx, 6);
  });

  it("foreshortens a limb pointing at the camera instead of faking its length", () => {
    // Mostly +z (away from the lens) leaves little in the image plane, and the
    // ghost should draw short — that is what the operator's own arm looks like
    // in the same pose.
    const out = buildGhostSides(
      shoulders(),
      { left: { upper: [0.2, 0, 0.98], fore: [0.2, 0, 0.98] }, right: null },
      { aspect: 1, leftMatched: false, rightMatched: false },
    );
    const [sx, sy] = out.left!.pose[0];
    const [ex, ey] = out.left!.pose[1];
    const drawn = Math.hypot(ex - sx, ey - sy);
    const full = buildGhostSides(
      shoulders(), { left: GHOST_DOWN, right: null },
      { aspect: 1, leftMatched: false, rightMatched: false },
    );
    const fullLen = Math.hypot(full.left!.pose[1][0] - full.left!.pose[0][0],
                               full.left!.pose[1][1] - full.left!.pose[0][1]);
    expect(drawn).toBeLessThan(fullLen * 0.3);
  });

  it("returns nothing when there is no body to anchor to", () => {
    expect(buildGhostSides(
      { landmarks: [] }, { left: GHOST_DOWN, right: GHOST_DOWN },
      { aspect: 1, leftMatched: false, rightMatched: false },
    )).toEqual({ left: null, right: null });
  });

  it("returns nothing when the operator is too small in frame to measure", () => {
    const lm = shoulders().landmarks[0];
    lm[12] = { ...lm[11] };     // both shoulders on the same point
    expect(buildGhostSides(
      { landmarks: [lm] }, { left: GHOST_DOWN, right: GHOST_DOWN },
      { aspect: 1, leftMatched: false, rightMatched: false },
    )).toEqual({ left: null, right: null });
  });

  it("omits a side the backend has no pose for", () => {
    const out = buildGhostSides(
      shoulders(), { left: GHOST_DOWN, right: null },
      { aspect: 1, leftMatched: true, rightMatched: false },
    );
    expect(out.left!.matched).toBe(true);
    expect(out.right).toBeNull();
  });
});

// ---- jaw trace recording -----------------------------------------------

describe("JawTraceRecorder", () => {
  it("records nothing until started", () => {
    const r = new JawTraceRecorder();
    r.push(0, 0.5);
    expect(r.count).toBe(0);
    expect(r.recording).toBe(false);
  });

  it("skips lost-face samples without ending the window", () => {
    // A blink or a decimated frame must not truncate a capture.
    const r = new JawTraceRecorder();
    r.start();
    r.push(0, 0.3);
    r.push(100, null);
    r.push(200, 0.4);
    expect(r.stop()).toEqual([[0, 0.3], [200, 0.4]]);
  });

  it("drops the previous window on restart", () => {
    const r = new JawTraceRecorder();
    r.start();
    r.push(0, 0.9);
    r.stop();
    r.start();
    r.push(0, 0.1);
    expect(r.stop()).toEqual([[0, 0.1]]);
  });

  it("stops recording after stop", () => {
    const r = new JawTraceRecorder();
    r.start();
    r.stop();
    r.push(0, 0.5);
    expect(r.count).toBe(0);
  });
});
