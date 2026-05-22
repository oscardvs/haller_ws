/**
 * MediaPipe Tasks for Web wrapper.
 *
 * Two responsibilities:
 *   1. Lazy-load HandLandmarker + PoseLandmarker from a single WASM bundle.
 *   2. Fuse their outputs into the `KeypointFrame` shape the backend expects.
 *
 * Coordinates: MediaPipe `worldLandmarks` are metres-relative-to-hip-centre
 * (pose) or metres-relative-to-hand-centre (hand). We pass them through as-is;
 * the backend handles any re-rooting.
 */
import {
  HandLandmarker,
  PoseLandmarker,
  FilesetResolver,
  type NormalizedLandmark,
  type Landmark,
  type HandLandmarkerResult,
  type PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";

// Indices into the 33-point Pose Landmarker output, per MediaPipe docs.
const POSE_LEFT_SHOULDER = 11;
const POSE_RIGHT_SHOULDER = 12;
const POSE_LEFT_ELBOW = 13;
const POSE_RIGHT_ELBOW = 14;
const POSE_LEFT_WRIST = 15;
const POSE_RIGHT_WRIST = 16;

// Indices into the 21-point Hand Landmarker output, per MediaPipe docs.
const HAND_WRIST = 0;
const HAND_THUMB_TIP = 4;
const HAND_INDEX_MCP = 5;
const HAND_INDEX_TIP = 8;
const HAND_MIDDLE_MCP = 9;
const HAND_PINKY_MCP = 17;

export type Vec3 = [number, number, number];

export type SideFrame = {
  pose: { shoulder: Vec3; elbow: Vec3; wrist: Vec3 };
  hand: {
    wrist: Vec3; thumb_tip: Vec3; index_tip: Vec3;
    index_mcp: Vec3; middle_mcp: Vec3; pinky_mcp: Vec3;
  };
  confidence: number;
};

export type KeypointFrame = {
  type: "keypoints";
  ts_ms: number;
  dead_man: boolean;
  pinch_calib?: {
    left?:  { min_m: number; max_m: number };
    right?: { min_m: number; max_m: number };
  };
  left:  SideFrame | null;
  right: SideFrame | null;
};

function _xyz(p: Landmark | NormalizedLandmark | undefined): Vec3 {
  if (!p) return [0, 0, 0];
  return [p.x, p.y, (p as Landmark).z ?? 0];
}

function _vis(p: Landmark | NormalizedLandmark | undefined): number {
  if (!p) return 0;
  return (p as Landmark).visibility ?? 0;
}

/** Pure function: combine MediaPipe results into a backend-shaped KeypointFrame. */
export function fuseLandmarkResults(
  pose: Pick<PoseLandmarkerResult, "worldLandmarks">,
  hands: Pick<HandLandmarkerResult, "worldLandmarks" | "handednesses">,
): { left: SideFrame | null; right: SideFrame | null } {
  const poseLm = pose.worldLandmarks?.[0];
  let left: SideFrame | null = null;
  let right: SideFrame | null = null;

  // Pair hands to sides via the handedness label.
  const handByLabel: Record<"Left" | "Right", { lm: Landmark[]; score: number } | null> = {
    Left: null, Right: null,
  };
  hands.worldLandmarks?.forEach((lm, i) => {
    const label = hands.handednesses?.[i]?.[0]?.categoryName as "Left" | "Right" | undefined;
    const score = hands.handednesses?.[i]?.[0]?.score ?? 0;
    if (label && (label === "Left" || label === "Right")) {
      handByLabel[label] = { lm: lm as Landmark[], score };
    }
  });

  if (poseLm) {
    const buildSide = (
      sIdx: number, eIdx: number, wIdx: number,
      handLabel: "Left" | "Right",
    ): SideFrame | null => {
      const h = handByLabel[handLabel];
      if (!h) return null;
      const handLm = h.lm;
      const poseConf = Math.min(_vis(poseLm[sIdx]), _vis(poseLm[eIdx]), _vis(poseLm[wIdx]));
      return {
        pose: {
          shoulder: _xyz(poseLm[sIdx]),
          elbow:    _xyz(poseLm[eIdx]),
          wrist:    _xyz(poseLm[wIdx]),
        },
        hand: {
          wrist:      _xyz(handLm[HAND_WRIST]),
          thumb_tip:  _xyz(handLm[HAND_THUMB_TIP]),
          index_tip:  _xyz(handLm[HAND_INDEX_TIP]),
          index_mcp:  _xyz(handLm[HAND_INDEX_MCP]),
          middle_mcp: _xyz(handLm[HAND_MIDDLE_MCP]),
          pinky_mcp:  _xyz(handLm[HAND_PINKY_MCP]),
        },
        confidence: Math.min(poseConf, h.score),
      };
    };
    left  = buildSide(POSE_LEFT_SHOULDER,  POSE_LEFT_ELBOW,  POSE_LEFT_WRIST,  "Left");
    right = buildSide(POSE_RIGHT_SHOULDER, POSE_RIGHT_ELBOW, POSE_RIGHT_WRIST, "Right");
  }
  return { left, right };
}

/** Stateful runner: loads the WASM bundle once, then runs inference per frame. */
export class MediaPipeRunner {
  private hand: HandLandmarker | null = null;
  private pose: PoseLandmarker | null = null;

  async load(): Promise<void> {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm",
    );
    this.hand = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
    });
    this.pose = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
    });
  }

  detect(video: HTMLVideoElement, timestamp_ms: number) {
    if (!this.hand || !this.pose) {
      throw new Error("MediaPipeRunner.load() not called");
    }
    const hands = this.hand.detectForVideo(video, timestamp_ms);
    const pose = this.pose.detectForVideo(video, timestamp_ms);
    return { hands, pose };
  }

  close() {
    this.hand?.close();
    this.pose?.close();
    this.hand = null;
    this.pose = null;
  }
}
