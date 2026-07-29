// hmi/frontend/lib/api.ts
import { BACKEND_URL } from "./config";

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const body = await res.json();
      detail = body.error ?? body.detail ?? detail;
    } catch {
      /* ignore parse error */
    }
    throw new Error(`HTTP ${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BACKEND_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return handle<T>(res);
}

export async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BACKEND_URL}${path}`);
  return handle<T>(res);
}

// Convenience wrappers
export type ArmGoal = Record<string, number>;

export type CameraInfo = {
  id: string;
  role: "wrist" | "base";
  // "sim_camera" is what the backend reports for a MuJoCo-rendered view
  // (config.*-sim.yaml). It was missing from this union while the backend had
  // been returning it all along.
  source: "placeholder" | "opencv" | "mjpeg" | "webrtc" | "sim_camera";
  arm_id?: string | null;
  active: boolean;
  width: number;
  height: number;
  fps: number;
};

export const cameraSnapshotUrl = (id: string) =>
  `${BACKEND_URL}/cameras/${encodeURIComponent(id)}/snapshot`;
export const cameraStreamUrl = (id: string) =>
  `${BACKEND_URL}/cameras/${encodeURIComponent(id)}/stream`;

export const api = {
  health: () => getJson<{ status: string }>("/health"),
  config: () => getJson<{
    version: string;
    arms: { id: string; model: string; port: string; mode: string }[];
    cameras: { id: string; role: string; source: string; arm_id?: string }[];
  }>("/config"),
  cameras: () => getJson<{ cameras: CameraInfo[] }>("/cameras"),
  cmdVel: (linear: number, angular: number) =>
    postJson<{ ok: true; linear: number; angular: number }>("/base/cmd_vel", { linear, angular }),
  armGoal: (armId: string, goal: ArmGoal) =>
    postJson<{ ok: true; sent: ArmGoal }>(`/arm/${armId}/goal`, goal),
  armMode: (armId: string, mode: "auto" | "manual" | "stop") =>
    postJson<{ ok: true; mode: string }>(`/arm/${armId}/mode`, { mode }),
  armPreset: (armId: string, name: string) =>
    postJson<{ ok: true; sent: ArmGoal }>(`/arm/${armId}/preset`, { name }),
  armPresetRecord: (armId: string, name: string) =>
    postJson<{ ok: true; saved: ArmGoal }>(`/arm/${armId}/preset/record`, { name }),
  armPresetsList: (armId: string) =>
    getJson<{ names: string[] }>(`/arm/${armId}/presets`),
  armPresetDelete: async (armId: string, name: string) => {
    const res = await fetch(
      `${BACKEND_URL}/arm/${armId}/preset/${encodeURIComponent(name)}`,
      { method: "DELETE" },
    );
    if (!res.ok) {
      let detail = `${res.status}`;
      try {
        const body = await res.json();
        detail = body.error ?? body.detail ?? detail;
      } catch {
        /* noop */
      }
      throw new Error(`HTTP ${res.status}: ${detail}`);
    }
    return res.json() as Promise<{ ok: true }>;
  },
  armHome: (armId: string) =>
    postJson<{ ok: true; sent: ArmGoal }>(`/arm/${armId}/home`, {}),
  armTorque: (armId: string, enabled: boolean) =>
    postJson<{ ok: true; torque: boolean }>(`/arm/${armId}/torque`, { enabled }),
  teleopStatus: () =>
    getJson<TeleopStatus>("/teleop"),
  teleopStart: (leader: string, follower: string, hz = 60) =>
    postJson<{ ok: true } & TeleopStatus>("/teleop/start", { leader, follower, hz }),
  teleopStop: () =>
    postJson<{ ok: true } & TeleopStatus>("/teleop/stop", {}),
  humanTeleopStatus: () =>
    getJson<HumanTeleopStatus>("/teleop/human"),
  humanTeleopStart: (body: {
    left_arm: string; right_arm: string; swap: boolean;
    hz?: number; clutch_source?: "spacebar" | "mouth";
  }) => postJson<{ ok: true } & HumanTeleopStatus>("/teleop/human/start", body),
  humanTeleopStop: () =>
    postJson<{ ok: true } & HumanTeleopStatus>("/teleop/human/stop", {}),
  humanTeleopSwap: (swap: boolean) =>
    postJson<{ ok: true } & HumanTeleopStatus>("/teleop/human/swap", { swap }),
  humanTeleopCalibrate: (body: {
    left?: PinchCalibSide; right?: PinchCalibSide;
    mouth?: MouthCalib;
  }) => postJson<{ ok: true }>("/teleop/human/calibrate", body),
  /** Derive mouth thresholds from recorded traces, and/or replay one through
   *  the real clutch. The browser records; the backend decides — the policy
   *  has exactly one implementation and it is the one that will run. */
  mouthAnalyze: (body: {
    talk?: JawTrace; open?: JawTrace; verify?: JawTrace; calib?: MouthCalib;
  }) => postJson<MouthAnalysis>("/teleop/human/mouth/analyze", body),
  recordStatus: () => getJson<RecordStatus>("/record/status"),
  recordStart: (repoId: string, task: string) =>
    postJson<{ ok: true } & RecordStatus>("/record/start", { repo_id: repoId, task }),
  recordStop: (save: boolean) =>
    postJson<{ ok: true } & RecordStatus>("/record/stop", { save }),
  estop: () => postJson<{ ok: true }>("/estop", {}),
};

export type TeleopStatus = {
  running: boolean;
  leader: string | null;
  follower: string | null;
  hz: number;
  tick_count?: number;
  last_error?: string | null;
  started_at?: number | null;
};

export type HumanTeleopState =
  | "idle" | "armed" | "tracking" | "acquiring" | "driving";

/** Per-side authority. `driving` is the only one that writes to an arm. */
export type SideAuthority = "held" | "acquiring" | "driving";

/** [t_ms, jawOpen] samples in arrival order. */
export type JawTrace = [number, number][];

/** Sustained levels over a hold window — never peaks. See backend safety.py. */
export type MouthCalib = {
  talk_hold: number;
  open_hold: number;
  talk_peak?: number | null;
};

export type MouthCaptureStats = {
  n: number;
  duration_ms: number;
  interval_ms: number | null;
  samples_per_window: number | null;
  peak: number | null;
  floor: number | null;
  sustained: number | null;
};

export type MouthVerdict = {
  /** The safety claim, as a boolean: normal speech must leave this false. */
  engaged: boolean;
  first_engage_ms: number | null;
  engaged_samples: number;
  n: number;
  peak: number | null;
  sustained: number | null;
  t_engage: number | null;
  /** t_engage minus what the trace sustained. Negative means it would drive. */
  margin: number | null;
};

export type MouthAnalysis = {
  window_ms?: number;
  talk?: MouthCaptureStats;
  open?: MouthCaptureStats;
  calib?: MouthCalib | null;
  thresholds?: { t_engage: number; t_release: number } | null;
  ok?: boolean;
  problems?: string[];
  verify?: MouthVerdict;
};

/** Why a side is where it is. `no_tracking` is the one that used to be
 *  invisible: a countdown silently restarting looks exactly like a frozen one. */
export type AcquireReason =
  | "clutch_open" | "no_tracking" | "matching" | "counting" | "driving" | "idle";

export type AcquireSide = {
  authority: SideAuthority;
  reason: AcquireReason;
  matched: boolean;
  /** Joints still outside tolerance, so the operator is told what to move. */
  blocking: string[];
  error_deg: Record<string, number>;
  tol_deg: Record<string, number>;
  /** Countdown to the earliest handover; null unless acquiring. */
  remaining_ms: number | null;
  /** 0..1 through the rate ramp; null unless driving. */
  ramp: number | null;
  /** The arm's pose as unit upper-arm/forearm vectors in the operator's own
   *  frame — what the ghost overlay is drawn from. */
  ghost: { upper: number[]; fore: number[] } | null;
};

export type PinchCalibSide = { min_m: number; max_m: number };

export type JointReason = "ok" | "rate_capped" | "clamped" | "held";

/** Why the clutch is in the state it reports. Mirrors the backend vocabulary
 *  in haller_hmi/human_teleop.py exactly — same seven strings, no others. */
export type ClutchReason =
  | "engaged" | "below_threshold" | "holding" | "stale"
  | "uncalibrated" | "spacebar_mode" | "source_mismatch";

export type JointDiag = {
  /** What the retargeter asked for, in degrees. null when the joint is held. */
  target: number | null;
  committed: number;
  reason: JointReason;
};

export type HumanTeleopStatus = {
  running: boolean;
  state: HumanTeleopState;
  left_arm: string | null;
  right_arm: string | null;
  swap: boolean;
  started_at: number | null;
  last_error: string | null;
  tracking: {
    left:  { age_ms: number | null; lost: boolean };
    right: { age_ms: number | null; lost: boolean };
  };
  frame_age_ms?: number;
  goal_deg?: { left?: Record<string, number>; right?: Record<string, number> };
  joints?: {
    left?:  Record<string, JointDiag>;
    right?: Record<string, JointDiag>;
  };
  /** Optional because a backend older than acquisition omits it. */
  acquire?: {
    acquire_ms: number;
    match_dwell_ms: number;
    left: AcquireSide;
    right: AcquireSide;
  };
  /** Optional because a backend older than the mouth clutch omits it. */
  clutch?: {
    source: "spacebar" | "mouth";
    jaw_open: number | null;
    t_engage: number | null;
    t_release: number | null;
    engaged: boolean;
    stale: boolean;
    reason: ClutchReason;
  };
};

export type RecordStatus = {
  recording: boolean;
  repo_id: string | null;
  task: string | null;
  episode_frames: number;
  started_at: number | null;
  last_error: string | null;
};
