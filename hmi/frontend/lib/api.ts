// hmi/frontend/lib/api.ts
import { BACKEND_URL } from "./config";

/** Thrown by every wrapper below. Carries the status code because callers act
 *  on it: 409 is "the robot refused, and the reason is worth showing" — a
 *  state you can often clear — while 404/501 mean this backend cannot do it at
 *  all. A UI that cannot tell those apart has to treat both as failures. */
export class ApiError extends Error {
  readonly status: number;
  /** The backend's own reason, without the `HTTP nnn:` prefix. A 409 from the
   *  robot is usually a sentence worth showing verbatim. */
  readonly detail: string;
  constructor(status: number, detail: string) {
    super(`HTTP ${status}: ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const body = await res.json();
      detail = body.error ?? body.detail ?? detail;
    } catch {
      /* ignore parse error */
    }
    throw new ApiError(res.status, detail);
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

export async function deleteJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BACKEND_URL}${path}`, { method: "DELETE" });
  return handle<T>(res);
}

/** `?repo_id=…` only when one was chosen — the endpoints default to the
 *  recorder's current repo, and sending an empty string is not the same ask. */
function repoQuery(repoId?: string | null): string {
  return repoId ? `?repo_id=${encodeURIComponent(repoId)}` : "";
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
  // "operator" when the camera looks back at whoever drives the arms (the
  // tower mast cam). The headset HUD mirrors such a view for display only.
  facing?: "work" | "operator";
  /** Whether this camera lands in recorded episodes. Runtime state, not the
   *  config-frozen flag: it is toggled by POST /cameras/{id}/record between
   *  takes. Optional because a backend that predates that route omits it. */
  record?: boolean;
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
  /** Include or exclude this camera from recorded episodes. 409 while an
   *  episode is open — the feature set is frozen at start_episode, so a
   *  camera added mid-take would have no column to land in. */
  cameraRecord: (id: string, record: boolean) =>
    postJson<{ id: string; record: boolean }>(
      `/cameras/${encodeURIComponent(id)}/record`, { record }),
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
  armPresetDelete: (armId: string, name: string) =>
    deleteJson<{ ok: true }>(
      `/arm/${armId}/preset/${encodeURIComponent(name)}`),
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
  /** Sim leader → follower: the operator drags a MuJoCo arm in the native
   *  viewer and the follower tracks it. Bring-up without a headset. */
  simTeleopStart: (body: SimTeleopStart) =>
    postJson<SimTeleopStatus>("/teleop/sim/start", body),
  simTeleopStop: () => postJson<SimTeleopStatus>("/teleop/sim/stop", {}),
  simTeleopStatus: () => getJson<SimTeleopStatus>("/teleop/sim/status"),
  humanTeleopStatus: () =>
    getJson<HumanTeleopStatus>("/teleop/human"),
  /** Start a session. Either arm may be null for a SINGLE-ARM session: that
   *  hand's controller is ignored and nothing is ever written to that side.
   *  At least one must be given. */
  humanTeleopStart: (body: {
    left_arm: string | null; right_arm: string | null; hz?: number;
  }) => postJson<{ ok: true } & HumanTeleopStatus>("/teleop/human/start", body),
  /** Switch the collision/workspace guard on or off, live. Off still
   *  MEASURES — `collision.slack_m` keeps updating — it just stops holding
   *  steps back. 409 when the rig has no mount geometry to guard with. */
  humanTeleopCollision: (enabled: boolean) =>
    postJson<{ ok: true; collision: CollisionStatus }>(
      "/teleop/human/collision", { enabled }),
  humanTeleopHome: () =>
    postJson<{ ok: true; sides: ("left" | "right")[] }>("/teleop/human/home", {}),
  humanTeleopStop: () =>
    postJson<{ ok: true } & HumanTeleopStatus>("/teleop/human/stop", {}),
  recordStatus: () => getJson<RecordStatus>("/record/status"),
  recordStart: (repoId: string, task: string) =>
    postJson<{ ok: true } & RecordStatus>("/record/start", { repo_id: repoId, task }),
  recordStop: (save: boolean) =>
    postJson<{ ok: true } & RecordStatus>("/record/stop", { save }),
  /** Episodes already on disk, read from the dataset meta. `repoId` omitted
   *  means the recorder's current-or-last repo. */
  recordEpisodes: (repoId?: string | null) =>
    getJson<EpisodesResponse>(`/record/episodes${repoQuery(repoId)}`),
  /** Every dataset under the lerobot home, for the repo picker. */
  recordRepos: () => getJson<ReposResponse>("/record/repos"),
  /** Pop the newest episode, in place. **409 is a refusal, not a failure** —
   *  an episode is open, it is the only one, the metadata disagrees with
   *  info.json, or the take shares its video file with an earlier one. The
   *  detail says which, and it is worth showing verbatim. */
  recordDeleteLastEpisode: (repoId?: string | null) =>
    deleteJson<DeletedEpisode>(`/record/episodes/last${repoQuery(repoId)}`),
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

/** One of the two leader sources /teleop/sim/start accepts. */
export type SimTeleopStart = {
  follower: string;
  hz?: number;
  leader:
    | { source: "mouse"; arm_name: string }
    | { source: "replay"; dataset_path: string };
};

export type SimTeleopStatus = {
  running: boolean;
  follower?: string | null;
  hz?: number;
  tick_count?: number;
  last_error?: string | null;
};

export type HumanTeleopState =
  | "idle" | "armed" | "tracking" | "acquiring" | "driving";

/** Per-side authority. `driving` is the only one that writes to an arm. */
export type SideAuthority = "held" | "acquiring" | "driving";

/** Why a side is where it is. `no_tracking` is the one that used to be
 *  invisible: a countdown silently restarting looks exactly like a frozen one.
 *  `no_arm` is the single-arm session's absent side — it never acquires and is
 *  never written, which is a configuration, not a fault. */
export type AcquireReason =
  | "clutch_open" | "no_tracking" | "counting" | "driving" | "idle" | "no_arm";

export type AcquireSide = {
  authority: SideAuthority;
  reason: AcquireReason;
  /** Countdown to the earliest handover; null unless acquiring. */
  remaining_ms: number | null;
  /** 0..1 through the rate ramp; null unless driving. */
  ramp: number | null;
};

export type JointReason = "ok" | "rate_capped" | "clamped" | "held" | "collision";

/** Why the clutch is in the state it reports. Mirrors the backend vocabulary
 *  in haller_hmi/human_teleop.py. `vr_grip_mode` is a controller grip that is
 *  simply not held: the resting state, not a fault. */
export type ClutchReason = "engaged" | "holding" | "stale" | "vr_grip_mode";

export type JointDiag = {
  /** What the mapper asked for, in degrees. null when the joint is held. */
  target: number | null;
  committed: number;
  reason: JointReason;
};

export type HumanTeleopStatus = {
  running: boolean;
  state: HumanTeleopState;
  left_arm: string | null;
  right_arm: string | null;
  started_at: number | null;
  last_error: string | null;
  tracking: {
    left:  { age_ms: number | null; lost: boolean };
    right: { age_ms: number | null; lost: boolean };
  };
  frame_age_ms?: number;
  /** The commanded joint targets — the recorder's `action` column. */
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
  clutch?: {
    engaged: boolean;
    /** Which sides `engaged` actually covers — the two controller grips. */
    sides?: { left: boolean; right: boolean };
    reason: ClutchReason;
  };
  /** Bimanual collision guard. Absent on backends that predate it;
   *  `enabled: false` means no guard is wired for this config. */
  collision?: CollisionStatus;
};

export type CollisionStatus = {
  /** Whether the guard is currently allowed to clamp a step. */
  enabled: boolean;
  /** Whether it COULD be enabled at all. False when the rig has no mount
   *  geometry for every arm, which is a one-way state: a guard that has no
   *  geometry for an arm would pass every check for it. Shown separately so
   *  a UI can tell "off, flip it back on" from "the switch does nothing
   *  here". Absent on backends that predate the runtime switch. */
  available?: boolean;
  /** Gap left before the guard clamps, metres. Negative while inside the
   *  margin (escape-only regime — the guard blocks approach, never retreat). */
  slack_m?: number;
  /** The binding constraint, e.g. "left:hand|right:hand" or "right:tip_floor". */
  worst?: string;
  /** True on ticks where the guard actually shortened the commanded step. */
  limited?: boolean;
  /** Fraction of the wanted step that survived (1 = untouched). */
  alpha?: number;
  margin_m?: number;
};

export type RecordStatus = {
  recording: boolean;
  repo_id: string | null;
  task: string | null;
  episode_frames: number;
  /** Ticks seen but not recorded (stale camera / missing arm telemetry). */
  skipped_frames: number;
  started_at: number | null;
  last_error: string | null;
};

/** One episode as the dataset meta records it. */
export type EpisodeInfo = {
  index: number;
  frames: number;
  task: string | null;
  length_s: number;
};

export type EpisodesResponse = {
  repo_id: string | null;
  episodes: EpisodeInfo[];
  total_frames: number;
  size_bytes: number;
  /** Where the dataset actually lives, under the lerobot home. */
  root?: string;
};

export type RepoInfo = {
  repo_id: string;
  episodes: number;
  frames?: number;
  size_bytes: number;
};

export type ReposResponse = { repos: RepoInfo[]; root?: string };

/** What the pop actually did. The totals are post-delete, so a UI can settle
 *  without a second read. */
export type DeletedEpisode = {
  deleted_index: number;
  repo_id: string;
  deleted_frames: number;
  total_episodes: number;
  total_frames: number;
};
