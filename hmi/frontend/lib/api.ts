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
  /** "The driver's page is back" — hold the running session open long enough
   *  for the operator to get into XR again after a reload. `ok: false` is an
   *  answer, not an error: the session this token named is gone. */
  humanTeleopReattach: (token: string) =>
    postJson<{ ok: boolean } & HumanTeleopStatus>(
      "/teleop/human/reattach", { token }),
  /** Re-deal the sim bench. `seed` null means fresh entropy; passing one back
   *  reproduces that scene exactly, which is how a re-record gets the SAME
   *  bench rather than a new one.
   *
   *  `home_arms` is left false here on purpose: sending the arms home mid-
   *  session moves the ROBOT, and the endpoint 409s outright while an episode
   *  is open (homing under a rolling recorder would splice a move nobody
   *  demonstrated into the action column). A reset between takes moves the
   *  bench and nothing else. */
  simSceneReset: (seed?: number | null) =>
    postJson<{ ok: true } & SimSceneSnapshot>(
      "/sim/scene/reset", seed == null ? {} : { seed }),
  recordStatus: () => getJson<RecordStatus>("/record/status"),
  recordStart: (repoId: string, task: string) =>
    postJson<{ ok: true } & RecordStatus>("/record/start", { repo_id: repoId, task }),
  /** Open the dataset and hold the start gate: schema, camera set and arm set
   *  frozen, measured fps written into info.json, episode index resolved, and
   *  no frames written yet. Every refusal lives here — colliding camera keys,
   *  an unknown repo, a rate below the gate — because a refusal at the moment
   *  the operator commits to a take is a lost take. */
  recordArm: (repoId: string, task: string) =>
    postJson<RecordStatus>("/record/arm", { repo_id: repoId, task }),
  /** Begin writing frames. 409 unless armed. */
  recordRoll: () => postJson<RecordStatus>("/record/roll", {}),
  /**
   * End the take. `rearm` is OPTIONAL and defaults false, so `{save}` alone
   * means exactly what it has always meant — the cockpit's stop button and the
   * Record popover both call it that way and predate the headset's state
   * machine.
   *
   *   save  rearm   result
   *   true  false   save            -> idle,  index advances
   *   false true    re-record       -> armed, SAME index
   *   false false   discard         -> idle,  index unchanged
   *   true  true    save + go again -> armed, next index
   */
  recordStop: (save: boolean, rearm?: boolean) =>
    postJson<{ ok: true } & RecordStatus>("/record/stop",
      rearm === undefined ? { save } : { save, rearm }),
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
  /** Why the session ended when nobody asked it to — set by the backend's own
   *  auto-stop, and readable AFTER `running` has already gone false. Null when
   *  the operator stopped it themselves. */
  stopped_reason?: string | null;
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

/**
 * Where a take is in the arm -> roll cycle.
 *
 * `armed` is the kit's start gate: full-rate teleop with the dataset open, the
 * camera set, feature schema and arm set frozen, and ZERO frames written. It
 * exists so the first second of every episode is not the operator reaching for
 * a controller.
 *
 * There are exactly three, and `"recording"` is spelled that way so
 * `recording === (state === "recording")` holds — every call site that already
 * reads the boolean stays correct. The headset's end-of-take PROMPT is NOT one
 * of these: the server does not know the operator is looking at a prompt, and
 * frames are still being written throughout it, so a UI that modelled it as a
 * server state would draw a stopped take that is still recording.
 *
 * Optional because a backend that predates the gate reports only the boolean.
 */
export type RecordState = "idle" | "armed" | "recording";

/** What `GET /sim/scene` and `POST /sim/scene/reset` both answer with.
 *
 *  `last_seed` is the seed the bench was RESET FROM, and it is null whenever
 *  the caller passed none — the server seeds from fresh entropy and does not
 *  invent a number to report back, so that scene cannot be asked for again.
 *  A caller that needs to reproduce a bench must therefore draw the seed
 *  ITSELF and pass it; see the take loop in `VRTeleopPanel`. */
export type SimSceneSnapshot = {
  cubes: { name: string; pos: number[]; quat: number[]; rgba: number[] }[];
  lights: { name: string; pos: number[]; diffuse: number[] }[];
  cameras: { name: string; pos: number[] }[];
  last_seed: number | null;
  randomized: boolean;
  mirrored: boolean;
  reset_count: number;
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
  /** Whether anything scored this take at all. False means `success` is null
   *  rather than false — the cockpit must not print FAILED for a rig that
   *  never had an opinion. Returned by recorder.py today. */
  auto_scored?: boolean;
  success?: boolean | null;
  success_frames?: number;
  state?: RecordState;
  /** Which index the next save lands on. Shown 1-based beside it everywhere. */
  episode_index?: number | null;
  /** Why the take in progress can no longer be saved — a degraded read, a
   *  stale camera. Set means the frames so far are not a demonstration. */
  invalidated_reason?: string | null;
  /** The MEASURED sample rate against the real bus, and the rate the recorder
   *  was ASKED for. They are shown side by side rather than one being
   *  reported as the truth: `fps` used to be `1 / telemetry._period`, a number
   *  that was declared and never measured, and every timestamp in every
   *  episode was synthesised from it. A gap between these two is the bug
   *  becoming visible. */
  fps_measured?: number | null;
  fps_declared?: number | null;
  /** Ticks dropped, ATTRIBUTED to the source that dropped them.
   *
   *  Nested rather than one flat map, for two reasons. "Which camera is
   *  starving the take" and "which arm went stale" are different questions
   *  with different fixes, and a flat map throws that attribution away at the
   *  type level. And the two namespaces can collide: the arms are `left` and
   *  `right`, and nothing stops a camera being named for a side — one key,
   *  two meanings, last writer wins, and the panel reports a confident wrong
   *  number. */
  drops?: RecordDrops;
  /** Advisory, mid-take. `record_rate` means the take is sparser than declared
   *  — the rows written are honest rows with real timestamps, so saving still
   *  works and whether it is worth keeping is the operator's call. */
  alerts?: RecordAlert[];
  /** The recorder's faithfulness bound: a take is refused when
   *  `|measured − fps| / fps` exceeds this. A SYMMETRIC TOLERANCE — a
   *  half-width, not a floor — so it is read by `recordRateTolerance` and
   *  compared two-sided. Absent on a backend that predates it, and that
   *  absence has NO fallback on purpose. */
  record_rate_tolerance?: number;
};

export type RecordDrops = {
  cameras?: Record<string, number>;
  arms?: Record<string, number>;
};

/**
 * One entry of `RecordStatus.alerts`, as `recorder.py::_rate_alerts()` emits
 * it — all eight keys, always present when an alert exists.
 *
 * **This type declared `detail` and `since` until 2026-08-27 and the backend
 * has never sent either.** It matched the wire on `code` alone, 1 key of 8,
 * and it type-checked the whole time because both phantoms were optional. It
 * survived because nothing reads it: an unread type cannot render `undefined`,
 * so the cost was deferred to whoever wrote the first consumer, who would have
 * reached for `detail` — the only text-shaped field on offer — and drawn an
 * empty warning row while the operator's sentence sat in `message`.
 *
 * The correct shape was in the tree the entire time. `lib/telemetry.ts`
 * declares the same producer for the telemetry frame and renders it in
 * `AlertsPopover`; that one works. This is now aligned to the wire, which is
 * the superset — the telemetry declaration names the four fields that panel
 * draws, and the four below it does not are the numbers behind the sentence.
 */
export type RecordAlert = {
  /** `"warn"` today. Widened because the popover colours on `error` too. */
  level: "warn" | (string & {});
  code: "record_rate" | (string & {});
  /** Which subsystem raised it — `"recorder"` here. Shown, so alerts from
   *  different sources stay tellable apart on one list. */
  source: string;
  /** The operator-facing sentence, written by the backend and shown verbatim.
   *  **Not `detail`.** */
  message: string;
  /** Null until the tick bus has measured anything — an alert can be raised
   *  from the declared side before a rate exists. */
  measured_hz: number | null;
  /** The DECLARED rate the breach is measured against, never the measured one.
   *  Non-null whenever an alert exists: `_rate_alerts()` returns `[]` while
   *  `fps_declared` is None. */
  fps: number;
  /** The symmetric half-width the breach exceeded — the same quantity
   *  `recordRateTolerance()` reads, repeated here so an alert carries the bound
   *  it was judged against rather than requiring the reader to hold both. */
  tolerance: number;
  /** How long the breach has held, in seconds. **Not `since`** — this is a
   *  duration, not a timestamp, and treating it as one dates the alert to 1970
   *  plus a few seconds. */
  held_s: number;
};

/**
 * The recorder's faithfulness tolerance, or `null` if it does not publish one.
 *
 * **`null` is deliberate and there is no fallback number.** The obvious one —
 * reusing the `0.9` floor fraction this file carried until the gate became a
 * tolerance — is the worst available answer: `0.9` read as a TOLERANCE means
 * ±90%, a band no real rate can fall outside, so the warning would not become
 * wrong, it would stop existing. A check that cannot fire in either direction
 * is dead code shaped like a safety check, and this one sits next to a readout
 * an operator trusts.
 *
 * That is the same fallback question the compare cap answers the other way,
 * and the difference is what happens when the fallback is WRONG. A stale
 * `compare_max_keys` is self-correcting — too high and the backend refuses in
 * words the pane displays, too low and the request is merely split more
 * finely. A wrong rate band shows the operator a wrong number with nothing to
 * contradict it. So: name the degraded state, never paper over it.
 */
export function recordRateTolerance(
  status: RecordStatus | null | undefined,
): number | null {
  const t = status?.record_rate_tolerance;
  return typeof t === "number" && Number.isFinite(t) && t > 0 ? t : null;
}

/**
 * Whether the measured rate is FAITHFUL to the declared one — the shape the
 * recorder enforces.
 *
 * `null` means NOT ANSWERABLE, and it covers two different unknowns that must
 * both stay off the screen as warnings: `fps_measured` is null until 30
 * samples have been seen — and an operator shown a rate warning in the first
 * second after boot learns to ignore rate warnings — or the backend publishes
 * no tolerance, in which case this UI has no band and must say so rather than
 * invent one.
 *
 * Two-sided because `|measured − fps| / fps > tol` refuses a rate that is too
 * FAST as readily as one too slow, and `measured >= declared * g` cannot
 * express that at any value of `g`. A dataset whose timestamps run quick is as
 * dishonest as one that runs slow — the frames are stamped from a rate the rig
 * did not achieve either way.
 */
/**
 * Decimals a rate readout needs for the BAND it is judged against to be
 * visible in it.
 *
 * A readout coloured as a warning while showing two numbers that look equal
 * teaches the operator that the warning is spurious. With `d` decimals there
 * is a rate OUTSIDE the tolerance that still renders as the declared one
 * whenever `fps < 10^(2-d)`:
 *
 *     d=0   collides at every rate below 100 — 30.15 Hz renders "30/30"
 *     d=1   collides below 10 — 5.025 Hz renders "5.0" against a declared 5
 *     d=2   collides below 1, which no session runs at
 *
 * `d=0` predicts the `RATE 30/30 fps` defect the headset track had already
 * observed, which is what makes this an instrument rather than an argument.
 * One decimal is safe at 30 and silently broken at 5 — and 10 Hz and below are
 * reachable today through `POST /teleop/human/start {hz}`. So the decimal count
 * is a CADENCE-COUPLED CONSTANT, and two is the first value that is not
 * calibrated for one cadence. Ported from Track D's derivation.
 */
export const RATE_DECIMALS = 2;

/** A measured rate, at a resolution that can show the tolerance band. */
export function formatHz(v: number): string {
  return v.toFixed(RATE_DECIMALS);
}

export function recordRateFaithful(
  status: RecordStatus | null | undefined,
): boolean | null {
  const measured = status?.fps_measured;
  const declared = status?.fps_declared;
  const tol = recordRateTolerance(status);
  if (
    tol === null ||
    typeof measured !== "number" || typeof declared !== "number" || declared <= 0
  ) {
    return null;
  }
  return Math.abs(measured - declared) / declared <= tol;
}

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
