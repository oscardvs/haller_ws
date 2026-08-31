// hmi/frontend/lib/lab.ts
/**
 * The Lab API — dataset review and training runs, under `/lab`.
 *
 * Separate from `lib/api.ts` on purpose. That file is the ROBOT: arms,
 * cameras, teleop, the recorder's live status — things that are true right
 * now and are gone if you look away. This one is the WORKBENCH: what was
 * collected, what was marked, what was trained. The two have different
 * failure meanings. A robot call that 404s is a broken build; a `/lab` call
 * that 404s is a backend that predates the Lab, and every surface here has to
 * degrade to "this build has no Lab" rather than to an error page.
 *
 * The transport is shared — `getJson`/`postJson`/`ApiError` come from
 * `lib/api.ts` — so a 409 means the same thing on both: the backend refused,
 * and the reason is a sentence worth showing.
 */
import { ApiError, deleteJson, getJson, postJson } from "./api";
import { BACKEND_URL } from "./config";

/* ─── query helpers ───────────────────────────────────────────────────── */

/** Build `?a=1&b=2`, dropping every param that is null/undefined/"".
 *
 *  Same discipline as `lib/api.ts::repoQuery`: an omitted param and an empty
 *  one are different asks. `filter_mark=` is not "no filter", it is a filter
 *  for the empty mark, and the server would be right to return nothing. */
export function qs(params: Record<string, string | number | boolean | null | undefined>): string {
  const out = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === null || v === undefined || v === "") continue;
    out.set(k, String(v));
  }
  const s = out.toString();
  return s ? `?${s}` : "";
}

/** True when the backend cannot do this at all — an old build, not a bad
 *  request. Every Lab surface treats this as "hide me", not "fail". */
export function isMissing(e: unknown): boolean {
  return e instanceof ApiError && (e.status === 404 || e.status === 501);
}

/** True when the backend refused because something else is in flight. The
 *  detail is the operator's next move, so it is always worth showing. */
export function isBusy(e: unknown): boolean {
  return e instanceof ApiError && e.status === 409;
}

/**
 * True when the backend read the request and refused it on its merits.
 *
 * The rollout route is the reason this exists. Its 400s are not malformed
 * JSON — they are the rate gate ("that is a different dynamical system, not a
 * faster or slower one"), a checkpoint whose dataset has been pruned away, a
 * duration over the ceiling. Every one of them is a sentence the operator has
 * to read and act on, so they get a panel that persists, not a toast that
 * vanishes — the same treatment as a 409, arrived at from a different place.
 */
export function isRefused(e: unknown): boolean {
  return e instanceof ApiError && e.status === 400;
}

/**
 * True when the backend refused because this client is not on the loopback.
 *
 * `--host 0.0.0.0` is how the Quest reaches the HMI, and reaching it must not
 * also mean deleting a dataset or launching a job. Six routes are gated —
 * autoclass apply/revert, prune, train, stop, delete run — and everything
 * else, marking and tagging included, works from the LAN so triage can happen
 * from the headset. A 403 here is a policy, not a fault, and the operator's
 * next move is a real one: do it from the machine, or set
 * HALLER_ALLOW_REMOTE_CONTROL.
 */
export function isForbidden(e: unknown): boolean {
  return e instanceof ApiError && e.status === 403;
}

export const REMOTE_REFUSED =
  "refused from the network — this action only runs on the machine itself. " +
  "open the cockpit there, or set HALLER_ALLOW_REMOTE_CONTROL on the backend.";

/** The backend's own sentence, or the transport's. */
export function reason(e: unknown): string {
  if (e instanceof ApiError) return e.detail;
  return e instanceof Error ? e.message : String(e);
}

/* ─── datasets ────────────────────────────────────────────────────────── */

/** keep / reject, and the third state that is not a decision yet.
 *  `unset` is load-bearing: "I have not watched this one" and "I watched it
 *  and it is fine" are different facts, and a two-state mark loses the first. */
export type Mark = "keep" | "reject" | "unset";

/** The grader's opinion, which is never the mark. A verdict is a heuristic
 *  reading of the trace; a mark is the operator's judgement. Autoclassify is
 *  the only thing that turns one into the other, and it shows its work first. */
export type Verdict = "PASS" | "SUSPECT" | "FAIL";

/**
 * Which arms the take was recorded with, derived by the backend from the
 * dataset's OWN column names — never configured.
 *
 * `"solo"` is an unprefixed single-arm rig (the kit's `shoulder_pan.pos`);
 * `"left"`/`"right"` are a side-prefixed rig with one side present. A solo
 * dataset has no columns for the absent side, so this is a fact about the
 * schema and not a preference.
 */
export type Rig = "bimanual" | "left" | "right" | "solo";

/** How a rig reads in the UI. Lowercase, like every other label here. */
export function rigLabel(rig: Rig | string | null | undefined): string {
  switch (rig) {
    case "bimanual": return "bimanual";
    case "left": return "solo left";
    case "right": return "solo right";
    case "solo": return "single arm";
    default: return "rig unknown";
  }
}

/**
 * What unit a dataset's joint columns are in, as the polled listing says it.
 *
 * Three scalars and no sentences, because `/lab/datasets` is polled and the
 * backend deliberately keeps the joint-name lists and the operator-facing note
 * on the detail endpoint (`catalog._units_summary`). It is enough for a card
 * to warn; the reasons are one click away.
 *
 * `state_unit` is the block-level declaration, VERBATIM, and it over-claims:
 * an SO-101 gripper is `range_0_100` under every configuration
 * (lerobot's `so_follower.py:59` pins it with no `use_degrees` branch), so a
 * dataset that says `"deg"` is still mixing two unit systems inside one
 * 12-column vector. Nothing in the UI should render it as the unit of every
 * column; it is provenance, not a label.
 */
export type DatasetUnitsSummary = {
  /** Whether the dataset carries a `haller_joint_calibration` block at all.
   *  False is the foreign case: a corpus pulled off the Hub, which says
   *  nothing about what its numbers mean. */
  declared: boolean;
  state_unit: string | null;
  /** True only when EVERY column has a usable calibrated range, which is the
   *  precondition the backend's converter enforces before it will touch a row
   *  (`lab/units.py`). A dataset that is 11/12 calibrated is not 92 %
   *  convertible; it is not convertible, because a row with one column left
   *  in the other unit keeps its width and its plausible magnitudes and
   *  nothing downstream can see it. */
  convertible: boolean;
};

/** The whole provenance, as the detail endpoint carries it next to `features`.
 *
 *  `features` can say a column is `float32[12]`; it has no slot for what those
 *  twelve numbers MEAN, and LeRobot never asks. This is that slot. `note` is
 *  written server-side rather than composed here so the page and the
 *  co-training caller that refuses the same dataset describe it identically. */
export type DatasetUnits = DatasetUnitsSummary & {
  /** Where the answer came from: the metadata key, or `"undeclared"`. */
  source: string;
  joints_total: number;
  joints_calibrated: number;
  /** The columns with no usable calibrated range, BY NAME. "1 joint is
   *  uncalibrated" is a fact nobody can act on; "right_gripper" is a fixable
   *  dataset. */
  uncalibrated: string[];
  reason: string | null;
  note: string;
};

/** What a card or a header shows about the unit, or null when there is
 *  nothing to say. */
export type UnitsAlert = { label: string; note: string };

/** Said when the dataset declares nothing at all. Only reached when the
 *  summary shape is all the caller has: the detail endpoint ships its own
 *  sentence and it is the better one, because it also counts the joints. */
const UNITS_UNKNOWN_NOTE =
  "units unknown: this dataset does not declare what unit its joint columns " +
  "are in. it may be degrees or normalised [-100, 100], and the two look " +
  "identical in a plot. values are shown exactly as recorded and must not be " +
  "read as this robot's degrees.";

/** Said when the block is there but incomplete. */
const UNITS_PARTIAL_NOTE =
  "units partly declared: not every joint has a calibrated range, so there " +
  "is no exact conversion for a row. the values stay in whatever unit they " +
  "were recorded in and must not be read as this robot's degrees.";

/**
 * The warning to show about a dataset's units, or null when there is none.
 *
 * WHY THIS IS A WARNING AND NOT A LABEL. Every number the Lab draws (a
 * trace, a sweep bar, a gripper guide, a grasp threshold) comes out of
 * `observation.state` with no unit attached, and the page used to render all
 * of them identically. On a Haller recording they are degrees. On a foreign
 * SO-101 recording they may be normalised [-100, 100], and nothing about the
 * plot distinguishes the two: both are small signed numbers on
 * joint-shaped trajectories. Reading one as the other does not fail, it
 * silently moves every threshold and verdict derived from it.
 *
 * Takes either shape (`DatasetUnitsSummary` from a card, `DatasetUnits` from
 * the detail) and prefers the backend's own sentence when it is there, so the
 * two surfaces never explain one dataset in two different ways.
 *
 * Returns null when `convertible` is true and ONLY then: a dataset whose every
 * column has a calibrated range is one where the map to normalised is exact
 * and reversible, and there is nothing for an operator to be careful about.
 * Also null while the field is absent, which is a backend that predates it:
 * the same "hide me, do not fail" rule every other optional field here
 * follows.
 */
export function unitsAlert(
  units: (Partial<DatasetUnits> & DatasetUnitsSummary) | null | undefined,
): UnitsAlert | null {
  if (!units || units.convertible) return null;
  const declared = units.declared === true;
  return {
    label: declared ? "units partial" : "units unknown",
    // The server's sentence names the joints and counts them; the constants
    // here only stand in for the polled card, which is not sent either.
    note: units.note ?? (declared ? UNITS_PARTIAL_NOTE : UNITS_UNKNOWN_NOTE),
  };
}

export type DatasetMarks = {
  keep: number;
  reject: number;
  unset: number;
  /**
   * What the trainer will actually be handed: `keep + unset`.
   *
   * An UNSET episode counts as keep — "I have not judged this" is not
   * "throw it away" — so `keep` alone understates the training set the moment
   * anything is unmarked. This is the number that answers "how much am I
   * about to train on". Optional: a backend that predates it omits it, and
   * `trainableCount` falls back.
   */
  train?: number;
};

/** How many episodes a run on this dataset would see. See `DatasetMarks.train`. */
export function trainableCount(marks: DatasetMarks | null | undefined): number {
  if (!marks) return 0;
  return typeof marks.train === "number" ? marks.train : marks.keep + marks.unset;
}

/** One dataset as the shelf and the picker show it. */
export type DatasetSummary = {
  repo_id: string;
  task: string | null;
  episodes: number;
  frames: number;
  duration_s: number;
  size_bytes: number;
  marks: DatasetMarks;
  /** Left behind by an in-place prune. Never offered as a training source —
   *  training on the pre-prune copy silently undoes the prune. */
  is_backup: boolean;
  rig: Rig;
  /** Three scalars saying whether these numbers are this robot's degrees.
   *  Optional: a backend that predates it omits the field, and `unitsAlert`
   *  reads that as "nothing to say" rather than as a warning. */
  units?: DatasetUnitsSummary;
  /** At least one mark no longer describes the episode it was made about —
   *  an episode was pruned and the survivors renumbered. The marks on this
   *  dataset cannot be trusted until they are re-checked, which is a louder
   *  fact than it looks: training on a stale keep-set trains on whatever
   *  slid into that index. */
  stale?: boolean;
};

/** Where one episode lives inside a packed v3.0 mp4.
 *
 *  Optional throughout: a backend that serves one episode per response has
 *  no slice to declare, and `sliceFor` below falls back to the episode's own
 *  duration. Both shapes work; see its note. */
export type VideoSlice = {
  chunk_index?: number;
  file_index?: number;
  from_timestamp: number;
  to_timestamp: number;
};

/** One arm's measurements behind an episode verdict.
 *
 *  The episode's own `verdict` is the WORST arm's — which is right for a
 *  single badge and useless for deciding what went wrong, so the per-arm
 *  readings ride along. */
export type EpisodeArm = {
  side: string;
  verdict: Verdict;
  why: string;
  /** How many times the gripper closed. More than one is a retry. */
  closes: number;
  reopened: boolean;
  grip_min: number;
  grip_max: number;
  /** Worst commanded-vs-measured error, degrees. */
  tracking: number;
  /** Total joint travel, degrees. Near zero means the arm never moved. */
  sweep_total: number;
  sweep?: number[];
  /** The gripper thresholds this arm was GRADED with, in the dataset's own
   *  units. The chart's guides are these floats and nothing else — a second
   *  source for the same number is how a chart ends up drawing a guide the
   *  verdict printed beside it disagrees with. Measured on disk:
   *  local/so101_pick_cube is 40.0 / 70.0 exactly; the bimanual dataset,
   *  calibrated in degrees over [-9.97, 100.27], is 34.1254 / 67.1965. */
  closed_below?: number;
  open_above?: number;
};

export type LabEpisode = {
  /**
   * The stored episode index — 0-based, and what every endpoint takes.
   *
   * Spelled `episode_index` and never `index`, because LeRobot's own v3.0
   * parquet carries BOTH as different columns and `index` is the GLOBAL FRAME
   * index: on the real dataset, episode 1's first three frames have
   * episode_index [1,1,1], frame_index [0,1,2] and index [855,856,857].
   * A field called `index` here would read correctly and mean something else.
   */
  episode_index: number;
  /** What the operator calls it — `episode_index + 1`. The server owns the mapping so
   *  the UI never derives it, and both spellings are always shown together:
   *  Oscar numbers episodes 1-based in conversation and they are stored
   *  0-based, and that off-by-one is how the wrong demonstration gets
   *  deleted. See `epLabel`. */
  label: string | number;
  frames: number;
  duration_s: number;
  /** This episode's share of the dataset's frames, 0..1. A take that is 30%
   *  of the corpus dominates whatever is trained on it. */
  share: number;
  task: string | null;
  verdict: Verdict | null;
  /** Why the grader said that — one entry per arm, prefixed `"left: "` /
   *  `"right: "` on a multi-arm rig and bare on a solo one, plus the
   *  dominant-share note when there is one. A list rather than the kit's
   *  single `why`, because a take can be a clean left arm and a dead right
   *  arm at the same time and one string cannot say that. */
  reasons: string[];
  /** The structured form of `reasons`, for per-arm badges. Absent on a
   *  backend that predates it. */
  arms?: EpisodeArm[];
  mark: Mark;
  note: string | null;
  tags: string[];
  /** Present only when the backend packs several episodes per video file. */
  videos?: Record<string, VideoSlice>;
};

/** One feature column as `info.json` declares it. */
export type FeatureSpec = {
  dtype?: string;
  shape?: number[];
  names?: string[] | Record<string, unknown> | null;
};

/**
 * One gripper channel, as the trace endpoint isolates it.
 *
 * Carries its own thresholds, which is why nothing else has to look them up:
 * `closed_below` / `open_above` here are the exact floats `grade.py` graded
 * this episode with, index-aligned to the channel they describe. The episode's
 * `arms[]` carries the same pair, and taking it from there instead would be a
 * second source for one number — the shape this whole surface has been
 * avoiding. Measured on disk: 40.0 / 70.0 on the kit's dataset, 34.13 / 67.20
 * on Haller's degree-calibrated one.
 */
export type GripperChannel = {
  /** "left" / "right", or "" on a solo rig. */
  side: string;
  /** The column name, e.g. `gripper.pos` or `left_gripper`. */
  name: string;
  /** Its index into `state` / `action`. */
  index: number;
  closed_below: number;
  open_above: number;
  values: number[];
};

export type DatasetDetail = {
  repo_id: string;
  root: string;
  fps: number;
  robot_type: string;
  /** Every instruction the takes were recorded under. A LIST because lerobot
   *  allows several; `_first_task` is the backend's own rule for reading one
   *  out of it, and the rollout launcher applies the same one. */
  tasks: string[];
  /** Camera keys, WITHOUT the `observation.images.` prefix where the backend
   *  strips it — the picker shows whatever it is given. */
  video_keys: string[];
  features: Record<string, FeatureSpec>;
  /** The observation columns the launcher ticks before anyone touches it:
   *  the state vector and the cameras. Computed by the BACKEND — the same
   *  rule that validates the choice on the way back in — because a browser
   *  that derived its own would drift from it, and the drift would be a
   *  policy trained on an observation space the form did not show. */
  policy_inputs_default?: string[];
  rig: Rig;
  /** The whole unit provenance, beside the `features` it describes. Optional
   *  for the same reason as `DatasetSummary.units`. */
  units?: DatasetUnits;
  episodes: LabEpisode[];
};

/** Server-side sort keys. Sorting is a server concern here and not a browser
 *  one on purpose: a 70-seed campaign is thousands of episodes and the list
 *  is paged, so a browser sort would order one page and call it the answer. */
export type EpisodeSort =
  | "index" | "duration" | "frames" | "share" | "verdict" | "mark" | "task";
export type SortOrder = "asc" | "desc";

export type EpisodeQuery = {
  repo_id: string;
  sort?: EpisodeSort;
  order?: SortOrder;
  filter_mark?: Mark | null;
  filter_verdict?: Verdict | null;
  tag?: string | null;
  q?: string | null;
  offset?: number;
  limit?: number;
};

export type EpisodePage = { total: number; episodes: LabEpisode[] };

/** One episode's numeric trace.
 *
 *  `names` is the contract that makes this rig-agnostic: the channel COUNT
 *  varies (6 on a solo arm, 12 bimanual), so nothing downstream may assume
 *  five joints and a gripper the way the kit's chart did. */
export type Trace = {
  names: string[];
  /** Seconds from episode start, one per sample. */
  t: number[];
  /** Commanded, `[channel][sample]`. What teleop asked for. */
  action: number[][];
  /** Measured, `[channel][sample]`. What the arm did. */
  state: number[][];
  /** The gripper channels, already isolated so a chart does not have to guess
   *  which of twelve columns closes on the object — and carrying the
   *  thresholds each was graded against. */
  gripper?: GripperChannel[];
};

/** The held-out plan, straight from the server.
 *
 *  Never recomputed in the browser. Two implementations of "which episodes
 *  does the trainer not see" drift, and when they do the `val` badges lie
 *  about which demonstrations the policy has already learned — the one error
 *  in this whole surface that cannot be spotted by looking at it. */
export type SplitPlan = {
  order: number[];
  train_episodes: number[];
  eval_episodes: number[];
};

export type SplitMode = "random" | "recent";

/**
 * The four autoclassify modes.
 *
 * `grade` runs the rule ladder; `rules` evaluates an operator-authored
 * comparison expression; `knn` propagates marks from the ones already made;
 * `policy-loss` is a SORT ORDER and never a mark — it always returns an empty
 * diff and 400s on apply, because a high loss is as often a rare-but-correct
 * demonstration as a bad one.
 */
export type AutoclassMode = "grade" | "rules" | "knn" | "policy-loss";

export type AutoclassParams =
  | Record<string, never>
  | { reject_if?: string; keep_if?: string }
  | { k?: number; min_confidence?: number; propagate?: "mark" | "tags" }
  | { run_id?: string };

/** One proposed mark change, before anything is written.
 *
 *  There is no `label` here: the server sends the STORED index and the UI
 *  renders `Ep {index + 1} (idx {index})`. Deriving the display number in one
 *  place is what stops the two spellings drifting apart. */
export type AutoclassChange = {
  episode: number;
  from: Mark;
  to: Mark;
  why: string;
  /** 0..1. Shown, never thresholded silently — a cut-off the operator cannot
   *  see is a cut-off they cannot disagree with. */
  confidence: number;
};

/** `policy-loss` only: hardest-to-fit first. */
export type AutoclassRank = { episode: number; score: number; rank: number };

export type AutoclassPreview = {
  /** Binds this diff to the dataset state it was computed against. `apply`
   *  recomputes it and 409s on a mismatch — the operator approved a diff for a
   *  state, and applying it to a different one applies decisions they never
   *  saw. Hand it back untouched. */
  token: string;
  diff: AutoclassChange[];
  ranking?: AutoclassRank[];
  /** `policy-loss` is data-gated: it needs a run that wrote per-episode loss. */
  available?: boolean;
  reason?: string;
};

export type AutoclassApplied = {
  applied: number;
  /** Handle for the one-click revert. */
  batch: string;
};

/** Pruning is a detached job, not a synchronous edit — the survivors are
 *  renumbered and the video is re-encoded. Watch it under Train. */
export type PruneStarted = { run_id: string };

/* ─── runs ────────────────────────────────────────────────────────────── */

export type RunKind = "train" | "export" | "prune" | "rollout" | "record";
export type RunStatus =
  | "queued" | "running" | "done" | "failed" | "stopped" | "died" | "launch_failed";

export type PolicyType = "act" | "smolvla" | "pi0" | "diffusion";

/** What a training run was asked to do. Echoed back on the run so a finished
 *  run can be read months later without the launcher's state. */
export type TrainSpec = {
  repo_id: string;
  policy_type: PolicyType;
  steps: number;
  batch_size: number;
  eval_split: number;
  eval_seed: number;
  eval_mode: SplitMode;
  eval_steps: number;
  save_freq: number;
  num_workers: number;
  device: string;
  job_name: string;
  tags: string[];
  /** The kept set, resolved at launch. Passed to LeRobot as
   *  `--dataset.episodes`; sending it explicitly is what makes a run
   *  reproducible after the marks have moved on. */
  episodes?: number[];
  /** The observation columns the policy may read.
   *
   *  LeRobot derives this from the dataset when it is not sent, and takes
   *  EVERY `observation.*` column with no way to opt one out. That turned the
   *  2026-08-29 schema migration into a silent change of what ACT consumed —
   *  a per-episode clock among it. Sending the list pins the space:
   *  `--policy.input_features` is honoured when set.
   *
   *  Omitted, not defaulted, when the caller has no opinion: absent is
   *  LeRobot's own behaviour, and `[]` is refused rather than promoted to it.
   *  The backend resolves these names to shapes and stores THAT. */
  policy_inputs?: string[];
};

/* ─── the derived job name ────────────────────────────────────────────── */

/** What the checkpoint directory can carry, and what still reads as a name
 *  in a 26rem rail. Longer than this and the run list truncates the half
 *  that distinguishes one run from the next. */
const MAX_JOB_NAME = 60;

/** Everything the derived name spells out. One argument object rather than
 *  eight positional numbers: the day a field is added, a caller that forgot
 *  it fails to compile instead of silently naming runs after the wrong
 *  hyperparameter. */
export type JobNameParts = {
  repoId: string | null;
  policy: PolicyType | string;
  /** How many episodes the run is pinned to — NOT the dataset's total. Two
   *  runs over the same dataset before and after a prune are different runs
   *  and must not share a name. */
  episodes: number;
  steps: number;
  batch: number;
  evalSplit: number;
  seed: number;
  mode: SplitMode;
};

/** Anything that is not a plain word becomes an underscore, so the name is
 *  safe as a path segment wherever LeRobot puts it. */
const slug = (s: string) =>
  s.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");

/** `90000` → `90k`. Round thousands only: `1500` stays `1500`, because a step
 *  count that is not a round thousand is a debug run and `1.5k` reads as a
 *  rounding of something else. */
const stepTag = (n: number) => {
  const s = Math.max(0, Math.round(n));
  return s >= 1000 && s % 1000 === 0 ? `${s / 1000}k` : String(s);
};

/**
 * The name a run gets when the operator does not type one.
 *
 * It exists because the operator's own naming is `act_hilti_box_91` typed
 * once and then sticky forever: three runs at three different step counts
 * arrive in the list under one name, and the only way to tell them apart is
 * to open each one and read its spec. So the name says what the run IS —
 * policy, dataset, how many episodes, and every hyperparameter that changes
 * the result:
 *
 *     act_so101_pick_cube_91ep_90k_b16_ev10_s42
 *
 * `device` and `workers` are left out on purpose (they change how long it
 * takes, never what comes out), and so are the eval/save cadences. The split
 * SEED is in, but only under `random` — `recent` ignores it, and a name that
 * carries a number the run does not use invites the operator to change that
 * number expecting a different split.
 *
 * `taken` is the names already in use. A repeat of an identical config — a
 * relaunch after a crash, most often — gets `_v2`, `_v3`, which is both
 * unique and the true description of what it is. Pass the set you have; the
 * run id underneath is unique regardless, so a stale or missing set costs a
 * duplicate label and nothing more.
 *
 * The DATASET is what gets trimmed when the whole thing will not fit, never
 * the tail: the hyperparameters are the half that tells two runs apart.
 */
export function trainJobName(
  parts: JobNameParts,
  taken?: ReadonlySet<string>,
): string {
  const { repoId, policy, episodes, steps, batch, evalSplit, seed, mode } = parts;
  const head = slug(String(policy)) || "run";
  const tail = [
    episodes > 0 ? `${Math.round(episodes)}ep` : null,
    stepTag(steps),
    `b${Math.max(1, Math.round(batch))}`,
    evalSplit > 0 ? `ev${Math.round(evalSplit * 100)}` : "noeval",
    evalSplit > 0 ? (mode === "recent" ? "recent" : `s${Math.round(seed)}`) : null,
  ]
    .filter((x): x is string => x !== null)
    .join("_");

  const name = (room: number) => {
    const ds = slug(repoId ? repoId.split("/").pop() || repoId : "")
      .slice(0, Math.max(0, room))
      .replace(/_+$/, "");
    return [head, ds, tail].filter(Boolean).join("_");
  };

  // The dataset gets whatever the fixed half leaves over — and a `_vN` is
  // part of the fixed half, so a de-duplicated name trims the dataset one
  // character further rather than dropping the suffix that made it unique.
  const room = (extra: number) =>
    MAX_JOB_NAME - head.length - tail.length - 2 - extra;

  const base = name(room(0));
  if (!taken?.has(base)) return base;
  for (let v = 2; v < 1000; v += 1) {
    const suffix = `_v${v}`;
    const candidate = name(room(suffix.length)) + suffix;
    if (!taken.has(candidate)) return candidate;
  }
  return base;
}

/* ─── rollout ─────────────────────────────────────────────────────────── */

/**
 * What a rollout run is ASKED to do. `POST /lab/runs/rollout`.
 *
 * A rollout is a run like any other and NOT a bus handover: the detached
 * child loads the checkpoint and runs inference, and the SERVER keeps the
 * Feetech bus and commits the child's targets through the same chain every
 * other leader goes through — LPF, rate cap, clamp, collision guard,
 * workspace floors, E-STOP. Which is why launching one is a POST and not a
 * mode: nothing here hands anything over.
 *
 * Only `policy_path` is required, and every optional below is optional in the
 * same direction — leaving it out is the CORRECT value rather than a shortcut
 * past a decision. `control_hz` most of all: omitted, the server runs the
 * policy at the rate its training dataset was recorded at, which it reads
 * from the checkpoint itself. See `RolloutRecord`.
 */
export type RolloutSpec = {
  /** The model directory — `.../checkpoints/060000/pretrained_model`, which
   *  is exactly the path `Checkpoint.path` carries. */
  policy_path: string;
  duration_s: number;
  device: string;
  /**
   * The rate to drive the policy at, in Hz. OMIT IT to get the rate it was
   * trained at.
   *
   * Sending one is a declaration the server checks against the fps of the
   * dataset the checkpoint records, and a divergence is a 400 with a sentence
   * naming both numbers — not a warning. `allow_rate_mismatch` is the only
   * way past it, and the run then says so in its own spec forever.
   */
  control_hz?: number;
  /** Launch anyway when the declared rate is not the trained one. */
  allow_rate_mismatch?: boolean;
  /** Let the run continue when the MEASURED rate falls under the floor. A
   *  different gate on a different event: this one is the child's, checked
   *  over warmup inference cycles before a single target is sent. */
  allow_slow?: boolean;
  /**
   * Which arm an UNPREFIXED policy drives — `"left"` or `"right"`.
   *
   * Required exactly when the training dataset's columns carry no side
   * (`rig === "solo"`, the shape the kit's datasets have), and refused rather
   * than guessed by the child: the two arms are 40 cm apart, and a rollout
   * aimed at the wrong one is a collision, not a typo. Ignored on a rig whose
   * columns name their own side.
   */
  side?: string;
  /** The language instruction, for a policy conditioned on one. The dataset's
   *  own recorded task is the honest default. */
  task?: string;
  robot_type?: string;
};

/**
 * A rollout spec as the RUN echoes it back, with the rate decision stamped.
 *
 * The extra fields are the server's, written at launch whether or not
 * anything was overridden, so a run read months later says where its rate
 * came from instead of reading as a deliberate agreement between two numbers.
 * `control_hz_trained_source`/`_reason` are on the wire too and are
 * deliberately not typed here: they are the diagnostic half of a refusal that
 * never reached this client, and nothing renders them.
 */
export type RolloutRecord = RolloutSpec & {
  control_hz: number;
  /** The fps of the dataset the checkpoint was trained on, or null when that
   *  chain could not be walked — in which case the operator had to declare a
   *  rate and there was no second source to check it against. */
  control_hz_trained: number | null;
  /** `"trained_fps"` when the operator left the rate alone. Without this a
   *  run that got the right rate for free is indistinguishable from one where
   *  somebody typed it. */
  control_hz_declared_by: "request" | "trained_fps";
  control_hz_trained_repo_id: string | null;
  /** Whether the trained rate was ever MEASURED, and what was measured. A
   *  PASS against a dataset that only declared its fps is a declaration
   *  agreeing with a declaration. */
  control_hz_trained_measured: boolean;
  control_hz_trained_measured_hz: number | null;
  control_hz_mismatch_override: boolean;
  repo_id?: string;
};

/**
 * The ceiling and the default, in seconds — `runs.py::MAX_ROLLOUT_DURATION_S`
 * and `routes_runs.py::DEFAULT_ROLLOUT_DURATION_S`.
 *
 * Copies, and the SERVER is the authority: it refuses over its own ceiling at
 * the door, and the child refuses again on the spec. These two only shape the
 * spinner, so a drift shows up as the backend's refusal on a number the box
 * offered — never as a run that outlives what the operator agreed to.
 */
export const MAX_ROLLOUT_DURATION_S = 900;
export const DEFAULT_ROLLOUT_DURATION_S = 60;

/** A run as the LIST reports it — no full spec, just enough for a row. */
export type RunSummary = {
  id: string;
  kind: RunKind;
  name: string | null;
  status: RunStatus;
  /** ISO 8601 UTC, as `runs.py::_now()` writes it — `"2026-08-26T19:33:50+00:00"`,
   *  never a unix number. The catalog sorts these as STRINGS
   *  (`r.get("started_at") or ""`), which only works because they are. Typed
   *  `number` here until 2026-08-27, which type-checked against a backend that
   *  has never sent one: every `AT` cell and every elapsed read `—`. */
  started_at: string | null;
  finished_at: string | null;
  tags?: string[];
  /** A one-line rendering of the spec, made by the backend. Shown verbatim:
   *  the list must not re-derive a summary the detail view would word
   *  differently. */
  spec_summary?: string | null;
};

/** A run as `GET /lab/runs/{id}` reports it: everything the list has, plus
 *  what was actually asked for and how it ended. */
export type Run = RunSummary & {
  /** Both shapes, because `kind` is what says which one arrived and a run
   *  read from the list has neither in hand. `Record<string, unknown>` stays:
   *  a spec is whatever the launching route wrote, and this pane must not
   *  drop a field it has no type for. */
  spec: Partial<TrainSpec & RolloutRecord> & Record<string, unknown>;
  /** The child's command line. The one thing that answers "what did this
   *  actually run" months later. */
  argv?: string[];
  exit_code?: number | null;
  error?: string | null;
};

/** One row of the metrics stream. Keys are whatever the trainer logged —
 *  which is the point: `MetricGrid` charts every numeric key it finds rather
 *  than the one the kit hardcoded. */
export type MetricRow = Record<string, number | string | boolean | null>;

export type MetricsPage = {
  /** Byte offset to resume from. Opaque; hand it straight back. */
  offset: number;
  rows: MetricRow[];
};

export type LogPage = { offset: number; text: string };

export type Checkpoint = {
  /** `null` for LeRobot's `last` symlink — the backend sends it that way ON
   *  PURPOSE, as the only thing distinguishing `last` from the numbered
   *  checkpoint it points at (`_checkpoint_wire`). Typed `number` here until
   *  2026-08-27. */
  step: number | null;
  path: string;
  /** False for a checkpoint directory the runner created but never finished
   *  writing — offering it as a rollout source would fail at load. */
  has_model: boolean;
};

/** Downsampled series for the compare view: `runs[id][key] = [[x, y], ...]`. */
export type CompareMetrics = { runs: Record<string, Record<string, [number, number][]>> };

export type LabSystem = {
  disk_free_bytes: number;
  lerobot_home: string;
  runs_dir: string;
  runner_python: string;
  /** Whether `~/venvs/haller-lab` is actually there — the "can this box train
   *  at all" fact, and the one worth saying before a spec is accepted. */
  runner_python_exists: boolean;
  torch_available: boolean;
  lerobot_version: string | null;
  /** The compare endpoint's own caps, read from `compare.py::MAX_KEYS` and
   *  `MAX_RUNS` rather than re-declared. Optional because a backend older than
   *  `f7b862c` does not publish them, and that absent case is pinned — see
   *  `COMPARE_MAX_KEYS_FALLBACK`. */
  compare_max_keys?: number;
  compare_max_runs?: number;
};

/* ─── the client ──────────────────────────────────────────────────────── */

/** A video URL, not a fetch: the element streams it and needs the server's
 *  Range support, so this never goes through `getJson`. */
export const labVideoUrl = (repoId: string, key: string, episode: number) =>
  `${BACKEND_URL}/lab/datasets/video` +
  qs({ repo_id: repoId, key, episode });

export const lab = {
  system: () => getJson<LabSystem>("/lab/system"),

  datasets: () => getJson<{ datasets: DatasetSummary[] }>("/lab/datasets"),

  detail: (repoId: string) =>
    getJson<DatasetDetail>(`/lab/datasets/detail${qs({ repo_id: repoId })}`),

  episodes: (q: EpisodeQuery) =>
    getJson<EpisodePage>(`/lab/datasets/episodes${qs({ ...q })}`),

  trace: (repoId: string, episode: number) =>
    getJson<Trace>(`/lab/datasets/trace${qs({ repo_id: repoId, episode })}`),

  /** The held-out plan. See `SplitPlan` for why this is never recomputed. */
  split: (repoId: string, evalSplit: number, seed: number, mode: SplitMode) =>
    getJson<SplitPlan>(
      `/lab/datasets/split${qs({ repo_id: repoId, eval_split: evalSplit, seed, mode })}`,
    ),

  mark: (repoId: string, episode: number, status: Mark, note?: string) =>
    postJson<{ ok: true }>("/lab/datasets/mark",
      note === undefined
        ? { repo_id: repoId, episode, status }
        : { repo_id: repoId, episode, status, note }),

  /** Several episodes at once. Every field but `repo_id`/`episodes` is
   *  optional and omitted when absent — a bulk call that sends
   *  `status: undefined` as `null` would clear marks the operator only meant
   *  to tag. */
  bulk: (body: {
    repo_id: string;
    episodes: number[];
    status?: Mark;
    tags_add?: string[];
    tags_remove?: string[];
  }) => postJson<{ updated: number }>("/lab/datasets/bulk", prune(body)),

  autoclassPreview: (repoId: string, mode: AutoclassMode, params: AutoclassParams = {}) =>
    postJson<AutoclassPreview>("/lab/datasets/autoclass/preview", {
      repo_id: repoId, mode, params,
    }),

  /** Applies the diff the operator read, by TOKEN.
   *
   *  The token binds that diff to the dataset state it was computed against,
   *  and the server recomputes it here — a 409 means the dataset moved
   *  underneath the dialog, so what would be applied is not what was
   *  approved. Never retry a 409 by re-previewing behind the operator. */
  autoclassApply: (repoId: string, token: string) =>
    postJson<AutoclassApplied>("/lab/datasets/autoclass/apply", {
      repo_id: repoId, token,
    }),

  autoclassRevert: (repoId: string, batch: string) =>
    postJson<{ reverted: number }>("/lab/datasets/autoclass/revert", {
      repo_id: repoId, batch,
    }),

  /** Destroys episodes, as a DETACHED JOB — the survivors are renumbered and
   *  the video is re-encoded, so this returns a run id to watch, not a result.
   *
   *  `expect_episodes` is the guard: the server refuses if the set moved
   *  between the dialog opening and the click, which is exactly when a
   *  renumbering would make the operator delete a different take. */
  prune: (repoId: string, backup: boolean, expectEpisodes: number[]) =>
    postJson<PruneStarted>("/lab/datasets/prune", {
      repo_id: repoId, backup, expect_episodes: expectEpisodes,
    }),

  /**
   * Destroys a whole dataset. The most destructive route in the surface.
   *
   * `confirm` must equal `repo_id` byte for byte or the server returns 400 —
   * the typed confirmation is on the WIRE, not only in the dialog, so a
   * mis-wired client cannot delete a dataset by accident. It is also
   * loopback-only (403 from the LAN: this is not something to reach through a
   * headset) and 409s while a run or the recorder has the dataset open.
   *
   * It does NOT remove the `<name>_old` copy a prune leaves behind — that is a
   * second dataset with its own row and its own delete. `is_backup` is what
   * tells the operator the pair exists.
   *
   * There is no undo and no backup: this box has one NVMe and no external
   * media (verified 2026-08-26).
   */
  deleteDataset: (repoId: string) =>
    deleteJson<{ repo_id: string; root: string; freed_bytes: number }>(
      `/lab/datasets${qs({ repo_id: repoId, confirm: repoId })}`),

  runs: (filter?: { kind?: RunKind | null; status?: RunStatus | null }) =>
    getJson<{ runs: RunSummary[] }>(`/lab/runs${qs({ ...filter })}`),

  run: (id: string) => getJson<Run>(`/lab/runs/${encodeURIComponent(id)}`),

  train: (spec: TrainSpec) => postJson<{ id: string }>("/lab/runs/train", { spec }),

  /**
   * Runs a trained policy ON THE ARMS, as a detached job. Loopback-only.
   *
   * Answers with a run id, not a result — the same shape as `train` and
   * `prune`, and for the same reason: the thing it starts outlives the
   * request. The run is where the handshake, the measured control rate and
   * the streamed target count are reported, and `stopRun` is what ends it
   * early.
   */
  rollout: (spec: RolloutSpec) =>
    postJson<{ id: string }>("/lab/runs/rollout", { spec }),

  runMetrics: (id: string, offset: number) =>
    getJson<MetricsPage>(
      `/lab/runs/${encodeURIComponent(id)}/metrics${qs({ offset })}`),

  runLog: (id: string, offset: number) =>
    getJson<LogPage>(`/lab/runs/${encodeURIComponent(id)}/log${qs({ offset })}`),

  checkpoints: (id: string) =>
    getJson<{ checkpoints: Checkpoint[] }>(
      `/lab/runs/${encodeURIComponent(id)}/checkpoints`),

  stopRun: (id: string) =>
    postJson<{ ok: true }>(`/lab/runs/${encodeURIComponent(id)}/stop`, {}),

  deleteRun: (id: string) =>
    deleteJson<{ ok: true }>(`/lab/runs/${encodeURIComponent(id)}`),

  /** Downsampled series for several runs at once. `max_points` is the whole
   *  point of the endpoint: a 200k-step run has 200k rows and the compare
   *  chart is 600px wide. */
  compareMetrics: (ids: string[], keys: string[], maxPoints = 600) =>
    getJson<CompareMetrics>(
      `/lab/runs/metrics${qs({
        ids: ids.join(","), keys: keys.join(","), max_points: maxPoints,
      })}`),
};

/** Drop undefined keys so an optional field is absent from the JSON rather
 *  than present as null. */
function prune<T extends Record<string, unknown>>(body: T): Partial<T> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(body)) if (v !== undefined) out[k] = v;
  return out as Partial<T>;
}

/* ─── derived readings ────────────────────────────────────────────────── */

/**
 * The operator-facing number for a STORED episode index.
 *
 * One implementation, because there must only ever be one. Episodes are stored
 * 0-based and talked about 1-based; every surface shows `Ep 4` next to a muted
 * `idx 3`, and the day those two are computed in two places is the day a
 * dialog offers to delete the wrong demonstration. Endpoints that already
 * carry a `label` use it; the ones that only send an index — the autoclassify
 * diff — come here.
 */
export function epLabel(index: number): string {
  return `Ep ${index + 1}`;
}

/**
 * Where this episode sits in its video, or null when the backend did not say.
 *
 * There is NO fallback, and that is deliberate. A v3.0 dataset packs many
 * episodes into one mp4: on `local/so101_pick_cube`, 46 episodes live in 7
 * files, and episodes 2..6 are all inside `file-001.mp4` starting at 0.0,
 * 15.53, 33.70, 48.60 and 60.57 seconds. Guessing "this episode starts at 0
 * and runs for its own duration" would open episode 6 at second 0 of an
 * 82-second file and play five other takes at the operator — who would then
 * mark the wrong one. Silence is the only safe answer to a missing slice.
 */
export function sliceFor(ep: LabEpisode, key: string | null): VideoSlice | null {
  return (key ? ep.videos?.[key] : undefined) ?? null;
}

/**
 * A stable identity for "which FILE is loaded".
 *
 * Keyed on (chunk, file) and never on the URL, because the URL carries the
 * episode: the server resolves episode -> file, so two episodes in one file
 * have different URLs and the same source. Changing episode inside one packed
 * mp4 must be a seek, not a load — five of the first seven episodes on the
 * real dataset share a file, and getting this wrong costs a full re-buffer on
 * every J/L keypress of a 46-episode triage pass.
 */
export function videoSrcKey(repoId: string, ep: LabEpisode, key: string | null): string | null {
  const s = sliceFor(ep, key);
  if (!s) return null;
  if (s.chunk_index !== undefined && s.file_index !== undefined) {
    return `${repoId}|${key}|${s.chunk_index}|${s.file_index}`;
  }
  return `${repoId}|${key}|${ep.episode_index}`;
}

/**
 * Whether a trace is complete enough to draw.
 *
 * The contract says every trace carries names, t, state and action, and a
 * conforming backend always sends them. This exists for the one that does not:
 * a partial body arriving with a 200 makes `trace.names.map` throw INSIDE a
 * render, and a render-phase throw takes the whole review pane down — the
 * operator loses the episode list and the marking controls too, over a chart.
 * Treated as "no trace" instead, which is a state every chart already draws.
 */
export function isDrawableTrace(t: Trace | null | undefined): t is Trace {
  return (
    !!t &&
    Array.isArray(t.names) &&
    Array.isArray(t.t) &&
    Array.isArray(t.state) &&
    Array.isArray(t.action)
  );
}

/** Channel indices grouped by arm side, from `trace.names`.
 *
 *  The names are the only rig signal a chart gets: a solo dataset's columns
 *  are bare (`shoulder_pan.pos`), a bimanual one's are side-prefixed
 *  (`left_shoulder_pan`). Anything that hardcodes five joints is wrong on
 *  one of the two datasets on this disk. */
export function armGroups(names: string[]): { side: string; channels: number[] }[] {
  const groups = new Map<string, number[]>();
  names.forEach((n, i) => {
    const m = /^(left|right)[_.]/.exec(n);
    const side = m ? m[1] : "arm";
    const list = groups.get(side);
    if (list) list.push(i);
    else groups.set(side, [i]);
  });
  return [...groups].map(([side, channels]) => ({ side, channels }));
}

/** A channel's short display name: the side prefix and the `.pos` suffix are
 *  already carried by the row group and the axis unit. */
export function shortChannel(name: string): string {
  return name.replace(/^(left|right)[_.]/, "").replace(/\.pos$/, "").replace(/_/g, " ");
}

/** True for a channel that is a gripper. Named rather than positional: the
 *  kit took `state[state.length - 1]`, which is the gripper on a solo arm and
 *  the RIGHT gripper on a bimanual one — silently charting one hand. */
export function isGripperChannel(name: string): boolean {
  return /gripper/i.test(name);
}

/** Every numeric key present across the metric rows, in first-seen order.
 *
 *  Bookkeeping columns are excluded: they are the axes, not series. Charting
 *  `step` against `step` is a diagonal line that teaches nothing. */
const AXIS_KEYS = new Set([
  // LeRobot's MetricsTracker.to_dict emits FOUR counters before any metric —
  // read off lerobot 0.6.1: {steps, samples, episodes, epochs, ...metrics}.
  // All four climb monotonically forever, so charting one against steps draws
  // a perfect diagonal: not wrong, just a chart that can never say anything.
  // `episodes` is the trap, because the word means something else everywhere
  // else in this UI — here it counts episode-passes the trainer has consumed,
  // not the dataset's episode list.
  "steps", "samples", "episodes", "epochs",
  "step", "epoch", "wall_s", "t", "elapsed_s", "timestamp",
]);

export function metricKeys(rows: MetricRow[]): string[] {
  const seen: string[] = [];
  const have = new Set<string>();
  for (const row of rows) {
    for (const [k, v] of Object.entries(row)) {
      if (have.has(k) || AXIS_KEYS.has(k)) continue;
      if (typeof v !== "number" || !Number.isFinite(v)) continue;
      have.add(k);
      seen.push(k);
    }
  }
  return seen;
}

export type MetricAxis = "step" | "epoch" | "wall";

/** The x value for one row on the chosen axis, or null when this row cannot
 *  be placed on it — an eval row logged without an epoch is not at epoch 0. */
export function metricX(row: MetricRow, axis: MetricAxis): number | null {
  const pick = (...keys: string[]): number | null => {
    for (const k of keys) {
      const v = row[k];
      if (typeof v === "number" && Number.isFinite(v)) return v;
    }
    return null;
  };
  if (axis === "epoch") return pick("epoch", "epochs");
  if (axis === "wall") return pick("wall_s", "elapsed_s", "t");
  return pick("step", "steps");
}

/** Every axis a row could be placed on. Ordered as the picker shows them. */
const METRIC_AXES: MetricAxis[] = ["step", "epoch", "wall"];

/**
 * The keys worth CHARTING — those with at least one sample that can be placed
 * on some axis.
 *
 * `metricKeys` answers "what did this run log"; this answers "what can be
 * drawn", and they differ because of lerobot's bookkeeping rows. The first
 * line of a training `metrics.jsonl` is
 * `{"kind":"split","train_episodes":28,"eval_episodes":7}` — real numbers with
 * no step, no epoch and no wall, so they sit on NO axis and never will.
 * Charting them drew two permanently empty cells labelled with metrics the run
 * does not measure over time.
 *
 * The filter is "unplottable on EVERY axis", not "on the current one", and the
 * difference is the whole point: a key that draws on `step` but not `epoch`
 * keeps its chart, so the "no samples on this axis" message stays a true
 * statement with a remedy behind it. Dropping per-axis would make charts
 * appear and vanish as the operator switches axes, and would turn a message
 * that means "look on another axis" into one that had silently become a lie.
 */
export function plottableMetricKeys(rows: MetricRow[]): string[] {
  return metricKeys(rows).filter((key) =>
    rows.some((row) => {
      const v = row[key];
      if (typeof v !== "number" || !Number.isFinite(v)) return false;
      return METRIC_AXES.some((axis) => metricX(row, axis) !== null);
    }),
  );
}
