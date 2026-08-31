"use client";

/**
 * Launch a training run, and say out loud what it will train on.
 *
 * Every other control here is a number the operator can check afterwards by
 * reading the run's spec. The held-out split is not: once the run is going,
 * "which demonstrations has the policy already seen" is invisible, and a val
 * curve computed over episodes the trainer was also fitting looks like a
 * policy that generalises. So the split is fetched from `/lab/datasets/split`
 * — the SAME code path the runner uses — and never recomputed here. Two
 * implementations of that answer drift, and when they do the val badges lie
 * about which takes are held out, which is the one error on this whole
 * surface that cannot be spotted by looking at it.
 *
 * The dataset picker offers non-backup datasets only. A pre-prune backup is a
 * complete copy of the dataset as it was before the rejected takes were
 * removed; training on it silently undoes the prune.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import {
  epLabel,
  isBusy,
  isForbidden,
  isMissing,
  lab,
  reason,
  REMOTE_REFUSED,
  rigLabel,
  trainableCount,
  trainJobName,
  unitsAlert,
  type DatasetDetail,
  type DatasetSummary,
  type PolicyType,
  type Run,
  type SplitMode,
  type SplitPlan,
  type TrainSpec,
} from "@/lib/lab";
import {
  Button, Chip, Field, Note, NumberInput, Panel, PanelHead, Refusal, Select,
  TextInput, WarnBox,
} from "@/components/lab/ui";
import { TagChips } from "@/components/lab/TagChips";
import { useSticky } from "@/components/cockpit/lib";

const POLICIES: readonly PolicyType[] = ["act", "smolvla", "pi0", "diffusion"];
const DEVICES = ["cuda", "cpu"] as const;
const MODES: readonly SplitMode[] = ["random", "recent"];

/** Listing every held-out episode turns the summary into a wall of labels.
 *  Eight is enough to recognise the sample and to see it change when the seed
 *  does; the rest is a count. */
const MAX_LISTED = 8;

/** Long enough that dragging a spinner is one request rather than twelve,
 *  short enough that the sentence has settled before the eye reaches it. */
const SPLIT_DEBOUNCE_MS = 200;

type Form = {
  policy: PolicyType;
  steps: number;
  batch: number;
  evalSplit: number;
  seed: number;
  mode: SplitMode;
  evalEvery: number;
  saveEvery: number;
  workers: number;
  device: string;
  /** Empty means "use the derived name" — see `trainJobName`. The box shows
   *  the derived name as a placeholder so the operator can read it without it
   *  becoming an override, and `auto` next to the box is the way back once
   *  one has been typed. Sticky like the rest of the form, which is exactly
   *  why the way back has to exist: a name typed once outlives the run it was
   *  typed for and would otherwise re-label every run after it. */
  jobName: string;
  tags: string[];
};

const DEFAULTS: Form = {
  policy: "act",
  steps: 20000,
  batch: 8,
  evalSplit: 0.2,
  seed: 42,
  mode: "random",
  evalEvery: 1000,
  saveEvery: 5000,
  workers: 4,
  device: "cuda",
  jobName: "",
  tags: [],
};

const clamp = (v: number, lo: number, hi: number) =>
  Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : lo;

const ep = (n: number) => `${n} episode${n === 1 ? "" : "s"}`;

/** One settled answer for one split key. `plan: null, error: null` is the
 *  third outcome — the backend has no runner, which is a fact about the build
 *  and not about the key, so it is recorded as "asked, nothing to show". */
type PlanRead = { plan: SplitPlan | null; error: string | null };

export function TrainLauncher({
  datasets,
  repoId,
  onRepoId,
  onLaunched,
}: {
  datasets: DatasetSummary[];
  repoId: string | null;
  onRepoId: (r: string) => void;
  onLaunched: (run: Run) => void;
}) {
  // Sticky rather than component state: this panel unmounts every time the
  // operator looks at Review, and a launcher that forgets a tuned batch size
  // on a tab switch is one the operator stops trusting with the rest.
  const [form, setForm] = useSticky<Form>("lab.train.launcher", DEFAULTS);
  const patch = useCallback(
    (p: Partial<Form>) => setForm({ ...form, ...p }),
    [form, setForm],
  );

  const [busy, setBusy] = useState(false);
  /** The backend's own sentence when it refused (409). Persists until the next
   *  attempt — it names a state the operator has to clear, which is not a
   *  thing to say in a toast that vanishes. */
  const [refusal, setRefusal] = useState<string | null>(null);
  /** 404/501 from split or train: this build has no runner at all. A property
   *  of the backend, not of the form, so it never clears on its own. */
  const [noRunner, setNoRunner] = useState(false);

  /* Both caches are STATE, not refs. A ref cannot be read during render, so a
     hit could only reach the screen through a setState from inside the effect
     — a second render for an answer that was already in hand. As state the hit
     is just a lookup below. */
  const [plans, setPlans] = useState<ReadonlyMap<string, PlanRead>>(() => new Map());
  const [details, setDetails] = useState<ReadonlyMap<string, DatasetDetail>>(() => new Map());

  const trainable = datasets.filter((d) => d.is_backup === false);
  const picked = repoId ? datasets.find((d) => d.repo_id === repoId) ?? null : null;
  const pickedIsBackup = picked?.is_backup === true;

  const evalSplit = clamp(form.evalSplit, 0, 0.9);

  /* ── the split, from the server ───────────────────────────────────────
     Cached per (repo | split | seed | mode) so re-rendering the form — which
     happens on every keystroke in the job-name box — never refetches, and
     debounced so dragging a spinner is one request rather than one per step.

     The sentence below reads the plan out of that cache BY KEY rather than
     from a `plan` that gets dropped when the key changes: same effect, no
     window in which the previous plan is on screen under the new seed, which
     is the exact lie this panel exists to prevent. `planning` falls out of it
     — it is "this key has no answer yet" and nothing else. */
  const { seed, mode } = form;
  const splitKey =
    repoId && evalSplit > 0 ? `${repoId}|${evalSplit}|${seed}|${mode}` : null;
  const read = splitKey === null ? undefined : plans.get(splitKey);
  const plan = read?.plan ?? null;
  const planError = read?.error ?? null;
  const planning = splitKey !== null && read === undefined;

  useEffect(() => {
    if (!repoId || splitKey === null || plans.has(splitKey)) return;
    let cancelled = false;
    const settle = (r: PlanRead) => {
      if (!cancelled) setPlans((m) => new Map(m).set(splitKey, r));
    };
    const timer = setTimeout(async () => {
      try {
        settle({ plan: await lab.split(repoId, evalSplit, seed, mode), error: null });
      } catch (e) {
        if (cancelled) return;
        // 404/501 is a property of the build, not of this key — it gets the
        // banner, and the key settles with nothing to show.
        const gone = isMissing(e);
        if (gone) setNoRunner(true);
        settle({ plan: null, error: gone ? null : reason(e) });
      }
    }, SPLIT_DEBOUNCE_MS);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [repoId, splitKey, plans, evalSplit, seed, mode]);

  /* ── the dataset's own episode list ───────────────────────────────────
     Two things come out of it and neither can be guessed: the kept indices
     that go into the spec, and index → label. The label is the server's
     mapping (1-based in conversation, 0-based on disk), so deriving it from
     the index would name a different take in the summary than the one the
     trainer holds out. */
  const detail = repoId === null ? null : details.get(repoId) ?? null;

  // From the LISTING's three scalars, so the warning is on screen the moment a
  // dataset is picked rather than one request later. The detail's sentence is
  // read separately below and only for the tooltip: handing `detail` itself to
  // a helper makes the React compiler treat it as possibly mutated and it then
  // refuses to preserve the memos above that depend on it.
  const pickedUnits = unitsAlert(picked?.units);
  // The server's own sentence, which names the joints with no calibrated
  // range. A plain string read, for the reason just above.
  const pickedUnitsNote = detail?.units?.note;

  useEffect(() => {
    if (!repoId || details.has(repoId)) return;
    let cancelled = false;
    (async () => {
      try {
        const d = await lab.detail(repoId);
        if (!cancelled) setDetails((m) => new Map(m).set(repoId, d));
      } catch {
        // Not fatal and not worth a banner: the run still launches, it just
        // launches without a pinned episode list. See `keptIdx` below.
      }
    })();
    return () => { cancelled = true; };
  }, [repoId, details]);

  /** Everything the operator has not rejected — which is broader than the
   *  picker's `keep` count, because an unwatched take is not a rejected one.
   *  Sent explicitly so the run still means the same thing after the marks
   *  move on; omitted rather than guessed when the detail read failed. */
  const keptIdx = useMemo(
    () =>
      detail && detail.repo_id === repoId
        ? detail.episodes.filter((e) => e.mark !== "reject").map((e) => e.episode_index)
        : null,
    [detail, repoId],
  );

  /** index → the operator's number for it, as the SERVER spells it. The split
   *  plan carries indices only, so the summary would otherwise be naming
   *  `Ep 3` off index 3 — right today and a lie the moment the mapping is not
   *  `index + 1`. `epLabel` is the fallback for a failed detail read, and is
   *  the one implementation of that guess. */
  const labels = useMemo(() => {
    const m = new Map<number, string>();
    if (detail && detail.repo_id === repoId) {
      for (const e of detail.episodes) m.set(e.episode_index, `Ep ${e.label}`);
    }
    return m;
  }, [detail, repoId]);

  // `keep + unset`, because an unset episode is handed to the trainer — "I
  // have not judged this" is not "throw it away". The backend publishes the
  // same sum as `marks.train`; `trainableCount` prefers it and falls back.
  const keptCount = keptIdx?.length ?? trainableCount(picked?.marks);

  /* ── what the policy is allowed to look at ────────────────────────────
     LeRobot builds a policy's observation space out of the DATASET: every
     `observation.*` column becomes an input, and nothing can opt one out. So
     a column added by a schema migration becomes a policy input that nobody
     chose — which is what happened here on 2026-08-29, when `observation.
     effort`, `observation.base` and `observation.wall_clock` arrived and ACT
     started training on a per-episode CLOCK it could fit instead of looking
     at the image.

     Every observation column in the dataset, in the dataset's own order. */
  const obsColumns = useMemo(
    () =>
      detail && detail.repo_id === repoId
        ? Object.keys(detail.features ?? {}).filter((k) => k.startsWith("observation"))
        : [],
    [detail, repoId],
  );

  /* Per repo, because a column list is a fact about ONE dataset — sticky
     state would carry `observation.effort` onto a dataset that has no such
     column, and the launch would 400 on a name the operator never typed. An
     absent entry means "not touched", which is what makes the default below
     follow the dataset instead of freezing at first render. */
  const [inputsByRepo, setInputsByRepo] =
    useState<ReadonlyMap<string, readonly string[]>>(() => new Map());

  /** The ticked columns. The default comes from the BACKEND — the same rule
   *  that validates the choice on the way back in — so the form cannot show
   *  one space and the run train on another. */
  const chosenInputs = useMemo(() => {
    if (!repoId || obsColumns.length === 0) return null;
    const touched = inputsByRepo.get(repoId);
    const fallback = detail?.policy_inputs_default ?? obsColumns;
    // Filtered through `obsColumns` so the order is always the dataset's and
    // a stale name can never reach the spec.
    const wanted = new Set(touched ?? fallback);
    return obsColumns.filter((k) => wanted.has(k));
  }, [repoId, obsColumns, inputsByRepo, detail]);

  const toggleInput = useCallback(
    (key: string) => {
      if (!repoId || chosenInputs === null) return;
      const next = chosenInputs.includes(key)
        ? chosenInputs.filter((k) => k !== key)
        : obsColumns.filter((k) => k === key || chosenInputs.includes(k));
      setInputsByRepo((m) => new Map(m).set(repoId, next));
    },
    [repoId, chosenInputs, obsColumns],
  );

  const excluded = chosenInputs === null
    ? 0
    : obsColumns.length - chosenInputs.length;

  /* ── names already in use ─────────────────────────────────────────────
     Only so a relaunch of an identical config comes out as `_v2` rather than
     as a second row with the first one's name. Read once per mount — the
     panel remounts on every tab switch — and extended locally at launch,
     because the run that just went out is the one most likely to be repeated
     next. Never blocking and never surfaced: a failed read costs a duplicate
     LABEL, and the run id underneath it is the server's and unique either
     way. Unfiltered by status on purpose — a set that honoured the run
     list's filter chips would hand out a name that is on screen. */
  const [taken, setTaken] = useState<ReadonlySet<string>>(() => new Set());
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const { runs } = await lab.runs({ kind: "train" });
        if (cancelled) return;
        setTaken((prev) => {
          const next = new Set(prev);
          for (const r of runs) if (r.name) next.add(r.name);
          return next;
        });
      } catch {
        // See above: the counter is a courtesy, not a correctness guarantee.
      }
    })();
    return () => { cancelled = true; };
  }, []);

  /** What the run is called when nobody types a name: the policy, the
   *  dataset, and every hyperparameter that changes the outcome. See
   *  `trainJobName` — the spelling lives there, next to the spec it names, so
   *  this component is not a second place to look for it. */
  const derivedName = trainJobName(
    {
      repoId,
      policy: form.policy,
      episodes: keptCount,
      steps: form.steps,
      batch: form.batch,
      evalSplit,
      seed: form.seed,
      mode: form.mode,
    },
    taken,
  );
  const custom = form.jobName.trim();
  const jobName = custom || derivedName;

  const launch = async () => {
    if (!repoId || noRunner) return;
    setBusy(true);
    setRefusal(null);
    // The boxes can be emptied, which reads as 0. A run with 0 steps or a
    // batch of 0 fails a minute after launch instead of here, so the floors
    // from the inputs are applied to the spec as well.
    const spec: TrainSpec = {
      repo_id: repoId,
      policy_type: form.policy,
      steps: Math.max(1, Math.round(form.steps)),
      batch_size: Math.max(1, Math.round(form.batch)),
      eval_split: evalSplit,
      eval_seed: Math.round(form.seed),
      eval_mode: form.mode,
      eval_steps: Math.max(0, Math.round(form.evalEvery)),
      save_freq: Math.max(0, Math.round(form.saveEvery)),
      num_workers: clamp(Math.round(form.workers), 0, 32),
      device: form.device,
      job_name: jobName,
      tags: form.tags,
      ...(keptIdx ? { episodes: keptIdx } : {}),
      // Omitted rather than guessed when the detail read failed: absent means
      // "LeRobot's own rule", and sending a list built from a dataset this
      // form never managed to read would pin the wrong space confidently.
      ...(chosenInputs && chosenInputs.length > 0
        ? { policy_inputs: [...chosenInputs] }
        : {}),
    };
    try {
      const { id } = await lab.train(spec);
      toast.success(`run ${id} queued`);
      // This name is spent. Next launch of the same config derives `_v2`
      // without waiting for a list read that may never be taken — the panel
      // often gets folded away the moment the run starts.
      setTaken((prev) => new Set(prev).add(spec.job_name));
      // The POST answers with an id, not a run. Read the run back so the pane
      // gets the runner's own record; if that second call fails the launch
      // still happened, so hand over what the accepted POST proves — the id,
      // the spec, and `queued`, which is what a just-accepted run is.
      let run: Run;
      try {
        run = await lab.run(id);
      } catch {
        run = {
          id, kind: "train", name: spec.job_name, status: "queued",
          started_at: null, finished_at: null, tags: spec.tags, spec,
        };
      }
      onLaunched(run);
    } catch (e) {
      if (isMissing(e)) setNoRunner(true);
      // Launching is one of the loopback-gated routes: from the headset's
      // browser this 403s, and the operator's next move is a real one.
      else if (isForbidden(e)) setRefusal(REMOTE_REFUSED);
      else if (isBusy(e)) setRefusal(reason(e));
      else toast.error(`launch failed: ${reason(e)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    // `min-h-0 flex-1` so the Panel FILLS whatever height its parent caps it
    // at and its body scrolls inside. Without it the Panel sizes to its
    // content, overflows the cap, and gets clipped instead — which cost the
    // job-name field and the summary sentence off the bottom of the form.
    <Panel className="min-h-0 flex-1">
      <PanelHead title="new run" right={jobName} />

      <div className="flex min-h-0 flex-1 flex-col gap-2.5 overflow-y-auto p-3">
        <Field
          label="dataset"
          hint={
            picked && !pickedIsBackup ? (
              <>
                {picked.task ?? "no task recorded"} · {rigLabel(picked.rig)}
                {/* On the launcher because this is where the consequence is
                    paid: a policy trained on columns whose unit nobody
                    declared learns that unit, and the run it produces then
                    commands the arms in it. The chips on the shelf and in
                    review warn about reading the numbers; this warns about
                    fitting them. */}
                {pickedUnits && (
                  <>
                    {" · "}
                    <span
                      style={{ color: "var(--haller-warn)" }}
                      title={pickedUnitsNote ?? pickedUnits.note}
                      data-units-alert
                    >
                      {pickedUnits.label}
                    </span>
                  </>
                )}
              </>
            ) : pickedIsBackup ? (
              <span style={{ color: "var(--haller-warn)" }}>
                pre-prune backup — a run over it undoes the prune
              </span>
            ) : (
              "pre-prune backups are not offered"
            )
          }
        >
          <Select
            value={repoId ?? ""}
            onChange={(e) => { if (e.target.value) onRepoId(e.target.value); }}
          >
            {!repoId && <option value="">no dataset picked</option>}
            {/* A backup can still arrive as the pane's repoId — from a deep
                link, or from the shelf. Showing it is more honest than
                silently displaying the row above it. */}
            {pickedIsBackup && picked && (
              <option value={picked.repo_id}>{picked.repo_id} · backup</option>
            )}
            {trainable.map((d) => (
              <option key={d.repo_id} value={d.repo_id}>
                {d.repo_id} · {trainableCount(d.marks)} of {d.episodes} to train on
              </option>
            ))}
          </Select>
        </Field>

        {/* A prune renumbers the survivors. A mark made before that names an
            index, and the take now at that index is a different one — so the
            kept set below is a list of numbers, not of demonstrations. */}
        {picked?.stale === true && (
          <WarnBox>
            The marks on this dataset are stale — an episode was pruned and the
            survivors renumbered. Re-check them under Review before training,
            or the kept set pins whatever slid into those indices.
          </WarnBox>
        )}

        {/* The sentence this whole panel exists to be able to write — directly
            under the dataset it is about, and ABOVE the fold. It used to sit
            last, under the tags, which on a 1080p rail is the one part of the
            form nobody sees without scrolling: the panel's whole argument,
            filed where it could not be read. */}
        <Note className="rounded-md border border-border bg-muted px-2.5 py-2">
          <SplitSentence
            repoId={repoId}
            evalSplit={evalSplit}
            mode={form.mode}
            plan={plan}
            planning={planning}
            planError={planError}
            keptCount={keptCount}
            labels={labels}
          />
        </Note>

        {/* THREE columns in the 26rem rail, not two. Every control in here is
            a short number or a two-word select, and at 9.5rem they made five
            rows of a form that has to share the rail with the run list —
            300px of height spent on whitespace either side of `8`. */}
        <div className="grid gap-x-2.5 gap-y-2 [grid-template-columns:repeat(auto-fill,minmax(6.75rem,1fr))]">
          <Field label="policy">
            <Select
              value={form.policy}
              onChange={(e) => patch({ policy: e.target.value as PolicyType })}
            >
              {POLICIES.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </Select>
          </Field>

          <Field label="steps">
            <NumberInput
              value={form.steps}
              min={1}
              step={1000}
              onChange={(v) => patch({ steps: v })}
            />
          </Field>

          <Field label="batch">
            <NumberInput
              value={form.batch}
              min={1}
              onChange={(v) => patch({ batch: v })}
            />
          </Field>

          <Field label="eval split" hint="held-out episodes">
            <NumberInput
              value={form.evalSplit}
              min={0}
              max={0.9}
              step={0.05}
              onChange={(v) => patch({ evalSplit: clamp(v, 0, 0.9) })}
            />
          </Field>

          <Field label="split seed" hint="same seed = same split">
            <span className="flex min-w-0 items-center gap-1.5">
              <NumberInput
                value={form.seed}
                step={1}
                onChange={(v) => patch({ seed: v })}
              />
              <Button
                aria-label="reshuffle the held-out sample"
                title="draw a different held-out sample"
                onClick={() => patch({ seed: Math.floor(Math.random() * 100000) })}
              >
                ↻
              </Button>
            </span>
          </Field>

          <Field label="held out" hint="how the eval set is chosen">
            <Select
              value={form.mode}
              onChange={(e) => patch({ mode: e.target.value as SplitMode })}
            >
              {MODES.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </Select>
          </Field>

          <Field label="eval every" hint="steps (0 = off)">
            <NumberInput
              value={form.evalEvery}
              min={0}
              step={100}
              onChange={(v) => patch({ evalEvery: v })}
            />
          </Field>

          <Field label="save every" hint="steps">
            <NumberInput
              value={form.saveEvery}
              min={0}
              step={1000}
              onChange={(v) => patch({ saveEvery: v })}
            />
          </Field>

          <Field label="workers">
            <NumberInput
              value={form.workers}
              min={0}
              max={32}
              onChange={(v) => patch({ workers: v })}
            />
          </Field>

          <Field label="device">
            <Select
              value={form.device}
              onChange={(e) => patch({ device: e.target.value })}
            >
              {DEVICES.map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </Select>
          </Field>
        </div>

        {/* The observation space, ticked rather than inherited. LeRobot takes
            EVERY `observation.*` column when nobody says otherwise, so this
            row is the only place a column added by a migration can be kept
            out of the policy. */}
        {chosenInputs !== null && (
          <div
            className="flex flex-col gap-1.5"
            role="group"
            aria-label="policy inputs"
          >
            <span className="label-tracked text-muted-foreground">
              policy inputs
            </span>
            <div className="flex flex-wrap gap-1.5">
              {obsColumns.map((key) => (
                <Chip
                  key={key}
                  on={chosenInputs.includes(key)}
                  onClick={() => toggleInput(key)}
                  title={
                    chosenInputs.includes(key)
                      ? `${key} — the policy reads this`
                      : `${key} — held out of the observation space`
                  }
                >
                  {key.replace(/^observation\./, "")}
                </Chip>
              ))}
            </div>
            <span className="text-[10px] text-pretty text-muted-foreground">
              {chosenInputs.length === 0 ? (
                <span style={{ color: "var(--haller-warn)" }}>
                  nothing ticked — the run will launch with LeRobot&apos;s own
                  set, which is every column above
                </span>
              ) : excluded > 0 ? (
                <>
                  {excluded} column{excluded === 1 ? "" : "s"} held out of the
                  observation space · untick anything the policy should not be
                  able to fit
                </>
              ) : (
                "every observation column goes to the policy · untick the ones it should not be able to fit"
              )}
            </span>
          </div>
        )}

        {/* Derived, not blank. The name is what the run list shows, and a
            name that does not move when the hyperparameters do is how three
            different runs end up looking like one. Typing still wins — this
            is a default, not a rule — and `auto` is the way back. */}
        <Field
          label="job name"
          hint={
            custom
              ? "your name — clear the box, or press auto, to follow the settings again"
              : "follows the dataset and the settings above · names the checkpoint directory"
          }
        >
          <span className="flex min-w-0 items-center gap-1.5">
            <TextInput
              value={form.jobName}
              placeholder={derivedName}
              onChange={(e) => patch({ jobName: e.target.value })}
            />
            <Button
              disabled={!custom}
              onClick={() => patch({ jobName: "" })}
              title={
                custom
                  ? `drop this name and follow the settings: ${derivedName}`
                  : "already following the settings"
              }
            >
              auto
            </Button>
          </span>
        </Field>

        <div className="flex flex-col gap-1.5" role="group" aria-label="run tags">
          <span className="label-tracked text-muted-foreground">tags</span>
          <TagChips
            tags={form.tags}
            onAdd={(t) => patch({ tags: form.tags.includes(t) ? form.tags : [...form.tags, t] })}
            onRemove={(t) => patch({ tags: form.tags.filter((x) => x !== t) })}
          />
        </div>

        {noRunner && <Refusal>this backend has no lab runner</Refusal>}
        {refusal && <Refusal>refused: {refusal}</Refusal>}
      </div>

      <div className="flex shrink-0 items-center justify-between gap-2 border-t border-border px-3 py-2.5">
        <span className="min-w-0 truncate font-mono text-[10px] text-muted-foreground">
          {keptIdx
            ? `${ep(keptIdx.length)} pinned to this run`
            : repoId
              ? "episode list unavailable — the runner picks the set"
              : "—"}
        </span>
        <Button
          tone="primary"
          disabled={busy || !repoId || noRunner}
          onClick={launch}
          title={
            noRunner
              ? "this backend has no lab runner"
              : repoId
                ? undefined
                : "pick a dataset first"
          }
        >
          {busy ? "starting…" : "start training"}
        </Button>
      </div>
    </Panel>
  );
}

/* ---- the English summary ------------------------------------------------ */

/**
 * What the trainer will and will not see, in a sentence.
 *
 * Every number below comes from the server's plan. The only arithmetic here
 * is `length` and the eight-label cap.
 */
function SplitSentence({
  repoId,
  evalSplit,
  mode,
  plan,
  planning,
  planError,
  keptCount,
  labels,
}: {
  repoId: string | null;
  evalSplit: number;
  mode: SplitMode;
  plan: SplitPlan | null;
  planning: boolean;
  planError: string | null;
  keptCount: number;
  labels: Map<number, string>;
}) {
  if (!repoId) {
    return <>Pick a dataset above and this line will name the episodes it holds out.</>;
  }
  if (planError) {
    return <>The held-out plan could not be read: {planError}</>;
  }
  if (evalSplit <= 0) {
    return (
      <>
        Trains on {ep(keptCount)}; no held-out episodes — eval loss will not be
        plotted.
      </>
    );
  }
  if (!plan) {
    return <>{planning ? "Resolving the held-out plan…" : "No held-out plan yet."}</>;
  }

  const held = plan.eval_episodes;
  const trains = plan.train_episodes.length;
  if (held.length === 0) {
    return (
      <>
        Trains on {ep(trains)}; the split held nothing back at this fraction —
        eval loss will not be plotted.
      </>
    );
  }

  const listed = held.slice(0, MAX_LISTED);
  const rest = held.length - listed.length;
  const how = mode === "recent" ? "the most recent" : "a random sample";
  // The server's own spelling where the detail read gave us one; `epLabel` —
  // the single shared implementation of index → operator number — where it
  // did not.
  const named = listed.map((i) => labels.get(i) ?? epLabel(i));

  return (
    <>
      Trains on {ep(trains)}; holds out {held.length} for eval loss — {how}:{" "}
      {named.join(", ")}
      {rest > 0 ? `… (+${rest} more)` : ""}.
    </>
  );
}
