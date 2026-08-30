"use client";

/**
 * Roll out a trained checkpoint — the one control on this surface that moves
 * an arm.
 *
 * **Nothing is handed over here.** The run this launches is a `/lab/runs`
 * route like any other: a detached child loads the checkpoint, runs inference
 * and streams target degrees to the server over loopback, and the SERVER
 * keeps the Feetech bus and commits those targets through the same chain
 * every other leader goes through. That is why this is a dialog over the Lab
 * and not a mode in the cockpit — the policy is a leader, like the Quest, and
 * the commit chain does not care which one is talking.
 *
 * ## The rate is the load-bearing number, so this box never picks one
 *
 * A policy trained at 30 Hz and driven at 25 is not a slower policy: its
 * action deltas are sized for 33 ms steps and would be applied over 40. The
 * server owns that check — it is the only place that can read both the
 * declared rate and the fps of the dataset the checkpoint records — and the
 * default here is therefore to DECLARE NOTHING and let it use the trained
 * rate. Choosing "declare a rate" is a deliberate act that can be refused,
 * and the refusal is quoted verbatim because it names both numbers.
 *
 * The fps shown beside that choice comes from the dataset the TRAINING RUN
 * recorded, read through `lab.detail`. It is displayed as what it is — a fact
 * about that dataset — and never sent: the spec carries `control_hz` only
 * when the operator typed one. Two paths to the same number is how a UI ends
 * up reassuring an operator about a check it did not make.
 *
 * ## Which step
 *
 * Opened from a checkpoint row, this dialog runs the row that was clicked and
 * shows no picker: the list behind it IS the choice. Opened from a surface
 * with no such list — the compare view, where the question is "that run won,
 * run it" — the same choice has to be made somewhere, so the caller hands the
 * whole set and the header becomes a select. One dialog either way; the only
 * difference is whether the step was already answered.
 *
 * ## Which arm
 *
 * A dataset whose columns carry no side prefix (`rig === "solo"` — the shape
 * the kit's own datasets have) cannot say which arm it was recorded from, so
 * the child refuses to guess and this dialog will not launch without an
 * answer. The rig read here can only ever cause the QUESTION to be asked; the
 * child re-derives the rig from the dataset itself and refuses on its own, so
 * a wrong read here costs a needless field and never a wrong arm.
 */
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";

import {
  DEFAULT_ROLLOUT_DURATION_S, MAX_ROLLOUT_DURATION_S,
  isBusy, isForbidden, isMissing, isRefused, lab, reason, REMOTE_REFUSED,
  rigLabel,
  type Checkpoint, type DatasetDetail, type Rig, type RolloutSpec, type Run,
} from "@/lib/lab";
import {
  Button, Dialog, Field, Note, NumberInput, Refusal, Select, TextInput, WarnBox,
} from "@/components/lab/ui";
import { checkpointName, stepLabel } from "@/components/lab/CheckpointList";

const DEVICES = ["cuda", "cpu"] as const;

/** The rate the run is driven at, and where it came from. `trained` sends no
 *  `control_hz` at all — see the note above. */
type RateSource = "trained" | "declared";

/** The rate to prefill the box with when the dataset could not be read. Only
 *  ever a STARTING POINT for a number the operator then owns: it is offered
 *  under "declare a rate", where the server checks it and refuses. */
const FALLBACK_HZ = 30;

/** Persists until the operator acts on it, so it is a panel and not a toast.
 *  Same shape as `PruneDialog`'s, plus the rung this route brings: a 400 here
 *  is the rate gate — a decision, not a fault — and reads as one. */
type Failure = { text: string; tone: "warn" | "fault" };

function failureOf(e: unknown): Failure {
  if (isMissing(e)) return { text: "this backend cannot run a rollout", tone: "warn" };
  if (isForbidden(e)) return { text: REMOTE_REFUSED, tone: "warn" };
  if (isRefused(e) || isBusy(e)) return { text: reason(e), tone: "warn" };
  return { text: reason(e), tone: "fault" };
}

const hz = (n: number) => `${Number(n.toFixed(3))} Hz`;

export function RolloutDialog({
  checkpoint,
  checkpoints,
  repoId,
  onClose,
  onLaunched,
}: {
  /** The one this opens on, and the one it launches unless the operator
   *  changes it below. */
  checkpoint: Checkpoint;
  /**
   * Every checkpoint the step may be changed to WITHOUT leaving the dialog,
   * `checkpoint` among them. Omitted where the caller already showed a list
   * and the operator clicked a row — see "Which step" above.
   *
   * Only loadable ones belong here: a partial directory is a step this dialog
   * would then have to refuse, and offering it in a select is offering a
   * choice that cannot be taken.
   */
  checkpoints?: Checkpoint[];
  /** The dataset the run that WROTE this checkpoint trained on, off its spec.
   *  Null when the run does not record one — then the rig question is asked
   *  rather than answered. */
  repoId: string | null;
  onClose: () => void;
  onLaunched: (run: Run) => void;
}) {
  /** The step to run, by PATH — the path is what the route loads and the
   *  only thing about a checkpoint that is unique. */
  const [path, setPath] = useState(checkpoint.path);
  const [duration, setDuration] = useState(DEFAULT_ROLLOUT_DURATION_S);
  const [rateSource, setRateSource] = useState<RateSource>("trained");
  const [declaredHz, setDeclaredHz] = useState<number | null>(null);
  const [device, setDevice] = useState<string>("cuda");
  const [side, setSide] = useState("");
  /** The operator's instruction, or null for "whatever the dataset recorded".
   *  Null rather than a prefilled string so the dataset read cannot race a
   *  box someone is already typing in, and so a CLEARED box stays cleared —
   *  `""` is an answer here, not an absence. */
  const [taskEdit, setTaskEdit] = useState<string | null>(null);
  const [allowMismatch, setAllowMismatch] = useState(false);
  const [allowSlow, setAllowSlow] = useState(false);

  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<Failure | null>(null);

  /** What is actually launched. Falls back to the one the dialog opened on, so
   *  a caller whose list does not contain it still launches the checkpoint it
   *  asked for rather than nothing. */
  const choices = checkpoints ?? [checkpoint];
  const active = choices.find((c) => c.path === path) ?? checkpoint;

  /** The training dataset, for the three facts this dialog cannot invent: the
   *  rig (does it have to ask for a side), the fps (what "as trained" will
   *  mean), and the instruction the demonstrations were recorded under. */
  const [detail, setDetail] = useState<DatasetDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);

  useEffect(() => {
    if (!repoId) return;
    let cancelled = false;
    (async () => {
      try {
        const d = await lab.detail(repoId);
        if (!cancelled) setDetail(d);
      } catch (e) {
        if (!cancelled) setDetailError(reason(e));
      }
    })();
    return () => { cancelled = true; };
  }, [repoId]);

  /** `tasks[0]` is the house rule for "the dataset's instruction" —
   *  `routes_datasets._first_task` takes the same one, because a take here is
   *  driven against one. */
  const recordedTask = detail?.tasks?.[0] ?? "";
  const task = taskEdit ?? recordedTask;

  const rig: Rig | null = detail?.rig ?? null;
  const trainedFps = detail?.fps ?? null;
  /** An unprefixed dataset names no arm, so the spec must. Unknown is not
   *  "no": the field is offered either way and only the GATE needs certainty. */
  const needsSide = rig === "solo";
  const sideMissing = needsSide && side === "";

  /** What goes on the wire when a rate is declared. Never defaulted silently:
   *  substituting a rate is the one thing this dialog must not do. */
  const declared = declaredHz ?? trainedFps ?? FALLBACK_HZ;
  const rateBad = rateSource === "declared" && !(declared > 0);

  const seconds = Math.min(MAX_ROLLOUT_DURATION_S, Math.max(1, Math.round(duration)));

  const launch = useCallback(async () => {
    setBusy(true);
    setFailure(null);
    const spec: RolloutSpec = {
      policy_path: active.path,
      duration_s: seconds,
      device,
      // Absent on the `trained` path — that absence IS the request for the
      // trained rate, and it is what makes the run's own record say the rate
      // was not somebody's choice.
      ...(rateSource === "declared"
        ? { control_hz: declared, ...(allowMismatch ? { allow_rate_mismatch: true } : {}) }
        : {}),
      ...(allowSlow ? { allow_slow: true } : {}),
      ...(side ? { side } : {}),
      ...(task.trim() ? { task: task.trim() } : {}),
      // The dataset's, not the operator's — there is no box for it. Sent only
      // when the dataset was actually read, so the policy is shown the robot
      // type its training data carried rather than a literal from this file.
      ...(detail?.robot_type ? { robot_type: detail.robot_type } : {}),
    };
    try {
      const { id } = await lab.rollout(spec);
      toast.success(`rollout ${id} queued`);
      // The POST answers with an id, not a run. Read it back so the pane gets
      // the runner's own record; if that second call fails the launch still
      // happened, so hand over what the accepted POST proves.
      let run: Run;
      try {
        run = await lab.run(id);
      } catch {
        run = {
          id, kind: "rollout", name: null, status: "queued",
          started_at: null, finished_at: null, tags: [], spec,
        };
      }
      onLaunched(run);
      onClose();
    } catch (e) {
      setFailure(failureOf(e));
    } finally {
      setBusy(false);
    }
  }, [
    allowMismatch, allowSlow, active.path, declared, detail, device,
    onClose, onLaunched, rateSource, seconds, side, task,
  ]);

  // A directory with no model file fails at load, three seconds in. Refused
  // at the door instead, for the same reason the route refuses an over-long
  // duration: a button that launches a doomed run and reports it dead
  // afterwards is worse than one that will not launch it.
  const blocked = busy || sideMissing || rateBad || !active.has_model;
  const footer = (
    <>
      <Button onClick={onClose} disabled={busy}>cancel</Button>
      <Button
        tone="danger"
        onClick={launch}
        disabled={blocked}
        title={
          sideMissing
            ? "this policy's columns name no arm — say which one it drives"
            : rateBad
              ? "a declared rate must be greater than zero"
              : undefined
        }
      >
        {busy ? "starting…" : "start rollout · the arm moves"}
      </Button>
    </>
  );

  return (
    <Dialog title="roll out a policy" onClose={onClose} footer={footer} wide>
      {/* One line for WHICH policy, whether or not it can be changed, with
          the path underneath it either way — the path is what gets loaded,
          and it is the only thing that tells two runs' `060000` apart. */}
      <div className="flex flex-col gap-1">
        {choices.length > 1 ? (
          <Field
            label="checkpoint"
            hint="the newest step this run wrote is picked for you"
          >
            <Select
              value={active.path}
              aria-label="checkpoint to roll out"
              onChange={(e) => setPath(e.target.value)}
            >
              {choices.map((c) => (
                <option key={c.path} value={c.path}>{stepLabel(c)}</option>
              ))}
            </Select>
          </Field>
        ) : (
          <span className="label-micro text-muted-foreground">{stepLabel(active)}</span>
        )}
        <span className="font-mono text-[10px] break-all text-muted-foreground">
          {active.path}
        </span>
      </div>

      <WarnBox tone="fault">
        <span className="label-micro">the arm moves for up to {seconds} s</span>
        <div className="mt-2 text-pretty">
          The policy is the leader for the length of this run — not your hand.
          It streams targets and the server commits them through the same
          low-pass, rate cap, clamp, collision guard, workspace floors and
          E-STOP as teleop, but a freshly-trained policy is less predictable
          than an operator, not more. Clear the workspace and stay in reach of
          the header E-STOP. <b>stop</b> on the run ends it early; it ends
          itself at {seconds} s either way.
        </div>
      </WarnBox>

      {failure && <Refusal tone={failure.tone}>{failure.text}</Refusal>}

      {!active.has_model && (
        <Refusal>
          this checkpoint directory has no model file — the child will refuse to
          load it. Pick one that is not marked partial.
        </Refusal>
      )}

      <div className="grid gap-2.5 [grid-template-columns:repeat(auto-fill,minmax(9.5rem,1fr))]">
        <Field label="duration" hint={`seconds · ${MAX_ROLLOUT_DURATION_S} s ceiling`}>
          <NumberInput
            value={duration}
            min={1}
            max={MAX_ROLLOUT_DURATION_S}
            step={10}
            aria-label="rollout duration in seconds"
            onChange={setDuration}
          />
        </Field>

        <Field
          label="control rate"
          hint={
            rateSource === "trained"
              ? trainedFps !== null
                ? `the server reads it from the checkpoint — ${hz(trainedFps)} on this dataset`
                : "the server reads it from the checkpoint"
              : "checked against the trained rate, and refused if they differ"
          }
        >
          <Select
            value={rateSource}
            aria-label="control rate source"
            onChange={(e) => setRateSource(e.target.value as RateSource)}
          >
            <option value="trained">as trained</option>
            <option value="declared">declare a rate</option>
          </Select>
        </Field>

        {rateSource === "declared" && (
          <Field label="hz" hint="what this run is driven at">
            <NumberInput
              value={declared}
              min={0}
              step={1}
              aria-label="control rate in hz"
              onChange={(v) => setDeclaredHz(v)}
            />
          </Field>
        )}

        <Field label="device">
          <Select
            value={device}
            aria-label="inference device"
            onChange={(e) => setDevice(e.target.value)}
          >
            {DEVICES.map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </Select>
        </Field>

        {/* Offered whenever the columns might not name an arm — which includes
            "the dataset could not be read". Hidden only when the rig is known
            to carry its own side, where the child ignores this outright. */}
        {(rig === null || needsSide) && (
          <Field
            label="arm"
            hint={
              needsSide
                ? "required — this dataset's columns name no side"
                : "only used if this policy's columns name no side"
            }
          >
            <Select
              value={side}
              aria-label="arm the policy drives"
              onChange={(e) => setSide(e.target.value)}
            >
              <option value="">{needsSide ? "pick one" : "from the dataset"}</option>
              <option value="left">left</option>
              <option value="right">right</option>
            </Select>
          </Field>
        )}
      </div>

      <Field
        label="task"
        hint="the instruction, for a policy conditioned on one. ACT ignores it."
      >
        <TextInput
          value={task}
          aria-label="task instruction"
          placeholder={recordedTask || "none recorded"}
          onChange={(e) => setTaskEdit(e.target.value)}
          spellCheck={false}
        />
      </Field>

      {/* Both loosen a gate that exists for a reason, so both say what they
          let past rather than naming themselves. */}
      {rateSource === "declared" && (
        <Override
          checked={allowMismatch}
          onChange={setAllowMismatch}
          disabled={busy}
          label="launch even if this is not the rate it was trained at"
          note="Recorded in the run's spec forever, and visible in the run list."
        />
      )}
      <Override
        checked={allowSlow}
        onChange={setAllowSlow}
        disabled={busy}
        label="keep going if the measured rate falls under the floor"
        note={
          "A different gate on a different event: the child measures its own " +
          "inference rate over warmup cycles, before a single target is sent, " +
          "and refuses below the floor. Off, a slow box costs no motion at all."
        }
      />

      <Note className="rounded-md border border-border bg-muted px-2.5 py-2">
        <RolloutSentence
          name={checkpointName(active.path)}
          seconds={seconds}
          rateSource={rateSource}
          declared={declared}
          trainedFps={trainedFps}
          device={device}
          side={side}
          rig={rig}
        />
      </Note>

      <Note>
        A rollout claims the bus for its own run: while it streams, teleop and
        calibration are refused, and it is refused in turn if a session is
        already live — so this can never take an arm out from under an
        operator, or be taken out from under itself.
        {repoId !== null && detailError !== null && (
          <>
            {" "}The training dataset could not be read ({detailError}), so the
            rig and the trained rate are unknown here — the child derives both
            for itself and refuses if this spec disagrees with them.
          </>
        )}
        {repoId === null && (
          <>
            {" "}This run records no dataset, so the rig and the trained rate
            are unknown here — the child derives both for itself and refuses if
            this spec disagrees with them.
          </>
        )}
      </Note>
    </Dialog>
  );
}

/** A checkbox that lets something past, with the sentence for what it lets
 *  past underneath it. */
function Override({
  checked,
  onChange,
  disabled,
  label,
  note,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
  label: string;
  note: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="flex cursor-pointer items-center gap-2">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          disabled={disabled}
          className="h-3.5 w-3.5 shrink-0 accent-[var(--haller-warn)]"
        />
        <span className="label-micro text-foreground">{label}</span>
      </label>
      <Note className="pl-5.5">{note}</Note>
    </div>
  );
}

/* ---- the English summary ------------------------------------------------ */

/**
 * What is about to happen, in a sentence.
 *
 * The same instinct as `TrainLauncher`'s split sentence: the operator should
 * be able to read the run back off the dialog before starting it, in the same
 * words the run list will use afterwards.
 */
function RolloutSentence({
  name,
  seconds,
  rateSource,
  declared,
  trainedFps,
  device,
  side,
  rig,
}: {
  name: string;
  seconds: number;
  rateSource: RateSource;
  declared: number;
  trainedFps: number | null;
  device: string;
  side: string;
  rig: Rig | null;
}) {
  const rate =
    rateSource === "trained"
      ? trainedFps !== null
        ? `at the rate it was trained at (${hz(trainedFps)})`
        : "at the rate it was trained at"
      : `at ${hz(declared)}`;

  const arm =
    side !== ""
      ? `, driving the ${side} arm`
      : rig === "solo"
        ? ", on an arm you have not named yet"
        : rig !== null
          ? `, on the arms its columns name (${rigLabel(rig)})`
          : "";

  const mismatch =
    rateSource === "declared" && trainedFps !== null && declared !== trainedFps;

  return (
    <>
      Runs {name} {rate} for {seconds} s on {device}
      {arm}.
      {mismatch && (
        <>
          {" "}That is not the {hz(trainedFps)} this dataset was recorded at —
          the server will refuse unless the override above is ticked.
        </>
      )}
    </>
  );
}
