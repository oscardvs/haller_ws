"use client";

/**
 * One run, in full: what it was asked to do, what it has logged, what it wrote
 * to disk, and what its stdout says.
 *
 * This component owns THE ONLY TIMER on the Train page. Metrics and log are
 * pulled by BYTE OFFSET and appended — the server hands back an opaque offset
 * and it goes straight back on the next call, so a 200k-step run is not
 * re-downloaded every two seconds to show two new lines.
 *
 * The poll is armed only while the watched run's status is `running`, and it
 * is torn down on unmount and the moment that status stops being `running`: a
 * poll that outlives the run it was watching is how a page ends up making four
 * requests a second at 3am. A transient failure is swallowed and retried on the
 * next tick — a dropped request must not blank a run that is still training.
 *
 * WHAT GETS THE HEIGHT. This column is a fixed slice of a fixed-viewport
 * shell, so every panel that is always open is height taken from the one the
 * operator is actually reading. Three of them were: a four-line header, the
 * charts, and a full-height log — which left the metrics panel about 230px,
 * one row of a three-row grid, with the rest behind a scrollbar. So the
 * forensic lines (argv, the rollout's policy path) fold behind a toggle, the
 * log is a DRAWER that opens on demand, and the charts take everything left.
 * A run with no metrics to draw — a rollout, an export, a prune — inverts
 * that: its stdout is the only account it has, so the drawer is forced open
 * and the question never arises.
 *
 * EVERY buffer here is per-run — both byte offsets, both append targets, the
 * armed delete — so a different run is a different component: the caller keys
 * this on `runId` and the whole set resets at once, refs included. Rendering it
 * without that key inherits the previous run's log.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { toast } from "sonner";

import {
  lab, isBusy, isForbidden, isMissing, reason, metricX, plottableMetricKeys,
  REMOTE_REFUSED,
  type Checkpoint, type MetricRow, type Run, type RunStatus, type RolloutRecord,
  type TrainSpec,
} from "@/lib/lab";
import { Button, Empty, Panel, PanelHead, Refusal, Stat } from "@/components/lab/ui";
import { fmtDuration } from "@/components/lab/charts/svg";
import { StatusPill, epochSeconds, fullWhen, runLabel, shortWhen } from "@/components/lab/RunList";
import { MetricGrid } from "@/components/lab/MetricGrid";
import { RunVitals } from "@/components/lab/RunVitals";
import { CheckpointList } from "@/components/lab/CheckpointList";
import { RolloutDialog } from "@/components/lab/RolloutDialog";
import { RunLogTail } from "@/components/lab/RunLogTail";
import { TagChips } from "@/components/lab/TagChips";
import { useSticky } from "@/components/cockpit/lib";

const POLL_MS = 2000;

export function RunDetail({
  runId,
  onChanged,
  onDeleted,
  onLaunched,
}: {
  runId: string | null;
  onChanged?: (r: Run) => void;
  onDeleted?: (id: string) => void;
  /** A rollout started from one of this run's checkpoints. A DIFFERENT run
   *  than the one on screen — the caller decides whether to follow it. */
  onLaunched?: (r: Run) => void;
}) {
  const [run, setRun] = useState<Run | null>(null);
  const [rows, setRows] = useState<MetricRow[]>([]);
  const [log, setLog] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [missing, setMissing] = useState(false);
  /** A refusal the operator has to read and act on — a 403 from the LAN, a 409
   *  from a run that is already stopping. It persists until the next attempt,
   *  which is why it is a box and not a toast. */
  const [refusal, setRefusal] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [armed, setArmed] = useState(false);
  /** The checkpoint the operator asked to roll out, which is what opens the
   *  launcher. Null closes it. */
  const [rollout, setRollout] = useState<Checkpoint | null>(null);
  /** The clock the elapsed readout is measured against, advanced by the poll
   *  below. Read from `Date.now()` during render it was impure AND wrong: a
   *  running run's elapsed froze at whatever moment React last happened to
   *  re-render it. A FINISHED run carries `finished_at` and needs no clock, so
   *  nothing here ticks for one. */
  const [now, setNow] = useState(0);
  /** argv and the rollout's policy path. Per-run state, not sticky: they are
   *  read once when something looks wrong, and a page that reopens them on
   *  every run has just moved the clutter back. */
  const [detailsOpen, setDetailsOpen] = useState(false);
  /** The log drawer. STICKY, because this is a preference about how the
   *  operator works — someone chasing a stall wants the tail open on every run
   *  they click through, and re-opening it each time is the kind of small
   *  friction that gets a surface abandoned. */
  const [logOpen, setLogOpen] = useSticky<boolean>("lab.train.log.open", false);

  /** Opaque resume offsets. Handed straight back, never interpreted. */
  const metricOff = useRef(0);
  const logOff = useRef(0);
  /** Bumped by every pull. Two can overlap — the stop button pulls by hand
   *  while the interval is mid-flight — and the one that started first must
   *  not land last and append rows the newer read already has. */
  const gen = useRef(0);
  const alive = useRef(true);
  const loaded = useRef(false);
  const lastStatus = useRef<RunStatus | null>(null);

  const changed = useRef(onChanged);
  const removed = useRef(onDeleted);
  useEffect(() => {
    changed.current = onChanged;
    removed.current = onDeleted;
  });

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const pull = useCallback(async () => {
    if (!runId) return;
    const g = gen.current;
    try {
      // One round trip's worth of work. The two streams are allowed to fail on
      // their own: a build with no metrics route must still show the run.
      const [m, l, r] = await Promise.all([
        lab.runMetrics(runId, metricOff.current).catch(() => null),
        lab.runLog(runId, logOff.current).catch(() => null),
        lab.run(runId),
      ]);
      if (!alive.current || g !== gen.current) return;
      if (m) {
        metricOff.current = m.offset;
        if (m.rows.length > 0) setRows((prev) => prev.concat(m.rows));
      }
      if (l) {
        logOff.current = l.offset;
        if (l.text) setLog((prev) => prev + l.text);
      }
      loaded.current = true;
      setError(null);
      setMissing(false);
      setRun(r);
      // Same batch as the run it describes, so `now` and `started_at` are
      // never one render out of step with each other.
      if (r.status === "running") setNow(Date.now() / 1000);
      // The list above us holds its own copy of this row; a status it has not
      // seen is the one thing it cannot work out for itself.
      if (lastStatus.current !== r.status) {
        lastStatus.current = r.status;
        changed.current?.(r);
      }
    } catch (e) {
      if (!alive.current || g !== gen.current) return;
      if (isMissing(e)) {
        setMissing(true);
        return;
      }
      // Only a first read that never landed is worth showing.
      if (!loaded.current) setError(reason(e));
    }
  }, [runId]);

  const live = run?.status === "running";

  useEffect(() => {
    if (!runId) return;
    void pull();
    // Armed ONLY while the run is running. `live` is in the dependency list so
    // the interval is torn down the instant the status changes — and the
    // re-run leaves one final pull behind it, which is what catches the last
    // lines the process wrote on its way out.
    if (!live) return;
    const t = setInterval(() => {
      void pull();
    }, POLL_MS);
    return () => clearInterval(t);
  }, [runId, live, pull]);

  const logLines = useMemo(() => {
    if (!log) return 0;
    let n = 1;
    for (let i = 0; i < log.length; i++) if (log.charCodeAt(i) === 10) n += 1;
    return n;
  }, [log]);

  /** The last non-blank line, for the head of a CLOSED drawer. A collapsed log
   *  that shows nothing makes the operator open it to find out whether
   *  anything is happening, which defeats collapsing it.
   *
   *  Split on CARRIAGE RETURNS as well as newlines. lerobot's progress bar is
   *  tqdm, which redraws by writing `\r` and never a newline, so the whole
   *  hour of it is ONE line by `\n` — and the head rendered forty overlapping
   *  frames of `Training: 2%| | 2201/90000 …` run together. A `\r` segment is
   *  a frame, and the last frame is the current one. */
  const lastLogLine = useMemo(() => {
    if (!log) return "";
    const frames = log.split(/[\r\n]/);
    for (let i = frames.length - 1; i >= 0; i--) {
      const s = frames[i].trim();
      if (s) return s;
    }
    return "";
  }, [log]);

  const lastStep = useMemo(() => {
    for (let i = rows.length - 1; i >= 0; i--) {
      const x = metricX(rows[i], "step");
      if (x !== null) return x;
    }
    return null;
  }, [rows]);

  const label = run ? runLabel(run) : (runId ?? "—");
  /** `kind` is what says which shape arrived; both are partial, because a
   *  run's spec is whatever the launching route wrote. */
  const spec: Partial<TrainSpec & RolloutRecord> = run?.spec ?? {};
  const isRollout = run?.kind === "rollout";
  /** Only the trainer writes `metrics.jsonl` and only the trainer writes
   *  checkpoints. Asked as "can this run answer the question" rather than
   *  "is it a rollout", because export, prune and a rollout are all runs with
   *  neither, and each of them got the same two empty panels. */
  const writesMetrics = run?.kind === "train";
  const writesCheckpoints = run?.kind === "train";
  /**
   * The panel appears when there is something to DRAW, or while a run that
   * could still log one is going.
   *
   * "Rows arrived" was the first rule and it was wrong on the case that
   * matters: a rollout writes its rate record into `metrics.jsonl`, so
   * `rows.length > 0` was true and the panel opened on a full-height
   * "no numeric keys logged yet" above the log that had the actual account.
   * `plottableMetricKeys` is the predicate `MetricGrid` itself draws by, so
   * the panel and its contents now agree about whether there is a chart.
   */
  const showMetrics =
    plottableMetricKeys(rows).length > 0 || (writesMetrics && live);
  /** The drawer is forced open for a run whose stdout is its only account —
   *  a rollout, an export, a prune. `logOpen` is the operator's preference and
   *  only gets asked when there is something else to look at. */
  const logExpanded = !showMetrics || logOpen;
  const argv = run?.argv?.join(" ") ?? null;
  /** `finished_at` where the run has one, otherwise the poll's clock. `now` is
   *  0 until the first tick of a RUNNING run — a run that has neither has no
   *  elapsed to report, and a made-up one would read as a run that started in
   *  1970. */
  const until = epochSeconds(run?.finished_at) ?? (now > 0 ? now : null);
  const startedSecs = epochSeconds(run?.started_at);
  const elapsed =
    startedSecs !== null && until !== null ? until - startedSecs : null;

  const refuse = (e: unknown, cannot: string) => {
    if (isForbidden(e)) setRefusal(REMOTE_REFUSED);
    else if (isBusy(e)) setRefusal(reason(e));
    else if (isMissing(e)) setRefusal(cannot);
    else toast.error(reason(e));
  };

  const doStop = async () => {
    if (!runId) return;
    setBusy(true);
    setRefusal(null);
    try {
      await lab.stopRun(runId);
      toast.success("stop requested");
      // `stopRun` answers `{ ok }`, not a run record — the authoritative
      // status is whatever the next read says it is.
      await pull();
    } catch (e) {
      refuse(e, "this backend cannot stop a run");
    } finally {
      if (alive.current) setBusy(false);
    }
  };

  const doDelete = async () => {
    if (!runId) return;
    setArmed(false);
    setBusy(true);
    setRefusal(null);
    try {
      await lab.deleteRun(runId);
      toast.success(`deleted ${label}`);
      removed.current?.(runId);
    } catch (e) {
      refuse(e, "this backend cannot delete a run");
    } finally {
      if (alive.current) setBusy(false);
    }
  };

  if (!runId) {
    return (
      <Panel className="flex-1">
        <Empty>select a run</Empty>
      </Panel>
    );
  }

  if (missing) {
    return (
      <Panel className="flex-1">
        <Empty>this run is gone, or this backend has no lab</Empty>
      </Panel>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-hidden">
      <Panel className="shrink-0">
        <PanelHead title="run">
          <span className="min-w-0 flex-1 truncate font-mono text-[11px]" title={runId}>
            {label}
          </span>
          {run && <StatusPill status={run.status} />}
          {run?.status === "running" && (
            <Button tone="danger" disabled={busy} onClick={doStop} title="SIGTERM the child">
              stop
            </Button>
          )}
          {/* Arm, then confirm — the same two-step the Dataset tab's
              delete-last uses, and for the same reason: nothing destructive on
              this surface is one click away from a mis-aimed pointer. */}
          {run && run.status !== "running" && (
            armed ? (
              <>
                <Button tone="danger" disabled={busy} onClick={doDelete}>
                  confirm · delete
                </Button>
                <Button onClick={() => setArmed(false)}>cancel</Button>
              </>
            ) : (
              <Button tone="danger" disabled={busy} onClick={() => setArmed(true)}>
                delete
              </Button>
            )
          )}
        </PanelHead>

        <div className="flex flex-col gap-2 p-2.5">
          {error && !run && <Refusal tone="fault">{error}</Refusal>}
          {refusal && <Refusal>{refusal}</Refusal>}
          {run?.error && <Refusal tone="fault">{run.error}</Refusal>}

          {run === null ? (
            <span className="font-mono text-[10px] text-muted-foreground">reading…</span>
          ) : (
            <>
              <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                <Stat label="kind" value={run.kind} />
                {/* A rollout's spec answers different questions than a train
                    run's, and rendering the train fields against it printed a
                    row of — that read as a run with nothing in it. */}
                {isRollout ? <RolloutStats spec={spec} /> : (
                  <>
                    <Stat label="policy" value={spec.policy_type ?? "—"} />
                    <Stat label="episodes" value={spec.episodes?.length ?? "—"} />
                    <Stat label="steps" value={spec.steps ?? "—"} />
                    <Stat label="batch" value={spec.batch_size ?? "—"} />
                    <Stat label="device" value={spec.device ?? "—"} />
                    <Stat
                      label="eval split"
                      value={
                        typeof spec.eval_split === "number"
                          ? `${Math.round(spec.eval_split * 100)}%`
                          : "—"
                      }
                    />
                  </>
                )}
                {/* Was its own line under the row. A dataset id is short and
                    this row is where every other "what was this run" fact
                    already is. */}
                <span title={spec.repo_id ?? undefined} className="min-w-0">
                  <Stat
                    label="repo"
                    value={
                      <span className="inline-block max-w-[16rem] truncate align-bottom">
                        {spec.repo_id ?? "—"}
                      </span>
                    }
                  />
                </span>
                <Stat label="ran" value={fmtDuration(elapsed)} />
                <span title={fullWhen(run.started_at)}>
                  <Stat label="started" value={shortWhen(run.started_at)} />
                </span>
                {typeof run.exit_code === "number" && (
                  <Stat
                    label="exit"
                    value={run.exit_code}
                    colour={
                      run.exit_code === 0 ? "var(--haller-live)" : "var(--haller-fault)"
                    }
                  />
                )}
              </div>

              {/* Tags read-only: the frozen `/lab` contract has no route that
                  writes a run's tags. They are set at launch, in the spec.

                  One line with the disclosure, rather than a line each: three
                  rows of small grey text above the charts is the clutter this
                  layout was losing. */}
              {/* The checkpoint this rollout loaded — VISIBLE, not folded. It
                  is the subject of the run, the way `repo` is a training run's;
                  a rollout draws no charts, so nothing is competing for the
                  line. The repo in the row above is the dataset it was TRAINED
                  on, which the route resolves from the checkpoint rather than
                  from anything the operator had open. */}
              {isRollout && (
                <div className="flex items-baseline gap-2">
                  <span className="label-micro shrink-0 text-muted-foreground">policy</span>
                  <span
                    className="min-w-0 truncate font-mono text-[10px]"
                    title={spec.policy_path ?? undefined}
                  >
                    {spec.policy_path ?? "—"}
                  </span>
                </div>
              )}

              <div className="flex min-w-0 items-center gap-2">
                {run.tags && run.tags.length > 0 && <TagChips tags={run.tags} />}
                {argv && (
                  <button
                    type="button"
                    onClick={() => setDetailsOpen(!detailsOpen)}
                    aria-expanded={detailsOpen}
                    title="the command line this run actually executed"
                    className="label-micro ml-auto inline-flex shrink-0 items-center gap-1 text-muted-foreground transition-colors hover:text-foreground"
                  >
                    {detailsOpen ? (
                      <ChevronDown size={11} aria-hidden />
                    ) : (
                      <ChevronRight size={11} aria-hidden />
                    )}
                    argv
                  </button>
                )}
              </div>

              {/* Months later argv is the only thing that answers "what did
                  this actually run", so open it WRAPS rather than truncating:
                  half a command line cannot be copied. */}
              {detailsOpen && argv && (
                <div className="min-w-0 rounded-md border border-border bg-[var(--haller-inset)] p-2 font-mono text-[10px] break-all text-muted-foreground">
                  {argv}
                </div>
              )}

              {/* Step, percentage, ETA and the headline numbers — the reading
                  this page did not have. Only for a run that logs metrics; a
                  prune has no progress to report and would get a bar stuck at
                  nothing. */}
              {writesMetrics && (
                <RunVitals
                  rows={rows}
                  steps={typeof spec.steps === "number" ? spec.steps : null}
                  lastStep={lastStep}
                  elapsed={elapsed}
                  live={live}
                />
              )}
            </>
          )}
        </div>
      </Panel>

      {/* ONE PANEL PER QUESTION THE RUN CAN ANSWER, and the log takes what is
          left. Three fixed panels in a column this short crumbled: metrics
          squeezed to an empty sliver with a scrollbar, the checkpoint list
          showing four of six inside its own scroll, and the log — the only
          account a rollout HAS — clipped off the bottom edge behind a third
          one. A kit-trained run has no metrics.jsonl at all, so its metrics
          panel was a promise about a file that does not exist.

          So a panel appears when the run can fill it: metrics for a training
          run that has logged or still might, checkpoints for a run that
          writes them. Everything else is stdout, and stdout gets the room. */}
      <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-hidden">
        {showMetrics && (
          // `relative` because MetricGrid's maximised chart positions its
          // overlay against this panel. It takes the whole column now: the
          // charts ARE the reading while a run trains, and the tail below is
          // for catching the line that explains a stall — which is a thing you
          // go and look for, not a thing you watch.
          <Panel className="relative min-h-0 flex-1">
            {/* No reading in the head: the grid's own control strip already
                says how many keys and rows, and the step count moved up to the
                progress rule where it belongs. */}
            <PanelHead title="metrics" />
            {/* `overflow-hidden`, not `auto`. The grid divides this box among
                its cells rather than stacking 130px charts past the bottom of
                it, so a scrollbar here would mean the fit failed. */}
            <div className="min-h-0 flex-1 overflow-hidden p-2.5">
              <MetricGrid rows={rows} steps={spec.steps ?? null} />
            </div>
          </Panel>
        )}

        {/* Its own list scrolls inside a capped height, so it takes what it
            needs and never squeezes the log to nothing. */}
        {run && writesCheckpoints && (
          <CheckpointList
            runId={runId}
            status={run.status}
            onRollout={setRollout}
          />
        )}

        <Panel className={logExpanded ? "min-h-0 flex-[0.7]" : "shrink-0"}>
          <PanelHead
            title="log"
            right={logLines > 0 ? `${logLines} lines` : undefined}
          >
            {/* Closed, the head carries the last line. A collapsed log that
                shows nothing makes you open it to find out whether anything is
                happening, which is the opposite of collapsing it. */}
            {!logExpanded && lastLogLine && (
              <span
                className="min-w-0 flex-1 truncate font-mono text-[10px] text-muted-foreground"
                title={lastLogLine}
              >
                {lastLogLine}
              </span>
            )}
            {showMetrics && (
              <button
                type="button"
                onClick={() => setLogOpen(!logOpen)}
                aria-expanded={logExpanded}
                title={logExpanded ? "close the log and give the charts the room" : "open the log"}
                className="label-micro inline-flex shrink-0 items-center gap-1 text-muted-foreground transition-colors hover:text-foreground"
              >
                {logExpanded ? (
                  <ChevronDown size={11} aria-hidden />
                ) : (
                  <ChevronRight size={11} aria-hidden />
                )}
                {logExpanded ? "close" : "open"}
              </button>
            )}
          </PanelHead>
          {logExpanded && (
            <div className="flex min-h-0 flex-1 flex-col p-2.5">
              <RunLogTail text={log} fill />
            </div>
          )}
        </Panel>
      </div>

      {rollout && (
        <RolloutDialog
          checkpoint={rollout}
          // The dataset THIS run trained on. The launcher uses it to know
          // whether it must ask which arm the policy drives; the run it starts
          // resolves its own from the checkpoint.
          repoId={spec.repo_id ?? null}
          onClose={() => setRollout(null)}
          onLaunched={(r) => {
            setRollout(null);
            onLaunched?.(r);
          }}
        />
      )}
    </div>
  );
}

/* ---- a rollout's own stats ---------------------------------------------- */

/**
 * What a rollout was asked to do, and where its rate came from.
 *
 * The rate is two facts, not one: what it ran at, and what the policy was
 * trained at. A run that got them for free reads the same as one where
 * somebody typed a number and ticked the override, unless both are shown —
 * which is exactly why the route stamps `control_hz_declared_by` and
 * `control_hz_mismatch_override` whether or not anything was overridden.
 */
function RolloutStats({ spec }: { spec: Partial<RolloutRecord> }) {
  const trained = spec.control_hz_trained;
  const override = spec.control_hz_mismatch_override === true;
  return (
    <>
      <span
        title={
          spec.control_hz_declared_by === "request"
            ? "declared in the request"
            : "the rate the policy was trained at"
        }
      >
        <Stat
          label="rate"
          value={typeof spec.control_hz === "number" ? `${spec.control_hz} Hz` : "—"}
          colour={override ? "var(--haller-warn)" : undefined}
        />
      </span>
      <span
        title={
          spec.control_hz_trained_measured === true
            ? `measured on the training dataset (${spec.control_hz_trained_measured_hz ?? "?"} Hz)`
            : "declared by the training dataset, never measured"
        }
      >
        <Stat
          label="trained at"
          value={typeof trained === "number" ? `${trained} Hz` : "unknown"}
          colour={override ? "var(--haller-warn)" : undefined}
        />
      </span>
      <Stat
        label="asked for"
        value={typeof spec.duration_s === "number" ? `${spec.duration_s} s` : "—"}
      />
      <Stat label="device" value={spec.device ?? "—"} />
      {spec.side ? <Stat label="arm" value={spec.side} /> : null}
      {spec.allow_slow === true && (
        <Stat label="rate floor" value="waived" colour="var(--haller-warn)" />
      )}
    </>
  );
}
