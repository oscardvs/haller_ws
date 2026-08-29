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
 * EVERY buffer here is per-run — both byte offsets, both append targets, the
 * armed delete — so a different run is a different component: the caller keys
 * this on `runId` and the whole set resets at once, refs included. Rendering it
 * without that key inherits the previous run's log.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import {
  lab, isBusy, isForbidden, isMissing, reason, metricX, REMOTE_REFUSED,
  type Checkpoint, type MetricRow, type Run, type RunStatus, type RolloutRecord,
  type TrainSpec,
} from "@/lib/lab";
import { Button, Empty, Panel, PanelHead, Refusal, Stat } from "@/components/lab/ui";
import { fmtDuration } from "@/components/lab/charts/svg";
import { StatusPill, epochSeconds, fullWhen, runLabel, shortWhen } from "@/components/lab/RunList";
import { MetricGrid } from "@/components/lab/MetricGrid";
import { CheckpointList } from "@/components/lab/CheckpointList";
import { RolloutDialog } from "@/components/lab/RolloutDialog";
import { RunLogTail } from "@/components/lab/RunLogTail";
import { TagChips } from "@/components/lab/TagChips";

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

              {/* The checkpoint this rollout loaded. The repo below it is the
                  dataset it was TRAINED on, which the route resolves from the
                  checkpoint rather than from anything the operator had open. */}
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

              <div className="flex items-baseline gap-2">
                <span className="label-micro shrink-0 text-muted-foreground">repo</span>
                <span
                  className="min-w-0 truncate font-mono text-[10px]"
                  title={spec.repo_id ?? undefined}
                >
                  {spec.repo_id ?? "—"}
                </span>
              </div>

              {/* The child's command line. Months later it is the only thing
                  that answers "what did this actually run". */}
              {argv && (
                <div className="flex items-baseline gap-2">
                  <span className="label-micro shrink-0 text-muted-foreground">argv</span>
                  <span className="min-w-0 truncate font-mono text-[10px] text-muted-foreground" title={argv}>
                    {argv}
                  </span>
                </div>
              )}

              {/* Read-only: the frozen `/lab` contract has no route that writes
                  a run's tags. They are set at launch, in the spec. */}
              {run.tags && run.tags.length > 0 && <TagChips tags={run.tags} />}
            </>
          )}
        </div>
      </Panel>

      {/* A ROLLOUT'S STDOUT IS THE WHOLE ACCOUNT. It logs no metrics and
          writes no checkpoints — what it has is the handshake, the measured
          control rate, the rate alerts, the target count, and the traceback
          when it refuses. Given the train layout it got a metrics panel
          promising "waiting for the first logged step…" about a stream that
          does not exist, an empty checkpoints panel offering to roll out a
          rollout, and the one thing worth reading squeezed into a 240px strip
          under both. So the log takes the column. */}
      {isRollout ? (
        <Panel className="min-h-0 flex-1">
          <PanelHead
            title="log"
            right={logLines > 0 ? `${logLines} lines` : undefined}
          />
          <div className="flex min-h-0 flex-1 flex-col p-2.5">
            <RunLogTail text={log} fill />
          </div>
        </Panel>
      ) : (
      /* Metrics get whatever height the column has; checkpoints and the log
         take what they need below, capped. Before this split the three shared
         one long scroll, and reading the log meant scrolling past every
         chart. The metrics Panel is `relative` because MetricGrid's maximised
         chart positions its overlay against it. */
      <div className="grid min-h-0 flex-1 grid-rows-[minmax(0,1fr)_auto] gap-2 overflow-hidden">
        <Panel className="relative">
          <PanelHead
            title="metrics"
            right={
              lastStep !== null
                ? `step ${Math.round(lastStep)}${spec.steps ? ` / ${spec.steps}` : ""}`
                : undefined
            }
          />
          <div className="min-h-0 flex-1 overflow-y-auto p-2.5">
            <MetricGrid rows={rows} steps={spec.steps ?? null} live={live} />
          </div>
        </Panel>

        <div className="flex max-h-[45%] min-h-0 flex-col gap-2 overflow-y-auto">
          {run && (
            <CheckpointList
              runId={runId}
              status={run.status}
              onRollout={setRollout}
            />
          )}

          <Panel className="shrink-0">
            <PanelHead title="log" right={logLines > 0 ? `${logLines} lines` : undefined} />
            <div className="p-2.5">
              <RunLogTail text={log} />
            </div>
          </Panel>
        </div>
      </div>
      )}

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
        label="for"
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
