"use client";

/**
 * The rollout this surface just started, while it is still going.
 *
 * The Train tab does not need this: a rollout launched there lands in a run
 * list beside a detail pane that already polls it, with a stop button in its
 * header. The compare view has neither — it is a page of curves, reachable by
 * bookmark on a second monitor — so a launch from there would otherwise put an
 * arm in motion and then say nothing at all about it. The E-STOP in the header
 * stops the robot; this is the control that ends THIS RUN, which is a
 * different act with a different aftermath.
 *
 * ONE AT A TIME, deliberately. The bus lease refuses a second rollout while
 * one is streaming, so a list of them could only ever be a list of finished
 * runs — and this strip exists to watch the live one. Launching again replaces
 * what is here; the runs themselves are all still in the Train tab's list,
 * which is what "dismiss" says.
 *
 * The poll is armed only while the run is queued or running and is torn down
 * the moment it is not — the same rule `RunDetail` follows, for the same
 * reason: nothing here should still be asking the backend for a run that
 * finished forty minutes ago.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import {
  isForbidden, isMissing, lab, reason, REMOTE_REFUSED, type Run,
} from "@/lib/lab";
import { Button, Panel, PanelHead, Refusal, Stat } from "@/components/lab/ui";
import { checkpointName } from "@/components/lab/CheckpointList";
import { StatusPill, epochSeconds, runLabel } from "@/components/lab/RunList";
import { fmtDuration } from "@/components/lab/charts/svg";

const POLL_MS = 2000;

export function RolloutWatch({
  run,
  onDismiss,
}: {
  /** The launched run. The caller keys this component on its id, so the state
   *  below is per-rollout and a second launch starts clean. */
  run: Run;
  onDismiss: () => void;
}) {
  const [cur, setCur] = useState<Run>(run);
  /** The clock elapsed is measured against, advanced by the poll. Read from
   *  `Date.now()` during render it would be both impure and stuck — the same
   *  trap `RunDetail` documents. */
  const [now, setNow] = useState(0);
  const [refusal, setRefusal] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const id = cur.id;
  const live = cur.status === "queued" || cur.status === "running";

  const pull = useCallback(async () => {
    try {
      const r = await lab.run(id);
      if (!alive.current) return;
      setCur(r);
      setNow(Date.now() / 1000);
    } catch {
      // A dropped read must not blank a run that is still streaming. The next
      // tick asks again; a run that is really gone stops being polled when its
      // last known status goes terminal, not when one request fails.
    }
  }, [id]);

  useEffect(() => {
    if (!live) return;
    // The interval is the whole poll — there is no immediate read on mount.
    // The record this strip opens on came from the launch a moment ago, so the
    // first tick is a refresh rather than the first thing known about the run,
    // and a setState in the effect body would be a cascading render for two
    // seconds of a status that is already on screen.
    const t = setInterval(() => {
      void pull();
    }, POLL_MS);
    return () => clearInterval(t);
  }, [live, pull]);

  const doStop = async () => {
    setBusy(true);
    setRefusal(null);
    try {
      await lab.stopRun(id);
      toast.success("stop requested");
      // Read back by hand rather than waiting out the interval: this is the
      // button that stops an arm, and two seconds of an unchanged status pill
      // reads as a click that did nothing.
      await pull();
    } catch (e) {
      if (isForbidden(e)) setRefusal(REMOTE_REFUSED);
      else if (isMissing(e)) setRefusal("this backend cannot stop a run");
      else setRefusal(reason(e));
    } finally {
      if (alive.current) setBusy(false);
    }
  };

  const spec = cur.spec ?? {};
  const startedSecs = epochSeconds(cur.started_at);
  const until = epochSeconds(cur.finished_at) ?? (now > 0 ? now : null);
  const elapsed =
    startedSecs !== null && until !== null ? Math.max(0, until - startedSecs) : null;
  const asked = typeof spec.duration_s === "number" ? spec.duration_s : null;
  /** How far through the duration it was ASKED for — the only progress a
   *  rollout has, since a policy reports no steps. Null while it is queued and
   *  has no start to measure from. */
  const frac =
    elapsed !== null && asked !== null && asked > 0
      ? Math.min(1, elapsed / asked)
      : null;

  return (
    <Panel
      className="shrink-0"
      // The one panel on this page that is a machine in motion rather than a
      // reading of one that stopped. It says so before it is read.
      style={
        cur.status === "running"
          ? { boxShadow: "0 0 0 1px var(--haller-fault)" }
          : undefined
      }
    >
      <PanelHead title="rollout">
        <span className="min-w-0 flex-1 truncate font-mono text-[11px]" title={cur.id}>
          {runLabel(cur)}
        </span>
        <StatusPill status={cur.status} />
        {cur.status === "running" && (
          <Button
            tone="danger"
            disabled={busy}
            onClick={doStop}
            title="SIGTERM the child — the policy stops streaming and the arm holds"
          >
            stop
          </Button>
        )}
        {!live && (
          <Button
            onClick={onDismiss}
            title="clear this strip — the run itself stays in the Train tab's list"
          >
            dismiss
          </Button>
        )}
      </PanelHead>

      <div className="flex flex-col gap-2 p-2.5">
        {refusal && <Refusal>{refusal}</Refusal>}
        {cur.error && <Refusal tone="fault">{cur.error}</Refusal>}

        {cur.status === "running" && (
          <span className="label-micro" style={{ color: "var(--haller-fault)" }}>
            the arm is moving — stop ends this run, E-STOP ends everything
          </span>
        )}

        <div className="flex items-center gap-2.5">
          {frac !== null && (
            <span
              role="progressbar"
              aria-label="rollout progress"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(frac * 100)}
              className="h-1.5 min-w-0 flex-1 overflow-hidden rounded-full bg-[var(--haller-inset)]"
            >
              <span
                className="block h-full rounded-full transition-[width] duration-500 ease-out"
                style={{
                  width: `${frac * 100}%`,
                  background: live ? "var(--haller-fault)" : "var(--muted-foreground)",
                }}
              />
            </span>
          )}
          <span className="flex shrink-0 items-baseline gap-2.5 font-mono text-[10px] whitespace-nowrap text-muted-foreground">
            <span data-num className="tabular-nums text-foreground">
              {elapsed === null ? "—" : fmtDuration(elapsed)}
            </span>
            {asked !== null && (
              <span>
                {"of "}
                <span data-num className="tabular-nums">{asked} s</span>
                <span className="label-micro pl-1.5">asked for</span>
              </span>
            )}
          </span>
        </div>

        {/* Every field is skipped when the run does not carry it. A rollout
            read back from the server carries all of them; the record this
            strip opens on is whatever the launch could prove, and a `—` for
            the rest would be a claim about the run rather than about what has
            been read of it yet. */}
        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1.5">
          {typeof spec.policy_path === "string" && (
            <span title={spec.policy_path}>
              <Stat label="policy" value={checkpointName(spec.policy_path)} />
            </span>
          )}
          {typeof spec.control_hz === "number" && (
            <Stat label="rate" value={`${spec.control_hz} Hz`} />
          )}
          {typeof spec.side === "string" && <Stat label="arm" value={spec.side} />}
          {typeof spec.device === "string" && <Stat label="device" value={spec.device} />}
          {typeof cur.exit_code === "number" && (
            <Stat
              label="exit"
              value={cur.exit_code}
              colour={cur.exit_code === 0 ? "var(--haller-live)" : "var(--haller-fault)"}
            />
          )}
        </div>
      </div>
    </Panel>
  );
}
