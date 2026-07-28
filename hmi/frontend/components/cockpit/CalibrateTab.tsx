"use client";

/**
 * Calibration, in-band. Arm cards on the left, the running session on the right.
 *
 * Unlike the modal wizard on /settings, leaving this tab does NOT abort the
 * session — the session belongs to the backend, and an operator who glances at
 * Cameras mid-sweep should not come back to an arm that silently reverted to
 * manual. Coming back re-attaches. The cost is that a forgotten session stays
 * open, which is why the other arms' buttons say so and the Operate tab's arm
 * cards show "calibrating · joints owned by the wizard".
 */
import { useCallback, useEffect, useState } from "react";

import { BACKEND_URL } from "@/lib/config";
import {
  fetchCalibrationStatus,
  type CalibrationStatusResponse,
} from "@/lib/calibration";
import {
  useCalibrationSession,
  confirmCopy,
  calibrationBlockedReason,
} from "@/lib/useCalibrationSession";
import { useTelemetry } from "@/lib/telemetry";

const STEPS = [
  { key: "homing", label: "neutral" },
  { key: "sweeping", label: "sweep" },
  { key: "review", label: "review" },
] as const;

export function CalibrateTab({ armIds }: { armIds: string[] }) {
  const [status, setStatus] = useState<CalibrationStatusResponse | null>(null);
  const [epoch, setEpoch] = useState(0);
  const refresh = useCallback(() => setEpoch((n) => n + 1), []);

  // Every arm must be in manual — calibration takes torque off, and an arm in
  // auto could be commanded mid-sweep by whatever is driving it.
  const allManual = useTelemetry((s) => {
    const arms = s.lastFrame?.arms;
    if (!arms) return false;
    const present = armIds.filter((id) => arms[id]);
    return present.length === armIds.length && present.every((id) => arms[id].mode === "manual");
  });

  const session = useCalibrationSession({
    armId: null,
    autoStart: false,
    abortOnUnmount: false,
    onClose: refresh,
  });

  useEffect(() => {
    let cancelled = false;
    fetchCalibrationStatus(BACKEND_URL)
      .then((s) => {
        if (!cancelled) setStatus(s);
      })
      .catch(() => {
        /* offline — the cards fall back to "unknown" rather than lying "ok" */
      });
    return () => {
      cancelled = true;
    };
  }, [epoch, session.armId]);

  const backendSession = status?.current_session ?? null;
  const inSession = session.armId !== null || backendSession !== null;
  const blockedReason = calibrationBlockedReason(allManual, inSession);
  const canStart = blockedReason === undefined;
  const copy = confirmCopy(session.phase);
  const stepIndex = STEPS.findIndex((s) => s.key === session.phase);

  return (
    <div className="grid min-h-0 grid-cols-[300px_minmax(0,1fr)] gap-2 overflow-hidden p-2">
      <div className="flex min-h-0 flex-col gap-2 overflow-y-auto">
        {armIds.map((id) => {
          const arm = status?.arms.find((a) => a.id === id);
          const mine = session.armId === id;
          const has = arm?.has_file ?? false;
          return (
            <div
              key={id}
              className="flex flex-col gap-2 rounded-lg bg-card p-2.5 shadow-[0_0_0_1px_var(--border)]"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="font-mono text-[12px] font-semibold tracking-[0.14em] uppercase">
                  {id}
                </span>
                <span
                  className="inline-flex h-5 items-center rounded-full px-2 label-micro"
                  style={{
                    background: has
                      ? "var(--haller-live-soft)"
                      : "oklch(0.62 0.245 27 / 0.18)",
                    color: has ? "var(--haller-live)" : "var(--haller-fault)",
                  }}
                >
                  {arm ? (has ? "ok" : "missing") : "…"}
                </span>
              </div>
              <span className="text-[11px] text-muted-foreground">
                {arm
                  ? has
                    ? `Calibrated · ${new Date((arm.mtime ?? 0) * 1000).toLocaleString()}`
                    : "No calibration file"
                  : "status unavailable"}
              </span>
              <span className="font-mono text-[9px] break-all text-muted-foreground">
                {arm?.path ?? "—"}
              </span>
              <button
                type="button"
                disabled={!canStart || mine}
                title={blockedReason}
                onClick={() => session.start(id)}
                className={
                  "h-7 rounded-md label-micro tracking-[0.12em] disabled:opacity-55 " +
                  (canStart
                    ? "bg-primary text-primary-foreground"
                    : "bg-secondary text-muted-foreground")
                }
              >
                {mine ? "in session…" : "calibrate"}
              </button>
            </div>
          );
        })}

        {blockedReason && (
          <div className="rounded-lg border border-[var(--haller-warn)] p-2.5 text-[11px] text-[var(--haller-warn)]">
            {blockedReason}
          </div>
        )}
      </div>

      <div className="relative flex min-h-0 flex-col overflow-hidden rounded-lg bg-card shadow-[0_0_0_1px_var(--border)]">
        <div className="flex h-8.5 shrink-0 items-center gap-2.5 border-b border-border px-3">
          <span className="label-tracked text-muted-foreground">
            {session.armId ? `Calibrate · ${session.armId}` : "Calibration"}
          </span>
          <span aria-hidden className="h-px flex-1 bg-border" />
          {STEPS.map((s, i) => (
            <span
              key={s.key}
              className="inline-flex items-center gap-1.5 label-micro"
              style={{
                color:
                  session.armId && i <= stepIndex
                    ? "var(--haller-live)"
                    : "var(--muted-foreground)",
              }}
            >
              <span
                aria-hidden
                className="h-1.5 w-1.5 rounded-[1px]"
                style={{
                  backgroundColor:
                    session.armId && i <= stepIndex
                      ? "var(--haller-live)"
                      : "var(--haller-rail)",
                }}
              />
              {s.label}
            </span>
          ))}
        </div>

        {session.sessionAborted ? (
          <SessionEnded
            armId={session.armId ?? "—"}
            onAcknowledge={() => {
              session.acknowledgeAborted();
              refresh();
            }}
          />
        ) : session.armId === null ? (
          <Idle />
        ) : (
          <Running session={session} />
        )}

        {session.confirmOpen && (
          <div className="absolute inset-0 z-80 flex items-center justify-center bg-[oklch(0_0_0/0.55)] p-4">
            <div
              role="alertdialog"
              aria-label={copy.title}
              className="flex w-[min(460px,100%)] flex-col gap-2.5 rounded-lg border border-border bg-popover p-4 shadow-[0_24px_60px_oklch(0_0_0/0.5)]"
            >
              <span className="font-mono text-[12px] font-semibold">{copy.title}</span>
              <span className="text-[11px] text-pretty text-muted-foreground">
                {copy.body}
              </span>
              <div className="mt-1 flex items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={session.keep}
                  className="h-7 rounded-md border border-border bg-secondary px-3 label-micro tracking-[0.12em]"
                >
                  {copy.keep}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    session.discard();
                    refresh();
                  }}
                  className="h-7 rounded-md border border-[var(--haller-fault)] px-3 label-micro tracking-[0.12em] text-[var(--haller-fault)]"
                >
                  Discard
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Idle() {
  return (
    <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-2 p-4">
      <span className="label-tracked text-muted-foreground">No calibration session</span>
      <span className="max-w-[420px] text-center text-[11px] text-pretty text-muted-foreground">
        Every arm must be in manual before a session can start. Calibration
        takes torque off, records a neutral pose and a range-of-motion sweep,
        then writes a new calibration file — the previous one is kept as a{" "}
        <span className="font-mono">.bak-&lt;ts&gt;</span> sibling.
      </span>
    </div>
  );
}

/** Distinct from an error string on purpose. An error means the step failed;
 *  this means the session itself is gone, nothing was written, and retrying
 *  the step cannot help. */
function SessionEnded({
  armId,
  onAcknowledge,
}: {
  armId: string;
  onAcknowledge: () => void;
}) {
  return (
    <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-2.5 p-4">
      <span className="inline-flex items-center gap-2 rounded-md border border-[var(--haller-fault)] px-2.5 py-1.5 font-mono text-[11px] text-[var(--haller-fault)]">
        <span
          aria-hidden
          className="h-1.5 w-1.5 rounded-[1px] bg-[var(--haller-fault)]"
        />
        session ended by the backend
      </span>
      <span className="max-w-[440px] text-center text-[11px] text-pretty text-muted-foreground">
        The calibration block for <span className="font-mono">{armId}</span>{" "}
        disappeared mid-session — a bus error, or the arm was unplugged. Nothing
        was written. Re-seat the cable and check the port, then start a new
        session; retrying the step will not fix this.
      </span>
      <button
        type="button"
        onClick={onAcknowledge}
        className="h-7 rounded-md border border-border bg-secondary px-3.5 label-micro tracking-[0.12em]"
      >
        clear session
      </button>
    </div>
  );
}

function Running({
  session,
}: {
  session: ReturnType<typeof useCalibrationSession>;
}) {
  const table = tableFor(session);
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2.5 overflow-hidden p-3">
      <span className="text-[11px] text-pretty text-muted-foreground">
        {table.help}
      </span>

      {(session.error || session.blockError) && (
        <div className="shrink-0 rounded-md border border-[var(--haller-fault)] px-2.5 py-1.5 font-mono text-[10px] text-[var(--haller-fault)]">
          {session.blockError ? `Bus error: ${session.blockError}` : session.error}
        </div>
      )}

      <div className="flex min-h-0 flex-col overflow-hidden rounded-md border border-border">
        <div
          className="grid gap-2 border-b border-border bg-muted px-2.5 py-1.5 label-micro text-muted-foreground"
          style={{ gridTemplateColumns: table.cols }}
        >
          {table.head.map((h, i) => (
            <span key={h} className={i === 0 ? "" : "text-right"}>
              {h}
            </span>
          ))}
        </div>
        <div className="min-h-0 overflow-y-auto">
          {session.joints.map((j) => (
            <div
              key={j}
              className="grid gap-2 border-b border-border px-2.5 py-1 font-mono text-[10px] tabular-nums"
              style={{ gridTemplateColumns: table.cols }}
            >
              {table.row(j).map((cell, i) => (
                <span
                  key={i}
                  className={
                    "truncate " +
                    (i === 0 ? "text-foreground" : "text-right text-muted-foreground")
                  }
                >
                  {cell}
                </span>
              ))}
            </div>
          ))}
        </div>
      </div>

      <div className="mt-auto flex shrink-0 items-center gap-2">
        <button
          type="button"
          disabled={session.busy || session.sessionAborted}
          onClick={session.advance}
          className="h-7 rounded-md bg-primary px-3.5 label-micro tracking-[0.12em] text-primary-foreground disabled:opacity-50"
        >
          {table.advance}
        </button>
        <button
          type="button"
          onClick={session.attemptClose}
          className="h-7 rounded-md border border-border px-3 label-micro tracking-[0.12em] text-muted-foreground"
        >
          cancel
        </button>
        <span className="ml-auto font-mono text-[10px] text-muted-foreground">
          arm {session.armId} · torque off · back-drivable
        </span>
      </div>
    </div>
  );
}

function tableFor(session: ReturnType<typeof useCalibrationSession>) {
  const dash = (n: number | undefined) => (n === undefined ? "–" : String(n));
  if (session.phase === "sweeping") {
    return {
      cols: "1fr 80px 80px 80px",
      head: ["joint", "min", "pos", "max"],
      row: (j: string) => [
        j,
        dash(session.mins[j]),
        dash(session.ticks[j]),
        dash(session.maxes[j]),
      ],
      help: "Step 2 of 3 — range of motion. Wiggle every joint to its physical limits; the table records the extremes. Finish when every joint has both a min and a max.",
      advance: "done sweeping",
    };
  }
  if (session.phase === "review") {
    return {
      cols: "1fr 108px 108px 108px",
      head: ["joint", "min", "max", "offset"],
      row: (j: string) => {
        const p = session.proposed?.[j];
        const c = session.current?.[j];
        return [
          j,
          `${c?.range_min ?? "—"} → ${p?.range_min ?? "—"}`,
          `${c?.range_max ?? "—"} → ${p?.range_max ?? "—"}`,
          `${c?.homing_offset ?? "—"} → ${p?.homing_offset ?? "—"}`,
        ];
      },
      help: "Step 3 of 3 — review. The previous calibration file is preserved as a .bak-<ts> sibling before the new one is written.",
      advance: "save calibration",
    };
  }
  return {
    cols: "1fr 90px",
    head: ["joint", "ticks"],
    row: (j: string) => [j, dash(session.ticks[j])],
    help: "Step 1 of 3 — set neutral pose. Move the arm by hand into the pose you want to be 0°. Torque is off; the arm is back-drivable.",
    advance: "capture neutral",
  };
}
