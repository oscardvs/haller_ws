"use client";

/**
 * Leader → follower teleop bridge: back-drive one arm by hand, the other
 * mirrors it. Lives in the command bar so it can be started and watched from
 * any tab.
 *
 * Not the headset path — that is the Teleop tab, and this popover links to it.
 * Manual joint control on both participating arms is disabled while this runs
 * — the arm cards read the same telemetry and lock themselves; see ArmCard's
 * lockFor().
 */
import { useEffect, useState, type RefObject } from "react";
import { toast } from "sonner";

import { api } from "@/lib/api";
import { useTelemetry } from "@/lib/telemetry";
import { Popover, PopoverHeader } from "./Popover";
import type { TabId } from "./lib";

const MIN_HZ = 1;
const MAX_HZ = 200;

export function TeleopPopover({
  armIds,
  onClose,
  triggerRef,
  onTab,
}: {
  armIds: string[];
  onClose: () => void;
  triggerRef: RefObject<HTMLButtonElement | null>;
  onTab: (t: TabId) => void;
}) {
  const running = useTelemetry((s) => s.lastFrame?.teleop?.running ?? false);
  const liveLeader = useTelemetry((s) => s.lastFrame?.teleop?.leader ?? null);
  const liveFollower = useTelemetry((s) => s.lastFrame?.teleop?.follower ?? null);
  const ticks = useTelemetry((s) => s.lastFrame?.teleop?.tick_count);
  const liveHz = useTelemetry((s) => s.lastFrame?.teleop?.hz);
  const startedAt = useTelemetry((s) => s.lastFrame?.teleop?.started_at ?? null);
  const lastError = useTelemetry((s) => s.lastFrame?.teleop?.last_error ?? null);

  const [draftLeader, setDraftLeader] = useState(() => armIds[0] ?? "");
  const [draftFollower, setDraftFollower] = useState(
    () => armIds.find((id) => id !== armIds[0]) ?? "",
  );
  const [hz, setHz] = useState("60");
  const [error, setError] = useState<string | null>(null);
  const now = useNow(running);

  // Derived, not mirrored into state. While the bridge runs, the pair IS what
  // the backend reports — including a session someone else started from
  // another browser. Only when idle does the operator's draft speak.
  const leader = running && liveLeader ? liveLeader : draftLeader;
  // A bridge from an arm to itself is not a thing, so the follower slides off
  // a leader that collides with it rather than being corrected after a render.
  const follower =
    running && liveFollower
      ? liveFollower
      : draftFollower === leader
        ? (armIds.find((id) => id !== leader) ?? draftFollower)
        : draftFollower;

  const uptime =
    running && startedAt
      ? `${Math.max(0, Math.floor(now / 1000 - startedAt))} s`
      : "—";

  const act = async () => {
    if (running) {
      try {
        await api.teleopStop();
        toast.message("teleop stopped");
      } catch (e) {
        toast.error(`teleop stop failed: ${(e as Error).message}`);
      }
      return;
    }
    const rate = Number(hz);
    if (!Number.isFinite(rate) || rate < MIN_HZ || rate > MAX_HZ) {
      setError(`rate must be ${MIN_HZ}–${MAX_HZ} Hz`);
      toast.error(`teleop start failed: rate must be ${MIN_HZ}–${MAX_HZ} Hz`);
      return;
    }
    setError(null);
    try {
      await api.teleopStart(leader, follower, rate);
      toast.success(`teleop started: ${leader} → ${follower} @ ${rate} Hz`);
    } catch (e) {
      toast.error(`teleop start failed: ${(e as Error).message}`);
    }
  };

  const tooFewArms = armIds.length < 2;

  return (
    <Popover
      onClose={onClose}
      triggerRef={triggerRef}
      label="Teleop bridge"
      className="bottom-[42px] left-2 w-[min(560px,calc(100%-16px))]"
    >
      <PopoverHeader title="Teleop bridge · leader → follower" onClose={onClose} />

      {tooFewArms ? (
        <p className="label-micro text-muted-foreground">
          teleop needs ≥2 enabled arms in <code>hmi/backend/config.yaml</code>
        </p>
      ) : (
        <div className="flex flex-wrap items-center gap-2">
          <span className="label-micro text-muted-foreground">Leader</span>
          <Select
            ariaLabel="leader arm"
            value={leader}
            onChange={setDraftLeader}
            options={armIds}
            disabled={running}
          />
          <button
            type="button"
            title="swap leader and follower"
            disabled={running}
            onClick={() => {
              setDraftLeader(follower);
              setDraftFollower(leader);
            }}
            className="h-7 rounded-sm border border-border bg-secondary px-2.5 disabled:opacity-50"
          >
            ⇄
          </button>
          <span className="label-micro text-muted-foreground">Follower</span>
          <Select
            ariaLabel="follower arm"
            value={follower}
            onChange={setDraftFollower}
            options={armIds.filter((id) => id !== leader)}
            disabled={running}
          />
          <span className="label-micro text-muted-foreground">Rate</span>
          <input
            value={running && typeof liveHz === "number" ? String(liveHz) : hz}
            onChange={(e) => setHz(e.target.value)}
            disabled={running}
            inputMode="numeric"
            aria-label="teleop rate in Hz"
            className="h-7 w-[58px] rounded-sm border border-input bg-background px-2 text-right font-mono text-[11px] disabled:opacity-50"
          />
          <span className="font-mono text-[11px] text-muted-foreground">Hz</span>
          <button
            type="button"
            onClick={act}
            className={
              "h-7 rounded-md px-3.5 label-micro tracking-[0.12em] " +
              (running
                ? "border border-[var(--haller-fault)] bg-[oklch(0.62_0.245_27/0.2)] text-[var(--haller-fault)]"
                : "bg-primary text-primary-foreground")
            }
          >
            {running ? "stop" : "start"}
          </button>
        </div>
      )}

      <div className="mt-2.5 flex flex-wrap items-center gap-3.5 border-t border-border pt-2.5 font-mono text-[10px] text-muted-foreground">
        <span>
          ticks{" "}
          <span className="text-foreground" data-num>
            {running && typeof ticks === "number" ? ticks : "—"}
          </span>
        </span>
        <span>
          rate{" "}
          <span className="text-foreground" data-num>
            {running && typeof liveHz === "number" ? `${liveHz.toFixed(0)} Hz` : "—"}
          </span>
        </span>
        <span>
          uptime{" "}
          <span className="text-foreground" data-num>
            {uptime}
          </span>
        </span>
        <button
          type="button"
          onClick={() => onTab("teleop")}
          className="ml-auto text-[11px] text-foreground hover:text-[var(--haller-live)]"
        >
          Quest teleop →
        </button>
      </div>

      {(error ?? lastError) && (
        <div className="mt-2 rounded-md border border-[var(--haller-fault)] px-2.5 py-2 font-mono text-[10px] text-[var(--haller-fault)]">
          {error ? error : `last error: ${lastError}`}
        </div>
      )}

      <p className="mt-2 text-[11px] text-pretty text-muted-foreground">
        Back-drive the leader by hand — the follower mirrors it. Manual joint
        control on both participating arms is disabled while the bridge runs.
      </p>
    </Popover>
  );
}

/** Wall clock, ticking only while there is an uptime worth counting. Reading
 *  Date.now() straight in the render body makes the component impure and, more
 *  practically, means the uptime only advances when something else re-renders. */
function useNow(active: boolean): number {
  const [now, setNow] = useState(0);
  useEffect(() => {
    if (!active) return;
    const tick = () => setNow(Date.now());
    const t = setInterval(tick, 1000);
    // First value lands a tick late rather than synchronously in the effect
    // body, which keeps this out of the cascading-render pattern.
    const first = setTimeout(tick, 0);
    return () => {
      clearInterval(t);
      clearTimeout(first);
    };
  }, [active]);
  return now;
}

function Select({
  value,
  onChange,
  options,
  ariaLabel,
  disabled,
}: {
  value: string;
  onChange: (v: string) => void;
  options: string[];
  ariaLabel: string;
  disabled?: boolean;
}) {
  return (
    <select
      aria-label={ariaLabel}
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
      className="h-7 rounded-sm border border-input bg-background px-2 font-mono text-[11px] disabled:opacity-50"
    >
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  );
}
