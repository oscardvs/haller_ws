"use client";

/**
 * 34px command bar: the two controls that must be reachable from every tab
 * (the teleop bridge and the recorder), plus a context hint and a census.
 *
 * Both controls open a popover rather than acting directly — starting a teleop
 * bridge or a take from a 24px strip you might click by accident is not a
 * thing this surface should offer.
 */
import { useEffect, useState, type RefObject } from "react";

import { useTelemetry } from "@/lib/telemetry";
import { useRecorder } from "@/lib/recorder";
import { slugify, type TabId, type Viewport } from "./lib";

export function CommandBar({
  tab,
  pop,
  onToggleTeleop,
  onToggleRec,
  teleopRef,
  recRef,
  census,
  version,
  viewport,
}: {
  tab: TabId;
  pop: string | null;
  onToggleTeleop: () => void;
  onToggleRec: () => void;
  teleopRef: RefObject<HTMLButtonElement | null>;
  recRef: RefObject<HTMLButtonElement | null>;
  census: string;
  version: string;
  viewport: Viewport;
}) {
  const link = useTelemetry((s) => s.link);
  const teleopRunning = useTelemetry((s) => s.lastFrame?.teleop?.running ?? false);
  const teleopLeader = useTelemetry((s) => s.lastFrame?.teleop?.leader ?? null);
  const teleopFollower = useTelemetry((s) => s.lastFrame?.teleop?.follower ?? null);
  const teleopHz = useTelemetry((s) => s.lastFrame?.teleop?.hz);
  const recording = useRecorder((s) => s.status?.recording ?? false);
  const frames = useRecorder((s) => s.status?.episode_frames ?? 0);

  const clock = useClock();

  const teleopStatus = teleopRunning
    ? `${teleopLeader} → ${teleopFollower}${typeof teleopHz === "number" ? ` · ${teleopHz.toFixed(0)} Hz` : ""}`
    : "idle";
  const recStatus = recording ? `rec · ${frames} frames` : "standby";

  return (
    <div className="flex items-center gap-2 overflow-hidden border-t border-border bg-[var(--haller-chrome)] px-2.5">
      <button
        ref={teleopRef}
        type="button"
        onClick={onToggleTeleop}
        aria-expanded={pop === "teleop"}
        className={
          "inline-flex h-6 shrink-0 items-center gap-2 rounded-sm border border-border px-2.5 " +
          (pop === "teleop" ? "bg-muted" : "hover:bg-muted/50")
        }
      >
        <span className="label-micro text-muted-foreground">Teleop</span>
        <span
          className="font-mono text-[10px] whitespace-nowrap"
          style={{
            color: teleopRunning ? "var(--haller-live)" : "var(--muted-foreground)",
          }}
        >
          {teleopStatus}
        </span>
        <span className="text-[9px] text-muted-foreground">▴</span>
      </button>

      <button
        ref={recRef}
        type="button"
        onClick={onToggleRec}
        aria-expanded={pop === "rec"}
        className={
          "inline-flex h-6 shrink-0 items-center gap-2 rounded-sm border border-border px-2.5 " +
          (pop === "rec" ? "bg-muted" : "hover:bg-muted/50")
        }
      >
        <span className="label-micro text-muted-foreground">Record</span>
        <span
          className="inline-flex items-center gap-1.5 font-mono text-[10px] whitespace-nowrap"
          style={{
            color: recording ? "var(--haller-fault)" : "var(--muted-foreground)",
          }}
        >
          <span
            className={
              "h-1.5 w-1.5 rounded-full " + (recording ? "animate-haller-rec" : "")
            }
            style={{
              backgroundColor: recording
                ? "var(--haller-fault)"
                : "var(--muted-foreground)",
            }}
          />
          {recStatus}
        </span>
        <span className="text-[9px] text-muted-foreground">▴</span>
      </button>

      <span aria-hidden className="h-3.5 w-px shrink-0 bg-border" />

      <span className="min-w-0 truncate font-mono text-[10px] text-muted-foreground">
        {hintFor(tab, link, viewport)}
      </span>

      <span className="ml-auto flex shrink-0 items-center gap-3 font-mono text-[10px] whitespace-nowrap text-muted-foreground">
        <span>{census}</span>
        <span>cfg {version}</span>
        <span data-num>{clock}</span>
      </span>
    </div>
  );
}

/** What the operator most needs to know on this tab, right now. Link trouble
 *  outranks every per-tab hint: with the socket down the readouts are frozen
 *  and every command will fail, which changes the meaning of the whole screen. */
function hintFor(
  tab: TabId,
  link: "live" | "reconnecting" | "disconnected",
  viewport: Viewport,
): string {
  if (link === "disconnected") {
    return "link down — readouts frozen, commands will fail until the websocket reconnects";
  }
  if (link === "reconnecting") {
    return "link unstable — readouts are held at em-dash until frames resume";
  }
  switch (tab) {
    case "operate":
      // The layout drops things at these sizes; saying which beats leaving the
      // operator to wonder where the second arm went.
      if (viewport.short) {
        return "short viewport — wrist tiles are collapsed to their label strip";
      }
      if (viewport.compact) {
        return "narrow layout — one arm at a time, pick it above the card";
      }
      return "drag joints to command · wasd or arrows drive the base";
    case "human":
      return "hold SPACE (or open MOUTH) to close the dead-man — this tab only";
    case "calibrate":
      return "every arm must be in manual before a session starts";
    case "cameras":
      return "the Operate tab's chips pick which of these is the primary view";
    case "dataset":
      return "start human teleop first — the take logs its commanded targets as action";
    case "settings":
      return "config is read-only — edit hmi/backend/config.yaml and restart";
  }
}

/** Wall clock at 1 Hz. The design ticked this at 10 Hz with centiseconds; the
 *  rail's `t` cell already carries sub-second resolution from the frame's own
 *  timestamp, which is the one that means something. */
function useClock(): string {
  const [now, setNow] = useState(() => "--:--:--");
  useEffect(() => {
    const read = () =>
      setNow(new Date().toLocaleTimeString([], { hour12: false }));
    read();
    const t = setInterval(read, 1000);
    return () => clearInterval(t);
  }, []);
  return now;
}

/** Shared by the Record popover and the Dataset tab. */
export function repoIdFor(hfUser: string, task: string): string {
  return `${hfUser || "local"}/haller_${slugify(task)}`;
}
