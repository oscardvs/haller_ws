"use client";

/** Compact recorder: rename the take, start/stop, jump to the full workspace.
 *  Everything here also exists on the Dataset tab — this is the version you
 *  reach without leaving the tab you are working on. */
import type { RefObject } from "react";

import { useRecorder } from "@/lib/recorder";
import { useTelemetry } from "@/lib/telemetry";
import { Popover, PopoverHeader } from "./Popover";
import { repoIdFor } from "./CommandBar";
import { startTake, stopTake, NO_TELEOP_WARNING } from "./recorderActions";
import type { TabId } from "./lib";

export function RecordPopover({
  onClose,
  triggerRef,
  onTab,
}: {
  onClose: () => void;
  triggerRef: RefObject<HTMLButtonElement | null>;
  onTab: (t: TabId) => void;
}) {
  const task = useRecorder((s) => s.task);
  const hfUser = useRecorder((s) => s.hfUser);
  const setTask = useRecorder((s) => s.setTask);
  const recording = useRecorder((s) => s.status?.recording ?? false);
  const frames = useRecorder((s) => s.status?.episode_frames ?? 0);
  const busy = useRecorder((s) => s.busy);
  const teleopRunning = useTelemetry((s) => s.lastFrame?.human_teleop?.running ?? false);

  const repoId = repoIdFor(hfUser, task);

  return (
    <Popover
      onClose={onClose}
      triggerRef={triggerRef}
      label="Recorder"
      className="bottom-[42px] left-2 w-[min(480px,calc(100%-16px))]"
    >
      <PopoverHeader title="Recorder" onClose={onClose} />

      <div className="flex items-center gap-2">
        <input
          value={task}
          onChange={(e) => setTask(e.target.value)}
          disabled={recording}
          aria-label="task description"
          className="h-7 min-w-0 flex-1 rounded-md border border-input bg-background px-2.5 font-mono text-[11px] disabled:opacity-50"
        />
        <button
          type="button"
          disabled={busy}
          onClick={() =>
            recording ? stopTake(true) : startTake(repoId, task, teleopRunning)
          }
          className={
            "h-7 shrink-0 rounded-md px-3 label-micro tracking-[0.12em] disabled:opacity-50 " +
            (recording
              ? "border border-[var(--haller-fault)] bg-[oklch(0.62_0.245_27/0.2)] text-[var(--haller-fault)]"
              : "bg-primary text-primary-foreground")
          }
        >
          {recording ? "stop & save" : "start recording"}
        </button>
        {recording && (
          <button
            type="button"
            disabled={busy}
            onClick={() => stopTake(false)}
            className="h-7 shrink-0 rounded-md border border-border bg-secondary px-2.5 label-micro tracking-[0.12em] disabled:opacity-50"
          >
            discard
          </button>
        )}
      </div>

      {!recording && !teleopRunning && (
        <p className="mt-2 text-[11px] text-pretty text-[var(--haller-warn)]">
          {NO_TELEOP_WARNING}
        </p>
      )}

      <div className="mt-2.5 flex items-baseline gap-3 font-mono text-[10px] text-muted-foreground">
        <span>
          frames{" "}
          <span className="text-foreground" data-num>
            {frames}
          </span>
        </span>
        <span className="min-w-0 truncate">{repoId}</span>
      </div>

      <div className="mt-2.5 border-t border-border pt-2.5">
        <button
          type="button"
          onClick={() => onTab("data")}
          className="font-mono text-[11px] hover:text-[var(--haller-live)]"
        >
          Open data workspace →
        </button>
      </div>
    </Popover>
  );
}
