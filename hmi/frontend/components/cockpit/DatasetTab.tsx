"use client";

/**
 * Dataset: what a take will contain, and the controls to record one.
 *
 * The left grid answers a question the old RecordingPanel could not: which
 * cameras actually end up in the dataset. The recorder fixes its schema at
 * start from the cameras that can yield RGB frames, so placeholder slots and
 * sim views are drawn dimmed and marked "not in take" — finding that out after
 * a recording session is an expensive way to learn it.
 */
import { useTelemetry } from "@/lib/telemetry";
import { useRecorder } from "@/lib/recorder";
import { cameraStreamUrl, type CameraInfo } from "@/lib/api";
import { repoIdFor } from "./CommandBar";
import { gridPlan } from "./cameraGrid";
import { startTake, stopTake, NO_TELEOP_WARNING } from "./recorderActions";

/** Recorder.start_episode() builds its feature set from cameras that can
 *  actually produce frames; a placeholder would break add_frame every tick. */
function loggedInTake(c: CameraInfo): boolean {
  return c.active && c.source !== "placeholder";
}

export function DatasetTab({ cameras }: { cameras: CameraInfo[] }) {
  const task = useRecorder((s) => s.task);
  const hfUser = useRecorder((s) => s.hfUser);
  const setTask = useRecorder((s) => s.setTask);
  const setHfUser = useRecorder((s) => s.setHfUser);
  const recording = useRecorder((s) => s.status?.recording ?? false);
  const frames = useRecorder((s) => s.status?.episode_frames ?? 0);
  const skipped = useRecorder((s) => s.status?.skipped_frames ?? 0);
  const lastError = useRecorder((s) => s.status?.last_error ?? null);
  const busy = useRecorder((s) => s.busy);
  const teleopRunning = useTelemetry((s) => s.lastFrame?.human_teleop?.running ?? false);
  const armCount = useTelemetry((s) => Object.keys(s.lastFrame?.arms ?? {}).length);

  const repoId = repoIdFor(hfUser, task);
  const logged = cameras.filter(loggedInTake);
  const plan = gridPlan(cameras);

  return (
    <div className="grid min-h-0 grid-cols-[minmax(0,1fr)_360px] gap-2 overflow-hidden p-2">
      <div className="flex min-h-0 flex-col overflow-hidden rounded-lg bg-card shadow-[0_0_0_1px_var(--border)]">
        <div className="flex h-8.5 shrink-0 items-center justify-between border-b border-border px-3">
          <span className="label-tracked text-muted-foreground">Take composition</span>
          <span className="font-mono text-[10px] text-muted-foreground">
            {armCount} arm{armCount === 1 ? "" : "s"} · {logged.length} camera
            {logged.length === 1 ? "" : "s"} · base logged per frame
          </span>
        </div>
        <div
          className="grid min-h-0 flex-1 auto-rows-[minmax(0,1fr)] gap-2 overflow-y-auto p-2.5"
          style={{ gridTemplateColumns: `repeat(${plan.columns}, minmax(0,1fr))` }}
        >
          {cameras.map((c) => {
            const inTake = loggedInTake(c);
            return (
              <div
                key={c.id}
                className={
                  "flex min-h-0 flex-col overflow-hidden rounded-md border border-border bg-[var(--haller-inset)] " +
                  (inTake ? "" : "opacity-45")
                }
                style={{ gridColumn: `span ${plan.span(c)}` }}
              >
                <span
                  className="shrink-0 truncate px-2 py-1.5 font-mono text-[9px]"
                  style={{
                    color: inTake ? "var(--foreground)" : "var(--muted-foreground)",
                  }}
                >
                  {c.id}
                  {inTake ? "" : " · not in take"}
                </span>
                <div className="relative flex min-h-0 flex-1 items-center justify-center">
                  {c.active ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={cameraStreamUrl(c.id)}
                      alt={`${c.id} live feed`}
                      className="absolute inset-0 h-full w-full object-contain"
                    />
                  ) : (
                    <>
                      <span className="scanlines absolute inset-0" aria-hidden />
                      <span className="relative font-mono text-[9px] tracking-[0.14em] uppercase text-muted-foreground opacity-70">
                        {c.source === "placeholder" ? "reserved slot" : "no feed"}
                      </span>
                    </>
                  )}
                </div>
              </div>
            );
          })}
          {cameras.length === 0 && (
            <span className="label-micro text-muted-foreground">
              no cameras configured — the take will hold state, action and base only
            </span>
          )}
        </div>
      </div>

      <div className="flex min-h-0 flex-col overflow-hidden rounded-lg bg-card shadow-[0_0_0_1px_var(--border)]">
        <div className="flex h-8.5 shrink-0 items-center justify-between border-b border-border px-3">
          <span className="label-tracked text-muted-foreground">Recorder</span>
          <span
            className="inline-flex h-5 items-center gap-1.5 rounded-full px-2 font-mono text-[10px]"
            style={{
              background: recording
                ? "oklch(0.62 0.245 27 / 0.18)"
                : "var(--secondary)",
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
            {recording ? `rec · ${frames} frames` : "standby"}
          </span>
        </div>

        <div className="flex min-h-0 flex-col gap-3 overflow-y-auto p-3">
          <p className="text-[11px] text-pretty text-muted-foreground">
            Logs both arms + cameras + base into a LeRobotDataset from inside
            the HMI. Start human-pose teleop first, then record and demonstrate
            with the dead-man held — the recorder logs the teleop&apos;s
            commanded joint targets as <span className="font-mono">action</span>{" "}
            and the measured joints as{" "}
            <span className="font-mono">observation.state</span>.
          </p>

          <label className="flex flex-col gap-1.5">
            <span className="label-tracked text-muted-foreground">Task</span>
            <input
              value={task}
              onChange={(e) => setTask(e.target.value)}
              disabled={recording}
              placeholder="What should the policy learn?"
              className="h-7.5 w-full rounded-md border border-input bg-background px-2.5 font-mono text-[11px] disabled:opacity-50"
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className="label-tracked text-muted-foreground">HF user</span>
            <input
              value={hfUser}
              onChange={(e) => setHfUser(e.target.value)}
              disabled={recording}
              placeholder="osrdvs"
              className="h-7.5 w-full rounded-md border border-input bg-background px-2.5 font-mono text-[11px] disabled:opacity-50"
            />
          </label>

          <div className="rounded-md border border-border bg-muted p-2.5 font-mono text-[10px] break-all">
            {repoId}
          </div>

          <div className="flex items-baseline gap-2.5 font-mono text-[11px]">
            <span className="label-micro text-muted-foreground">frames</span>
            <span data-num className="tabular-nums">
              {frames}
            </span>
            {/* Nonzero means some ticks were dropped (stale camera / missing arm
                telemetry) — the take has gaps; flag it rather than hide it. */}
            {skipped > 0 && (
              <span className="tabular-nums text-[var(--haller-warn)]">
                {skipped} skipped
              </span>
            )}
          </div>

          {/* Warning, not a block: the backend does not enforce this, so a
              disabled button here would look like an invariant without being
              one — and a bring-up take with no teleop is legitimate. */}
          {!recording && !teleopRunning && (
            <div className="rounded-md border border-[var(--haller-warn)] px-2.5 py-2 text-[11px] text-pretty text-[var(--haller-warn)]">
              {NO_TELEOP_WARNING}
            </div>
          )}

          {lastError && (
            <div className="rounded-md border border-[var(--haller-fault)] px-2.5 py-2 font-mono text-[10px] break-all text-[var(--haller-fault)]">
              recorder error: {lastError}
            </div>
          )}

          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={busy}
              onClick={() =>
                recording ? stopTake(true) : startTake(repoId, task, teleopRunning)
              }
              className={
                "h-8 flex-1 rounded-md label-micro tracking-[0.12em] disabled:opacity-50 " +
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
                className="h-8 rounded-md border border-border bg-secondary px-3 label-micro tracking-[0.12em] disabled:opacity-50"
              >
                discard take
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
