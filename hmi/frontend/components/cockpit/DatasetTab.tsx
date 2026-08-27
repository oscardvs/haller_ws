"use client";

/**
 * Dataset: what a take will contain, the controls to record one, and what is
 * already on disk.
 *
 * The camera grid answers a question no status line can: which cameras end up
 * in the dataset. That is now a decision, not a reading — `record` is runtime
 * state per camera, and the recorder freezes its feature set from it at
 * `start_episode`. Finding out after a session that the mast cam was not in
 * the take is an expensive way to learn it, so the toggle and the live tile
 * are the same object.
 *
 * The episode browser reads the dataset meta off disk. It is deliberately
 * read-only apart from delete-last: lerobot owns this directory, and the one
 * edit worth offering is undoing the take you just realised was bad.
 */
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";

import { useTelemetry } from "@/lib/telemetry";
import { useRecorder } from "@/lib/recorder";
import {
  api, ApiError, cameraStreamUrl, recordRateFaithful, recordRateTolerance,
  type CameraInfo, type EpisodesResponse, type RepoInfo,
} from "@/lib/api";
import { repoIdFor } from "./CommandBar";
import { gridPlan } from "./cameraGrid";
import { startTake, stopTake, NO_TELEOP_WARNING } from "./recorderActions";

/** What a camera contributes to the next take.
 *
 *  `capable` is a fact about the camera — the recorder builds its feature set
 *  from cameras that can actually produce frames, and a placeholder would
 *  break add_frame every tick. `inTake` is the operator's choice on top of it.
 *  `known` is false against a backend that predates the runtime toggle, where
 *  capability is all there is and the switch would be a lie. */
function takeStateFor(c: CameraInfo) {
  const capable = c.active && c.source !== "placeholder";
  const known = typeof c.record === "boolean";
  return { capable, known, inTake: known ? c.record === true : capable };
}

function formatBytes(n: number | undefined): string {
  if (typeof n !== "number" || !Number.isFinite(n)) return "—";
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(v >= 100 ? 0 : 1)} ${units[i]}`;
}

export function DatasetTab({
  cameras,
  onCameraRecord,
}: {
  cameras: CameraInfo[];
  /** Lifts the accepted toggle back into the cockpit's one camera list, so
   *  every tab reads the same answer to "is this camera in the take". */
  onCameraRecord: (id: string, record: boolean) => void;
}) {
  const task = useRecorder((s) => s.task);
  const hfUser = useRecorder((s) => s.hfUser);
  const setTask = useRecorder((s) => s.setTask);
  const setHfUser = useRecorder((s) => s.setHfUser);
  const recording = useRecorder((s) => s.status?.recording ?? false);
  const frames = useRecorder((s) => s.status?.episode_frames ?? 0);
  const skipped = useRecorder((s) => s.status?.skipped_frames ?? 0);
  const lastError = useRecorder((s) => s.status?.last_error ?? null);
  // The rate pair. Optional on the wire, so a backend that predates the
  // measured sampler simply draws no rate row rather than an em-dash.
  const fpsMeasured = useRecorder((s) => s.status?.fps_measured ?? null);
  const fpsDeclared = useRecorder((s) => s.status?.fps_declared ?? null);
  // Two-sided, and null when the band is unknowable — either the rate is
  // not measured yet or the backend publishes no tolerance. Neither is a
  // warning, and neither may be drawn as one.
  const rateOk = useRecorder((s) => recordRateFaithful(s.status));
  const tol = useRecorder((s) => recordRateTolerance(s.status));
  const liveRepo = useRecorder((s) => s.status?.repo_id ?? null);
  const busy = useRecorder((s) => s.busy);
  const teleopRunning = useTelemetry((s) => s.lastFrame?.human_teleop?.running ?? false);
  const armCount = useTelemetry((s) => Object.keys(s.lastFrame?.arms ?? {}).length);

  const repoId = repoIdFor(hfUser, task);
  const logged = cameras.filter((c) => takeStateFor(c).inTake);
  const plan = gridPlan(cameras);

  const toggleRecord = useCallback(
    async (c: CameraInfo, next: boolean) => {
      try {
        const r = await api.cameraRecord(c.id, next);
        onCameraRecord(r.id, r.record);
      } catch (e) {
        toast.error(`${c.id}: ${(e as Error).message}`);
      }
    },
    [onCameraRecord],
  );

  return (
    <div className="grid min-h-0 grid-cols-[minmax(0,1fr)_380px] gap-2 overflow-hidden p-2">
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
          {cameras.map((c) => (
            <CameraTakeTile
              key={c.id}
              camera={c}
              span={plan.span(c)}
              recording={recording}
              onToggle={toggleRecord}
            />
          ))}
          {cameras.length === 0 && (
            <span className="label-micro text-muted-foreground">
              no cameras configured — the take will hold state, action and base only
            </span>
          )}
        </div>
      </div>

      <div className="grid min-h-0 grid-rows-[auto_minmax(0,1fr)] gap-2 overflow-hidden">
        <div className="flex shrink-0 flex-col overflow-hidden rounded-lg bg-card shadow-[0_0_0_1px_var(--border)]">
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

          <div className="flex flex-col gap-2.5 p-3">
            <p className="text-[11px] text-pretty text-muted-foreground">
              Logs both arms + the cameras above + base into a LeRobotDataset.
              Start a teleop session first, then record and demonstrate with a
              grip held — the recorder logs the session&apos;s commanded joint
              targets as <span className="font-mono">action</span> and the
              measured joints as{" "}
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

            {/* MEASURED against DECLARED, both of them, always.

                `fps` used to be `1 / telemetry._period` — the rate telemetry was
                ASKED for, never measured against a real bus — and every
                timestamp in every episode was synthesised from it. Printing one
                number here would reproduce that: it would look like a
                measurement whichever of the two it was. The gap between them IS
                the defect, so the gap is what is drawn. The threshold is the
                recorder's own, read from its status rather than chosen here. */}
            {typeof fpsDeclared === "number" && (
              <div className="flex items-baseline gap-2.5 font-mono text-[11px]">
                <span className="label-micro text-muted-foreground">rate</span>
                {fpsMeasured === null || fpsMeasured === undefined ? (
                  <span className="text-muted-foreground">
                    measuring… <span data-num className="tabular-nums">{fpsDeclared}</span> declared
                  </span>
                ) : (
                  <span
                    className="tabular-nums"
                    style={rateOk === false ? { color: "var(--haller-warn)" } : undefined}
                    title={
                      rateOk === false && tol !== null
                        ? `outside ±${(tol * 100).toFixed(1)}% of the declared rate — ` +
                          "the recorder refuses to open an episode here"
                        : tol === null
                          ? "measured / declared — this backend publishes no rate tolerance"
                          : "measured / declared"
                    }
                  >
                    <span data-num>{fpsMeasured.toFixed(1)}</span>
                    <span className="text-muted-foreground"> / </span>
                    <span data-num>{fpsDeclared}</span> Hz
                  </span>
                )}
              </div>
            )}

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

        <EpisodeBrowser
          recording={recording}
          draftRepo={repoId}
          liveRepo={liveRepo}
        />
      </div>
    </div>
  );
}

/* ---- one camera, with its take membership ------------------------------- */

function CameraTakeTile({
  camera,
  span,
  recording,
  onToggle,
}: {
  camera: CameraInfo;
  span: number;
  recording: boolean;
  onToggle: (c: CameraInfo, next: boolean) => void;
}) {
  const { capable, known, inTake } = takeStateFor(camera);
  // Asked for but unable to deliver: the recorder drops the whole tick when a
  // required camera has no fresh image, so this would cost every frame.
  const willFail = inTake && !capable;
  const colour = willFail
    ? "var(--haller-warn)"
    : inTake
      ? "var(--foreground)"
      : "var(--muted-foreground)";

  return (
    <div
      className={
        "flex min-h-0 flex-col overflow-hidden rounded-md border border-border bg-[var(--haller-inset)] " +
        (inTake ? "" : "opacity-45")
      }
      style={{ gridColumn: `span ${span}` }}
      data-camera-id={camera.id}
    >
      <div className="flex h-6.5 shrink-0 items-center gap-1.5 px-2">
        <span className="min-w-0 flex-1 truncate font-mono text-[9px]" style={{ color: colour }}>
          {camera.id}
          {willFail ? " · no feed" : ""}
        </span>
        {known ? (
          <button
            type="button"
            role="switch"
            aria-checked={inTake}
            aria-label={`record ${camera.id}`}
            disabled={recording}
            title={
              recording
                ? "the feature set is frozen while an episode is open"
                : inTake
                  ? "drop this camera from the next take"
                  : "add this camera to the next take"
            }
            onClick={() => onToggle(camera, !inTake)}
            className={
              "inline-flex h-4.5 shrink-0 items-center gap-1 rounded-full border px-1.5 label-micro disabled:opacity-50 " +
              (inTake
                ? "border-[var(--haller-live)] text-[var(--haller-live)]"
                : "border-border text-muted-foreground")
            }
          >
            <span
              aria-hidden
              className="h-1 w-1 rounded-full"
              style={{
                backgroundColor: inTake
                  ? "var(--haller-live)"
                  : "var(--muted-foreground)",
              }}
            />
            rec
          </button>
        ) : (
          <span className="shrink-0 label-micro text-muted-foreground" title="this backend has no runtime record toggle">
            {inTake ? "in take" : "not in take"}
          </span>
        )}
      </div>
      <div className="relative flex min-h-0 flex-1 items-center justify-center">
        {camera.active ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={cameraStreamUrl(camera.id)}
            alt={`${camera.id} live feed`}
            className="absolute inset-0 h-full w-full object-contain"
          />
        ) : (
          <>
            <span className="scanlines absolute inset-0" aria-hidden />
            <span className="relative font-mono text-[9px] tracking-[0.14em] uppercase text-muted-foreground opacity-70">
              {camera.source === "placeholder" ? "reserved slot" : "no feed"}
            </span>
          </>
        )}
      </div>
    </div>
  );
}

/* ---- what is already on disk -------------------------------------------- */

function EpisodeBrowser({
  recording,
  draftRepo,
  liveRepo,
}: {
  recording: boolean;
  /** The repo the next take would write to. */
  draftRepo: string;
  /** The repo the recorder is writing to now, or last wrote to. */
  liveRepo: string | null;
}) {
  const [repos, setRepos] = useState<RepoInfo[] | null>(null);
  /** null = "whatever the backend calls current", which is what the endpoints
   *  default to. Only an explicit pick pins a repo. */
  const [pick, setPick] = useState<string | null>(null);
  const [data, setData] = useState<EpisodesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [armed, setArmed] = useState(false);
  /** The backend's own words for why the last pop was refused. Cleared on any
   *  refresh, because every refusal it reports is about a state that can
   *  change — an open episode, a lone episode, a foreign dataset. */
  const [refusal, setRefusal] = useState<string | null>(null);
  /** Set only when the backend cannot delete at all (404/501), which is a
   *  property of the build, not of the dataset. */
  const [deleteUnsupported, setDeleteUnsupported] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const refresh = useCallback(() => {
    setRefusal(null);
    setEpoch((n) => n + 1);
  }, []);

  useEffect(() => {
    let cancelled = false;
    api.recordRepos()
      .then((r) => { if (!cancelled) setRepos(r.repos); })
      .catch(() => { if (!cancelled) setRepos([]); });
    return () => { cancelled = true; };
  }, [epoch]);

  // `recording` is in the dependency list so that stopping a take refreshes
  // the list — the episode that just landed is the one the operator wants to
  // see, and a stale count here reads as "it did not save".
  useEffect(() => {
    let cancelled = false;
    api.recordEpisodes(pick)
      .then((r) => { if (!cancelled) { setData(r); setError(null); } })
      .catch((e: Error) => {
        if (cancelled) return;
        setData(null);
        // 400 here is not a fault: it is the recorder saying it has never
        // opened a repo, which is simply true of a backend that has just come
        // up. `e.message` would render it as "HTTP 400: …", and an operator who
        // reads a status code on a fresh cockpit goes looking for a broken
        // build instead of picking a dataset. The backend's own sentence, plus
        // the move that clears it.
        setError(
          e instanceof ApiError && e.status === 404
            ? "this backend has no episode listing"
            : e instanceof ApiError && e.status === 400
              ? `${e.detail} — pick one above, or start a take`
              : e.message,
        );
      });
    return () => { cancelled = true; };
  }, [pick, epoch, recording]);

  const episodes = data?.episodes ?? [];
  // Highest index, not last in the array: "delete last" means the newest
  // episode, and naming the wrong one on a confirm button is how the operator
  // deletes a take they wanted.
  const last = episodes.reduce<typeof episodes[number] | null>(
    (best, e) => (best === null || e.index > best.index ? e : best),
    null,
  );
  const shown = data?.repo_id ?? pick ?? liveRepo ?? draftRepo;

  const doDelete = async () => {
    if (!last) return;
    setBusy(true);
    setArmed(false);
    try {
      const r = await api.recordDeleteLastEpisode(pick);
      toast.success(
        `deleted episode ${r.deleted_index} · ${r.deleted_frames} frames · ` +
        `${r.total_episodes} left`,
      );
      refresh();
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        // A refusal, not a failure. The pop is in-place and conservative: it
        // declines rather than leave a dataset lerobot can no longer resume,
        // and the reason names which guard tripped. Keep the button — every
        // one of those states can be cleared.
        setRefusal(e.detail);
      } else if (e instanceof ApiError && (e.status === 404 || e.status === 501)) {
        setDeleteUnsupported(
          "this backend cannot delete episodes — drop the take by hand, or record over it",
        );
      } else {
        toast.error(`delete failed: ${(e as Error).message}`);
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex min-h-0 flex-col overflow-hidden rounded-lg bg-card shadow-[0_0_0_1px_var(--border)]">
      <div
        className="flex h-8.5 shrink-0 items-center justify-between gap-2 border-b border-border px-3"
        title={data?.root ?? undefined}
      >
        <span className="label-tracked shrink-0 text-muted-foreground">On disk</span>
        <span className="font-mono text-[10px] whitespace-nowrap text-muted-foreground">
          {episodes.length} ep · {data?.total_frames ?? "—"} frames ·{" "}
          {formatBytes(data?.size_bytes)}
        </span>
      </div>

      <div className="flex shrink-0 items-center gap-2 border-b border-border px-3 py-2">
        <span className="label-micro shrink-0 text-muted-foreground">Repo</span>
        <select
          aria-label="dataset repo"
          value={pick ?? ""}
          onChange={(e) => {
            setPick(e.target.value || null);
            // Dropped here rather than in the effect, so the list never shows
            // one repo's episodes under another repo's name while the new
            // read is in flight.
            setData(null);
            setArmed(false);
          }}
          className="h-7 min-w-0 flex-1 rounded-sm border border-input bg-background px-2 font-mono text-[10px]"
        >
          <option value="">current · {shown}</option>
          {(repos ?? []).map((r) => (
            <option key={r.repo_id} value={r.repo_id}>
              {r.repo_id} · {r.episodes} ep
              {typeof r.frames === "number" ? ` · ${r.frames} frames` : ""}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={refresh}
          aria-label="refresh episode list"
          className="h-7 shrink-0 rounded-sm border border-border bg-secondary px-2 label-micro"
        >
          ↻
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {error ? (
          <div className="p-3 font-mono text-[10px] text-pretty text-muted-foreground">
            {error}
          </div>
        ) : episodes.length === 0 ? (
          <div className="relative flex h-full min-h-[80px] items-center justify-center">
            <span className="scanlines absolute inset-0" aria-hidden />
            <span className="relative font-mono text-[10px] tracking-[0.14em] uppercase text-muted-foreground opacity-70">
              {data === null ? "reading…" : "no episodes yet"}
            </span>
          </div>
        ) : (
          <>
            <div className="sticky top-0 grid grid-cols-[38px_minmax(0,1fr)_50px_46px] gap-2 border-b border-border bg-muted px-2.5 py-1 label-micro text-muted-foreground">
              <span>ep</span>
              <span>task</span>
              <span className="text-right">frames</span>
              <span className="text-right">len</span>
            </div>
            {episodes.map((e) => (
              <div
                key={e.index}
                className="grid grid-cols-[38px_minmax(0,1fr)_50px_46px] gap-2 border-b border-border px-2.5 py-1 font-mono text-[10px] tabular-nums"
              >
                <span data-num>{e.index}</span>
                <span className="truncate text-muted-foreground" title={e.task ?? ""}>
                  {e.task ?? "—"}
                </span>
                <span className="text-right" data-num>{e.frames}</span>
                <span className="text-right text-muted-foreground" data-num>
                  {Number.isFinite(e.length_s) ? `${e.length_s.toFixed(1)}s` : "—"}
                </span>
              </div>
            ))}
          </>
        )}
      </div>

      <div className="flex shrink-0 flex-wrap items-center gap-2 border-t border-border px-3 py-2">
        {deleteUnsupported ? (
          <span className="text-[10px] text-pretty text-muted-foreground">
            {deleteUnsupported}
          </span>
        ) : armed ? (
          <>
            <button
              type="button"
              disabled={busy}
              onClick={doDelete}
              className="h-7 rounded-md border border-[var(--haller-fault)] bg-[oklch(0.62_0.245_27/0.2)] px-3 label-micro tracking-[0.12em] text-[var(--haller-fault)] disabled:opacity-50"
            >
              confirm · delete ep {last?.index}
            </button>
            <button
              type="button"
              onClick={() => setArmed(false)}
              className="h-7 rounded-md border border-border bg-secondary px-3 label-micro tracking-[0.12em]"
            >
              cancel
            </button>
          </>
        ) : (
          <button
            type="button"
            disabled={recording || last === null}
            title={
              recording
                ? "an episode is open — stop the take first"
                : last === null
                  ? "nothing recorded yet"
                  : undefined
            }
            onClick={() => setArmed(true)}
            className="h-7 rounded-md border border-border bg-secondary px-3 label-micro tracking-[0.12em] disabled:opacity-50"
          >
            delete last episode
          </button>
        )}
        {refusal && (
          <span className="min-w-0 flex-1 text-[10px] text-pretty text-[var(--haller-warn)]">
            refused: {refusal}
          </span>
        )}
      </div>
    </div>
  );
}
