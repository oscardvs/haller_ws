"use client";

/**
 * Resuming a dataset instead of starting one.
 *
 * repo_id is picked here, not typed — and the pick PINS it. Recomposing
 * `haller_${slugify(task)}` from a resumed task forks the dataset it was
 * meant to extend (LeRobot keys tasks by string; the slug rule cannot
 * reproduce a repo it did not name), so the pick writes the dataset's own
 * task string into the shared recorder draft AND sets its `repoIdOverride`;
 * every record-start consumer — the recorder card, the Record popover, the
 * headset's A/X hold — then aims at the picked repo through
 * `effectiveRepoId`. "new dataset" clears both, so starting fresh stays one
 * gesture.
 *
 * The card derives rather than mirrors: the select shows whichever on-disk
 * repo the current draft resolves to, so editing the task in the recorder —
 * the deliberate way to leave a dataset; `setTask` drops the pin — visibly
 * flips the pick back to "new dataset" instead of letting the two disagree.
 */
import { useCallback, useEffect, useState } from "react";

import { api, type EpisodesResponse, type RepoInfo } from "@/lib/api";
import { effectiveRepoId, useRecorder } from "@/lib/recorder";
import { Button, Panel, PanelHead } from "@/components/lab/ui";

/** Select value for the new-dataset choice. */
const NEW = "";

/** What a pick can fall back to when the dataset has no episodes to read a
 *  task from: the repo name with the slug undone. */
function unslug(repoId: string): string {
  const name = repoId.slice(repoId.indexOf("/") + 1);
  return name.replace(/^haller_/, "").replace(/_/g, " ");
}

/** The episode the next take would extend — highest index, not last row:
 *  "newest" is a fact about the dataset, not about array order. */
function newestOf(r: EpisodesResponse) {
  return r.episodes.reduce<EpisodesResponse["episodes"][number] | null>(
    (best, e) => (best === null || e.index > best.index ? e : best),
    null,
  );
}

/** The rate the dataset was written at, off its newest episode. Rounded: a
 *  372-frame 12.4s take was written at 30 fps, not 30.0-vs-band trivia.
 *  Null when it cannot be known — a new dataset, an empty one, a zero-length
 *  episode — because a guessed rate is worse than none. */
export function datasetFpsOf(r: EpisodesResponse | null): number | null {
  const newest = r ? newestOf(r) : null;
  if (!newest || !Number.isFinite(newest.length_s) || newest.length_s <= 0) {
    return null;
  }
  return Math.round(newest.frames / newest.length_s);
}

export function CollectResumeCard({
  onDatasetFps,
}: {
  /** Lifts the picked dataset's write rate to the pane, so the session card
   *  can steer the session rate and the recorder can warn before a take is
   *  refused. Null for a new dataset or an unreadable one. */
  onDatasetFps?: (fps: number | null) => void;
}) {
  const task = useRecorder((s) => s.task);
  const hfUser = useRecorder((s) => s.hfUser);
  const repoIdOverride = useRecorder((s) => s.repoIdOverride);
  const setTask = useRecorder((s) => s.setTask);
  const setHfUser = useRecorder((s) => s.setHfUser);
  const setRepoIdOverride = useRecorder((s) => s.setRepoIdOverride);
  const recording = useRecorder((s) => s.status?.recording ?? false);

  const [repos, setRepos] = useState<RepoInfo[] | null>(null);
  /** The repo the operator last picked here, and the task string the pick
   *  wrote. Kept so a draft that STILL disagrees with its pin can be flagged
   *  rather than silently redirected — and so an operator edit afterwards is
   *  NOT flagged. */
  const [picked, setPicked] = useState<string | null>(null);
  const [pickedTask, setPickedTask] = useState<string | null>(null);
  const [detail, setDetail] = useState<EpisodesResponse | null>(null);
  const [epoch, setEpoch] = useState(0);
  const refresh = useCallback(() => setEpoch((n) => n + 1), []);

  // The repo the next take writes to: the pin when a dataset was picked, the
  // task/hfUser composition otherwise.
  const repoId = effectiveRepoId({ repoIdOverride, hfUser, task });
  const matched = repos?.some((r) => r.repo_id === repoId) ? repoId : null;
  const shown = matched ?? picked;

  // `recording` is a dependency so the take that just stopped re-reads the
  // disk: the episode it landed is the one the operator wants counted.
  useEffect(() => {
    let cancelled = false;
    api.recordRepos()
      .then((r) => { if (!cancelled) setRepos(r.repos); })
      .catch(() => { if (!cancelled) setRepos([]); });
    return () => { cancelled = true; };
  }, [epoch, recording]);

  useEffect(() => {
    if (!shown) { setDetail(null); return; }
    let cancelled = false;
    api.recordEpisodes(shown)
      .then((r) => { if (!cancelled) setDetail(r); })
      .catch(() => { if (!cancelled) setDetail(null); });
    return () => { cancelled = true; };
  }, [shown, epoch, recording]);

  // The fps follows the DETAIL, not the pick: a take that just stopped can
  // have changed which episode is newest, and a "new dataset" choice has no
  // detail at all — both land here as the same null.
  const datasetFps = shown ? datasetFpsOf(detail) : null;
  useEffect(() => {
    onDatasetFps?.(datasetFps);
  }, [datasetFps, onDatasetFps]);

  const pick = async (repo: string) => {
    const slash = repo.indexOf("/");
    const user = slash > 0 ? repo.slice(0, slash) : "";
    const fallback = unslug(repo);
    setHfUser(user);
    // Order is the store's rule: setTask CLEARS the override (a task edit is
    // the deliberate way to leave a dataset), so the task lands first and the
    // resume pin second. The pin is what makes START RECORDING append to the
    // picked repo instead of composing a new one from the task string.
    setTask(fallback);
    setRepoIdOverride(repo);
    setPicked(repo);
    setPickedTask(fallback);
    try {
      const r = await api.recordEpisodes(repo);
      const t = newestOf(r)?.task ?? fallback;
      // Correct the fallback to the dataset's own task string — unless the
      // operator has typed since the pick, in which case their edit wins.
      const cur = useRecorder.getState();
      if (t !== fallback && cur.task === fallback && cur.hfUser === user) {
        setTask(t);
        setRepoIdOverride(repo);
        setPickedTask(t);
      }
    } catch {
      /* no episode listing on this backend — the un-slugged name stands */
    }
  };

  const choose = (v: string) => {
    if (v === NEW) {
      setPicked(null);
      setPickedTask(null);
      setTask("");
      setHfUser("");
      setRepoIdOverride(null);
      return;
    }
    void pick(v);
  };

  // Derived, never stored: a draft that resolves to an on-disk repo shows
  // that repo; the pick shows while its task is still landing; anything else
  // is a new dataset — including the deliberate task edit that left one.
  const value =
    matched ??
    (picked !== null && task === pickedTask &&
    repos?.some((r) => r.repo_id === picked)
      ? picked
      : NEW);
  const drifted = picked !== null && task === pickedTask && repoId !== picked;
  const newest = detail ? newestOf(detail) : null;

  return (
    <Panel>
      <PanelHead
        title="dataset"
        right={repos === null ? "reading…" : `${repos.length} on disk`}
      >
        <Button
          tone="ghost"
          onClick={refresh}
          aria-label="refresh repo list"
          title="re-read /record/repos"
          className="ml-auto"
        >
          ↻
        </Button>
      </PanelHead>

      <div className="flex flex-col gap-2 p-3">
        {/* Frozen with the rest of the draft while a take is open — the
            recorder's own task/user inputs disable the same way. */}
        <select
          aria-label="resume dataset"
          disabled={recording}
          value={value}
          onChange={(e) => choose(e.target.value)}
          className="h-7.5 w-full rounded-md border border-input bg-background px-2.5 font-mono text-[11px] disabled:opacity-50"
        >
          <option value={NEW}>new dataset — clear the draft</option>
          {(repos ?? []).map((r) => (
            <option key={r.repo_id} value={r.repo_id}>
              {r.repo_id} · {r.episodes} ep
              {typeof r.frames === "number" ? ` · ${r.frames} frames` : ""}
            </option>
          ))}
        </select>

        {shown && detail && (
          <div className="flex flex-col gap-1 font-mono text-[10px]">
            {/* Read-only on purpose: the string the episodes already carry.
                Changing it is editing the recorder's task field, i.e. a new
                dataset — that path stays deliberate. */}
            <span className="truncate text-muted-foreground" title={newest?.task ?? undefined}>
              task on disk: <span className="text-foreground">{newest?.task ?? "—"}</span>
            </span>
            <span className="text-muted-foreground">
              next episode will be <span data-num className="text-foreground">#{(newest?.index ?? -1) + 1}</span>
            </span>
          </div>
        )}

        {drifted && (
          <div className="rounded-md border border-[var(--haller-warn)] px-2.5 py-2 font-mono text-[10px] text-pretty text-[var(--haller-warn)]">
            the draft writes to {repoId}, not the picked {picked} — the resume
            pin was dropped; pick again or start a new dataset
          </div>
        )}

        <p className="text-[11px] text-pretty text-muted-foreground">
          Picking resumes: the take lands in the picked repo, with the
          dataset&apos;s own task string in the recorder draft. Edit that task
          only when you mean a new dataset — LeRobot keys tasks by string, and
          a variant forks silently.
        </p>
      </div>
    </Panel>
  );
}
