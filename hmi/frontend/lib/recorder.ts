// hmi/frontend/lib/recorder.ts
"use client";

/**
 * Recorder status + the take's draft fields, in a store rather than in the
 * cockpit's root state.
 *
 * Two surfaces need this at once — the always-visible Record button in the
 * command bar and the Dataset tab — and it polls once a second. Held at the
 * cockpit root, that poll would re-render all six tabs' worth of tree every
 * second for a frame counter. Here, only the components that select a changed
 * field re-render, which is the same reason lib/telemetry.ts is a store.
 *
 * `task` and `hfUser` live here too so the compact Record popover and the full
 * Dataset panel edit one draft, not two that silently disagree.
 *
 * `repoIdOverride` is the resume half of that draft: picking an existing
 * dataset pins the take to THAT repo_id, because LeRobot keys tasks by string
 * and recomposing `haller_${slugify(task)}` from a resumed task would fork the
 * dataset it was meant to extend. The override is deliberately fragile —
 * `setTask` clears it, so any task edit (the deliberate way to leave a
 * dataset) drops the pin and the repo is composed again. Writers therefore
 * set the task FIRST and the override SECOND; every reader goes through
 * `effectiveRepoId`, never `repoIdFor` alone.
 */
import { create } from "zustand";

import { api, type RecordStatus } from "./api";

const POLL_MS = 1000;

// The take draft persists so a take can be started from inside the headset
// (VRTeleopPanel's A/X hold) with whatever the operator last typed into the
// cockpit, instead of forcing them out of VR to re-draft it.
const TASK_LS_KEY = "haller.recorder.task.v1";
const HFUSER_LS_KEY = "haller.recorder.hfUser.v1";
const REPOID_LS_KEY = "haller.recorder.repoId.v1";

/** Dataset repo slug — the one answer to what a composed take is called,
 *  whichever surface started it. */
export function slugify(s: string): string {
  return (
    s.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 60) ||
    "task"
  );
}

/** The composed repo id: what a take is called when no resume pin is set. */
export function repoIdFor(hfUser: string, task: string): string {
  return `${hfUser || "local"}/haller_${slugify(task)}`;
}

/** The repo the next take actually writes to: the resume pin when one is
 *  set, the composed id otherwise. The ONLY read path record-start consumers
 *  may use — a consumer that recomposes on its own splits the dataset. */
export function effectiveRepoId(draft: {
  repoIdOverride: string | null;
  hfUser: string;
  task: string;
}): string {
  return draft.repoIdOverride ?? repoIdFor(draft.hfUser, draft.task);
}

function readLS(key: string): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeLS(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* storage full / blocked — the draft just won't survive a reload */
  }
}

function removeLS(key: string): void {
  try {
    window.localStorage.removeItem(key);
  } catch {
    /* same as writeLS — a draft that cannot persist simply does not */
  }
}

type Store = {
  status: RecordStatus | null;
  busy: boolean;
  task: string;
  hfUser: string;
  /** The resume pin: an exact repo_id picked from disk, or null for "compose
   *  from task/hfUser". Cleared by `setTask` — see the module docstring. */
  repoIdOverride: string | null;
  setTask: (v: string) => void;
  setHfUser: (v: string) => void;
  setRepoIdOverride: (v: string | null) => void;
  refresh: () => Promise<void>;
  startPolling: () => () => void;
  start: (repoId: string, task: string) => Promise<RecordStatus>;
  stop: (save: boolean) => Promise<RecordStatus>;
};

let pollTimer: ReturnType<typeof setInterval> | null = null;
let subscribers = 0;

export const useRecorder = create<Store>((set, get) => ({
  status: null,
  busy: false,
  task: readLS(TASK_LS_KEY) ?? "Pick the red cube and place it in the box",
  hfUser: readLS(HFUSER_LS_KEY) ?? "",
  repoIdOverride: readLS(REPOID_LS_KEY),

  setTask: (v) => {
    // A task edit is the deliberate way to leave a resumed dataset, so it
    // drops the pin: the repo is composed from here on. This is why a resume
    // writes the task FIRST and the override SECOND.
    set({ task: v, repoIdOverride: null });
    writeLS(TASK_LS_KEY, v);
    removeLS(REPOID_LS_KEY);
  },
  setHfUser: (v) => {
    set({ hfUser: v });
    writeLS(HFUSER_LS_KEY, v);
  },
  setRepoIdOverride: (v) => {
    set({ repoIdOverride: v });
    if (v === null) removeLS(REPOID_LS_KEY);
    else writeLS(REPOID_LS_KEY, v);
  },

  refresh: async () => {
    try {
      const next = await api.recordStatus();
      const prev = get().status;
      // Bail out when nothing moved, so an idle recorder is not a 1 Hz
      // re-render source for the command bar.
      if (
        prev &&
        prev.recording === next.recording &&
        prev.episode_frames === next.episode_frames &&
        prev.skipped_frames === next.skipped_frames &&
        prev.repo_id === next.repo_id &&
        prev.last_error === next.last_error
      ) {
        return;
      }
      set({ status: next });
    } catch {
      /* backend not ready or recorder unavailable — keep the last known status
         rather than flapping the UI to "unknown" on one failed poll */
    }
  },

  /** Ref-counted: several components may mount wanting fresh status, and only
   *  the last one to leave should stop the timer. */
  startPolling: () => {
    subscribers += 1;
    void get().refresh();
    if (!pollTimer) {
      pollTimer = setInterval(() => void get().refresh(), POLL_MS);
    }
    return () => {
      subscribers -= 1;
      if (subscribers <= 0 && pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
        subscribers = 0;
      }
    };
  },

  start: async (repoId, task) => {
    set({ busy: true });
    try {
      const s = await api.recordStart(repoId, task);
      set({ status: s });
      return s;
    } finally {
      set({ busy: false });
    }
  },

  stop: async (save) => {
    set({ busy: true });
    try {
      const s = await api.recordStop(save);
      set({ status: s });
      return s;
    } finally {
      set({ busy: false });
    }
  },
}));
