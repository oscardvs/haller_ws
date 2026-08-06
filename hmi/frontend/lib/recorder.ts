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
 */
import { create } from "zustand";

import { api, type RecordStatus } from "./api";

const POLL_MS = 1000;

type Store = {
  status: RecordStatus | null;
  busy: boolean;
  task: string;
  hfUser: string;
  setTask: (v: string) => void;
  setHfUser: (v: string) => void;
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
  task: "Pick the red cube and place it in the box",
  hfUser: "",

  setTask: (v) => set({ task: v }),
  setHfUser: (v) => set({ hfUser: v }),

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
