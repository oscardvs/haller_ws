"use client";

/**
 * One run list, read the one way the cockpit reads it.
 *
 * Extracted from `TrainPane` the day the Runs tab arrived: two panes now ask
 * "which jobs has the backend launched", and two copies of the poll is how one
 * of them ends up showing a finished run as still burning the GPU.
 *
 * The shape of the read is deliberate and carried over verbatim:
 *
 * - One read, TAGGED with what it answers (`key`). A poll that was in flight
 *   when the filter changed must not paint the old filter's rows over the new
 *   one's, and `loading` is derived from the tag rather than raised in the
 *   effect body — "this filter has no answer yet" is what the spinner means.
 * - `quiet` is the poll. A dropped request while a run is training must not
 *   blank a list that is still perfectly good, so a quiet failure is kept and
 *   retried on the next tick.
 * - The poll is armed only while some row says `running`. A list that never
 *   refreshes shows a finished run as live; one that always refreshes makes
 *   requests at 3am against a page nobody is reading.
 * - `now` rides along with each read so a running row's duration is measured
 *   against the moment its status was read, never a render-time clock.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  isMissing, lab, reason,
  type RunKind, type RunStatus, type RunSummary,
} from "@/lib/lab";

/** The same cadence `RunDetail` polls one run at. Matching it deliberately:
 *  a list that lags the detail view by a different interval shows a run as
 *  `running` in the row the operator is reading `done` in. */
export const LIST_POLL_MS = 2000;

export function useRunList({
  kind,
  status,
}: {
  kind: RunKind | null;
  status: RunStatus | null;
}): {
  runs: RunSummary[];
  error: string | null;
  loading: boolean;
  /** Epoch seconds of the last read — the clock running rows measure against. */
  now: number;
  /** 404/501 on `/lab/runs`: this build predates the Lab. */
  noLab: boolean;
  refetch: () => void;
} {
  const [read, setRead] = useState<{
    key: string;
    runs: RunSummary[];
    error: string | null;
    now: number;
  } | null>(null);
  const [noLab, setNoLab] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const refetch = useCallback(() => setRefreshKey((k) => k + 1), []);

  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);

  /** Bumped on every list read — the in-flight guard described above. */
  const gen = useRef(0);

  /** What one read answers: the SERVER-side filter plus the manual refresh
   *  generation. A read tagged with anything else would satisfy a filter it
   *  was not taken under. */
  const listKey = `${kind ?? "any"}|${status ?? "any"}|${refreshKey}`;

  const loadRuns = useCallback(
    async (quiet: boolean) => {
      const g = (gen.current += 1);
      try {
        const { runs: rows } = await lab.runs({ kind, status });
        if (!alive.current || g !== gen.current) return;
        setRead({ key: listKey, runs: rows, error: null, now: Date.now() / 1000 });
      } catch (e) {
        if (!alive.current || g !== gen.current) return;
        if (isMissing(e)) {
          setNoLab(true);
          setRead({ key: listKey, runs: [], error: null, now: Date.now() / 1000 });
        } else if (!quiet) {
          // The rows already on screen are kept: a failed refresh is not
          // evidence that the runs are gone.
          setRead((prev) => ({
            key: listKey,
            runs: prev?.runs ?? [],
            error: reason(e),
            now: prev?.now ?? 0,
          }));
        }
      }
    },
    [kind, status, listKey],
  );

  useEffect(() => {
    void loadRuns(false);
  }, [loadRuns]);

  const runs = useMemo(() => read?.runs ?? [], [read]);

  const anyRunning = runs.some((r) => r.status === "running");
  useEffect(() => {
    if (!anyRunning || noLab) return;
    const t = setInterval(() => { void loadRuns(true); }, LIST_POLL_MS);
    return () => clearInterval(t);
  }, [anyRunning, noLab, loadRuns]);

  return {
    runs,
    error: read?.error ?? null,
    loading: read === null || read.key !== listKey,
    now: read?.now ?? 0,
    noLab,
    refetch,
  };
}
