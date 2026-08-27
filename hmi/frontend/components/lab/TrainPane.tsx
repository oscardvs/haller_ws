"use client";

/**
 * Train: launch a run, and watch the ones already going.
 *
 * The layout is a lab notebook, not a dashboard — the launcher and the run
 * list share the left rail because "what am I about to run" and "what did I
 * already run" are the same question asked twice, and the right column is one
 * run in full.
 *
 * TWO timers would be one too many. `RunDetail` owns the per-run poll (metrics
 * and log by byte offset, armed only while that run is running); the only
 * thing this pane adds is a refetch of the LIST on the same 2 s cadence, armed
 * only while some row says `running`. A list that never refreshes shows a
 * finished run as still burning the GPU; a list that always refreshes makes
 * four requests a second at 3am against a page nobody is reading.
 *
 * `kind` and `status` are filtered by the SERVER — `GET /lab/runs` takes both —
 * and only the free-text box is filtered here, because it is a substring match
 * over rows that have already arrived.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { toast } from "sonner";

import {
  isMissing, lab, reason,
  type DatasetSummary, type Run, type RunSummary,
} from "@/lib/lab";
import { Button, Empty, Panel, PanelHead, Refusal } from "@/components/lab/ui";
import { useSticky } from "@/components/cockpit/lib";
import { TrainLauncher } from "@/components/lab/TrainLauncher";
import { RunFilters, DEFAULT_RUN_FILTERS, type RunFilterState } from "@/components/lab/RunFilters";
import { RunList, runLabel } from "@/components/lab/RunList";
import { PaneBoundary } from "@/components/lab/PaneBoundary";
import { RunDetail } from "@/components/lab/RunDetail";

/** The same cadence `RunDetail` polls one run at. Matching it deliberately:
 *  a list that lags the detail view by a different interval shows a run as
 *  `running` in the row the operator is reading `done` in. */
const LIST_POLL_MS = 2000;

export function TrainPane({
  repoId,
  onOpenDataset,
}: {
  repoId: string | null;
  onOpenDataset: (repoId: string) => void;
}) {
  /* Sticky, not persisted. This pane unmounts every time the operator glances
     at Review or at Cameras, and coming back to a cleared compare set and a
     deselected run is the small betrayal that makes a surface untrustworthy.
     A reload is still a fresh session. */
  const [filters, setFilters] = useSticky<RunFilterState>("lab.train.filters", DEFAULT_RUN_FILTERS);
  const [selected, setSelected] = useSticky<string | null>("lab.train.run", null);
  const [compare, setCompare] = useSticky<Set<string>>("lab.train.compare", new Set<string>());

  /* The launcher's dataset. Seeded from the pane's `repoId` so opening a
     dataset from the shelf and switching to train lands on that one, but the
     honoured value is REMEMBERED rather than re-applied on every mount — a
     pick made in the launcher must survive a tab switch. */
  const [picked, setPicked] = useSticky<string | null>("lab.train.repo", repoId);
  const [seededFrom, setSeededFrom] = useSticky<string | null>("lab.train.repo.seed", repoId);
  useEffect(() => {
    if (repoId && repoId !== seededFrom) {
      setSeededFrom(repoId);
      setPicked(repoId);
    }
  }, [repoId, seededFrom, setSeededFrom, setPicked]);

  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [dsError, setDsError] = useState<string | null>(null);
  /** One list read, TAGGED with what it answers. `loading` is derived from the
   *  tag rather than raised at the top of the effect: "this filter has no
   *  answer yet" is what the spinner means, and raising it in the effect body
   *  is a cascading render on every read. */
  const [read, setRead] = useState<{
    key: string;
    runs: RunSummary[];
    error: string | null;
  } | null>(null);
  /** 404/501 anywhere in `/lab`: this build predates the Lab. A property of
   *  the backend, so it replaces the pane rather than reading as a failure. */
  const [noLab, setNoLab] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const refetch = useCallback(() => setRefreshKey((k) => k + 1), []);

  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);

  /** Bumped on every list read. A poll that was in flight when the filter
   *  changed must not paint the old filter's rows over the new one's. */
  const gen = useRef(0);

  /* ── the dataset list, once ───────────────────────────────────────────
     Handed to the launcher untouched: which datasets are trainable and how
     they are worded is the launcher's decision, not this pane's. */
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const { datasets: ds } = await lab.datasets();
        if (cancelled) return;
        setDatasets(ds);
        setDsError(null);
      } catch (e) {
        if (cancelled) return;
        if (isMissing(e)) setNoLab(true);
        else setDsError(reason(e));
      }
    })();
    return () => { cancelled = true; };
  }, []);

  /* ── the run list ─────────────────────────────────────────────────────
     `quiet` is the poll. A dropped request while a run is training must not
     blank a list that is still perfectly good, so a quiet failure is kept and
     retried on the next tick. */
  /** What one list read answers: the SERVER-side filter, plus the manual
   *  refresh generation. A read tagged with anything else would satisfy a
   *  filter it was not taken under. */
  const listKey = `${filters.kind ?? "any"}|${filters.status ?? "any"}|${refreshKey}`;

  const loadRuns = useCallback(
    async (quiet: boolean) => {
      const g = (gen.current += 1);
      try {
        const { runs: rows } = await lab.runs({ kind: filters.kind, status: filters.status });
        if (!alive.current || g !== gen.current) return;
        setRead({ key: listKey, runs: rows, error: null });
      } catch (e) {
        if (!alive.current || g !== gen.current) return;
        if (isMissing(e)) {
          setNoLab(true);
          setRead({ key: listKey, runs: [], error: null });
        } else if (!quiet) {
          // The rows already on screen are kept: a failed refresh is not
          // evidence that the runs are gone.
          setRead((prev) => ({ key: listKey, runs: prev?.runs ?? [], error: reason(e) }));
        }
      }
    },
    [filters.kind, filters.status, listKey],
  );

  useEffect(() => {
    void loadRuns(false);
  }, [loadRuns]);

  const runs = useMemo(() => read?.runs ?? [], [read]);
  const error = read?.error ?? null;
  const loading = read === null || read.key !== listKey;

  const anyRunning = runs.some((r) => r.status === "running");
  useEffect(() => {
    if (!anyRunning || noLab) return;
    const t = setInterval(() => { void loadRuns(true); }, LIST_POLL_MS);
    return () => clearInterval(t);
  }, [anyRunning, noLab, loadRuns]);

  /** The only browser-side filter. Name and id, which is exactly what the box
   *  says it searches. */
  const shown = useMemo(() => {
    const q = filters.q.trim().toLowerCase();
    if (!q) return runs;
    return runs.filter(
      (r) =>
        r.id.toLowerCase().includes(q) ||
        (r.name ?? "").toLowerCase().includes(q),
    );
  }, [runs, filters.q]);

  /* ── wiring ───────────────────────────────────────────────────────────── */

  /** The list holds its own copy of the row `RunDetail` is polling; a status
   *  it has not seen is the one thing it cannot work out for itself. The first
   *  read of a row we already agree with is not a change and buys no request. */
  const runsRef = useRef(runs);
  useEffect(() => { runsRef.current = runs; });
  const onRunChanged = useCallback(
    (r: Run) => {
      const row = runsRef.current.find((x) => x.id === r.id);
      if (row && row.status === r.status) return;
      refetch();
    },
    [refetch],
  );

  const onLaunched = useCallback(
    (run: Run) => {
      setSelected(run.id);
      refetch();
      // The launcher already said "queued". This one names what the right
      // column just switched to, which is the fact this pane owns.
      toast.success(`watching ${runLabel(run)}`);
    },
    [refetch, setSelected],
  );

  const onDeleted = useCallback(
    (id: string) => {
      if (selected === id) setSelected(null);
      if (compare.has(id)) {
        const next = new Set(compare);
        next.delete(id);
        setCompare(next);
      }
      refetch();
    },
    [compare, refetch, selected, setCompare, setSelected],
  );

  const toggleCompare = useCallback(
    (id: string) => {
      const next = new Set(compare);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      setCompare(next);
    },
    [compare, setCompare],
  );

  if (noLab) {
    return (
      <div className="grid min-h-0 overflow-hidden p-2">
        <Panel>
          <PanelHead title="train" />
          <Empty>this backend has no lab</Empty>
        </Panel>
      </div>
    );
  }

  const compareIds = [...compare];
  const listRight = filters.q.trim()
    ? `${shown.length} of ${runs.length} match`
    : `${runs.length} run${runs.length === 1 ? "" : "s"}`;

  return (
    <div className="grid min-h-0 grid-cols-[24rem_minmax(0,1fr)] gap-2 overflow-hidden p-2">
      <div className="grid min-h-0 grid-rows-[auto_auto_minmax(0,1fr)] gap-2 overflow-hidden">
        {/* The launcher's form is taller than a short viewport, and an auto
            grid row would let it push the run list to nothing. Capped here so
            it scrolls inside its own Panel instead — 60vh leaves the filters
            and four or five run rows visible on the 720px-high case the
            cockpit already calls `short`. */}
        <div className="flex max-h-[60vh] min-h-0 flex-col gap-1.5 overflow-hidden">
          {dsError && <Refusal>datasets could not be read: {dsError}</Refusal>}
          <div className="flex shrink-0 items-center justify-end">
            <Button
              disabled={!picked}
              onClick={() => { if (picked) onOpenDataset(picked); }}
              title={
                picked
                  ? `open ${picked} under review`
                  : "pick a dataset below first"
              }
            >
              review this dataset →
            </Button>
          </div>
          <TrainLauncher
            datasets={datasets}
            repoId={picked}
            onRepoId={setPicked}
            onLaunched={onLaunched}
          />
        </div>

        <Panel className="shrink-0">
          <RunFilters value={filters} onChange={setFilters} />
        </Panel>

        <Panel>
          <PanelHead title="runs" right={listRight} />
          <RunList
            runs={shown}
            loading={loading}
            error={error}
            selectedId={selected}
            onSelect={setSelected}
            compare={compare}
            onToggleCompare={toggleCompare}
          />
        </Panel>
      </div>

      <div className="grid min-h-0 grid-rows-[auto_minmax(0,1fr)] gap-2 overflow-hidden">
        {/* Compare is a deep link, not a mode: the ticked set is the whole
            state of `/lab/compare`, so it opens in its own tab and the url is
            the thing worth keeping. One run overlaid on nothing is just the
            run, which is why the link needs two. */}
        <div className="flex h-8.5 shrink-0 items-center gap-2 overflow-hidden rounded-lg bg-card px-3 shadow-[0_0_0_1px_var(--border)]">
          <span className="label-tracked shrink-0 text-muted-foreground">compare</span>
          <span className="min-w-0 flex-1 truncate font-mono text-[10px] text-muted-foreground">
            <span data-num className="tabular-nums text-foreground">{compare.size}</span>
            {" selected for compare"}
            {compare.size < 2 && " — needs at least two"}
          </span>
          {compare.size > 0 && (
            <Button
              tone="ghost"
              onClick={() => setCompare(new Set<string>())}
              title="untick every run"
            >
              clear
            </Button>
          )}
          {compare.size >= 2 && (
            <Link
              href={`/lab/compare?runs=${compareIds.map(encodeURIComponent).join(",")}`}
              target="_blank"
              rel="noreferrer"
              className="label-micro shrink-0 text-muted-foreground transition-colors hover:text-[var(--haller-live)]"
            >
              open compare ↗
            </Link>
          )}
        </div>

        <div className="flex min-h-0 flex-col overflow-hidden">
          {/* Keyed: every buffer in there is per-run — two byte offsets, the
              metric rows, the log text, the armed delete — and a remount
              resets the whole set at once rather than a list of resets that
              has to be kept in step with the state above it. */}
          <PaneBoundary what="the run detail">
            <RunDetail
              key={selected ?? "none"}
              runId={selected}
              onChanged={onRunChanged}
              onDeleted={onDeleted}
            />
          </PaneBoundary>
        </div>
      </div>
    </div>
  );
}
