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
 * thing this pane adds is `useRunList`, which refetches the LIST on the same
 * 2 s cadence, armed only while some row says `running`. The read's shape —
 * the tagged answer, the quiet poll, the in-flight guard — lives in the hook
 * now, because the Runs tab reads the same list.
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
  type DatasetSummary, type Run,
} from "@/lib/lab";
import { Button, Empty, Panel, PanelHead, Refusal } from "@/components/lab/ui";
import { useSticky } from "@/components/cockpit/lib";
import { TrainLauncher } from "@/components/lab/TrainLauncher";
import { RunFilters, DEFAULT_RUN_FILTERS, type RunFilterState } from "@/components/lab/RunFilters";
import { RunList, runLabel } from "@/components/lab/RunList";
import { PaneBoundary } from "@/components/lab/PaneBoundary";
import { RunDetail } from "@/components/lab/RunDetail";
import { useRunList } from "@/components/lab/useRunList";

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
  /* The launcher is a form you use occasionally; the run list is what you
     navigate with. Left permanently open it took 60vh of the rail and left the
     list showing two rows of four, so it folds away — sticky, because an
     operator who is rolling out rather than training should not have to fold
     it again after every glance at Review. */
  const [launcherOpen, setLauncherOpen] = useSticky<boolean>("lab.train.launcher.open", true);
  const [seededFrom, setSeededFrom] = useSticky<string | null>("lab.train.repo.seed", repoId);
  useEffect(() => {
    if (repoId && repoId !== seededFrom) {
      setSeededFrom(repoId);
      setPicked(repoId);
    }
  }, [repoId, seededFrom, setSeededFrom, setPicked]);

  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [dsError, setDsError] = useState<string | null>(null);
  /** 404/501 on the DATASET read: this build predates the Lab. The run list's
   *  own copy of the same fact comes back from the hook, and either one
   *  replaces the pane rather than reading as a failure. */
  const [dsNoLab, setDsNoLab] = useState(false);

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
        if (isMissing(e)) setDsNoLab(true);
        else setDsError(reason(e));
      }
    })();
    return () => { cancelled = true; };
  }, []);

  /* ── the run list ─────────────────────────────────────────────────────
     The tagged read, the quiet poll and the running-only cadence all live in
     the hook — see `useRunList`. */
  const {
    runs, error, loading, now, noLab: listNoLab, refetch,
  } = useRunList({ kind: filters.kind, status: filters.status });
  const noLab = dsNoLab || listNoLab;

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
    <div className="grid min-h-0 grid-cols-[26rem_minmax(0,1fr)] gap-2 overflow-hidden p-2">
      <div className="grid min-h-0 grid-rows-[auto_minmax(0,1fr)] gap-2 overflow-hidden">
        {/* The launcher's form is taller than a short viewport, and an auto
            grid row would let it push the run list to nothing. Capped so it
            scrolls inside its own Panel instead — and 40vh rather than the
            original 60, because at 60 the run list below it was two rows on
            an 800px-high window, which is not a list. */}
        <div className="flex max-h-[40vh] min-h-0 flex-col gap-1.5 overflow-hidden">
          {dsError && <Refusal>datasets could not be read: {dsError}</Refusal>}
          <div className="flex shrink-0 items-center justify-between gap-2">
            <Button
              tone="ghost"
              onClick={() => setLauncherOpen(!launcherOpen)}
              aria-expanded={launcherOpen}
              title={
                launcherOpen
                  ? "fold the launcher away and give the rail to the run list"
                  : "open the training launcher"
              }
            >
              {/* Not "new run" both ways: open, the panel below already wears
                  that title, and two of them read as two things. */}
              {launcherOpen ? "▾ hide" : "▸ new run"}
            </Button>
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
          {launcherOpen && (
            <TrainLauncher
              datasets={datasets}
              repoId={picked}
              onRepoId={setPicked}
              onLaunched={onLaunched}
            />
          )}
        </div>

        {/* Filters and the list are ONE panel: the filters only ever answer
            "which of these rows", and two cards with a gap between them read
            as two unrelated things. */}
        <Panel>
          <PanelHead title="runs" right={listRight} />
          <RunFilters value={filters} onChange={setFilters} />
          <RunList
            runs={shown}
            loading={loading}
            error={error}
            selectedId={selected}
            onSelect={setSelected}
            compare={compare}
            onToggleCompare={toggleCompare}
            now={now}
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
              // A rollout launched off one of this run's checkpoints is a NEW
              // run, and the same handover the launcher above already uses:
              // select it, refetch the list, say what the right column just
              // became. The checkpoint's own run keeps its place in the list.
              onLaunched={onLaunched}
            />
          </PaneBoundary>
        </div>
      </div>
    </div>
  );
}
