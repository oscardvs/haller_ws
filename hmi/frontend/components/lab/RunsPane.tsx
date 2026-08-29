"use client";

/**
 * Runs — trained models read side by side, inside the cockpit.
 *
 * The comparison itself is `ComparePane`, unchanged: `/lab/compare?runs=a,b,c`
 * stays the shareable, bookmarkable form of the same reading, and the ↗ link
 * in the strip builds exactly that url from the ticked set. What this pane
 * adds is the picking: the run list on the left is the SAME list the Train
 * tab shows, and the ticked set is the SAME sticky set, so two runs ticked
 * while watching them train are already overlaid when the operator lands here.
 *
 * The whole ROW is the toggle, not just the checkbox. On the Train tab a row
 * click answers "show me this run" and the tick is the secondary act; here
 * there is no detail column and adding-to-the-overlay is the only thing a
 * click can mean. `selectedId` stays null for the same reason — this pane has
 * no notion of "the" run.
 *
 * The right column scrolls. `ComparePane` is the one surface in the cockpit
 * allowed to, because the chart count is set by whatever the trainer logged,
 * and clipping a metric because it landed in row four would hide exactly the
 * run that failed.
 */
import { useCallback, useMemo, useState } from "react";
import Link from "next/link";

import { Button, Empty, Panel, PanelHead } from "@/components/lab/ui";
import { useSticky } from "@/components/cockpit/lib";
import { ComparePane } from "@/components/lab/ComparePane";
import { PaneBoundary } from "@/components/lab/PaneBoundary";
import { RunFilters, DEFAULT_RUN_FILTERS, type RunFilterState } from "@/components/lab/RunFilters";
import { RunList } from "@/components/lab/RunList";
import { useRunList } from "@/components/lab/useRunList";

export function RunsPane() {
  /* Sticky, not persisted — same contract as every Lab pane. The filters are
     this pane's own (`lab.runs.*`): Train filters "what am I watching", this
     filters "what am I comparing", and one following the other around would
     make a chip clicked over there quietly empty the list over here. The
     COMPARE SET is deliberately shared with Train (`lab.train.compare`) —
     the tick means "overlay this run" on both surfaces. */
  const [filters, setFilters] = useSticky<RunFilterState>("lab.runs.filters", DEFAULT_RUN_FILTERS);
  const [compare, setCompare] = useSticky<Set<string>>("lab.train.compare", new Set<string>());

  const { runs, error, loading, now, noLab, refetch } = useRunList({
    kind: filters.kind,
    status: filters.status,
  });

  /** Remounts `ComparePane`, whose read is keyed on the id set: while a run is
   *  still training its curves grow under a comparison that was read once, and
   *  a button beats a poll on a pane that is mostly read after the fact. */
  const [curveGen, setCurveGen] = useState(0);

  /** The only browser-side filter — name and id, which is exactly what the
   *  box says it searches. Same seam as the Train tab. */
  const shown = useMemo(() => {
    const q = filters.q.trim().toLowerCase();
    if (!q) return runs;
    return runs.filter(
      (r) =>
        r.id.toLowerCase().includes(q) ||
        (r.name ?? "").toLowerCase().includes(q),
    );
  }, [runs, filters.q]);

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
          <PanelHead title="runs" />
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
      <Panel>
        <PanelHead title="runs" right={listRight}>
          <Button tone="ghost" onClick={refetch} title="re-read the run list">
            refresh
          </Button>
        </PanelHead>
        <RunFilters value={filters} onChange={setFilters} />
        <RunList
          runs={shown}
          loading={loading}
          error={error}
          selectedId={null}
          onSelect={toggleCompare}
          compare={compare}
          onToggleCompare={toggleCompare}
          now={now}
        />
      </Panel>

      <div className="grid min-h-0 grid-rows-[auto_minmax(0,1fr)] gap-2 overflow-hidden">
        <div className="flex h-8.5 shrink-0 items-center gap-2 overflow-hidden rounded-lg bg-card px-3 shadow-[0_0_0_1px_var(--border)]">
          <span className="label-tracked shrink-0 text-muted-foreground">compare</span>
          <span className="min-w-0 flex-1 truncate font-mono text-[10px] text-muted-foreground">
            <span data-num className="tabular-nums text-foreground">{compare.size}</span>
            {" ticked"}
            {compare.size === 1 && " — one run is a reading, two is a comparison"}
          </span>
          {compare.size > 0 && (
            <>
              <Button
                tone="ghost"
                onClick={() => setCurveGen((g) => g + 1)}
                title="re-read the curves — a run still training has grown since the last read"
              >
                re-read curves
              </Button>
              <Button
                tone="ghost"
                onClick={() => setCompare(new Set<string>())}
                title="untick every run"
              >
                clear
              </Button>
              <Link
                href={`/lab/compare?runs=${compareIds.map(encodeURIComponent).join(",")}`}
                target="_blank"
                rel="noreferrer"
                title="the url is the whole state — bookmark it, paste it into a note"
                className="label-micro shrink-0 text-muted-foreground transition-colors hover:text-[var(--haller-live)]"
              >
                open as link ↗
              </Link>
            </>
          )}
        </div>

        {compare.size === 0 ? (
          <Panel>
            <Empty>
              tick a run on the left to read it here — two or more to compare
            </Empty>
          </Panel>
        ) : (
          <div className="min-h-0 overflow-y-auto">
            <PaneBoundary what="the run comparison">
              <ComparePane key={`${curveGen}`} runIds={compareIds} />
            </PaneBoundary>
          </div>
        )}
      </div>
    </div>
  );
}
