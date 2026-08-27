"use client";

/**
 * Several training runs read side by side.
 *
 * The whole page hangs off the URL — `/lab/compare?runs=a,b,c` — because the
 * question it answers ("did the larger batch actually help?") outlives the
 * session that asked it. A comparison you can paste into a note is a
 * comparison you still have next week.
 *
 * Only the metrics EVERY run logged are charted. A key one run has and the
 * others do not is not a comparison, it is one curve with an empty grid cell
 * beside it, and silently charting it invites the reader to compare a line
 * against nothing. The dropped keys are named rather than hidden — that a run
 * logged something the others did not is itself worth knowing.
 *
 * This is the one surface in the cockpit allowed to scroll: it is a deep link,
 * not a control surface, nothing here stops the robot, and the number of
 * charts is set by the trainer's log rather than by a layout.
 */
import Link from "next/link";
import { Fragment, useEffect, useMemo, useState } from "react";

import { CompareChart } from "@/components/lab/charts/CompareChart";
import { fmtNum, seriesColor } from "@/components/lab/charts/svg";
import { HparamDiff } from "@/components/lab/HparamDiff";
import { PaneBoundary } from "@/components/lab/PaneBoundary";
import { Empty, Note, Panel, PanelHead, Refusal, Segmented } from "@/components/lab/ui";
import {
  isBusy, isMissing, lab, metricKeys, reason,
  type CompareMetrics, type Run, type RunStatus,
} from "@/lib/lab";

/** What the server downsamples to. A 200k-step run has 200k rows and the
 *  widest chart in the grid is ~600 CSS px, so anything past this is points
 *  drawn on top of each other. */
const MAX_POINTS = 600;

/** The x axis `/lab/runs/metrics` emits: `metrics.jsonl` is one object per
 *  logged step, and the server pairs each value with that step. */
const X_LABEL = "step";

/** A run's status as a colour.
 *
 *  `ui.tsx` carries a shared vocabulary for marks and verdicts but not for run
 *  status, so the mapping is stated here against the same status tokens the
 *  rest of the cockpit uses: green for a run that is alive or finished clean,
 *  amber for one a human stopped, red for one that ended by itself in a way
 *  nobody asked for. `died` is red and not amber on purpose — a dead pid with
 *  no result is not a clean finish. */
const STATUS_COLOR: Record<RunStatus, string> = {
  queued: "var(--haller-rail)",
  running: "var(--haller-live)",
  done: "var(--haller-live)",
  stopped: "var(--haller-warn)",
  failed: "var(--haller-fault)",
  died: "var(--haller-fault)",
  launch_failed: "var(--haller-fault)",
};

type Loaded = {
  runs: Run[];
  /** Ids the backend does not have — a deleted run in a shared URL. */
  gone: string[];
  /** Runs that have logged no metric yet. They keep their colour and their
   *  legend row; they just have nothing to draw. */
  silent: string[];
  /** Keys every run that has logged anything logged. */
  shared: string[];
  /** Keys only some of them logged, named so the omission is not a mystery. */
  dropped: string[];
  series: CompareMetrics["runs"];
  /** The backend has runs but no cross-run metrics endpoint. */
  seriesMissing: boolean;
  /** The metrics request was REFUSED (not absent) — the backend's own sentence.
   *  Kept inside the loaded state on purpose: a refusal here costs the CURVES,
   *  never the page. A real ACT run logs 12 numeric keys against a cap of 8, so
   *  this is the ordinary path for the only kind of run worth comparing, and
   *  throwing it to the outer catch blanked the run list, the legend and the
   *  hparam diff over a chart nobody could draw. */
  seriesRefusal: string | null;
};

export function ComparePane({ runIds }: { runIds: string[] }) {
  // The parent rebuilds `runIds` from the query string on every render, so the
  // effect keys off the joined string rather than the array identity.
  const idKey = runIds.join(",");
  const ids = useMemo(() => idKey.split(",").filter(Boolean), [idKey]);

  /** One read of one url, TAGGED with the ids it answers — the loaded state or
   *  the way it failed. Tagged rather than blanked at the top of the effect:
   *  blanking there is a setState in the effect body, and a result that
   *  outlives its url labels one set of runs with another's numbers. */
  const [read, setRead] = useState<{
    key: string;
    loaded: Loaded | null;
    error: string | null;
    refusal: string | null;
    noLab: boolean;
  } | null>(null);
  const [log, setLog] = useState(true);
  /** Default ON, unlike the metric grid. Three overlaid noisy series is a
   *  hairball; on this page the reader is after the SHAPE of the divergence,
   *  not the value at a step. The raw series stays drawn underneath. */
  const [smoothing, setSmoothing] = useState(0.6);

  const answer = read !== null && read.key === idKey ? read : null;
  const state = answer?.loaded ?? null;
  const error = answer?.error ?? null;
  const refusal = answer?.refusal ?? null;
  const noLab = answer?.noLab ?? false;
  const loading = ids.length > 0 && answer === null;

  useEffect(() => {
    if (ids.length === 0) return;
    let cancelled = false;
    const settle = (r: {
      loaded?: Loaded;
      error?: string;
      refusal?: string;
      noLab?: boolean;
    }) => {
      if (cancelled) return;
      setRead({
        key: idKey,
        loaded: r.loaded ?? null,
        error: r.error ?? null,
        refusal: r.refusal ?? null,
        noLab: r.noLab ?? false,
      });
    };

    (async () => {
      try {
        const records = await Promise.all(ids.map(runOrNull));
        if (cancelled) return;
        const runs = records.filter((r): r is Run => r !== null);
        const gone = ids.filter((_, i) => records[i] === null);
        if (runs.length === 0) {
          settle({ noLab: true });
          return;
        }

        // One page of each run's metrics is enough to learn its key set: the
        // trainer logs the same keys every step, and reading the whole stream
        // to find that out would pull megabytes to build a header row.
        const keySets = await Promise.all(runs.map(keysOrNone));
        if (cancelled) return;

        const union: string[] = [];
        for (const keys of keySets) {
          for (const k of keys) if (!union.includes(k)) union.push(k);
        }
        const logged = keySets.filter((k) => k.length > 0).map((k) => new Set(k));
        const shared = logged.length > 0
          ? union.filter((k) => logged.every((s) => s.has(k)))
          : [];
        const dropped = union.filter((k) => !shared.includes(k));
        const silent = runs.filter((_, i) => keySets[i].length === 0).map((r) => r.id);
        const speaking = runs.filter((_, i) => keySets[i].length > 0).map((r) => r.id);

        let series: CompareMetrics["runs"] = {};
        let seriesMissing = false;
        let seriesRefusal: string | null = null;
        if (shared.length > 0 && speaking.length > 0) {
          try {
            const res = await lab.compareMetrics(speaking, shared, MAX_POINTS);
            series = res.runs ?? {};
          } catch (e) {
            if (isMissing(e)) seriesMissing = true;
            else seriesRefusal = reason(e);
          }
        }
        if (cancelled) return;
        settle({
          loaded: {
            runs, gone, silent, shared, dropped, series, seriesMissing, seriesRefusal,
          },
        });
      } catch (e) {
        if (cancelled) return;
        if (isBusy(e)) settle({ refusal: reason(e) });
        else if (isMissing(e)) settle({ noLab: true });
        else settle({ error: reason(e) });
      }
    })();

    return () => { cancelled = true; };
  }, [ids, idKey]);

  const scaleMode: "log" | "linear" = log ? "log" : "linear";
  // Memoised, not `??`-defaulted inline: a fresh [] per render is a new
  // identity, and `summary` below is a useMemo keyed on both of them.
  const runs = useMemo(() => state?.runs ?? [], [state]);
  const shared = useMemo(() => state?.shared ?? [], [state]);
  // An empty bordered strip reads as a panel that failed to load something.
  const hasNotes = Boolean(
    refusal || error || state?.seriesMissing || state?.seriesRefusal ||
    (state?.gone.length ?? 0) > 0 || (state?.silent.length ?? 0) > 0 ||
    (state?.dropped.length ?? 0) > 0 || runs.length === 1,
  );
  const summary = useMemo(
    () => shared.map((key) => {
      const lower = lowerIsBetter(key);
      const cells = runs.map((r) => summarise(state?.series[r.id]?.[key], lower));
      return {
        key,
        lower,
        cells,
        bestWin: extremeOf(cells.map((c) => c.best), lower),
        finalWin: extremeOf(cells.map((c) => c.final), lower),
      };
    }),
    [shared, runs, state],
  );

  if (ids.length === 0) {
    return (
      <Panel>
        <PanelHead title="Compare runs" />
        <div className="p-3">
          <Note>
            No runs in the url. Tick the compare box on two runs in the Train
            tab&apos;s run list, then open the link it builds.
          </Note>
        </div>
      </Panel>
    );
  }

  if (noLab) {
    return (
      <Panel>
        <PanelHead title="Compare runs" right={`${ids.length} asked for`} />
        <div className="p-3">
          <Note>
            this backend has no lab, or these runs are gone —{" "}
            <span className="font-mono">{ids.join(", ")}</span>
          </Note>
        </div>
      </Panel>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <Panel>
        <PanelHead
          title="Runs"
          right={`${runs.length}/${ids.length} · ${shared.length} shared metric${shared.length === 1 ? "" : "s"}`}
        />
        {loading && runs.length === 0 ? (
          <Empty>reading runs…</Empty>
        ) : (
          <div className="flex flex-col">
            {runs.map((r, i) => (
              <RunLegendRow key={r.id} run={r} colour={seriesColor(i)} />
            ))}
          </div>
        )}
        {hasNotes && (
          <div className="flex flex-col gap-1.5 border-t border-border px-3 py-2.5">
            {refusal && <Refusal>refused: {refusal}</Refusal>}
            {error && <Refusal tone="fault">compare failed: {error}</Refusal>}
            {state && state.gone.length > 0 && (
              <Note>
                gone from the runs dir, so not drawn:{" "}
                <span className="font-mono">{state.gone.join(", ")}</span>
              </Note>
            )}
            {runs.length === 1 && (
              <Note>
                One run. Everything below is drawn, but a comparison needs at
                least two — add another id to <span className="font-mono">runs=</span>.
              </Note>
            )}
            {state && state.silent.length > 0 && (
              <Note>
                no metrics logged yet, so out of the shared set:{" "}
                <span className="font-mono">{state.silent.join(", ")}</span>
              </Note>
            )}
            {state && state.dropped.length > 0 && (
              <Note>
                logged by some runs and not others, so not compared:{" "}
                <span className="font-mono">{state.dropped.join(", ")}</span>
              </Note>
            )}
            {state?.seriesMissing && (
              <Note>this backend cannot serve cross-run metrics — the curves are unavailable.</Note>
            )}
            {state?.seriesRefusal && (
              <Refusal>curves refused: {state.seriesRefusal}</Refusal>
            )}
          </div>
        )}
      </Panel>

      {shared.length > 0 && (
        <>
          <div className="flex items-center gap-2.5 px-1">
            <span className="label-micro text-muted-foreground">y scale</span>
            <Segmented
              label="y scale"
              className="w-[13rem]"
              value={scaleMode}
              onChange={(v) => setLog(v === "log")}
              options={[
                { value: "log", label: "log", hint: "ACT loss spans two decades — a linear axis flattens everything after the first fifty steps" },
                { value: "linear", label: "linear", hint: "raw values, evenly spaced" },
              ]}
            />
            <span className="label-micro ml-2 text-muted-foreground">smoothing</span>
            <input
              type="range"
              min={0}
              max={0.95}
              step={0.05}
              value={smoothing}
              aria-label="smoothing"
              title="the raw series stays drawn underneath — smoothing never hides a spike"
              onChange={(e) => setSmoothing(Number(e.target.value))}
              className="haller-range w-[9rem]"
              style={{ ["--pct" as string]: `${(smoothing / 0.95) * 100}%` }}
            />
            <span data-num className="font-mono text-[10px] tabular-nums text-muted-foreground">
              {smoothing.toFixed(2)}
            </span>
          </div>

          <div className="grid gap-3 [grid-template-columns:repeat(auto-fill,minmax(26rem,1fr))]">
            {shared.map((key) => (
              <PaneBoundary key={key} what={`the ${key} chart`}>
              <CompareChart
                key={key}
                metricKey={key}
                log={log}
                smoothing={smoothing}
                xLabel={X_LABEL}
                series={runs.map((r, i) => ({
                  id: r.id,
                  label: r.name ?? r.id,
                  color: seriesColor(i),
                  points: state?.series[r.id]?.[key] ?? [],
                }))}
              />
              </PaneBoundary>
            ))}
          </div>

          <Panel>
            <PanelHead
              title="Best / final"
              right={`${runs.length} run${runs.length === 1 ? "" : "s"} · ${shared.length} metric${shared.length === 1 ? "" : "s"}`}
            />
            {/* The table reads the RAW downsampled points, never the smoothed
                ones. A smoothed minimum is a value the run never reached, and
                "best loss 0.071" has to be a number that actually happened —
                the smoothing slider is for reading shape, not for moving the
                figures underneath it. */}
            <div className="overflow-x-auto">
              <table className="w-full border-collapse font-mono text-[10px]">
                <thead>
                  <tr className="border-b border-border label-micro text-muted-foreground">
                    <th scope="col" rowSpan={2} className="px-2.5 py-1 text-left font-semibold align-bottom">
                      run
                    </th>
                    {summary.map((s) => (
                      <th
                        key={s.key}
                        scope="colgroup"
                        colSpan={2}
                        className="border-l border-border px-2.5 py-1 text-left font-semibold"
                      >
                        {s.key}
                      </th>
                    ))}
                  </tr>
                  <tr className="border-b border-border label-micro text-muted-foreground">
                    {summary.map((s) => (
                      <Fragment key={s.key}>
                        <th
                          scope="col"
                          className="border-l border-border px-2.5 py-1 text-right font-semibold"
                          title={s.lower ? "min is better" : "max is better"}
                        >
                          best {s.lower ? "↓" : "↑"}
                        </th>
                        <th scope="col" className="px-2.5 py-1 text-right font-semibold">
                          final
                        </th>
                      </Fragment>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {runs.map((r, i) => (
                    <tr key={r.id} className="border-b border-border/60 last:border-0">
                      <th scope="row" className="max-w-[18rem] px-2.5 py-1 text-left font-normal" title={r.id}>
                        <span className="flex items-center gap-1.5">
                          <span
                            aria-hidden
                            className="h-1.5 w-1.5 shrink-0 rounded-[1px]"
                            style={{ backgroundColor: seriesColor(i) }}
                          />
                          <span className="truncate">{r.name ?? r.id}</span>
                        </span>
                      </th>
                      {summary.map((s) => {
                        const cell = s.cells[i];
                        return (
                          <Fragment key={s.key}>
                            <Num
                              v={cell.best}
                              win={cell.best !== null && cell.best === s.bestWin}
                              className="border-l border-border"
                            />
                            <Num
                              v={cell.final}
                              win={cell.final !== null && cell.final === s.finalWin}
                            />
                          </Fragment>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      )}

      {shared.length === 0 && !loading && runs.length > 0 && !state?.seriesMissing && (
        <Panel>
          <PanelHead title="Metrics" />
          <Empty>no metric these runs both logged</Empty>
        </Panel>
      )}

      <HparamDiff runs={runs} />
    </div>
  );
}

/* ---- one run in the legend ---------------------------------------------- */

function RunLegendRow({ run, colour }: { run: Run; colour: string }) {
  const label = run.name ?? run.id;
  const repo = typeof run.spec?.repo_id === "string" ? run.spec.repo_id : null;
  const policy = typeof run.spec?.policy_type === "string" ? run.spec.policy_type : null;
  return (
    <div className="flex items-center gap-2.5 border-b border-border/60 px-3 py-1.5 last:border-0">
      <span
        aria-hidden
        className="h-2.5 w-2.5 shrink-0 rounded-[2px]"
        style={{ backgroundColor: colour }}
      />
      <span className="min-w-0 flex-1 truncate font-mono text-[11px]" title={run.id}>
        {label}
      </span>
      {(policy || repo) && (
        <span className="hidden min-w-0 shrink truncate font-mono text-[10px] text-muted-foreground sm:inline">
          {[policy, repo].filter(Boolean).join(" · ")}
        </span>
      )}
      <StatusPill status={run.status} />
      {/* The Lab has exactly one deep link — this page — so there is no url
          that opens a single run. This goes back to the cockpit, where the
          run list and its detail pane live. */}
      <Link
        href="/"
        aria-label={`back to the cockpit, where ${label} is in the train tab`}
        title="the run list lives in the cockpit's Train tab"
        className="label-micro shrink-0 text-muted-foreground transition-colors hover:text-[var(--haller-live)]"
      >
        cockpit →
      </Link>
    </div>
  );
}

function StatusPill({ status }: { status: RunStatus }) {
  const colour = STATUS_COLOR[status] ?? "var(--haller-rail)";
  return (
    <span
      className="inline-flex h-4 shrink-0 items-center rounded-[3px] px-1 label-micro"
      style={{ color: colour, background: "color-mix(in oklch, " + colour + " 16%, transparent)" }}
    >
      {status}
    </span>
  );
}

/* ---- one number in the summary ------------------------------------------ */

function Num({
  v, win, className = "",
}: {
  v: number | null;
  win: boolean;
  className?: string;
}) {
  return (
    <td
      data-num
      className={"px-2.5 py-1 text-right tabular-nums " + className}
      style={win ? { color: "var(--haller-live)" } : undefined}
      title={v === null ? undefined : String(v)}
    >
      {fmtNum(v)}
    </td>
  );
}

/* ---- data ---------------------------------------------------------------- */

async function runOrNull(id: string): Promise<Run | null> {
  try {
    return await lab.run(id);
  } catch (e) {
    // 404 here is either a deleted run or a backend with no Lab at all. The
    // caller tells them apart by whether EVERY id came back empty.
    if (isMissing(e)) return null;
    throw e;
  }
}

async function keysOrNone(run: Run): Promise<string[]> {
  try {
    return metricKeys((await lab.runMetrics(run.id, 0)).rows);
  } catch (e) {
    if (isMissing(e)) return [];
    throw e;
  }
}

/** Smaller is better for a loss or an error, larger for everything else — a
 *  success rate, a reward, an accuracy. The key's name is all the server
 *  gives us, and reading the direction off the curve instead would call a run
 *  that got steadily worse the winner. */
function lowerIsBetter(key: string): boolean {
  return /loss|error/i.test(key);
}

/** The best and the last value of one series. `final` is the last FINITE
 *  point, not the last point: a run that logged a NaN on its way out did not
 *  end at NaN. */
function summarise(
  points: [number, number][] | undefined,
  lower: boolean,
): { best: number | null; final: number | null } {
  let best: number | null = null;
  let final: number | null = null;
  for (const p of points ?? []) {
    const y = p[1];
    if (!Number.isFinite(y)) continue;
    best = best === null ? y : lower ? Math.min(best, y) : Math.max(best, y);
    final = y;
  }
  return { best, final };
}

function extremeOf(values: (number | null)[], lower: boolean): number | null {
  let out: number | null = null;
  for (const v of values) {
    if (v === null) continue;
    out = out === null ? v : lower ? Math.min(out, v) : Math.max(out, v);
  }
  return out;
}
