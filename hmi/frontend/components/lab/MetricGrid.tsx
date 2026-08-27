"use client";

/**
 * Every numeric key the trainer logged, one chart each.
 *
 * The kit charted `loss` and threw the rest of the row away — `lr`,
 * `grad_norm`, `samples_per_s`, `gpu_mem_gb`, `l1_loss`, `kld_loss` and
 * `epochs` were all sitting in the same JSONL line and none of them were ever
 * drawn. So nothing here is hardcoded: `metricKeys()` discovers the keys, and
 * a policy that logs one nobody has heard of gets a chart like everything
 * else. `loss`/`eval_loss` lead when they exist, which is an ORDER preference
 * and not a filter.
 *
 * The three controls are the ones that decide whether a curve tells the truth:
 *   - LOG y by default, because ACT loss spans 7.25 → 0.068 and a linear axis
 *     flattens everything after step 50 into the baseline.
 *   - EMA smoothing that draws the RAW series underneath at low opacity, so
 *     smoothing can never hide the spike it was smoothing over.
 *   - step / epoch / wall-clock, where a row that cannot be placed on the
 *     chosen axis is DROPPED rather than plotted at zero.
 *
 * The grid is one instrument, not a wall of pictures: every chart shares one
 * x domain and one crosshair, so reading `loss` against `grad_norm` at the
 * same step is a glance rather than an arithmetic exercise.
 */
import { useMemo, useState } from "react";

import {
  metricKeys, metricX, type MetricAxis, type MetricRow,
} from "@/lib/lab";
import { Empty, Note, Segmented } from "@/components/lab/ui";
import {
  ChartLegend, LineChart, type HoverPoint, type Series,
} from "@/components/lab/charts/LineChart";
import {
  EVAL_COLOR, TRAIN_COLOR, ema, fmtDuration, fmtNum, fmtTick, lttb, seriesColor,
} from "@/components/lab/charts/svg";
import { useSticky } from "@/components/cockpit/lib";

/** Above this a 20rem-wide chart is drawing several samples per pixel. LTTB
 *  keeps the ones that carry the shape — plain striding drops exactly the
 *  spike you opened the chart to find. */
const MAX_POINTS = 1200;

const CHART_H = 130;

/** The one pair that shares a chart. The gap between train and eval loss IS
 *  the overfitting reading, and it is only legible on one set of axes. */
const PAIR: readonly string[] = ["loss", "eval_loss"];

const PAIR_COLOR: Record<string, string> = {
  loss: TRAIN_COLOR,
  eval_loss: EVAL_COLOR,
};

type BaseSeries = { key: string; color: string; xs: number[]; ys: number[] };
type Cell = {
  id: string;
  keys: string[];
  pair: boolean;
  series: BaseSeries[];
  /** Every sample strictly above zero — the only case where a log axis is a
   *  smaller axis rather than a wrong one. Computed once with the series, not
   *  per render: the crosshair re-renders this component on every pointer
   *  move and a full scan per chart per move is a scan too many. */
  positive: boolean;
};

type Scale = "log" | "lin";

const AXIS_OPTIONS: readonly { value: MetricAxis; label: string; hint: string }[] = [
  { value: "step", label: "step", hint: "optimiser steps — the axis the trainer logs against" },
  { value: "epoch", label: "epoch", hint: "passes over the dataset; rows with no epoch are dropped, not plotted at 0" },
  { value: "wall", label: "wall", hint: "seconds since launch — the axis that answers 'how long will this take'" },
];

const SCALE_OPTIONS: readonly { value: Scale; label: string; hint: string }[] = [
  {
    value: "log",
    label: "log",
    hint: "ACT loss spans 7.25 to 0.068; a log axis is the only one that shows the last two decades of it",
  },
  {
    value: "lin",
    label: "linear",
    hint: "a linear axis flattens everything after about step 50 against the baseline",
  },
];

export function MetricGrid({
  rows,
  steps,
}: {
  rows: MetricRow[];
  /** The planned length from the run's spec. Fixes the step axis so a run
   *  that is 5% done looks 5% done instead of rescaling every tick. */
  steps?: number | null;
}) {
  // Sticky so a glance at another tab does not reset the axis and the
  // smoothing the operator just dialled in.
  const [scale, setScale] = useSticky<Scale>("lab.metrics.scale", "log");
  const [axis, setAxis] = useSticky<MetricAxis>("lab.metrics.axis", "step");
  const [alpha, setAlpha] = useSticky<number>("lab.metrics.ema", 0);
  const [hover, setHover] = useState<HoverPoint | null>(null);

  /** One pass over the rows per key. Rows arrive append-only in axis order,
   *  which every consumer downstream assumes — `nearestIndex` binary-searches
   *  them and the shared x domain reads the ends. */
  const cells = useMemo<Cell[]>(() => {
    const keys = metricKeys(rows);
    if (keys.length === 0) return [];

    const seriesFor = (key: string, color: string): BaseSeries => {
      const xs: number[] = [];
      const ys: number[] = [];
      for (const row of rows) {
        const v = row[key];
        if (typeof v !== "number" || !Number.isFinite(v)) continue;
        const x = metricX(row, axis);
        // An eval row logged without an epoch is not at epoch 0.
        if (x === null) continue;
        xs.push(x);
        ys.push(v);
      }
      return { key, color, xs, ys };
    };

    const positive = (ss: BaseSeries[]) => ss.every((s) => s.ys.every((v) => v > 0));

    const out: Cell[] = [];
    const pairKeys = PAIR.filter((k) => keys.includes(k));
    if (pairKeys.length > 0) {
      const ss = pairKeys.map((k) => seriesFor(k, PAIR_COLOR[k]));
      out.push({ id: pairKeys.join(" · "), keys: pairKeys, pair: true, series: ss, positive: positive(ss) });
    }
    let i = 0;
    for (const k of keys) {
      if (PAIR.includes(k)) continue;
      const ss = [seriesFor(k, seriesColor(i))];
      out.push({ id: k, keys: [k], pair: false, series: ss, positive: positive(ss) });
      i += 1;
    }
    return out;
  }, [rows, axis]);

  /** Decimation and smoothing, cached on the data and the smoothing weight —
   *  NOT on the hover, which changes on every pointer move and would otherwise
   *  re-run an EMA over the whole run twenty times a second. */
  const drawn = useMemo(
    () =>
      cells.map((c) => {
        const out: Series[] = [];
        for (const s of c.series) {
          const rawIdx = lttb(s.xs, s.ys, MAX_POINTS);
          if (alpha > 0) {
            const sm = ema(s.ys, alpha);
            const smIdx = lttb(s.xs, sm, MAX_POINTS);
            // Raw underneath, smoothed on top. A spike the filter ate is still
            // on the chart, at a quarter opacity.
            out.push({
              id: `${s.key}~raw`,
              color: s.color,
              xs: rawIdx.map((i) => s.xs[i]),
              ys: rawIdx.map((i) => s.ys[i]),
              opacity: 0.25,
              width: 1,
            });
            out.push({
              id: s.key,
              label: s.key,
              color: s.color,
              xs: smIdx.map((i) => s.xs[i]),
              ys: smIdx.map((i) => sm[i]),
            });
          } else {
            out.push({
              id: s.key,
              label: s.key,
              color: s.color,
              xs: rawIdx.map((i) => s.xs[i]),
              ys: rawIdx.map((i) => s.ys[i]),
            });
          }
        }
        return out;
      }),
    [cells, alpha],
  );

  /** ONE x domain for the whole grid. The synchronised crosshair only means
   *  anything if every chart puts the same step at the same pixel — and a
   *  playhead drawn from another chart's domain would otherwise land outside
   *  this one's plot area. */
  const xDomain = useMemo<[number, number] | undefined>(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const c of cells) {
      for (const s of c.series) {
        if (s.xs.length === 0) continue;
        lo = Math.min(lo, s.xs[0]);
        hi = Math.max(hi, s.xs[s.xs.length - 1]);
      }
    }
    if (lo === Infinity) return undefined;
    if (axis === "step" && typeof steps === "number" && steps > 0) {
      return [Math.min(0, lo), Math.max(hi, steps)];
    }
    return [lo, hi];
  }, [cells, axis, steps]);

  const plotted = cells.some((c) => c.series.some((s) => s.xs.length > 0));

  if (rows.length === 0) {
    return (
      <div className="flex min-h-[100px] flex-col">
        <Empty>waiting for the first logged step…</Empty>
      </div>
    );
  }

  // Rows arrived, but nothing in them is a finite number — a stream of
  // strings is not a chart, and an empty grid says nothing about why.
  if (cells.length === 0) {
    return (
      <div className="flex min-h-[100px] flex-col">
        <Empty>no numeric keys logged yet</Empty>
      </div>
    );
  }

  const readout = hover?.values.filter((v) => !v.id.endsWith("~raw")) ?? [];

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <Segmented
          label="metric y scale"
          options={SCALE_OPTIONS}
          value={scale}
          onChange={setScale}
          className="shrink-0"
        />
        <Segmented
          label="metric x axis"
          options={AXIS_OPTIONS}
          value={axis}
          onChange={setAxis}
          className="shrink-0"
        />
        <label className="flex shrink-0 items-center gap-1.5">
          <span className="label-micro text-muted-foreground">ema</span>
          <input
            type="range"
            min={0}
            max={0.95}
            step={0.05}
            value={alpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
            aria-label="ema smoothing weight"
            title="forward-only exponential smoothing; the raw series stays drawn underneath"
            className="haller-range h-4 w-24"
            style={{ "--pct": `${(alpha / 0.95) * 100}%` } as React.CSSProperties}
          />
          <span data-num className="w-7 font-mono text-[10px] tabular-nums text-muted-foreground">
            {alpha.toFixed(2)}
          </span>
        </label>
        <span className="ml-auto shrink-0 font-mono text-[10px] text-muted-foreground">
          <span data-num className="tabular-nums">{cells.length}</span> keys ·{" "}
          <span data-num className="tabular-nums">{rows.length}</span> rows
        </span>
      </div>

      {/* The synchronised readout. One strip for the whole grid, because the
          crosshair is one crosshair. */}
      <div className="flex h-4 items-center gap-2 overflow-hidden font-mono text-[10px] whitespace-nowrap text-muted-foreground">
        {hover ? (
          <>
            <span data-num className="tabular-nums text-foreground">
              {xReadout(hover.x, axis)}
            </span>
            {readout.map((v) => (
              <span key={v.id} className="shrink-0">
                <span aria-hidden className="pr-2 opacity-40">·</span>
                <span style={{ color: v.color }}>{v.label ?? v.id}</span>{" "}
                <span data-num className="tabular-nums text-foreground">{fmtNum(v.y)}</span>
              </span>
            ))}
          </>
        ) : (
          <span className="opacity-60">hover any chart — the crosshair is shared across the grid</span>
        )}
      </div>

      {!plotted && (
        <Note>
          Nothing in this run is logged against{" "}
          <span className="font-mono">{axis === "wall" ? "wall-clock" : axis}</span> — the
          rows that carry no such value are dropped rather than stacked at zero. Try
          another axis.
        </Note>
      )}

      <div className="grid gap-2 [grid-template-columns:repeat(auto-fill,minmax(20rem,1fr))]">
        {cells.map((cell, ci) => {
          // A log axis on a series that reaches zero or below is not a smaller
          // axis, it is a wrong one — `lr` decays to 0 and a reward can be
          // negative. Those cells stay linear whatever the toggle says.
          const log = scale === "log" && cell.positive;
          return (
            <div
              key={cell.id}
              className="flex flex-col gap-1 rounded-md border border-border bg-[var(--haller-inset)] p-1.5"
            >
              <div className="flex min-w-0 items-baseline justify-between gap-2 px-0.5">
                {cell.pair ? (
                  <ChartLegend
                    className="min-w-0 shrink"
                    items={cell.series.map((s) => ({ label: s.key, color: s.color }))}
                  />
                ) : (
                  <span
                    className="label-micro min-w-0 truncate text-muted-foreground"
                    title={cell.id}
                  >
                    {cell.id}
                  </span>
                )}
                <span className="flex shrink-0 items-baseline gap-1.5">
                  {cell.series.map((s) => (
                    <span
                      key={s.key}
                      data-num
                      title={`last ${s.key}`}
                      className="font-mono text-[10px] tabular-nums"
                      style={{ color: cell.pair ? s.color : "var(--foreground)" }}
                    >
                      {fmtNum(s.ys[s.ys.length - 1])}
                    </span>
                  ))}
                </span>
              </div>

              <LineChart
                label={`${cell.keys.join(" and ")} against ${axis}`}
                series={drawn[ci]}
                height={CHART_H}
                log={log}
                xDomain={xDomain}
                xTicks={3}
                xTickFormat={axis === "wall" ? fmtDuration : fmtTick}
                playhead={hover?.x ?? null}
                onHover={setHover}
                empty="no samples on this axis"
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** The hovered x, worded for the axis it came from. */
function xReadout(x: number, axis: MetricAxis): string {
  if (axis === "wall") return `wall ${fmtDuration(x)}`;
  if (axis === "epoch") return `epoch ${x.toFixed(2)}`;
  return `step ${Math.round(x)}`;
}
