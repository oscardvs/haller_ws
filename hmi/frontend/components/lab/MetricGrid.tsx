"use client";

/**
 * Every numeric key the trainer logged, one chart each.
 *
 * The kit charted `loss` and threw the rest of the row away — `lr`,
 * `grad_norm`, `samples_per_s`, `gpu_mem_gb`, `l1_loss`, `kld_loss` and
 * `epochs` were all sitting in the same JSONL line and none of them were ever
 * drawn. So nothing here is hardcoded: `plottableMetricKeys()` discovers the
 * keys that can actually be placed on an axis, and
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
 *
 * THE GRID FITS THE PANEL. Every cell was 130px tall whatever the window was,
 * so nine keys made three rows in a panel with room for one and a half, and
 * the answer to "how is this run doing" was behind a scrollbar inside a card
 * inside a fixed-viewport shell. Rows are `minmax(MIN_CELL_H, 1fr)` against a
 * definite container height instead: every chart takes an equal share of
 * whatever the panel has, and the scrollbar comes back only when even the
 * minimum will not fit. Each cell measures its own plot box, which is why a
 * cell is a component rather than a loop body.
 *
 * Any cell can be maximised to fill the metrics panel — the same overlay the
 * Review pane's charts use (`data-chart-zoom`, Escape or the header toggle to
 * restore). The tiled charts stay mounted underneath, and the zoomed chart
 * gets its own hover probe: a tiled curve is a shape, and reading VALUES off
 * it is what the big chart is for.
 */
import { useEffect, useMemo, useState } from "react";
import { Maximize2, Minimize2 } from "lucide-react";

import {
  metricX, plottableMetricKeys, type MetricAxis, type MetricRow,
} from "@/lib/lab";
import { Empty, Note, Segmented } from "@/components/lab/ui";
import {
  ChartLegend, LineChart, useElementHeight,
  type HoverPoint, type LineChartProps, type Series,
} from "@/components/lab/charts/LineChart";
import {
  EVAL_COLOR, TRAIN_COLOR, ema, fmtDuration, fmtNum, fmtTick, lttb, seriesColor,
} from "@/components/lab/charts/svg";
import { useSticky } from "@/components/cockpit/lib";

/** Above this a tiled chart is drawing several samples per pixel. LTTB keeps
 *  the ones that carry the shape — plain striding drops exactly the spike you
 *  opened the chart to find. */
const MAX_POINTS = 1200;

/** The floor a cell may be squeezed to before the grid gives up and scrolls.
 *  Below about this the plot area is shorter than its own axis labels, and a
 *  chart nobody can read is worse than a scrollbar.
 *
 *  Measured against the case that has to fit: nine keys on a 1500px-wide
 *  window is four columns and THREE rows, and at 132 those three rows wanted
 *  50px more than the panel had — a scrollbar for one row of charts. */
const MIN_CELL_H = 112;

/** Track width, and the number that decides how tall the charts are: columns
 *  come out of `auto-fill`, rows are what is left over, so a track that is one
 *  hair too wide costs a whole row of height.
 *
 *  16rem, measured on the two windows this runs on. At 17rem a 1830px cockpit
 *  fits four columns, which makes nine keys THREE rows of 95px; at 16rem it
 *  fits five, which makes them two rows of 150. The plot keeps ~206px of width
 *  after the y-axis gutter, which is a curve, not a smear. */
const CELL_MIN_W = "16rem";

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
  /** The maximised cell's id, or null. Cell ids are the metric keys, which are
   *  stable as rows append, so a live run keeps drawing into the big chart. */
  const [zoom, setZoom] = useState<string | null>(null);

  // Escape restores the grid; the overlay's header toggle does the same, so
  // the two ways in are the two ways out. Same rule as the Review pane.
  useEffect(() => {
    if (zoom === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setZoom(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [zoom]);

  /** One pass over the rows per key. Rows arrive append-only in axis order,
   *  which every consumer downstream assumes — `nearestIndex` binary-searches
   *  them and the shared x domain reads the ends. */
  const cells = useMemo<Cell[]>(() => {
    const keys = plottableMetricKeys(rows);
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

  /** The maximised cell, and its index into `drawn`. Found by id on every
   *  render: rows append as the run trains, and an index captured at click
   *  time would slide onto a different metric. */
  const zi = zoom === null ? -1 : cells.findIndex((c) => c.id === zoom);
  const zoomed = zi >= 0 ? cells[zi] : null;

  if (rows.length === 0) {
    return (
      <div className="flex h-full min-h-[100px] flex-col">
        {/* Only reachable while a run that could still log one is going:
            `RunDetail` does not open this panel for a run with nothing left to
            draw, so "waiting" is never a promise to a run that has stopped. */}
        <Empty>waiting for the first logged step…</Empty>
      </div>
    );
  }

  // Rows arrived, but nothing in them is a finite number — a stream of
  // strings is not a chart, and an empty grid says nothing about why.
  if (cells.length === 0) {
    return (
      <div className="flex h-full min-h-[100px] flex-col">
        <Empty>no numeric keys logged yet</Empty>
      </div>
    );
  }

  const readout = hover?.values.filter((v) => !v.id.endsWith("~raw")) ?? [];

  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      <div className="flex shrink-0 flex-wrap items-center gap-2">
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
      <div className="flex h-4 shrink-0 items-center gap-2 overflow-hidden font-mono text-[10px] whitespace-nowrap text-muted-foreground">
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
        <Note className="shrink-0">
          Nothing in this run is logged against{" "}
          <span className="font-mono">{axis === "wall" ? "wall-clock" : axis}</span> — the
          rows that carry no such value are dropped rather than stacked at zero. Try
          another axis.
        </Note>
      )}

      {/* `1fr` rows against the panel's own height: the charts divide what is
          left after the controls rather than each insisting on 130px and
          pushing the rest below the fold. `overflow-y-auto` is the floor
          case — a window too short for MIN_CELL_H scrolls, which is the one
          honest answer left. */}
      <div
        className="grid min-h-0 flex-1 gap-2 overflow-y-auto"
        style={{
          gridTemplateColumns: `repeat(auto-fill, minmax(${CELL_MIN_W}, 1fr))`,
          gridAutoRows: `minmax(${MIN_CELL_H}px, 1fr)`,
        }}
      >
        {cells.map((cell, ci) => (
          <MetricCell
            key={cell.id}
            cell={cell}
            series={drawn[ci]}
            axis={axis}
            // A log axis on a series that reaches zero or below is not a
            // smaller axis, it is a wrong one — `lr` decays to 0 and a reward
            // can be negative. Those cells stay linear whatever the toggle says.
            log={scale === "log" && cell.positive}
            xDomain={xDomain}
            playhead={hover?.x ?? null}
            onHover={setHover}
            onZoom={setZoom}
          />
        ))}
      </div>

      {/* The maximised cell covers the whole metrics panel — positioned
          against the Panel, which is the nearest relative ancestor, same as
          the Review pane's zoom. The tiled grid stays mounted underneath so
          restoring is a reveal, not a re-measure. */}
      {zoomed && (
        <div
          data-chart-zoom={zoomed.id}
          className="absolute inset-0 z-10 flex flex-col bg-background"
        >
          <div className="flex h-8.5 shrink-0 items-center gap-2 border-b border-border px-3">
            {zoomed.pair ? (
              <ChartLegend
                className="min-w-0 shrink"
                items={zoomed.series.map((s) => ({ label: s.key, color: s.color }))}
              />
            ) : (
              <span className="label-tracked shrink-0 text-muted-foreground">
                {zoomed.id}
              </span>
            )}
            <span className="flex min-w-0 items-baseline gap-1.5">
              {zoomed.series.map((s) => (
                <span
                  key={s.key}
                  data-num
                  title={`last ${s.key}`}
                  className="font-mono text-[10px] tabular-nums"
                  style={{ color: zoomed.pair ? s.color : "var(--foreground)" }}
                >
                  {fmtNum(s.ys[s.ys.length - 1])}
                </span>
              ))}
            </span>
            <button
              type="button"
              onClick={() => setZoom(null)}
              aria-label={`restore ${zoomed.id} chart`}
              title="back to the grid (esc)"
              className="ml-auto inline-flex h-5.5 w-5.5 shrink-0 items-center justify-center rounded-[3px] text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              <Minimize2 size={11} aria-hidden />
            </button>
          </div>
          <ZoomedCell
            label={`${zoomed.keys.join(" and ")} against ${axis}`}
            series={drawn[zi]}
            log={scale === "log" && zoomed.positive}
            xDomain={xDomain}
            xTickFormat={axis === "wall" ? fmtDuration : fmtTick}
            empty="no samples on this axis"
          />
        </div>
      )}
    </div>
  );
}

/**
 * One tile: a label, the last value, and a plot that takes whatever height the
 * grid row gives it.
 *
 * A COMPONENT rather than a loop body for one reason — `useElementHeight` is a
 * hook, and the plot has to be measured per cell. The alternative was to
 * compute a height here from the column count, which means re-deriving in
 * JavaScript what `auto-fill` already decided in CSS, and being wrong about it
 * on every window the two disagree on.
 */
function MetricCell({
  cell,
  series,
  axis,
  log,
  xDomain,
  playhead,
  onHover,
  onZoom,
}: {
  cell: Cell;
  series: Series[];
  axis: MetricAxis;
  log: boolean;
  xDomain: [number, number] | undefined;
  playhead: number | null;
  onHover: (p: HoverPoint | null) => void;
  onZoom: (id: string) => void;
}) {
  const [box, setBox] = useState<HTMLDivElement | null>(null);
  const h = useElementHeight(box);
  return (
    <div className="flex min-h-0 flex-col gap-1 overflow-hidden rounded-md border border-border bg-[var(--haller-inset)] p-1.5">
      <div className="flex min-w-0 shrink-0 items-baseline justify-between gap-2 px-0.5">
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
          <button
            type="button"
            onClick={() => onZoom(cell.id)}
            aria-label={`maximize ${cell.id} chart`}
            title="fill the metrics panel (esc to restore)"
            className="inline-flex h-4.5 w-4.5 items-center justify-center rounded-[3px] text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <Maximize2 size={10} aria-hidden />
          </button>
        </span>
      </div>

      {/* Measured, not fixed. jsdom reports 0 and `useElementHeight` keeps its
          fallback, so the tests still get a drawable chart. */}
      <div ref={setBox} className="min-h-0 flex-1">
        <LineChart
          label={`${cell.keys.join(" and ")} against ${axis}`}
          series={series}
          height={h}
          log={log}
          xDomain={xDomain}
          xTicks={3}
          xTickFormat={axis === "wall" ? fmtDuration : fmtTick}
          playhead={playhead}
          onHover={onHover}
          empty="no samples on this axis"
        />
      </div>
    </div>
  );
}

/** The maximised cell's plot. Owns its own probe — the grid's shared
 *  crosshair stays with the tiled charts underneath — and filters the EMA's
 *  raw underlay out of the readout: it is drawn for honesty, but two entries
 *  per key is noise, and the grid's shared readout already drops it. */
function ZoomedCell(props: Omit<LineChartProps, "height" | "onHover" | "probe" | "playhead">) {
  const [box, setBox] = useState<HTMLDivElement | null>(null);
  const h = useElementHeight(box);
  const [probe, setProbe] = useState<HoverPoint | null>(null);
  return (
    <div className="flex min-h-0 flex-1 flex-col p-2.5">
      {/* Measured, not fixed: the overlay is whatever height the metrics
          panel happens to be, and jsdom still gets a real box. */}
      <div ref={setBox} className="min-h-0 flex-1">
        <LineChart
          {...props}
          height={h}
          xTicks={5}
          onHover={(p) =>
            setProbe(p && { ...p, values: p.values.filter((v) => !v.id.endsWith("~raw")) })
          }
          probe={probe}
        />
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
