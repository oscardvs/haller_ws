"use client";

/**
 * One metric, every run overlaid.
 *
 * The colour arrives from the caller and is never re-derived here: a run wears
 * the same swatch in the legend, in all twelve charts of the grid and in the
 * summary table, so the mapping is learned once. `seriesColor` walks
 * --chart-1..5, which nothing else in the cockpit uses — a compare curve can
 * therefore never be mistaken for a status colour.
 *
 * The y scale is the caller's too, and it is one decision for the whole page.
 * Half a grid on log and half on linear is a grid whose curves cannot be
 * compared by shape, which is the only thing a grid of small charts is for.
 */
import { ChartLegend, LineChart } from "@/components/lab/charts/LineChart";
import { Panel, PanelHead } from "@/components/lab/ui";
import { ema } from "./svg";

export function CompareChart({
  metricKey,
  series,
  log,
  xLabel,
  height = 200,
  smoothing = 0,
}: {
  metricKey: string;
  /** Already downsampled by the server — `[x, y]` per point, x on `xLabel`. */
  series: { id: string; label: string; color: string; points: [number, number][] }[];
  /** EMA weight, 0..1. 0 draws the raw series alone; above 0 the smoothed
   *  curve is drawn over a faint raw one, so smoothing can never hide the
   *  spike it is smoothing over. */
  smoothing?: number;
  log: boolean;
  xLabel: string;
  height?: number;
}) {
  const raw = series.map((s) => ({
    id: s.id,
    label: s.label,
    color: s.color,
    xs: s.points.map((p) => p[0]),
    ys: s.points.map((p) => p[1]),
  }));
  const plotted = raw.filter((s) => s.xs.length > 0);

  // Smoothed over faint raw, never smoothed instead of it.
  //
  // On a single run you read the VALUE; on three overlaid you read the SHAPE,
  // and per-step noise on grad_norm or samples_per_s is a hairball that hides
  // exactly the divergence this page exists to show. So smoothing is on by
  // default here where it is off in the metric grid — but the raw series stays
  // underneath, because a smoothed curve is an interpretation and the operator
  // has to be able to see what it was made from.
  const smoothed =
    smoothing > 0
      ? raw.map((s) => ({ ...s, id: `${s.id}:ema`, ys: ema(s.ys, smoothing) }))
      : [];
  const drawn =
    smoothing > 0
      ? [...raw.map((s) => ({ ...s, opacity: 0.22, width: 1 })), ...smoothed]
      : raw;

  return (
    <Panel>
      <PanelHead title={metricKey} right={xLabel} />
      <div className="flex min-h-0 flex-col gap-1.5 p-2.5">
        <LineChart
          series={drawn}
          height={height}
          log={log}
          xTicks={3}
          label={`${metricKey}, ${plotted.length} run${plotted.length === 1 ? "" : "s"}`}
          empty="no points"
        />
        <ChartLegend
          items={series.map((s) => ({ label: s.label, color: s.color }))}
          className="px-1"
        />
      </div>
    </Panel>
  );
}
