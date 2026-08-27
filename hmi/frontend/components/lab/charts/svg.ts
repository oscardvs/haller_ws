/**
 * Hand-written SVG charting. No library, no CDN, no new dependency.
 *
 * That is not thrift. This page is reached over a USB tether or a self-signed
 * tailnet cert with no egress guarantee, and a chart that renders only when a
 * CDN answers is a chart that is blank exactly when the operator is furthest
 * from a working network. The kit's `data.js` reached the same conclusion for
 * the same reason and drew its own paths.
 *
 * Everything here is pure: numbers in, geometry and strings out. React never
 * appears, so the arithmetic that decides whether a loss curve lies is
 * testable without a DOM.
 */

export type Pad = { l: number; r: number; t: number; b: number };

/** Room for a 4-character y label at 9px and a one-line x axis. Charts share
 *  one padding so their plot areas align down a column — a metric grid whose
 *  gutters disagree reads as several unrelated charts. */
export const PAD: Pad = { l: 42, r: 8, t: 10, b: 18 };

export type Scale = {
  w: number;
  h: number;
  pad: Pad;
  /** Data → px along x. */
  x: (v: number) => number;
  /** Data → px along y. Already log-mapped when the scale is logarithmic. */
  y: (v: number) => number;
  /** px → data along x, for hover. */
  xInvert: (px: number) => number;
  xDomain: [number, number];
  yDomain: [number, number];
  log: boolean;
};

/** Smallest positive value a log scale will plot. Below this a loss is noise
 *  and the axis would stretch decades to show it. */
const LOG_FLOOR = 1e-9;

/**
 * A linear or log scale over `[w, h]` with `pad` reserved.
 *
 * A log scale maps through log10 *inside* `y`, so callers pass raw data
 * either way and cannot forget. Non-positive values on a log scale are
 * clamped to the floor rather than dropped: dropping them silently shortens
 * the series, which looks like the run stopped.
 */
export function scale(
  w: number, h: number,
  xDomain: [number, number], yDomain: [number, number],
  opts: { log?: boolean; pad?: Pad } = {},
): Scale {
  const pad = opts.pad ?? PAD;
  const log = opts.log === true;
  const [x0, x1] = xDomain;
  const [ry0, ry1] = yDomain;
  const y0 = log ? Math.log10(Math.max(LOG_FLOOR, ry0)) : ry0;
  const y1 = log ? Math.log10(Math.max(LOG_FLOOR, ry1)) : ry1;
  const xSpan = x1 - x0 || 1;
  const ySpan = y1 - y0 || 1;
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;
  return {
    w, h, pad, log, xDomain, yDomain,
    x: (v) => pad.l + ((v - x0) / xSpan) * plotW,
    y: (v) => {
      const m = log ? Math.log10(Math.max(LOG_FLOOR, v)) : v;
      return pad.t + (1 - (m - y0) / ySpan) * plotH;
    },
    xInvert: (px) => x0 + ((px - pad.l) / (plotW || 1)) * xSpan,
  };
}

/** [min, max] over finite values, or null when there are none. */
export function extent(values: readonly number[]): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo === Infinity ? null : [lo, hi];
}

/** Widen a domain by a fraction of its span, so the extremes are not drawn
 *  on the frame. A zero-span domain gets ±1, not ±0 — a flat series still
 *  needs a plot area to sit in. */
export function padDomain(
  d: [number, number], frac = 0.06,
): [number, number] {
  const [lo, hi] = d;
  const span = hi - lo;
  if (span === 0) return [lo - 1, hi + 1];
  return [lo - span * frac, hi + span * frac];
}

/** Round-number ticks inside [lo, hi], at most `count` of them. */
export function niceTicks(lo: number, hi: number, count = 4): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return [lo];
  const raw = (hi - lo) / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return out;
}

/**
 * Decade ticks for a log axis, INSIDE the domain.
 *
 * ACT loss spans 7.25 → 0.068, which is two decades and change; a linear axis
 * flattens everything after the first fifty steps into the baseline.
 *
 * The bracketing decades are dropped rather than drawn. A tick outside the
 * domain maps to a y outside the plot box, and since the chart's `<svg>` does
 * not clip (the hover crosshair needs to overhang), such a label renders on
 * top of whichever chart happens to sit above or below it in a metric grid —
 * a number floating in another chart's cell, belonging to neither. A log axis
 * with two gridlines is honest; one with a label that is not on it is not.
 *
 * A domain inside a single decade has no decade to show, so it falls back to
 * its own endpoints — an axis with no labels at all is worse than an axis
 * labelled with the range it covers.
 */
export function logTicks(lo: number, hi: number): number[] {
  const a = Math.ceil(Math.log10(Math.max(LOG_FLOOR, lo)));
  const b = Math.floor(Math.log10(Math.max(LOG_FLOOR, hi)));
  const out: number[] = [];
  for (let p = a; p <= b; p++) {
    const v = Math.pow(10, p);
    if (v >= lo && v <= hi) out.push(v);
  }
  return out.length > 0 ? out : [lo, hi];
}

/** A polyline through the points, skipping non-finite ones by breaking the
 *  path. A gap in the data must read as a gap, not as a straight line across
 *  the hole it left. */
export function linePath(
  xs: readonly number[], ys: readonly number[], sc: Scale,
): string {
  let d = "";
  let pen = false;
  const n = Math.min(xs.length, ys.length);
  for (let i = 0; i < n; i++) {
    const xv = xs[i];
    const yv = ys[i];
    if (!Number.isFinite(xv) || !Number.isFinite(yv)) { pen = false; continue; }
    d += (pen ? "L" : "M") + sc.x(xv).toFixed(1) + " " + sc.y(yv).toFixed(1);
    pen = true;
  }
  return d;
}

/** A horizontal rule at `v`, as a path — cheaper than an element per guide
 *  when a chart draws several. */
export function hRule(v: number, sc: Scale): string {
  const y = sc.y(v).toFixed(1);
  return `M${sc.pad.l} ${y}L${sc.w - sc.pad.r} ${y}`;
}

/** A vertical rule at data-x `v` — the playhead and the hover crosshair. */
export function vRule(v: number, sc: Scale): string {
  const x = sc.x(v).toFixed(1);
  return `M${x} ${sc.pad.t}L${x} ${sc.h - sc.pad.b}`;
}

/**
 * Exponential moving average, forward pass only.
 *
 * `alpha` is the smoothing weight in 0..1, where 0 is "no smoothing" and
 * approaching 1 is "the curve barely moves". Forward-only is deliberate: a
 * centred filter would let a later step influence an earlier one, which on a
 * live run means the last plotted point moves after the fact.
 */
export function ema(ys: readonly number[], alpha: number): number[] {
  if (alpha <= 0) return [...ys];
  const a = Math.min(0.99, alpha);
  const out: number[] = [];
  let acc: number | null = null;
  for (const v of ys) {
    if (!Number.isFinite(v)) { out.push(NaN); continue; }
    acc = acc === null ? v : acc * a + v * (1 - a);
    out.push(acc);
  }
  return out;
}

/**
 * Largest-Triangle-Three-Buckets downsample to `maxPoints`.
 *
 * Plain stride sampling drops whichever points fall between strides, and on a
 * loss curve that is exactly the spike you wanted to see. LTTB keeps the
 * points that carry the shape, so a decimated curve and the full one tell the
 * same story. Returns the surviving indices.
 */
export function lttb(
  xs: readonly number[], ys: readonly number[], maxPoints: number,
): number[] {
  const n = Math.min(xs.length, ys.length);
  if (maxPoints >= n || maxPoints < 3) return range(n);
  const out = [0];
  const bucket = (n - 2) / (maxPoints - 2);
  let a = 0;
  for (let i = 0; i < maxPoints - 2; i++) {
    const lo = Math.floor((i + 1) * bucket) + 1;
    const hi = Math.min(n - 1, Math.floor((i + 2) * bucket) + 1);
    // Centroid of the NEXT bucket — the third corner of the triangle.
    let cx = 0;
    let cy = 0;
    for (let j = lo; j < hi; j++) { cx += xs[j]; cy += ys[j]; }
    const count = Math.max(1, hi - lo);
    cx /= count;
    cy /= count;

    const start = Math.floor(i * bucket) + 1;
    const end = Math.floor((i + 1) * bucket) + 1;
    let best = start;
    let bestArea = -1;
    for (let j = start; j < Math.min(end, n - 1); j++) {
      const area = Math.abs(
        (xs[a] - cx) * (ys[j] - ys[a]) - (xs[a] - xs[j]) * (cy - ys[a]),
      );
      if (area > bestArea) { bestArea = area; best = j; }
    }
    out.push(best);
    a = best;
  }
  out.push(n - 1);
  return out;
}

function range(n: number): number[] {
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = i;
  return out;
}

/** Index of the sample nearest data-x `x`. Binary search: the hover crosshair
 *  runs on every pointer move over a series that can be tens of thousands of
 *  points. Assumes `xs` ascending, which every axis here is. */
export function nearestIndex(xs: readonly number[], x: number): number {
  if (xs.length === 0) return -1;
  let lo = 0;
  let hi = xs.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] > x) hi = mid;
    else lo = mid;
  }
  return Math.abs(xs[lo] - x) <= Math.abs(xs[hi] - x) ? lo : hi;
}

/** A number sized for a 9px tick label or a stat cell. Exponential at both
 *  extremes because `0.0000034` and `12400000` both blow a column open. */
export function fmtNum(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e4 || a < 1e-3)) return v.toExponential(1);
  return v.toFixed(digits);
}

/** A tick label: integers stay bare, decades on a log axis stay decades. */
export function fmtTick(v: number): string {
  if (!Number.isFinite(v)) return "";
  if (Number.isInteger(v) && Math.abs(v) < 1e5) return String(v);
  const a = Math.abs(v);
  if (a >= 1e5 || (a !== 0 && a < 1e-2)) return v.toExponential(0);
  return String(Number(v.toFixed(3)));
}

/**
 * A tick formatter for a seconds axis, with the precision the SPAN needs.
 *
 * `v.toFixed(0)` is right for a 28-second take and useless for a 0.3-second
 * one, where every tick rounds to "0s" and the axis says nothing. Both exist
 * on this disk: the bimanual dataset's first episode is 9 frames long.
 */
export function secondsTickFormat(span: number): (v: number) => string {
  const decimals = !Number.isFinite(span) || span >= 10 ? 0 : span >= 1 ? 1 : 2;
  return (v: number) => `${v.toFixed(decimals)}s`;
}

/** Seconds as `1m 04s` / `47s` — durations, not clock times. */
export function fmtDuration(s: number | null | undefined): string {
  if (s === null || s === undefined || !Number.isFinite(s)) return "—";
  const total = Math.round(s);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const sec = total % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m) return `${m}m ${String(sec).padStart(2, "0")}s`;
  return `${total}s`;
}

/** Bytes at one decimal. Matches `DatasetTab`'s formatter so the two dataset
 *  surfaces never print the same directory at two different sizes. */
export function fmtBytes(n: number | null | undefined): string {
  if (typeof n !== "number" || !Number.isFinite(n)) return "—";
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
  return `${v.toFixed(v >= 100 ? 0 : 1)} ${units[i]}`;
}

/** The compare ramp, in order. Five tokens the cockpit does not otherwise
 *  use, so nothing on this page competes with a status colour. Cycled past
 *  five: a sixth run repeats a hue, which is honest, where inventing a colour
 *  would not be. */
export const SERIES_COLORS = [
  "var(--chart-1)", "var(--chart-2)", "var(--chart-3)",
  "var(--chart-4)", "var(--chart-5)",
] as const;

export const seriesColor = (i: number) =>
  SERIES_COLORS[((i % SERIES_COLORS.length) + SERIES_COLORS.length) % SERIES_COLORS.length];

/** Train is live-green, eval is warn-amber. The pair is fixed everywhere a
 *  run is charted so the eye learns it once. */
export const TRAIN_COLOR = "var(--haller-live)";
export const EVAL_COLOR = "var(--haller-warn)";
