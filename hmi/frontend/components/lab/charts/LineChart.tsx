"use client";

/**
 * The one chart component. A thin wrapper over `svg.ts` — it owns measurement,
 * axes, hover and the empty state, and nothing else.
 *
 * It measures its container rather than scaling a fixed viewBox, because the
 * same chart is drawn at 700px in the review pane and at 320px in a
 * three-column metric grid. A fixed viewBox scales the TEXT with the frame,
 * and a 9px tick label in a 320px cell off an 800px viewBox renders at four
 * pixels — present, unreadable, and worse than absent.
 */
import { useCallback, useLayoutEffect, useRef, useState } from "react";

import {
  PAD, type Scale, extent, fmtTick, linePath, hRule, vRule, logTicks,
  nearestIndex, niceTicks, padDomain, scale,
} from "./svg";

export type Series = {
  id: string;
  label?: string;
  color: string;
  xs: number[];
  ys: number[];
  width?: number;
  /** Dashed strokes read as "not measured the same way" — an action trace
   *  against a state trace, a projection against a reading. */
  dashed?: boolean;
  opacity?: number;
};

/** A labelled horizontal reference line — a gripper threshold, a target. */
export type Guide = { at: number; label?: string; color?: string; dashed?: boolean };

export type HoverPoint = {
  /** Data-space x under the pointer. */
  x: number;
  /** Nearest sample per series, by series id. */
  values: { id: string; label?: string; color: string; x: number; y: number }[];
};

/** Rendered when the container has not been measured yet — jsdom and the
 *  first paint both land here. Wide enough that a chart in a test has a real
 *  plot area to put paths in. */
const FALLBACK_W = 640;

export function LineChart({
  series,
  height,
  log = false,
  yDomain,
  xDomain,
  guides = [],
  playhead = null,
  yTickFormat = fmtTick,
  xTickFormat = fmtTick,
  yTicks: yTickCount = 3,
  xTicks: xTickCount = 0,
  empty = "no data",
  label,
  onHover,
  className = "",
}: {
  series: Series[];
  height: number;
  log?: boolean;
  /** Overrides the computed y extent. Use when several charts must share one
   *  scale, never to hide an outlier. */
  yDomain?: [number, number];
  xDomain?: [number, number];
  guides?: Guide[];
  /** Data-space x for the playhead rule, or null for none. */
  playhead?: number | null;
  yTickFormat?: (v: number) => string;
  xTickFormat?: (v: number) => string;
  yTicks?: number;
  /** 0 draws only the right-hand end label, which is all a step axis needs. */
  xTicks?: number;
  empty?: string;
  /** Accessible name. Charts are the readout here, not decoration. */
  label: string;
  onHover?: (p: HoverPoint | null) => void;
  className?: string;
}) {
  const [box, setBox] = useState<HTMLDivElement | null>(null);
  const w = useElementWidth(box);
  const svgRef = useRef<SVGSVGElement>(null);

  const drawable = series.filter((s) => s.xs.length > 0 && s.ys.length > 0);
  const hasData = drawable.length > 0;

  const xs = drawable.flatMap((s) => s.xs);
  const ys = drawable.flatMap((s) => s.ys);
  const xExt = xDomain ?? extent(xs) ?? [0, 1];
  const rawY = yDomain ?? padDomain(extent(ys) ?? [0, 1]);
  // A log axis cannot show a non-positive floor; lift it to the smallest
  // positive sample rather than clamping every point onto one pixel.
  const yExt: [number, number] = log
    ? [Math.max(smallestPositive(ys), rawY[0] > 0 ? rawY[0] : smallestPositive(ys)), Math.max(rawY[1], 1e-9)]
    : rawY;

  const sc = scale(w, height, xExt, yExt, { log });
  const yTickValues = log ? logTicks(yExt[0], yExt[1]) : niceTicks(yExt[0], yExt[1], yTickCount);
  const xTickValues = xTickCount > 0 ? niceTicks(xExt[0], xExt[1], xTickCount) : [];

  const emit = useCallback(
    (e: React.PointerEvent<SVGSVGElement>) => {
      if (!onHover) return;
      const rect = svgRef.current?.getBoundingClientRect();
      if (!rect) return;
      const px = e.clientX - rect.left;
      if (px < sc.pad.l || px > sc.w - sc.pad.r) { onHover(null); return; }
      const x = sc.xInvert(px);
      onHover({
        x,
        values: drawable.map((s) => {
          const i = nearestIndex(s.xs, x);
          return {
            id: s.id, label: s.label, color: s.color,
            x: s.xs[i], y: s.ys[i],
          };
        }),
      });
    },
    // `sc` and `drawable` are rebuilt each render from props; depending on the
    // primitives they are derived from keeps this stable across re-renders
    // that changed nothing about the geometry.
    [onHover, sc, drawable],
  );

  return (
    <div ref={setBox} className={"relative w-full " + className}>
      <svg
        ref={svgRef}
        role="img"
        aria-label={label}
        viewBox={`0 0 ${w} ${height}`}
        width="100%"
        height={height}
        className="block overflow-visible"
        onPointerMove={onHover ? emit : undefined}
        onPointerLeave={onHover ? () => onHover(null) : undefined}
      >
        {/* Grid + y labels. Drawn first so every stroke lands on top of it. */}
        {yTickValues.map((v) => (
          <g key={`y${v}`}>
            <path
              d={hRule(v, sc)}
              stroke="var(--border)"
              strokeWidth={1}
              vectorEffect="non-scaling-stroke"
              fill="none"
            />
            <text
              x={sc.pad.l - 6}
              y={sc.y(v) + 3}
              textAnchor="end"
              fill="var(--muted-foreground)"
              fontSize={9}
              className="font-mono"
              style={{ fontVariantNumeric: "tabular-nums" }}
            >
              {yTickFormat(v)}
            </text>
          </g>
        ))}

        {xTickValues.map((v) => (
          <text
            key={`x${v}`}
            x={sc.x(v)}
            y={height - 5}
            textAnchor="middle"
            fill="var(--muted-foreground)"
            fontSize={9}
            className="font-mono"
            style={{ fontVariantNumeric: "tabular-nums" }}
          >
            {xTickFormat(v)}
          </text>
        ))}

        {guides.map((g, i) => (
          <g key={`g${i}`}>
            <path
              d={hRule(g.at, sc)}
              stroke={g.color ?? "var(--haller-rail)"}
              strokeWidth={1}
              strokeDasharray={g.dashed === false ? undefined : "3 3"}
              vectorEffect="non-scaling-stroke"
              fill="none"
            />
            {g.label && (
              <text
                x={sc.w - sc.pad.r - 2}
                y={sc.y(g.at) - 3}
                textAnchor="end"
                fill="var(--haller-rail)"
                fontSize={8}
                className="font-mono"
              >
                {g.label}
              </text>
            )}
          </g>
        ))}

        {hasData &&
          drawable.map((s) => (
            <path
              key={s.id}
              d={linePath(s.xs, s.ys, sc)}
              data-series={s.id}
              fill="none"
              stroke={s.color}
              strokeWidth={s.width ?? 1.4}
              strokeDasharray={s.dashed ? "4 3" : undefined}
              strokeLinejoin="round"
              strokeLinecap="round"
              opacity={s.opacity ?? 1}
              vectorEffect="non-scaling-stroke"
            />
          ))}

        {playhead !== null && playhead !== undefined && hasData && (
          <path
            d={vRule(playhead, sc)}
            data-playhead=""
            stroke="var(--haller-live)"
            strokeWidth={1}
            vectorEffect="non-scaling-stroke"
            fill="none"
          />
        )}

        {!hasData && (
          <text
            x={w / 2}
            y={height / 2}
            textAnchor="middle"
            fill="var(--muted-foreground)"
            fontSize={10}
            className="font-mono"
          >
            {empty}
          </text>
        )}
      </svg>
    </div>
  );
}

function smallestPositive(values: number[]): number {
  let lo = Infinity;
  for (const v of values) if (Number.isFinite(v) && v > 0 && v < lo) lo = v;
  return lo === Infinity ? 1e-6 : lo;
}

/**
 * The element's content width in CSS pixels.
 *
 * ResizeObserver where it exists and a single measurement where it does not —
 * jsdom has neither the observer nor layout, so a test would otherwise render
 * every chart into a zero-width plot area and assert nothing.
 */
export function useElementWidth(node: HTMLElement | null): number {
  const [w, setW] = useState(FALLBACK_W);

  useLayoutEffect(() => {
    if (!node) return;
    const read = () => {
      const next = node.clientWidth;
      if (next > 0) setW(next);
    };
    read();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(read);
    ro.observe(node);
    return () => ro.disconnect();
  }, [node]);

  return w;
}

/** A legend row that matches the chart's stroke vocabulary — a 9px dash in
 *  the series colour, dashed when the series is. */
export function ChartLegend({
  items,
  className = "",
}: {
  items: { label: string; color: string; dashed?: boolean }[];
  className?: string;
}) {
  return (
    <div className={"flex flex-wrap items-center gap-x-2.5 gap-y-1 " + className}>
      {items.map((it) => (
        <span
          key={it.label}
          className="inline-flex items-center gap-1 font-mono text-[9px] text-muted-foreground"
        >
          <svg width="10" height="4" aria-hidden className="shrink-0">
            <path
              d="M0 2H10"
              stroke={it.color}
              strokeWidth={2}
              strokeDasharray={it.dashed ? "2 2" : undefined}
            />
          </svg>
          {it.label}
        </span>
      ))}
    </div>
  );
}

/** Re-exported so a chart consumer has one import for the geometry it needs
 *  to place an overlay against. */
export type { Scale };
export { PAD, scale };
