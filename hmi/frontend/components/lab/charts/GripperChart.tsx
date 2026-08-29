"use client";

/**
 * The gripper, on its own scale and against its own dataset's thresholds.
 *
 * Two things the kit could not do. It took `state[state.length - 1]`, which is
 * the gripper on a solo arm and the RIGHT gripper on a bimanual one, so on
 * Haller's 12-dim dataset it silently charted one hand and never mentioned the
 * other. And it drew its guides at a hardcoded 40/70, which is only true of a
 * 0..100 gripper — a dataset recorded against a different calibration would be
 * read against thresholds that were never true of it.
 *
 * So: every channel `isGripperChannel` finds, and guides read from the
 * thresholds the GRADER used — `arms[].closed_below` / `open_above`, the exact
 * floats the verdict beside this chart was decided with. One source for one
 * number: reading the calibration separately and re-deriving the fractions
 * would let the chart draw a guide the verdict disagrees with. Measured on
 * disk: the kit's dataset grades at 40.0/70.0, and the bimanual one, whose
 * gripper is calibrated in DEGREES over [-9.97, 100.27], at 34.13/67.20.
 *
 * When the backend sends no thresholds there are NO guides and the legend says
 * so. An invented threshold is worse than a missing one: it looks like a
 * measurement.
 */
import { useMemo, useState } from "react";
import { Maximize2, Minimize2 } from "lucide-react";

import { isDrawableTrace, isGripperChannel, type Trace } from "@/lib/lab";
import { Panel, PanelHead } from "@/components/lab/ui";
import {
  ChartLegend, ProbeLineChart, useElementHeight, type Guide, type Series,
} from "./LineChart";
import { extent, padDomain, secondsTickFormat, seriesColor } from "./svg";

const HEIGHT = 104;

/** `left_gripper` / `gripper` — the side is the whole point of the label here,
 *  so `shortChannel` (which strips it) is deliberately not used. */
const gripperLabel = (name: string) =>
  name.replace(/\.pos$/, "").replace(/_/g, " ");

export function GripperChart({
  trace: rawTrace,
  playheadT,
  zoomed = false,
  onZoom,
}: {
  trace: Trace | null;
  /** Episode-relative seconds from the player, or null when nothing plays. */
  playheadT: number | null;
  /** Filling the analysis column as an overlay rather than tiling under the
   *  player. The plot height comes from the overlay, not from `HEIGHT`. */
  zoomed?: boolean;
  /** Toggles `zoomed`; absent, the header carries no zoom control. */
  onZoom?: (zoomed: boolean) => void;
}): React.ReactElement {
  // A partial body arriving with a 200 is "no trace", not a render-phase throw
  // that takes the review pane down with it. See `isDrawableTrace`.
  const trace = isDrawableTrace(rawTrace) ? rawTrace : null;

  // `trace.gripper` is the backend's own isolation of these columns, and it
  // carries the thresholds each channel was GRADED against — so the line and
  // the guides under it come from one place and cannot disagree. The scan is
  // the fallback for a backend that does not send it, and it can only produce
  // lines: there is nothing to draw a guide from, and an invented one looks
  // like a measurement.
  const channels = useMemo(() => {
    if (!trace) return [];
    const packed = trace.gripper;
    if (Array.isArray(packed) && packed.length > 0) {
      return packed.map((g) => ({
        name: g.name,
        ys: g.values ?? trace.state[g.index] ?? [],
        closed_below: g.closed_below,
        open_above: g.open_above,
      }));
    }
    return trace.names
      .map((name, i) => ({
        name,
        ys: trace.state[i] ?? [],
        closed_below: undefined as number | undefined,
        open_above: undefined as number | undefined,
      }))
      .filter((c) => isGripperChannel(c.name));
  }, [trace]);

  const { guides, uncalibrated } = useMemo(() => {
    const out: Guide[] = [];
    const seen = new Set<string>();
    const missing: string[] = [];
    for (const c of channels) {
      if (!Number.isFinite(c.closed_below) || !Number.isFinite(c.open_above)) {
        missing.push(c.name);
        continue;
      }
      for (const [at, label] of
           [[c.closed_below as number, "closed"], [c.open_above as number, "open"]] as const) {
        // Both arms usually share one calibration; drawing the same rule twice
        // makes a 1px line look like a 2px one.
        const key = `${label}:${at.toFixed(4)}`;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push({ at, label });
      }
    }
    return { guides: out, uncalibrated: missing };
  }, [channels]);

  const series: Series[] = channels.map((c, i) => ({
    id: c.name,
    label: gripperLabel(c.name),
    color: seriesColor(i),
    xs: trace?.t ?? [],
    ys: c.ys,
  }));

  // The guides join the extent: a gripper that never opens fully would push
  // its "open" line off the top of the chart, which is exactly the episode
  // the line was drawn for.
  const yDomain = useMemo<[number, number] | undefined>(() => {
    let lo = Infinity;
    let hi = -Infinity;
    const take = (v: number) => {
      if (!Number.isFinite(v)) return;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    };
    for (const c of channels) {
      const e = extent(c.ys);
      if (e) { take(e[0]); take(e[1]); }
    }
    for (const g of guides) take(g.at);
    return lo === Infinity ? undefined : padDomain([lo, hi]);
  }, [channels, guides]);

  const xDomain = useMemo<[number, number] | undefined>(() => {
    const t = trace?.t;
    if (!t || t.length === 0) return undefined;
    return [t[0], t[t.length - 1]];
  }, [trace]);

  // A 0.3-second episode and a 28-second one need different tick precision:
  // toFixed(0) turns the first into "0s 0s 0s". Both exist on this disk.
  const secondsTick = useMemo(
    () => secondsTickFormat(xDomain ? xDomain[1] - xDomain[0] : 0),
    [xDomain],
  );

  const note =
    channels.length === 0
      ? null
      : uncalibrated.length === channels.length
        ? "no graded thresholds in this dataset — no guides"
        : uncalibrated.length > 0
          ? `no graded thresholds for ${uncalibrated.map(gripperLabel).join(", ")}`
          : null;

  /** The zoomed plot's box. Measured rather than fixed: the overlay is
   *  whatever the analysis column happens to be tall. */
  const [plotBox, setPlotBox] = useState<HTMLDivElement | null>(null);
  const plotH = useElementHeight(plotBox);

  return (
    <Panel className={zoomed ? "min-h-0 flex-1" : "shrink-0"}>
      <PanelHead
        title="gripper"
        right={channels.length > 0 ? `${channels.length} ch` : "—"}
      >
        {onZoom && (
          <button
            type="button"
            onClick={() => onZoom(!zoomed)}
            aria-label={zoomed ? "restore gripper chart" : "maximize gripper chart"}
            aria-pressed={zoomed}
            title={zoomed ? "back to the tiled layout (esc)" : "fill the analysis column"}
            className="inline-flex h-5.5 w-5.5 shrink-0 items-center justify-center rounded-[3px] text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            {zoomed
              ? <Minimize2 size={11} aria-hidden />
              : <Maximize2 size={11} aria-hidden />}
          </button>
        )}
      </PanelHead>
      <div className={"px-2.5 py-2" + (zoomed ? " flex min-h-0 flex-1 flex-col" : "")}>
        <div ref={setPlotBox} className={zoomed ? "min-h-0 flex-1" : undefined}>
          <ProbeLineChart
            label="gripper position"
            series={series}
            height={zoomed ? plotH : HEIGHT}
            xDomain={xDomain}
            yDomain={yDomain}
            guides={guides}
            playhead={playheadT}
            xTicks={4}
            xTickFormat={secondsTick}
            empty={trace ? "no gripper channel" : "no trace"}
          />
        </div>
        <div className="flex shrink-0 flex-wrap items-center gap-x-2.5 gap-y-1 pl-[42px]">
          <ChartLegend
            items={channels.map((c, i) => ({
              label: gripperLabel(c.name),
              color: seriesColor(i),
            }))}
          />
          {note && (
            <span className="font-mono text-[9px] text-muted-foreground opacity-70">
              {note}
            </span>
          )}
        </div>
      </div>
    </Panel>
  );
}
