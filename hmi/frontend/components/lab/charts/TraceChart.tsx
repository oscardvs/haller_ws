"use client";

/**
 * The joint traces, one row per arm.
 *
 * Everything here is driven by `trace.names`. The channel COUNT varies — 6 on
 * the solo dataset, 12 on the bimanual one — so `armGroups` splits the columns
 * into rows and nothing counts joints. The gripper is excluded by name and
 * charted separately: a 0..100 gripper sharing an axis with a ±180° joint
 * flattens the joint into the baseline, which is how the kit's chart managed
 * to show a dead arm as a straight line and call it normal.
 *
 * The dashed overlay is the thing the kit fetched every episode and threw
 * away: `action` is what teleop commanded, `state` is what the arm did, and
 * the gap between them is tracking error. It is the most useful reading on
 * this panel and it cost one more series per channel to draw.
 */
import { useMemo } from "react";

import { armGroups, isDrawableTrace, isGripperChannel, shortChannel, type Trace } from "@/lib/lab";
import { Chip, Panel, PanelHead } from "@/components/lab/ui";
import { ChartLegend, LineChart, type Series } from "./LineChart";
import { extent, padDomain, secondsTickFormat, seriesColor } from "./svg";

/** Two arms have to fit above the fold with the player; one arm gets the room
 *  it frees. Both keep the same PAD, so the plot areas line up down the
 *  column. */
const HEIGHT_ONE = 128;
const HEIGHT_MANY = 92;

const sideLabel = (side: string) => (side === "arm" ? "arm" : `${side} arm`);

export function TraceChart({
  trace: given,
  playheadT,
  overlay,
  onOverlay,
}: {
  trace: Trace | null;
  /** Episode-relative seconds from the player, or null when nothing plays. */
  playheadT: number | null;
  overlay: boolean;
  onOverlay?: (v: boolean) => void;
}): React.ReactElement {
  // A partial body arriving with a 200 is "no trace", not a render-phase throw
  // that takes the review pane down with it. See `isDrawableTrace`.
  const trace = isDrawableTrace(given) ? given : null;

  const groups = useMemo(() => {
    if (!trace) return [];
    return armGroups(trace.names)
      .map((g) => ({
        side: g.side,
        channels: g.channels.filter((i) => !isGripperChannel(trace.names[i])),
      }))
      .filter((g) => g.channels.length > 0);
  }, [trace]);

  // One y domain across every row so the two arms are comparable, and one x
  // domain so the playhead lands on the same instant in both.
  const yDomain = useMemo<[number, number] | undefined>(() => {
    if (!trace) return undefined;
    let lo = Infinity;
    let hi = -Infinity;
    const take = (row: number[] | undefined) => {
      const e = extent(row ?? []);
      if (!e) return;
      if (e[0] < lo) lo = e[0];
      if (e[1] > hi) hi = e[1];
    };
    for (const g of groups) {
      for (const ci of g.channels) {
        take(trace.state[ci]);
        if (overlay) take(trace.action[ci]);
      }
    }
    return lo === Infinity ? undefined : padDomain([lo, hi]);
  }, [trace, groups, overlay]);

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

  const channels = groups.reduce((n, g) => n + g.channels.length, 0);
  const height = groups.length > 1 ? HEIGHT_MANY : HEIGHT_ONE;

  return (
    <Panel className="min-h-0 flex-1">
      <PanelHead
        title="traces"
        right={trace ? `${channels} ch · ${trace.t.length} pts` : "—"}
      >
        {onOverlay && (
          <Chip
            on={overlay}
            disabled={!trace}
            onClick={() => onOverlay(!overlay)}
            title="commanded action, dashed, over measured state — the gap is tracking error"
          >
            action overlay
          </Chip>
        )}
      </PanelHead>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {trace === null || groups.length === 0 ? (
          <div className="px-2.5 py-2">
            <LineChart
              label="joint traces"
              series={[]}
              height={HEIGHT_ONE}
              empty={trace ? "no joint channels" : "no trace"}
            />
          </div>
        ) : (
          groups.map((g, gi) => {
            const series: Series[] = [];
            g.channels.forEach((ci, k) => {
              const name = trace.names[ci];
              series.push({
                id: `state-${ci}`,
                label: shortChannel(name),
                // Coloured by position WITHIN the group, so the same joint is
                // the same colour on both arms.
                color: seriesColor(k),
                xs: trace.t,
                ys: trace.state[ci] ?? [],
              });
              if (overlay) {
                series.push({
                  id: `action-${ci}`,
                  label: `${shortChannel(name)} cmd`,
                  color: seriesColor(k),
                  xs: trace.t,
                  ys: trace.action[ci] ?? [],
                  dashed: true,
                  opacity: 0.55,
                });
              }
            });

            const last = gi === groups.length - 1;
            return (
              <div
                key={g.side}
                className="flex min-w-0 items-start gap-2 border-b border-border px-2.5 py-2 last:border-b-0"
              >
                <span className="label-micro w-[52px] shrink-0 pt-2 text-muted-foreground">
                  {sideLabel(g.side)}
                </span>
                <div className="min-w-0 flex-1">
                  <LineChart
                    label={`${sideLabel(g.side)} joint traces, degrees`}
                    series={series}
                    height={height}
                    xDomain={xDomain}
                    yDomain={yDomain}
                    playhead={playheadT}
                    yTickFormat={(v) => `${Math.round(v)}°`}
                    // Only the bottom row needs the seconds axis; the padding
                    // is reserved on every row regardless, so they stay aligned.
                    xTicks={last ? 4 : 0}
                    xTickFormat={secondsTick}
                    empty="no trace"
                  />
                  <ChartLegend
                    className="pl-[42px]"
                    items={[
                      ...g.channels.map((ci, k) => ({
                        label: shortChannel(trace.names[ci]),
                        color: seriesColor(k),
                      })),
                      ...(overlay
                        ? [{
                            label: "action",
                            color: "var(--muted-foreground)",
                            dashed: true,
                          }]
                        : []),
                    ]}
                  />
                </div>
              </div>
            );
          })
        )}
      </div>
    </Panel>
  );
}
