"use client";

/**
 * What a training run is doing RIGHT NOW — above the charts, in one strip.
 *
 * The charts answer "what shape is this run". None of them answers "how far
 * in is it, and when does it finish", and that reading was missing from the
 * Train page entirely: the step count lived in the metrics panel's right-hand
 * corner, and every live number was a 10px figure in the corner of a 130px
 * sparkline. An operator who wants to know whether to wait for a run or go and
 * do something else should not have to read a chart to find out.
 *
 * NOTHING HERE IS DEFAULTED TO ZERO. Every tile is the last FINITE sample of a
 * key the run actually logged, and a key the run never logged gets no tile —
 * a `grad_norm` reading 0.000 is a claim about the gradient, not about the
 * log, and the two are the opposite of each other on a run that has just
 * diverged. Same rule the metric grid draws by.
 *
 * ETA is the one derived number. It comes from the run's OWN progress —
 * elapsed wall-clock divided by the fraction of steps done — and NOT from
 * `samples_per_s`, which is instantaneous and swings by two orders of
 * magnitude between the first step (dataloader cold, 0.3 smp/s) and the
 * hundredth (84 smp/s). A number that would have said "4 days" for the first
 * two seconds of every run is worse than no number.
 */
import { useMemo } from "react";

import { plottableMetricKeys, type MetricRow } from "@/lib/lab";
import { Stat } from "@/components/lab/ui";
import { EVAL_COLOR, TRAIN_COLOR, fmtDuration, fmtNum } from "@/components/lab/charts/svg";

/**
 * The keys worth a tile, in the order an operator reads them: is it learning,
 * is it stable, how fast, on what.
 *
 * A SHORTLIST and not every key, deliberately — the grid below already draws
 * all of them with their last value in the corner, and a strip that repeats
 * nine numbers is a second grid rather than a summary. Anything not listed
 * stays a chart.
 */
const TILES: readonly { key: string; label: string; colour?: string }[] = [
  { key: "loss", label: "loss", colour: TRAIN_COLOR },
  { key: "eval_loss", label: "eval loss", colour: EVAL_COLOR },
  { key: "grad_norm", label: "grad norm" },
  { key: "lr", label: "lr" },
  { key: "samples_per_s", label: "smp/s" },
  { key: "gpu_mem_gb", label: "gpu gb" },
];

/** The last finite value of every key, in one backward pass. A live run
 *  appends thousands of rows and this runs on every poll. */
function lastValues(rows: MetricRow[], keys: readonly string[]): Map<string, number> {
  const want = new Set(keys);
  const out = new Map<string, number>();
  for (let i = rows.length - 1; i >= 0 && out.size < want.size; i--) {
    for (const [k, v] of Object.entries(rows[i])) {
      if (!want.has(k) || out.has(k)) continue;
      if (typeof v !== "number" || !Number.isFinite(v)) continue;
      out.set(k, v);
    }
  }
  return out;
}

export function RunVitals({
  rows,
  steps,
  lastStep,
  elapsed,
  live,
}: {
  rows: MetricRow[];
  /** The planned length from the run's spec, or null when it did not say. */
  steps: number | null;
  /** The furthest step any row reports, or null before the first one lands. */
  lastStep: number | null;
  /** Seconds since launch — the poll's clock while running, `finished_at`
   *  after. Null when the run has neither. */
  elapsed: number | null;
  live: boolean;
}) {
  /** Only keys this run can actually be charted on get a tile, so the strip
   *  and the grid below it never disagree about what was logged. */
  const plottable = useMemo(() => new Set(plottableMetricKeys(rows)), [rows]);
  const values = useMemo(
    () => lastValues(rows, TILES.map((t) => t.key)),
    [rows],
  );

  /** Fraction done, and the wall-clock left at the rate the run has averaged
   *  so far. Both null unless the spec said how long the run is: a run with no
   *  planned length has no percentage, and inventing one from the furthest
   *  step seen would read 100% at every moment. */
  const frac =
    steps !== null && steps > 0 && lastStep !== null
      ? Math.min(1, Math.max(0, lastStep / steps))
      : null;
  const eta =
    live && frac !== null && frac > 0 && frac < 1 && elapsed !== null && elapsed > 0
      ? (elapsed * (1 - frac)) / frac
      : null;

  const shown = TILES.filter((t) => plottable.has(t.key) && values.has(t.key));
  if (frac === null && lastStep === null && shown.length === 0) return null;

  return (
    <div className="flex flex-col gap-2 border-t border-border pt-2.5">
      {/* The progress rule. A bar rather than a percentage alone because
          "2.5%" and "25%" are one character apart at 10px and four hours
          apart in fact. */}
      <div className="flex items-center gap-2.5">
        {frac !== null && (
          <span
            role="progressbar"
            aria-label="training progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(frac * 100)}
            className="h-1.5 min-w-0 flex-1 overflow-hidden rounded-full bg-[var(--haller-inset)]"
          >
            <span
              className="block h-full rounded-full transition-[width] duration-500 ease-out"
              style={{
                width: `${frac * 100}%`,
                background: live ? "var(--haller-live)" : "var(--muted-foreground)",
              }}
            />
          </span>
        )}
        <span className="flex shrink-0 items-baseline gap-2.5 font-mono text-[10px] whitespace-nowrap text-muted-foreground">
          <span>
            <span data-num className="tabular-nums text-foreground">
              {lastStep === null ? "—" : Math.round(lastStep).toLocaleString("en-US")}
            </span>
            {steps !== null && (
              <>
                {" / "}
                <span data-num className="tabular-nums">
                  {steps.toLocaleString("en-US")}
                </span>
              </>
            )}
            <span className="label-micro pl-1.5">steps</span>
          </span>
          {frac !== null && (
            <span data-num className="tabular-nums text-foreground">
              {(frac * 100).toFixed(1)}%
            </span>
          )}
          {/* Only while it is going. A finished run's "time left" is zero, and
              printing it invites the question of why it is on screen. */}
          {eta !== null && (
            <span title="at the average rate since launch">
              <span className="label-micro pr-1.5">eta</span>
              <span data-num className="tabular-nums text-foreground">
                {fmtDuration(eta)}
              </span>
            </span>
          )}
        </span>
      </div>

      {shown.length > 0 && (
        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1.5">
          {shown.map((t) => (
            <Stat
              key={t.key}
              label={t.label}
              value={fmtNum(values.get(t.key))}
              colour={t.colour}
            />
          ))}
        </div>
      )}
    </div>
  );
}
