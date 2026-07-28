"use client";

/**
 * Read-only per-joint bar. Zero tick = 0° (positioned within the range
 * according to `min`/`max`, not necessarily at the midpoint — SO-101 joint
 * limits are frequently asymmetric), side ticks = limits, filled segment
 * spans from the zero tick to `commanded`, ghost tick = `intended` when it
 * diverges.
 *
 * Used in the human-teleop side panel; purely visual.
 */
export function ScopeBar({
  label, min, max, commanded, intended,
}: {
  label: string;
  min: number;
  max: number;
  commanded: number;
  intended?: number;
}) {
  const span = Math.max(max - min, 1e-6);
  const pct = (v: number) => Math.max(0, Math.min(100, ((v - min) / span) * 100));
  const cmdPct = pct(commanded);
  const zeroPct = pct(0);
  const fillLo = Math.min(zeroPct, cmdPct);
  const fillHi = Math.max(zeroPct, cmdPct);
  const intendedPct = intended === undefined ? null : pct(intended);
  const diverged = intended !== undefined && Math.abs(intended - commanded) > 0.5;

  return (
    <div className="flex items-center gap-2 text-[12px] font-mono">
      <span className="w-12 text-muted-foreground">{label}</span>
      <div className="relative h-2 flex-1 rounded-sm border border-border bg-card">
        {/* zero tick */}
        <div className="absolute top-0 bottom-0 w-px bg-border" style={{ left: `${zeroPct}%` }} />
        {/* commanded fill: spans from the zero tick to the commanded value */}
        <div
          data-fill
          className="absolute top-0 bottom-0 bg-[var(--instrument-line,oklch(80%_0.18_142))]"
          style={{
            left: `${fillLo}%`,
            width: `${fillHi - fillLo}%`,
          }}
        />
        {/* ghost tick (intended) */}
        {diverged && intendedPct !== null ? (
          <div
            data-ghost
            className="absolute top-[-2px] bottom-[-2px] w-px bg-foreground/70"
            style={{ left: `${intendedPct}%` }}
          />
        ) : null}
      </div>
      <span className="w-14 text-right tabular-nums">{commanded.toFixed(1)}°</span>
    </div>
  );
}
