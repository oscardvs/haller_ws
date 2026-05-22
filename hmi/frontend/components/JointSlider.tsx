// hmi/frontend/components/JointSlider.tsx
"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { Slider } from "@/components/ui/slider";

export type JointSliderProps = {
  name: string;
  pos: number;       // current observed position (deg)
  min: number;       // calibrated min (deg)
  max: number;       // calibrated max (deg)
  onChange: (value: number) => void;
  disabled?: boolean;
};

/**
 * JointSlider — calibrated range track with home (0°) tick.
 *
 *  Visual model:
 *    [-min ============== 0 (home tick) ============== +max]
 *                              ▲ thumb at current pos
 *
 *  - Whole calibrated range fills the track (the shadcn slider already does
 *    this since min/max are the calibrated limits). We add a 0° home tick
 *    and a "telemetry ghost" tick that shows the observed position even
 *    while the operator is dragging the thumb elsewhere.
 *  - Numeric readout in mono with tabular figures; current value highlighted.
 */
export function JointSlider({
  name,
  pos,
  min,
  max,
  onChange,
  disabled,
}: JointSliderProps) {
  // Locally controlled while user drags; snaps back to `pos` when released.
  const [local, setLocal] = useState<number | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // If the upstream telemetry changes more than 5° from our local value, accept it
  useEffect(() => {
    if (local === null) return;
    if (Math.abs(local - pos) > 5) setLocal(null);
  }, [pos, local]);

  const value = local ?? pos;
  const dragging = local !== null;

  // Helpers for positioning ticks/markers within the track (0..100%).
  const range = max - min;
  const pct = useMemo(() => {
    const clamp = (v: number) =>
      Math.max(0, Math.min(100, ((v - min) / range) * 100));
    return {
      home: range > 0 && min <= 0 && max >= 0 ? clamp(0) : null,
      ghost: clamp(pos),
    };
  }, [min, max, pos, range]);

  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-2">
        <span className="font-mono text-[11px] tracking-[0.04em] text-foreground/85">
          {name}
        </span>
        <span className="font-mono text-[11px] text-muted-foreground tabular-nums">
          <span
            className={
              dragging
                ? "text-[var(--haller-live)]"
                : "text-foreground"
            }
            data-num
          >
            {value.toFixed(1)}°
          </span>
          <span className="text-muted-foreground/60">
            {"  "}[{min.toFixed(0)}, {max.toFixed(0)}]
          </span>
        </span>
      </div>

      {/* Track wrapper — Slider sits on top of decorative ticks. */}
      <div className="relative h-5">
        {/* Home tick at 0° (only if the range includes zero) */}
        {pct.home !== null && (
          <span
            aria-hidden
            className="absolute top-1/2 -translate-y-1/2 z-[1] h-3 w-px bg-muted-foreground/60"
            style={{ left: `${pct.home}%` }}
          />
        )}
        {/* Telemetry ghost (where the joint actually IS) — diamond marker */}
        <span
          aria-hidden
          className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 z-[1] h-1.5 w-1.5 rotate-45 bg-[var(--haller-warn)]/80 ring-2 ring-background"
          style={{ left: `${pct.ghost}%` }}
          title={`telemetry pos=${pos.toFixed(1)}°`}
        />
        <Slider
          min={min}
          max={max}
          step={0.5}
          value={[value]}
          onValueChange={(next) => {
            const v = Array.isArray(next) ? next[0] : (next as number);
            setLocal(v);
            if (timer.current) clearTimeout(timer.current);
            timer.current = setTimeout(() => onChange(v), 50);
          }}
          onValueCommitted={() => {
            setLocal(null);
          }}
          disabled={disabled}
          className="relative z-[2]"
        />
      </div>
    </div>
  );
}
