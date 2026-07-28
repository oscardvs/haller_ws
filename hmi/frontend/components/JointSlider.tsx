// hmi/frontend/components/JointSlider.tsx
"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { Slider } from "@/components/ui/slider";

export type JointSliderProps = {
  name: string;
  pos: number;       // measured position from telemetry (deg)
  min: number;       // calibrated min (deg)
  max: number;       // calibrated max (deg)
  onChange: (value: number) => void;
  disabled?: boolean;
  /** Compact single-row form used by the cockpit's arm cards. */
  dense?: boolean;
};

/** Close enough to call the joint "arrived" and hand the thumb back to
 *  telemetry. Loose enough to survive servo dither, tight enough that the
 *  handover is invisible. */
const ARRIVED_DEG = 1.5;
/** Hard deadline on the reconcile. A joint that is blocked, unpowered or
 *  fighting a limit never arrives, and a thumb frozen at an unreachable
 *  commanded value forever is worse than one that admits where the arm is. */
const RECONCILE_MS = 2500;
/** Coalesce the POST while the thumb is moving. */
const SEND_DEBOUNCE_MS = 50;

/**
 * JointSlider — calibrated range track with a home (0°) tick.
 *
 *  The thumb is the COMMANDED position, held optimistically:
 *
 *    idle      → thumb follows telemetry
 *    dragging  → thumb is the operator's value, telemetry is ignored outright
 *    released  → thumb holds the commanded value until the arm reaches it
 *                (±ARRIVED_DEG) or RECONCILE_MS expires, then follows again
 *
 *  The "ignored outright" is the point. This used to drop the local value
 *  whenever telemetry diverged by more than 5°, which is precisely what
 *  happens during any fast drag: the arm lags, the guard fires, and the thumb
 *  snaps backwards out from under the pointer. Dragging is the one moment the
 *  feed must not win.
 *
 *  There is only one marker. The backend publishes a single number per joint,
 *  so a second "measured" tick would either sit under the thumb (idle) or
 *  imply a distinction the protocol does not carry.
 */
export function JointSlider({
  name,
  pos,
  min,
  max,
  onChange,
  disabled,
  dense,
}: JointSliderProps) {
  const [local, setLocal] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const sendTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const deadline = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Hand the thumb back once the arm has arrived. Adjusted during render
  // rather than in an effect: this is React's documented "reset state when a
  // prop changes" case, and doing it here means the released frame never
  // paints the commanded value one tick after the arm already matched it.
  if (local !== null && !dragging && Math.abs(local - pos) <= ARRIVED_DEG) {
    setLocal(null);
  }

  useEffect(
    () => () => {
      if (sendTimer.current) clearTimeout(sendTimer.current);
      if (deadline.current) clearTimeout(deadline.current);
    },
    [],
  );

  const value = local ?? pos;
  const commanding = local !== null;

  const range = max - min;
  const homePct = useMemo(
    () =>
      range > 0 && min <= 0 && max >= 0
        ? Math.max(0, Math.min(100, ((0 - min) / range) * 100))
        : null,
    [min, max, range],
  );

  const commit = (v: number) => {
    setDragging(false);
    if (sendTimer.current) clearTimeout(sendTimer.current);
    onChange(v);
    if (deadline.current) clearTimeout(deadline.current);
    // Backstop for a joint that never arrives — blocked, unpowered, or fighting
    // a limit. Better to admit where the arm is than to hold a lie forever.
    deadline.current = setTimeout(() => setLocal(null), RECONCILE_MS);
  };

  return (
    <div className={dense ? "" : "space-y-1.5"}>
      {!dense && (
        <div className="flex items-baseline justify-between gap-2">
          <span className="font-mono text-[11px] tracking-[0.04em] text-foreground/85">
            {name}
          </span>
          <span className="font-mono text-[11px] text-muted-foreground tabular-nums">
            <span
              className={commanding ? "text-[var(--haller-live)]" : "text-foreground"}
              data-num
            >
              {value.toFixed(1)}°
            </span>
            <span className="text-muted-foreground/60">
              {"  "}[{min.toFixed(0)}, {max.toFixed(0)}]
            </span>
          </span>
        </div>
      )}

      <div
        className={
          dense
            ? "grid h-[26px] grid-cols-[88px_minmax(0,1fr)_54px] items-center gap-2"
            : "relative h-5"
        }
      >
        {dense && (
          <span
            className="truncate font-mono text-[10px] text-foreground/80"
            title={`${name} [${min.toFixed(0)}, ${max.toFixed(0)}]`}
          >
            {name}
          </span>
        )}
        <div className="relative h-5">
          {homePct !== null && (
            <span
              aria-hidden
              className="absolute top-1/2 z-1 h-3 w-px -translate-y-1/2 bg-muted-foreground/60"
              style={{ left: `${homePct}%` }}
            />
          )}
          <Slider
            min={min}
            max={max}
            step={0.5}
            value={[value]}
            aria-label={name}
            onValueChange={(next) => {
              const v = Array.isArray(next) ? next[0] : (next as number);
              setDragging(true);
              if (deadline.current) clearTimeout(deadline.current);
              setLocal(v);
              if (sendTimer.current) clearTimeout(sendTimer.current);
              sendTimer.current = setTimeout(() => onChange(v), SEND_DEBOUNCE_MS);
            }}
            onValueCommitted={(next) => {
              const v = Array.isArray(next) ? next[0] : (next as number);
              commit(v);
            }}
            disabled={disabled}
            className="relative z-2"
          />
        </div>
        {dense && (
          <span
            data-num
            className={
              "text-right font-mono text-[10px] tabular-nums " +
              (disabled
                ? "text-muted-foreground"
                : commanding
                  ? "text-[var(--haller-live)]"
                  : "text-foreground")
            }
          >
            {value.toFixed(1)}°
          </span>
        )}
      </div>
    </div>
  );
}
