"use client";

/**
 * Mouth-clutch calibration. Two capture windows, mirroring PinchCalibrationStep:
 *   1. "talk" — start the window, speak normally for a few seconds, stop it.
 *      What is recorded is the MAX jawOpen your speech reached.
 *   2. "open" — start the window, hold a deliberate wide open, stop it.
 *      What is recorded is the MIN value you sustained.
 *
 * A window, not a click. An instantaneous sample is not what either capture
 * means, and the two directions fail in opposite directions:
 *
 *   - `open_min` sampled at an instant errs toward a PEAK, so the gap looks
 *     wider than it is and engaging gets harder. That fails safe.
 *   - `talk_max` sampled at an instant errs toward a TROUGH — nobody clicks at
 *     the peak of their own speech envelope, and the score only updates around
 *     10 Hz — so the gap again looks wider, which puts t_engage BELOW the
 *     operator's real speech maximum. That fails UNSAFE, and it undoes the
 *     speech-resistance the whole feature rests on.
 *
 * The gap between the two numbers is the entire safety margin. The backend
 * derives t_engage / t_release from it and refuses to arm when the separation
 * is under MIN_SEPARATION — this component mirrors that check so the operator
 * finds out at capture time rather than at start time.
 *
 * Neither this component nor the browser decides anything: the captured
 * numbers are raw MediaPipe blendshape scores, sent as-is.
 */
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

/** Must match safety.MOUTH_MIN_SEPARATION on the backend. */
export const MOUTH_MIN_SEPARATION = 0.25;

export type MouthCalib = {
  talk_max: number | null;
  open_min: number | null;
};

type Phase = "talk" | "open";

export function mouthCalibReady(v: MouthCalib): boolean {
  return (
    v.talk_max !== null &&
    v.open_min !== null &&
    v.open_min - v.talk_max >= MOUTH_MIN_SEPARATION
  );
}

export function MouthClutchCalibration({
  liveJawOpen,
  value,
  onChange,
}: {
  /** Current jawOpen score [0,1], or null when no face is tracked. */
  liveJawOpen: number | null;
  value: MouthCalib;
  onChange: (next: MouthCalib) => void;
}) {
  const [capturing, setCapturing] = useState<Phase | null>(null);
  const [running, setRunning] = useState<number | null>(null);

  // Fold every sample that arrives during an open window into the running
  // extreme. The parent re-renders this component on each jaw sample, so "the
  // window" is simply every render between start and stop. Null samples (no
  // face this frame) are skipped rather than ending the window — a blink or a
  // decimated frame must not truncate a capture.
  //
  // Folding with max/min rather than assigning is also what makes the effect
  // safe to run twice, as React does in development.
  useEffect(() => {
    if (capturing === null || liveJawOpen === null) return;
    setRunning((prev) =>
      prev === null
        ? liveJawOpen
        : capturing === "talk"
          ? Math.max(prev, liveJawOpen)
          : Math.min(prev, liveJawOpen),
    );
  }, [capturing, liveJawOpen]);

  const toggle = (phase: Phase) => {
    if (capturing === null) {
      setRunning(null);
      setCapturing(phase);
      return;
    }
    if (capturing !== phase) return;   // the other button is disabled anyway
    setCapturing(null);
    setRunning(null);
    // No face was tracked for the whole window: there is nothing to record,
    // and a stale previous capture is better than an invented one.
    if (running === null) return;
    onChange(
      phase === "talk"
        ? { ...value, talk_max: running }
        : { ...value, open_min: running },
    );
  };

  const both = value.talk_max !== null && value.open_min !== null;
  const separation = both ? value.open_min! - value.talk_max! : null;
  const ready = mouthCalibReady(value);
  const label = (phase: Phase, verb: string) =>
    capturing === phase
      ? `${phase} · stop${running === null ? "" : ` (${running.toFixed(2)})`}`
      : `${phase} · ${verb}`;

  return (
    <Card className="p-0">
      <CardContent className="p-3 flex flex-col gap-2 text-[12px] font-mono">
        <div className="flex justify-between">
          <span className="text-muted-foreground">clutch</span>
          <span>mouth</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">live jaw</span>
          <span className="tabular-nums">
            {liveJawOpen === null ? "—" : liveJawOpen.toFixed(2)}
          </span>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="outline" className="h-7 flex-1"
                  disabled={capturing === "open"}
                  onClick={() => toggle("talk")}>
            {label("talk", "capture")}
          </Button>
          <Button size="sm" variant="outline" className="h-7 flex-1"
                  disabled={capturing === "talk"}
                  onClick={() => toggle("open")}>
            {label("open", "capture")}
          </Button>
        </div>
        <div className="text-muted-foreground">
          {capturing === null
            ? "start already talking, or already open — not before. the window folds from your first sample, so opening after you click reads as closed"
            : capturing === "talk"
              ? "speak normally — recording the loudest your jaw gets"
              : "hold a deliberate wide open — recording the least you sustain"}
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">talk..open</span>
          <span className="tabular-nums">
            {value.talk_max === null ? "—" : value.talk_max.toFixed(2)}
            {" .. "}
            {value.open_min === null ? "—" : value.open_min.toFixed(2)}
          </span>
        </div>
        {both && !ready ? (
          <div className="text-[var(--instrument-warn,oklch(75%_0.16_70))]">
            too close ({separation!.toFixed(2)} &lt; {MOUTH_MIN_SEPARATION}) — open
            wider or speak quieter; the clutch will not arm
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
