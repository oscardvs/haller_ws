"use client";

/**
 * Mouth-clutch calibration. Two captures, mirroring PinchCalibrationStep:
 *   1. "talk" — speak normally, click → the max jawOpen your speech reaches.
 *   2. "open" — hold a deliberate wide open, click → the min sustained value.
 *
 * The gap between them is the entire safety margin. The backend derives
 * t_engage / t_release from it and refuses to arm when the separation is
 * under MIN_SEPARATION — this component mirrors that check so the operator
 * finds out at capture time rather than at start time.
 *
 * Neither this component nor the browser decides anything: the captured
 * numbers are raw MediaPipe blendshape scores, sent as-is.
 */
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

/** Must match safety.MOUTH_MIN_SEPARATION on the backend. */
export const MOUTH_MIN_SEPARATION = 0.25;

export type MouthCalib = {
  talk_max: number | null;
  open_min: number | null;
};

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
  const captureTalk = () => {
    if (liveJawOpen === null) return;
    onChange({ ...value, talk_max: liveJawOpen });
  };
  const captureOpen = () => {
    if (liveJawOpen === null) return;
    onChange({ ...value, open_min: liveJawOpen });
  };

  const both = value.talk_max !== null && value.open_min !== null;
  const separation = both ? value.open_min! - value.talk_max! : null;
  const ready = mouthCalibReady(value);

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
          <Button size="sm" variant="outline" className="h-7 flex-1" onClick={captureTalk}>
            talk · capture
          </Button>
          <Button size="sm" variant="outline" className="h-7 flex-1" onClick={captureOpen}>
            open · capture
          </Button>
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
