"use client";

/**
 * Per-hand pinch calibration. Two captures:
 *   1. "Open"  — hold hand open, click → max thumb-index distance.
 *   2. "Pinch" — pinch fully closed, click → min thumb-index distance.
 *
 * The caller passes a `liveDistance` (current thumb-index distance in metres,
 * derived from MediaPipe in the parent), and receives onChange callbacks when
 * a side captures its values.
 */
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export type PinchSide = "left" | "right";

export type PinchCalib = {
  min_m: number | null;
  max_m: number | null;
};

export function PinchCalibrationStep({
  liveDistance,
  side,
  value,
  onChange,
}: {
  liveDistance: number | null;
  side: PinchSide;
  value: PinchCalib;
  onChange: (next: PinchCalib) => void;
}) {
  const captureOpen = () => {
    if (liveDistance === null) return;
    onChange({ ...value, max_m: liveDistance });
  };
  const capturePinch = () => {
    if (liveDistance === null) return;
    onChange({ ...value, min_m: liveDistance });
  };
  const ready = value.min_m !== null && value.max_m !== null
    && value.max_m > value.min_m;

  return (
    <Card className="p-0">
      <CardContent className="p-3 flex flex-col gap-2 text-[12px] font-mono">
        <div className="flex justify-between">
          <span className="text-muted-foreground">side</span>
          <span>{side}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">live</span>
          <span>{liveDistance === null ? "—" : liveDistance.toFixed(3) + " m"}</span>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="outline" className="h-7 flex-1" onClick={captureOpen}>
            open · capture
          </Button>
          <Button size="sm" variant="outline" className="h-7 flex-1" onClick={capturePinch}>
            pinch · capture
          </Button>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">min..max</span>
          <span>
            {value.min_m === null ? "—" : value.min_m.toFixed(3)}
            {" .. "}
            {value.max_m === null ? "—" : value.max_m.toFixed(3)}
            {ready ? "" : " (incomplete)"}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
