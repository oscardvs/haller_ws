// hmi/frontend/components/JointSlider.tsx
"use client";
import { useEffect, useRef, useState } from "react";
import { Slider } from "@/components/ui/slider";

export type JointSliderProps = {
  name: string;
  pos: number;       // current observed position (deg)
  min: number;       // calibrated min (deg)
  max: number;       // calibrated max (deg)
  onChange: (value: number) => void;
  disabled?: boolean;
};

export function JointSlider({ name, pos, min, max, onChange, disabled }: JointSliderProps) {
  // Locally controlled while user drags; snaps back to `pos` when released.
  const [local, setLocal] = useState<number | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // If the upstream telemetry changes more than 5° from our local value, accept it
  useEffect(() => {
    if (local === null) return;
    if (Math.abs(local - pos) > 5) setLocal(null);
  }, [pos, local]);

  const value = local ?? pos;

  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-mono">{name}</span>
        <span className="text-xs font-mono text-muted-foreground">
          {value.toFixed(1)}° <span className="opacity-50">/ [{min.toFixed(0)}, {max.toFixed(0)}]</span>
        </span>
      </div>
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
      />
    </div>
  );
}
