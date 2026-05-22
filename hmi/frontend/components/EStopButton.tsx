// hmi/frontend/components/EStopButton.tsx
"use client";
import { api } from "@/lib/api";
import { useTelemetry } from "@/lib/telemetry";
import { toast } from "sonner";

/**
 * EStopButton — the most important affordance in the HMI.
 *
 *  - ALWAYS bright red, full opacity. Never visually disabled.
 *  - When alerts are present in telemetry, pulses to draw the eye.
 *  - Mechanical-looking pill, never a rounded blob; corner registration marks
 *    reinforce the "panel button" mental model.
 */
export function EStopButton({ className }: { className?: string }) {
  // Subscribe via a primitive selector so React's snapshot cache stays stable.
  const hasAlerts = useTelemetry((s) => (s.lastFrame?.alerts.length ?? 0) > 0);

  const handleClick = async () => {
    try {
      await api.estop();
      toast.error("E-STOP triggered — torque off, base zeroed");
    } catch (e) {
      toast.error(`E-STOP failed: ${(e as Error).message}`);
    }
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      aria-label="Emergency stop"
      className={[
        // Bright red. Full opacity, never disabled.
        "relative isolate inline-flex items-center gap-2 select-none",
        "h-8 px-3 rounded-[3px]",
        "bg-red-600 text-white",
        "font-mono text-[12px] font-bold tracking-[0.22em] uppercase",
        // Hard mechanical edge — outer ring + inset highlight.
        "border border-red-900",
        "shadow-[inset_0_1px_0_rgba(255,255,255,0.25),inset_0_-1px_0_rgba(0,0,0,0.4),0_0_0_1px_rgba(0,0,0,0.6)]",
        // Hover/active feedback. Stays opaque always.
        "hover:bg-red-500 active:translate-y-px active:bg-red-700",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-400 focus-visible:ring-offset-2 focus-visible:ring-offset-background",
        hasAlerts ? "animate-haller-pulse" : "",
        className ?? "",
      ].join(" ")}
    >
      <span
        aria-hidden
        className="inline-block h-2 w-2 rounded-full bg-white shadow-[0_0_8px_rgba(255,255,255,0.9)]"
      />
      E-STOP
    </button>
  );
}
