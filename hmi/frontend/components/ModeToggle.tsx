// hmi/frontend/components/ModeToggle.tsx
"use client";
import { api } from "@/lib/api";
import { toast } from "sonner";

type Mode = "auto" | "manual" | "stop";
const ORDER: Mode[] = ["auto", "manual", "stop"];

/**
 * Three-state mode toggle. Reads as a segmented control;
 * active segment is filled with the mode's status color.
 *
 *  auto   → emerald + radar/ping (VLA is driving)
 *  manual → blue (operator has the stick)
 *  stop   → red (torque off)
 */
export function ModeToggle({
  armId,
  mode,
  showPill = true,
}: {
  armId: string;
  mode: Mode;
  /** The cockpit's arm-card header is 34px and already colours the segmented
   *  control by mode, so the pill there is a second copy of the same fact in
   *  space the joint stack wants. Deep-link pages keep it. */
  showPill?: boolean;
}) {
  return (
    <div className="inline-flex items-center gap-2">
      {showPill && <ModePill mode={mode} />}
      <div
        role="group"
        aria-label="Arm mode"
        className="inline-flex rounded-[3px] border border-border bg-card overflow-hidden"
      >
        {ORDER.map((m, i) => {
          const active = m === mode;
          return (
            <button
              key={m}
              type="button"
              onClick={async () => {
                try {
                  await api.armMode(armId, m);
                } catch (e) {
                  toast.error(`mode change failed: ${(e as Error).message}`);
                }
              }}
              className={[
                "label-micro px-2 h-6 min-w-[52px]",
                "transition-colors",
                i > 0 ? "border-l border-border" : "",
                active
                  ? modeActiveClass(m)
                  : "text-muted-foreground hover:text-foreground hover:bg-muted/50",
              ].join(" ")}
            >
              {m}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function modeActiveClass(m: Mode) {
  switch (m) {
    case "auto":
      return "bg-[color:var(--haller-live-soft)] text-[var(--haller-live)]";
    case "manual":
      return "bg-[oklch(0.66_0.16_245/0.18)] text-[var(--haller-manual)]";
    case "stop":
      return "bg-[oklch(0.62_0.245_27/0.18)] text-[var(--haller-fault)]";
  }
}

function ModePill({ mode }: { mode: Mode }) {
  const dot =
    mode === "auto"
      ? "bg-[var(--haller-live)]"
      : mode === "manual"
      ? "bg-[var(--haller-manual)]"
      : "bg-[var(--haller-fault)]";
  const text =
    mode === "auto"
      ? "text-[var(--haller-live)]"
      : mode === "manual"
      ? "text-[var(--haller-manual)]"
      : "text-[var(--haller-fault)]";

  return (
    <span className="inline-flex items-center gap-1.5 px-2 h-6 rounded-[3px] border border-border bg-card">
      <span className="relative inline-flex h-1.5 w-1.5">
        {mode === "auto" && (
          <span className="absolute inline-flex h-full w-full rounded-full bg-[var(--haller-live)] opacity-70 animate-haller-ping" />
        )}
        <span className={`relative inline-flex h-1.5 w-1.5 rounded-full ${dot}`} />
      </span>
      <span className={`label-micro ${text}`}>{mode}</span>
    </span>
  );
}
