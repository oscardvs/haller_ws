"use client";

/**
 * Visual chip for the dead-man state, whichever source holds authority.
 * - `held`: lime "DRIVING — release to stop"
 * - !held & !lost: muted prompt naming the armed source
 * - lost: amber "HOLD — tracking lost" (outranks everything else)
 *
 * `reason` comes from the backend's clutch block and answers "why isn't it
 * engaging" without a terminal — the same idea as the per-joint reasons.
 */
export type ClutchSource = "spacebar" | "mouth";

/** Clutch reasons that explain nothing about a *disengaged* clutch, so they
 *  are never printed as a blocker:
 *    - "below_threshold" — a closed mouth. The resting state, not a fault.
 *    - "spacebar_mode"   — not a fault either; the prompt already says SPACE.
 *
 *  "engaged" is deliberately NOT on this list. It used to be, to mask a
 *  backend bug where the forced-disengage branch cleared `engaged` without
 *  revisiting the reason the mouth policy had just set — so the chip could
 *  read "DRIVE — open MOUTH (engaged)" for one frame. That branch now sets
 *  its own reason, and masking the combination permanently would suppress a
 *  real signal: a disengaged clutch still reporting reason "engaged" means
 *  the clutch block and the state machine disagree, which is worth seeing.
 */
const NON_BLOCKING: readonly string[] = ["below_threshold", "spacebar_mode"];

export function DeadManIndicator({
  held, trackingLost, source = "spacebar", reason,
}: {
  held: boolean;
  trackingLost: boolean;
  source?: ClutchSource;
  reason?: string;
}) {
  if (trackingLost) {
    return (
      <div className="font-mono text-[12px] px-3 py-1 rounded-sm border border-amber-500 text-amber-500">
        HOLD — tracking lost
      </div>
    );
  }
  if (held) {
    return (
      <div className="font-mono text-[12px] px-3 py-1 rounded-sm border border-[var(--instrument-line,oklch(80%_0.18_142))] text-[var(--instrument-line,oklch(80%_0.18_142))] animate-pulse">
        DRIVING — release to stop
      </div>
    );
  }
  const prompt = source === "mouth" ? "open MOUTH" : "hold SPACE";
  const blocked = reason && !NON_BLOCKING.includes(reason);
  return (
    <div className="font-mono text-[12px] px-3 py-1 rounded-sm border border-border text-muted-foreground">
      DRIVE — {prompt}
      {blocked ? ` (${reason})` : ""}
    </div>
  );
}
