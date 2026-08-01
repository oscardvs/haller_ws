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
export type ClutchSource = "spacebar" | "mouth" | "vr_grip";

/** Clutch reasons that explain nothing about a *disengaged* clutch, so they
 *  are never printed as a blocker:
 *    - "below_threshold" — a closed mouth. The resting state, not a fault.
 *    - "spacebar_mode"   — not a fault either; the prompt already says SPACE.
 *    - "vr_grip_mode"    — same, for a controller grip that simply isn't held.
 *
 *  "engaged" is deliberately NOT on this list. It used to be, to mask a
 *  backend bug where the forced-disengage branch cleared `engaged` without
 *  revisiting the reason the mouth policy had just set — so the chip could
 *  read "DRIVE — open MOUTH (engaged)" for one frame. That branch now sets
 *  its own reason, and masking the combination permanently would suppress a
 *  real signal: a disengaged clutch still reporting reason "engaged" means
 *  the clutch block and the state machine disagree, which is worth seeing.
 */
const NON_BLOCKING: readonly string[] = [
  "below_threshold", "spacebar_mode", "vr_grip_mode",
];

export function DeadManIndicator({
  held, trackingLost, source = "spacebar", reason, acquiring, remainingMs,
}: {
  held: boolean;
  trackingLost: boolean;
  source?: ClutchSource;
  reason?: string;
  /** The clutch is closed but authority has not transferred yet. */
  acquiring?: boolean;
  /** Countdown to the earliest handover, null once it has run out and the
   *  pose match is what is left. */
  remainingMs?: number | null;
}) {
  // Tracking loss is ranked BELOW acquiring and driving, because the sides are
  // independent: one hand out of frame while the other is mid-acquisition is
  // the normal single-arm case, and a chip reading "HOLD — tracking lost" over
  // a countdown that is visibly running just tells the operator the chip is
  // wrong. The per-side age pills carry which arm is missing; this only claims
  // the whole session is stalled when nothing at all is under way.
  if (trackingLost && !acquiring && !held) {
    return (
      <div className="font-mono text-[12px] px-3 py-1 rounded-sm border border-amber-500 text-amber-500">
        HOLD — tracking lost
      </div>
    );
  }
  // Ranked above `held` because during acquisition the clutch IS closed. The
  // operator has to be able to tell "asking" from "driving" at a glance —
  // reading a chip that says DRIVING while the arms are still frozen is how
  // someone learns to distrust the chip.
  if (acquiring) {
    const secs = remainingMs != null && remainingMs > 0
      ? ` ${(remainingMs / 1000).toFixed(1)}s`
      : "";
    return (
      <div className="font-mono text-[12px] px-3 py-1 rounded-sm border border-[var(--instrument-line,oklch(80%_0.18_142))]/50 text-[var(--instrument-line,oklch(80%_0.18_142))]/80">
        ACQUIRING{secs} — hold the pose
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
