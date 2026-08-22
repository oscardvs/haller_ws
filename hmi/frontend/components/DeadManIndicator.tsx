"use client";

/**
 * Visual chip for one side's dead-man state.
 *
 * - `no_arm`:  muted "NOT IN SESSION" — outranks everything, because a side
 *              with no arm has no dead-man to report on.
 * - lost:      amber "HOLD — tracking lost"
 * - acquiring: the countdown
 * - `held`:    live "DRIVING — release to stop"
 * - otherwise: the prompt, plus the reason when the reason is a blocker
 *
 * `reason` answers "why isn't it engaging" without a terminal — the same idea
 * as the per-joint reasons. It takes either vocabulary: the clutch block's
 * reason for a session-wide chip, or `acquire[side].reason` for a per-side one.
 */
export type ClutchSource = "vr_grip";

/** Reasons that explain nothing about a side that is simply not driving, so
 *  they are never printed as a blocker:
 *    - "vr_grip_mode" — a controller grip that isn't squeezed. The resting
 *      state, not a fault; the prompt already says GRIP.
 *    - "clutch_open"  — the same fact in the acquisition vocabulary.
 *    - "idle"         — no session; the chip's own text says so.
 *
 *  "engaged" is deliberately NOT on this list. It used to be, to mask a
 *  backend bug where the forced-disengage branch cleared `engaged` without
 *  revisiting the reason the clutch policy had just set — so the chip could
 *  read "DRIVE — squeeze GRIP (engaged)" for one frame. That branch now sets
 *  its own reason, and masking the combination permanently would suppress a
 *  real signal: a disengaged clutch still reporting reason "engaged" means
 *  the clutch block and the state machine disagree, which is worth seeing.
 */
const NON_BLOCKING: readonly string[] = ["vr_grip_mode", "clutch_open", "idle"];

/** The single-arm session's absent side. Not a fault and not a prompt — the
 *  operator chose it at start, and nothing is ever written to that arm. */
const NO_ARM = "no_arm";

export function DeadManIndicator({
  held, trackingLost, reason, acquiring, remainingMs,
}: {
  held: boolean;
  trackingLost: boolean;
  /** Only one clutch source survives; kept as a prop so callers read as
   *  explicit about which one they mean. */
  source?: ClutchSource;
  reason?: string;
  /** The clutch is closed but authority has not transferred yet. */
  acquiring?: boolean;
  /** Countdown to the earliest handover, null once it has run out and the
   *  pose match is what is left. */
  remainingMs?: number | null;
}) {
  // Ranked first: a side the session was started without. Every other line
  // this chip can draw — including "tracking lost", which is trivially true
  // for a hand nobody is reading — would be a claim about an arm that is not
  // in the session at all.
  if (reason === NO_ARM) {
    return (
      <div className="font-mono text-[12px] px-3 py-1 rounded-sm border border-border text-muted-foreground opacity-70">
        NOT IN SESSION — solo, this side is unused
      </div>
    );
  }
  // Tracking loss is ranked BELOW acquiring and driving, because the sides are
  // independent: one hand out of frame while the other is mid-acquisition is
  // the normal single-arm case, and a chip reading "HOLD — tracking lost" over
  // a countdown that is visibly running just tells the operator the chip is
  // wrong. The per-side age pills carry which arm is missing; this only claims
  // the whole session is stalled when nothing at all is under way.
  if (trackingLost && !acquiring && !held) {
    return (
      <div className="font-mono text-[12px] px-3 py-1 rounded-sm border border-[var(--haller-warn)] text-[var(--haller-warn)]">
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
      <div className="font-mono text-[12px] px-3 py-1 rounded-sm border border-[var(--haller-live)]/50 text-[var(--haller-live)]/80">
        ACQUIRING{secs} — hold the pose
      </div>
    );
  }
  if (held) {
    return (
      <div className="font-mono text-[12px] px-3 py-1 rounded-sm border border-[var(--haller-live)] text-[var(--haller-live)] animate-pulse">
        DRIVING — release to stop
      </div>
    );
  }
  const blocked = reason && !NON_BLOCKING.includes(reason);
  return (
    <div className="font-mono text-[12px] px-3 py-1 rounded-sm border border-border text-muted-foreground">
      DRIVE — squeeze GRIP
      {blocked ? ` (${reason})` : ""}
    </div>
  );
}
