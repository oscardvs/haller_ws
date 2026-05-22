"use client";

/**
 * Visual chip for the spacebar-driven dead-man state.
 * - `held`: lime "DRIVING — release to stop"
 * - !held & !lost: muted "DRIVE — hold SPACE"
 * - lost: amber "HOLD — tracking lost"
 */
export function DeadManIndicator({
  held, trackingLost,
}: {
  held: boolean;
  trackingLost: boolean;
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
  return (
    <div className="font-mono text-[12px] px-3 py-1 rounded-sm border border-border text-muted-foreground">
      DRIVE — hold SPACE
    </div>
  );
}
