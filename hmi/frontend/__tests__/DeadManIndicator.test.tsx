import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { DeadManIndicator } from "@/components/DeadManIndicator";

describe("DeadManIndicator", () => {
  it("prompts for the grip when nothing holds the dead-man", () => {
    render(<DeadManIndicator held={false} trackingLost={false} source="vr_grip" />);
    expect(screen.getByText(/squeeze GRIP/i)).toBeInTheDocument();
  });

  it("shows driving while the clutch is closed", () => {
    render(<DeadManIndicator held trackingLost={false} source="vr_grip" />);
    expect(screen.getByText(/DRIVING/i)).toBeInTheDocument();
  });

  it("tracking loss outranks the prompt", () => {
    render(<DeadManIndicator held={false} trackingLost source="vr_grip" />);
    expect(screen.getByText(/tracking lost/i)).toBeInTheDocument();
  });

  it("counts down while acquiring rather than claiming to drive", () => {
    // During acquisition the clutch IS closed but the arms are still frozen.
    render(
      <DeadManIndicator held trackingLost={false} acquiring remainingMs={600} />,
    );
    expect(screen.getByText(/ACQUIRING/i)).toBeInTheDocument();
    expect(screen.getByText(/0\.6s/)).toBeInTheDocument();
    expect(screen.queryByText(/DRIVING/i)).not.toBeInTheDocument();
  });

  it("surfaces why the clutch will not engage", () => {
    render(
      <DeadManIndicator held={false} trackingLost={false} reason="stale" />,
    );
    expect(screen.getByText(/stale/i)).toBeInTheDocument();
  });

  it("does not dress the resting state up as a fault", () => {
    // vr_grip_mode is what an unsqueezed grip reports every frame, and
    // clutch_open is the same fact in the acquisition vocabulary. Both are the
    // normal resting state, not something to explain to the operator.
    for (const reason of ["vr_grip_mode", "clutch_open"]) {
      const { unmount } = render(
        <DeadManIndicator held={false} trackingLost={false} reason={reason} />,
      );
      expect(screen.queryByText(new RegExp(reason))).not.toBeInTheDocument();
      unmount();
    }
  });

  it("surfaces rather than hides a disengaged clutch still reporting reason=engaged", () => {
    // The backend no longer emits {engaged: false, reason: "engaged"} — the
    // forced-disengage branch sets its own reason now. "engaged" was on the
    // non-blocking list purely to hide that bug; with the bug fixed, this
    // combination means the clutch block and the state machine disagree, and
    // silently swallowing it would hide a real fault from the operator.
    render(
      <DeadManIndicator held={false} trackingLost={false} reason="engaged" />,
    );
    expect(screen.queryByText(/DRIVING/i)).not.toBeInTheDocument();
    expect(screen.getByText(/squeeze GRIP/i)).toBeInTheDocument();
    expect(screen.getByText(/\(engaged\)/i)).toBeInTheDocument();
  });

  it("reports a solo session's absent side instead of a dead-man state", () => {
    // The hand on that side is ignored and the arm is never written, so every
    // other line this chip draws — "tracking lost" included — would be a claim
    // about an arm that is not in the session.
    render(
      <DeadManIndicator held={false} trackingLost reason="no_arm" />,
    );
    expect(screen.getByText(/NOT IN SESSION/i)).toBeInTheDocument();
    expect(screen.queryByText(/tracking lost/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/squeeze GRIP/i)).not.toBeInTheDocument();
  });
});
