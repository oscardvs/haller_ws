import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { DeadManIndicator } from "@/components/DeadManIndicator";

describe("DeadManIndicator", () => {
  it("names the spacebar when that source holds authority", () => {
    render(<DeadManIndicator held={false} trackingLost={false} source="spacebar" />);
    expect(screen.getByText(/hold SPACE/i)).toBeInTheDocument();
  });

  it("names the mouth when that source holds authority", () => {
    render(<DeadManIndicator held={false} trackingLost={false} source="mouth" />);
    expect(screen.getByText(/open MOUTH/i)).toBeInTheDocument();
  });

  it("shows driving regardless of source", () => {
    render(<DeadManIndicator held trackingLost={false} source="mouth" />);
    expect(screen.getByText(/DRIVING/i)).toBeInTheDocument();
  });

  it("tracking loss outranks the source prompt", () => {
    render(<DeadManIndicator held={false} trackingLost source="mouth" />);
    expect(screen.getByText(/tracking lost/i)).toBeInTheDocument();
  });

  it("surfaces why the mouth clutch will not engage", () => {
    render(
      <DeadManIndicator held={false} trackingLost={false} source="mouth"
                        reason="uncalibrated" />,
    );
    expect(screen.getByText(/uncalibrated/i)).toBeInTheDocument();
  });

  it("does not dress the resting state up as a fault", () => {
    // below_threshold is what a closed mouth reports every frame. It is the
    // normal resting state, not something to explain to the operator.
    render(
      <DeadManIndicator held={false} trackingLost={false} source="mouth"
                        reason="below_threshold" />,
    );
    expect(screen.queryByText(/below_threshold/i)).not.toBeInTheDocument();
  });

  it("surfaces rather than hides a disengaged clutch still reporting reason=engaged", () => {
    // The backend no longer emits {engaged: false, reason: "engaged"} — the
    // forced-disengage branch sets its own reason now. "engaged" was on the
    // non-blocking list purely to hide that bug; with the bug fixed, this
    // combination means the clutch block and the state machine disagree, and
    // silently swallowing it would hide a real fault from the operator.
    render(
      <DeadManIndicator held={false} trackingLost={false} source="mouth"
                        reason="engaged" />,
    );
    expect(screen.queryByText(/DRIVING/i)).not.toBeInTheDocument();
    expect(screen.getByText(/open MOUTH/i)).toBeInTheDocument();
    expect(screen.getByText(/\(engaged\)/i)).toBeInTheDocument();
  });
});
