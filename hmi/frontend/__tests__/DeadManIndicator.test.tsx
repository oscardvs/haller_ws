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

  it("never contradicts itself when the backend reports reason=engaged while disengaged", () => {
    // A mid-session source switch makes the backend emit
    // {engaged: false, reason: "engaged"} for exactly one frame: the mouth
    // policy sets the reason, then the forced-disengage branch clears
    // `engaged` without revisiting it. The chip must not claim to be driving
    // and must not print "(engaged)" underneath "DRIVE".
    render(
      <DeadManIndicator held={false} trackingLost={false} source="mouth"
                        reason="engaged" />,
    );
    expect(screen.queryByText(/DRIVING/i)).not.toBeInTheDocument();
    expect(screen.getByText(/open MOUTH/i)).toBeInTheDocument();
    expect(screen.queryByText(/\(engaged\)/i)).not.toBeInTheDocument();
  });
});
