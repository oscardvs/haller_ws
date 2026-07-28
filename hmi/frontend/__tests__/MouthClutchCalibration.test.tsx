import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { MouthClutchCalibration, mouthCalibReady } from "@/components/MouthClutchCalibration";

describe("mouthCalibReady", () => {
  it("is false until both captures exist", () => {
    expect(mouthCalibReady({ talk_max: null, open_min: null })).toBe(false);
    expect(mouthCalibReady({ talk_max: 0.1, open_min: null })).toBe(false);
    expect(mouthCalibReady({ talk_max: null, open_min: 0.9 })).toBe(false);
  });

  it("is false when speech overlaps the deliberate open", () => {
    // Separation 0.05 — the backend would refuse to arm on this.
    expect(mouthCalibReady({ talk_max: 0.50, open_min: 0.55 })).toBe(false);
  });

  it("is true with adequate separation", () => {
    expect(mouthCalibReady({ talk_max: 0.10, open_min: 0.90 })).toBe(true);
  });

  it("is false when open is below talk", () => {
    expect(mouthCalibReady({ talk_max: 0.90, open_min: 0.10 })).toBe(false);
  });
});

describe("MouthClutchCalibration", () => {
  it("captures the live score into talk_max", () => {
    const onChange = vi.fn();
    render(
      <MouthClutchCalibration
        liveJawOpen={0.22}
        value={{ talk_max: null, open_min: null }}
        onChange={onChange}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /talk/i }));
    expect(onChange).toHaveBeenCalledWith({ talk_max: 0.22, open_min: null });
  });

  it("captures the live score into open_min", () => {
    const onChange = vi.fn();
    render(
      <MouthClutchCalibration
        liveJawOpen={0.81}
        value={{ talk_max: 0.2, open_min: null }}
        onChange={onChange}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /open/i }));
    expect(onChange).toHaveBeenCalledWith({ talk_max: 0.2, open_min: 0.81 });
  });

  it("does not capture when no face is tracked", () => {
    const onChange = vi.fn();
    render(
      <MouthClutchCalibration
        liveJawOpen={null}
        value={{ talk_max: null, open_min: null }}
        onChange={onChange}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /talk/i }));
    expect(onChange).not.toHaveBeenCalled();
  });

  it("warns when the captured separation is too small to arm", () => {
    render(
      <MouthClutchCalibration
        liveJawOpen={0.5}
        value={{ talk_max: 0.50, open_min: 0.55 }}
        onChange={vi.fn()}
      />,
    );
    expect(screen.getByText(/too close/i)).toBeInTheDocument();
  });
});
