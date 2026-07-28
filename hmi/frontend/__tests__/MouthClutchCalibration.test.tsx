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

const talkButton = () => screen.getByRole("button", { name: /talk/i });
const openButton = () => screen.getByRole("button", { name: /open/i });

describe("MouthClutchCalibration", () => {
  // Spec 6.3: `talk` records the MAX jawOpen reached while speaking for a few
  // seconds, `open` the MIN sustained through a deliberate wide open. A single
  // sample taken at the instant of a click is not either of those, and the two
  // directions are not symmetric in how they fail. An instantaneous open_min
  // errs toward a peak, which widens the apparent gap and makes engaging
  // harder — it fails safe. An instantaneous talk_max errs toward a trough
  // (nobody clicks at the peak of their own speech envelope, and the score
  // only updates ~10 Hz), which also widens the apparent gap — and that puts
  // t_engage BELOW the operator's real speech maximum. It fails unsafe, and
  // takes the speech-resistance the whole feature rests on with it.

  it("records the maximum jawOpen across the talk window, not the last sample", () => {
    const onChange = vi.fn();
    const { rerender } = render(
      <MouthClutchCalibration
        liveJawOpen={0.18}
        value={{ talk_max: null, open_min: null }}
        onChange={onChange}
      />,
    );
    fireEvent.click(talkButton());              // window opens
    const speak = (v: number) =>
      rerender(
        <MouthClutchCalibration
          liveJawOpen={v}
          value={{ talk_max: null, open_min: null }}
          onChange={onChange}
        />,
      );
    speak(0.31);        // the peak of the speech envelope
    speak(0.24);
    speak(0.29);        // ...and the window closes somewhere off the peak
    fireEvent.click(talkButton());              // window closes
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith({ talk_max: 0.31, open_min: null });
  });

  it("records the minimum jawOpen across the open window, not the last sample", () => {
    const onChange = vi.fn();
    const value = { talk_max: 0.2, open_min: null };
    const { rerender } = render(
      <MouthClutchCalibration liveJawOpen={0.85} value={value} onChange={onChange} />,
    );
    fireEvent.click(openButton());
    const hold = (v: number) =>
      rerender(
        <MouthClutchCalibration liveJawOpen={v} value={value} onChange={onChange} />,
      );
    hold(0.62);         // the sag in a "sustained" open
    hold(0.79);
    fireEvent.click(openButton());
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith({ talk_max: 0.2, open_min: 0.62 });
  });

  it("ignores lost-face samples inside an otherwise good window", () => {
    const onChange = vi.fn();
    const value = { talk_max: null, open_min: null };
    const { rerender } = render(
      <MouthClutchCalibration liveJawOpen={0.30} value={value} onChange={onChange} />,
    );
    fireEvent.click(talkButton());
    const speak = (v: number | null) =>
      rerender(
        <MouthClutchCalibration liveJawOpen={v} value={value} onChange={onChange} />,
      );
    speak(null);        // face briefly lost mid-window
    speak(0.40);
    fireEvent.click(talkButton());
    expect(onChange).toHaveBeenCalledWith({ talk_max: 0.40, open_min: null });
  });

  it("commits nothing when no face was tracked for the whole window", () => {
    const onChange = vi.fn();
    render(
      <MouthClutchCalibration
        liveJawOpen={null}
        value={{ talk_max: null, open_min: null }}
        onChange={onChange}
      />,
    );
    fireEvent.click(talkButton());
    fireEvent.click(talkButton());
    expect(onChange).not.toHaveBeenCalled();
  });

  it("captures one direction at a time", () => {
    const onChange = vi.fn();
    render(
      <MouthClutchCalibration
        liveJawOpen={0.5}
        value={{ talk_max: null, open_min: null }}
        onChange={onChange}
      />,
    );
    fireEvent.click(talkButton());
    expect(openButton()).toBeDisabled();
    fireEvent.click(openButton());
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
