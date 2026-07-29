import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import {
  MouthClutchCalibration, mouthCalibReady, type MouthCalib,
} from "@/components/MouthClutchCalibration";
import { JawTraceRecorder } from "@/lib/mediapipe";
import type { MouthAnalysis } from "@/lib/api";

const EMPTY: MouthCalib = { talk_hold: null, open_hold: null, talk_peak: null };
const GOOD: MouthCalib = { talk_hold: 0.11, open_hold: 0.46, talk_peak: 0.44 };

describe("mouthCalibReady", () => {
  it("is false until both sustained levels exist", () => {
    expect(mouthCalibReady(EMPTY)).toBe(false);
    expect(mouthCalibReady({ ...EMPTY, talk_hold: 0.1 })).toBe(false);
    expect(mouthCalibReady({ ...EMPTY, open_hold: 0.9 })).toBe(false);
  });

  it("is false when sustained speech leaves no headroom", () => {
    // Separation 0.05 — the backend would refuse to arm on this.
    expect(mouthCalibReady({ talk_hold: 0.5, open_hold: 0.55, talk_peak: 0.6 }))
      .toBe(false);
  });

  it("is true with adequate separation", () => {
    expect(mouthCalibReady(GOOD)).toBe(true);
  });

  it("ignores the speech peak entirely", () => {
    // Two operators sustaining speech at the same level get the same verdict
    // however loud one of them gets mid-sentence. A peak is a transient; the
    // clutch is judged on what is held. This is the whole reason the shape
    // changed from {talk_max, open_min}.
    expect(mouthCalibReady({ ...GOOD, talk_peak: 0.05 }))
      .toBe(mouthCalibReady({ ...GOOD, talk_peak: 0.95 }));
  });

  it("is false when the open is below the speech level", () => {
    expect(mouthCalibReady({ talk_hold: 0.9, open_hold: 0.1, talk_peak: 0.95 }))
      .toBe(false);
  });
});

const btn = (name: RegExp) => screen.getByRole("button", { name });
const talkButton = () => btn(/talk/i);
const openButton = () => btn(/^open/i);
const verifyButton = () => btn(/verify/i);

/** Typed so `analyze.mock.calls[0][0]` is the request body, not `never` —
 *  what this component is FOR is sending an unreduced trace, so the tests have
 *  to be able to look at what it sent. */
type AnalyzeFn = NonNullable<
  React.ComponentProps<typeof MouthClutchCalibration>["analyze"]
>;

function setup(opts: {
  value?: MouthCalib;
  analysis?: MouthAnalysis;
} = {}) {
  const onChange = vi.fn();
  const recorder = new JawTraceRecorder();
  const analyze = vi.fn<AnalyzeFn>(async () => opts.analysis ?? ({
    ok: true,
    calib: { talk_hold: 0.11, open_hold: 0.46, talk_peak: 0.44 },
    thresholds: { t_engage: 0.26, t_release: 0.16 },
    problems: [],
  } as MouthAnalysis));
  render(
    <MouthClutchCalibration
      liveJawOpen={0.2}
      value={opts.value ?? EMPTY}
      onChange={onChange}
      recorder={recorder}
      analyze={analyze}
    />,
  );
  /** Stand in for the render loop feeding the recorder. */
  const speak = (samples: number[]) =>
    samples.forEach((v, i) => recorder.push(i * 100, v));
  return { onChange, recorder, analyze, speak };
}

describe("MouthClutchCalibration", () => {
  it("sends the whole recorded trace to the backend, not a folded number", async () => {
    // The statistic that matters is a windowed minimum, which cannot be
    // recovered from a max or a min. Reducing here would also put a second
    // copy of the safety policy in the browser.
    const { analyze, speak } = setup();
    fireEvent.click(talkButton());
    speak([0.05, 0.41, 0.07, 0.38, 0.06]);
    fireEvent.click(talkButton());
    await waitFor(() => expect(analyze).toHaveBeenCalled());
    expect(analyze.mock.calls[0][0].talk).toEqual([
      [0, 0.05], [100, 0.41], [200, 0.07], [300, 0.38], [400, 0.06],
    ]);
  });

  it("stores the calibration the backend derived", async () => {
    const { onChange, analyze, speak } = setup();
    fireEvent.click(talkButton());
    speak([0.05, 0.4]);
    fireEvent.click(talkButton());
    await waitFor(() => expect(analyze).toHaveBeenCalled());
    expect(onChange).toHaveBeenCalledWith({
      talk_hold: 0.11, open_hold: 0.46, talk_peak: 0.44,
    });
  });

  it("clears the calibration when a capture fails to produce thresholds", async () => {
    // Keeping the previous numbers after a recalibration the operator watched
    // fail would leave them driving on thresholds nobody meant to be using.
    const { onChange, speak } = setup({
      value: GOOD,
      analysis: { ok: false, calib: null, thresholds: null,
                  problems: ["talk capture is only 0.4s; hold it for at least 3s"] },
    });
    fireEvent.click(talkButton());
    speak([0.3]);
    fireEvent.click(talkButton());
    await waitFor(() =>
      expect(onChange).toHaveBeenCalledWith(
        { talk_hold: null, open_hold: null, talk_peak: null }),
    );
  });

  it("shows the backend's reason for refusing", async () => {
    const { speak } = setup({
      analysis: { ok: false, calib: null, thresholds: null,
                  problems: ["only 0.08 between sustained speech and a sustained open"] },
    });
    fireEvent.click(talkButton());
    speak([0.3]);
    fireEvent.click(talkButton());
    expect(await screen.findByText(/only 0.08 between sustained speech/i))
      .toBeInTheDocument();
  });

  it("commits nothing when no face was tracked for the whole window", async () => {
    const { onChange, analyze } = setup();
    fireEvent.click(talkButton());
    fireEvent.click(talkButton());          // no samples pushed at all
    expect(analyze).not.toHaveBeenCalled();
    expect(onChange).not.toHaveBeenCalled();
    expect(await screen.findByText(/no face was tracked/i)).toBeInTheDocument();
  });

  it("captures one window at a time", () => {
    const { analyze } = setup();
    fireEvent.click(talkButton());
    expect(openButton()).toBeDisabled();
    expect(verifyButton()).toBeDisabled();
    expect(analyze).not.toHaveBeenCalled();
  });

  it("will not run a speech test before there is a calibration to test", () => {
    setup();
    expect(verifyButton()).toBeDisabled();
  });

  it("reports a PASS when replayed speech never engages the clutch", async () => {
    const { speak } = setup({
      value: GOOD,
      analysis: {
        verify: {
          engaged: false, first_engage_ms: null, engaged_samples: 0, n: 60,
          peak: 0.44, sustained: 0.11, t_engage: 0.26, margin: 0.15,
        },
      },
    });
    fireEvent.click(verifyButton());
    speak([0.05, 0.44, 0.06]);
    fireEvent.click(verifyButton());
    expect(await screen.findByText(/PASS/)).toBeInTheDocument();
  });

  it("reports a FAIL loudly when replayed speech would have driven the arms", async () => {
    // This is the one result that must never be quiet: it means the operator's
    // speech can take the robot, and the calibration must not be used.
    const { speak } = setup({
      value: GOOD,
      analysis: {
        verify: {
          engaged: true, first_engage_ms: 1240, engaged_samples: 9, n: 60,
          peak: 0.61, sustained: 0.31, t_engage: 0.26, margin: -0.05,
        },
      },
    });
    fireEvent.click(verifyButton());
    speak([0.3, 0.35]);
    fireEvent.click(verifyButton());
    expect(await screen.findByText(/FAIL/)).toBeInTheDocument();
    expect(screen.getByText(/do not drive on this calibration/i)).toBeInTheDocument();
  });

  it("says so when a calibration has never been verified", () => {
    setup({ value: GOOD });
    expect(screen.getByText(/not yet verified against your speech/i))
      .toBeInTheDocument();
  });

  it("checks the speech test against the calibration being trusted", async () => {
    const { analyze, speak } = setup({
      value: GOOD,
      analysis: { verify: {
        engaged: false, first_engage_ms: null, engaged_samples: 0, n: 3,
        peak: 0.2, sustained: 0.1, t_engage: 0.26, margin: 0.16,
      } },
    });
    fireEvent.click(verifyButton());
    speak([0.1, 0.2]);
    fireEvent.click(verifyButton());
    await waitFor(() => expect(analyze).toHaveBeenCalled());
    expect(analyze.mock.calls[0][0].calib).toEqual({
      talk_hold: 0.11, open_hold: 0.46, talk_peak: 0.44,
    });
  });
});
