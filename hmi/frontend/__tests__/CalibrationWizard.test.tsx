// hmi/frontend/__tests__/CalibrationWizard.test.tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { CalibrationWizard } from "../components/CalibrationWizard";
import * as cal from "../lib/calibration";
import * as tele from "../lib/telemetry";

vi.mock("../lib/calibration");
vi.mock("../lib/telemetry");

beforeEach(() => {
  vi.resetAllMocks();
  (cal.startCalibration as any).mockResolvedValue({ ok: true, state: "homing" });
  (cal.abortCalibration as any).mockResolvedValue({ ok: true, state: "aborted" });
  (tele.useTelemetry as any).mockImplementation((sel: any) =>
    sel({ lastFrame: { arms: { right: { mode: "manual", torque: false,
      joints: {}, calibration: { state: "homing", ticks: { shoulder_pan: 2048 } } } } } }));
  (cal.fetchCalibrationStatus as any).mockResolvedValue({
    arms: [], current_session: { arm_id: "right", state: "homing" },
  });
});

describe("CalibrationWizard step 2 (sweeping)", () => {
  it("renders min | POS | max columns in the sweep step", async () => {
    (tele.useTelemetry as any).mockImplementation((sel: any) =>
      sel({ lastFrame: { arms: { right: { mode: "manual", torque: false, joints: {},
        calibration: { state: "sweeping",
                       ticks: { shoulder_pan: 2200 },
                       min:   { shoulder_pan: 1000 },
                       max:   { shoulder_pan: 3500 } } } } } }));
    (cal.fetchCalibrationStatus as any).mockResolvedValue({
      arms: [], current_session: { arm_id: "right", state: "sweeping" },
    });
    render(<CalibrationWizard armId="right" onClose={() => {}} />);
    expect(await screen.findByText("1000")).toBeInTheDocument();
    expect(await screen.findByText("2200")).toBeInTheDocument();
    expect(await screen.findByText("3500")).toBeInTheDocument();
  });
});

describe("CalibrationWizard step 1 (homing)", () => {
  it("renders the live ticks table", async () => {
    render(<CalibrationWizard armId="right" onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText(/shoulder_pan/)).toBeInTheDocument());
    expect(screen.getByText("2048")).toBeInTheDocument();
  });

  it("clicking Capture neutral calls the backend and advances to step 2", async () => {
    (cal.captureNeutral as any).mockResolvedValue({ ok: true, state: "sweeping", homing_offsets: {} });
    (tele.useTelemetry as any).mockImplementation((sel: any) =>
      sel({ lastFrame: { arms: { right: { mode: "manual", torque: false, joints: {},
        calibration: { state: "sweeping",
                       ticks: { shoulder_pan: 2048 },
                       min: { shoulder_pan: 2048 },
                       max: { shoulder_pan: 2048 } } } } } }));
    render(<CalibrationWizard armId="right" onClose={() => {}} />);
    fireEvent.click(await screen.findByRole("button", { name: /capture neutral/i }));
    await waitFor(() => expect(cal.captureNeutral).toHaveBeenCalledWith(expect.any(String), "right"));
    expect(await screen.findByRole("button", { name: /done sweeping/i })).toBeInTheDocument();
  });

  it("calls /abort exactly once on unmount", async () => {
    const { unmount } = render(<CalibrationWizard armId="right" onClose={() => {}} />);
    unmount();
    await waitFor(() => expect(cal.abortCalibration).toHaveBeenCalledTimes(1));
  });
});
