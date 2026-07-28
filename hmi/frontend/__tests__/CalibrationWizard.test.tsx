// hmi/frontend/__tests__/CalibrationWizard.test.tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
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

describe("CalibrationWizard step 3 (review)", () => {
  it("renders the diff table in review and Save calls saveCalibration", async () => {
    (tele.useTelemetry as any).mockImplementation((sel: any) =>
      sel({ lastFrame: { arms: { right: { mode: "manual", torque: false, joints: {} } } } }));
    (cal.fetchCalibrationStatus as any).mockResolvedValue({
      arms: [],
      current_session: {
        arm_id: "right", state: "review",
        proposed: { shoulder_pan: { id: 1, drive_mode: 0, homing_offset: 0, range_min: 500, range_max: 3500 } },
        current:  { shoulder_pan: { id: 1, drive_mode: 0, homing_offset: 0, range_min: 100, range_max: 200  } },
      },
    });
    (cal.saveCalibration as any).mockResolvedValue({ ok: true, state: "done", path: "/x", backup_path: "/x.bak-1" });

    render(<CalibrationWizard armId="right" onClose={() => {}} />);
    expect(await screen.findByText("500")).toBeInTheDocument();   // proposed min
    expect(await screen.findByText("3500")).toBeInTheDocument();  // proposed max
    expect(await screen.findByText("100")).toBeInTheDocument();   // current min
    fireEvent.click(screen.getByRole("button", { name: /save/i }));
    await waitFor(() => expect(cal.saveCalibration).toHaveBeenCalledWith(expect.any(String), "right"));
  });
});

describe("CalibrationWizard step 1 (homing)", () => {
  it("renders the live ticks table", async () => {
    render(<CalibrationWizard armId="right" onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText(/shoulder_pan/)).toBeInTheDocument());
    expect(screen.getByText("2048")).toBeInTheDocument();
  });

  it("clicking Capture neutral calls the backend and advances to step 2", async () => {
    // Telemetry reports "homing" here, because that is what it reports before
    // capture_neutral is called. It used to be mocked as "sweeping" from the
    // first render, which only rendered step 1 at all because the bootstrap
    // fetch happened to resolve after the phase-mirroring effect and overwrite
    // it — the assertion passed on a race. The wizard now takes whichever of
    // telemetry and its own optimistic advance is further along, so step 2
    // here is driven by the click, which is what this test is about.
    (cal.captureNeutral as any).mockResolvedValue({ ok: true, state: "sweeping", homing_offsets: {} });
    (tele.useTelemetry as any).mockImplementation((sel: any) =>
      sel({ lastFrame: { arms: { right: { mode: "manual", torque: false, joints: {},
        calibration: { state: "homing",
                       ticks: { shoulder_pan: 2048 } } } } } }));
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

describe("CalibrationWizard bus error handling", () => {
  it("displays calibration.error inline when present", async () => {
    (tele.useTelemetry as any).mockImplementation((sel: any) =>
      sel({ lastFrame: { arms: { right: { mode: "manual", torque: false, joints: {},
        calibration: { state: "sweeping",
                       ticks: { shoulder_pan: 2200 },
                       min: { shoulder_pan: 1000 },
                       max: { shoulder_pan: 3500 },
                       error: "bus disconnect" } } } } }));
    (cal.fetchCalibrationStatus as any).mockResolvedValue({
      arms: [], current_session: { arm_id: "right", state: "sweeping" },
    });
    render(<CalibrationWizard armId="right" onClose={() => {}} />);
    expect(await screen.findByText(/bus disconnect/i)).toBeInTheDocument();
  });

  it("shows session-aborted message when calibration block disappears mid-session", async () => {
    // First render: block is present (simulates initial sweeping state)
    let frame: any = { arms: { right: { mode: "manual", torque: false, joints: {},
      calibration: { state: "sweeping",
                     ticks: { shoulder_pan: 2048 },
                     min: { shoulder_pan: 2048 },
                     max: { shoulder_pan: 2048 } } } } };
    (tele.useTelemetry as any).mockImplementation((sel: any) => sel({ lastFrame: frame }));
    (cal.fetchCalibrationStatus as any).mockResolvedValue({
      arms: [], current_session: { arm_id: "right", state: "sweeping" },
    });
    const { rerender } = render(<CalibrationWizard armId="right" onClose={() => {}} />);
    // Wait for initial frame to be observed
    await screen.findByRole("button", { name: /done sweeping/i });
    // Simulate backend abort: calibration block disappears
    frame = { arms: { right: { mode: "manual", torque: false, joints: {} } } } as any;
    (tele.useTelemetry as any).mockImplementation((sel: any) => sel({ lastFrame: frame }));
    rerender(<CalibrationWizard armId="right" onClose={() => {}} />);
    expect(await screen.findByText(/calibration was aborted/i)).toBeInTheDocument();
  });
});

describe("CalibrationWizard confirm-before-cancel", () => {
  it("step 1 Cancel closes directly without confirmation", async () => {
    const onClose = vi.fn();
    render(<CalibrationWizard armId="right" onClose={onClose} />);
    fireEvent.click(await screen.findByRole("button", { name: /^cancel$/i }));
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(screen.queryByText(/discard the range/i)).not.toBeInTheDocument();
  });

  it("step 2 Cancel opens confirm dialog; Discard then closes", async () => {
    (tele.useTelemetry as any).mockImplementation((sel: any) =>
      sel({ lastFrame: { arms: { right: { mode: "manual", torque: false, joints: {},
        calibration: { state: "sweeping",
                       ticks: { shoulder_pan: 2048 },
                       min: { shoulder_pan: 2048 },
                       max: { shoulder_pan: 2048 } } } } } }));
    (cal.fetchCalibrationStatus as any).mockResolvedValue({
      arms: [], current_session: { arm_id: "right", state: "sweeping" },
    });
    const onClose = vi.fn();
    render(<CalibrationWizard armId="right" onClose={onClose} />);
    fireEvent.click(await screen.findByRole("button", { name: /^cancel$/i }));
    expect(await screen.findByText(/discard the range/i)).toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /^discard$/i }));
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
  });

  it("step 2 Cancel then Keep going does NOT close", async () => {
    (tele.useTelemetry as any).mockImplementation((sel: any) =>
      sel({ lastFrame: { arms: { right: { mode: "manual", torque: false, joints: {},
        calibration: { state: "sweeping",
                       ticks: { shoulder_pan: 2048 },
                       min: { shoulder_pan: 2048 },
                       max: { shoulder_pan: 2048 } } } } } }));
    (cal.fetchCalibrationStatus as any).mockResolvedValue({
      arms: [], current_session: { arm_id: "right", state: "sweeping" },
    });
    const onClose = vi.fn();
    render(<CalibrationWizard armId="right" onClose={onClose} />);
    fireEvent.click(await screen.findByRole("button", { name: /^cancel$/i }));
    await screen.findByText(/discard the range/i);
    fireEvent.click(screen.getByRole("button", { name: /keep going/i }));
    expect(onClose).not.toHaveBeenCalled();
  });
});
