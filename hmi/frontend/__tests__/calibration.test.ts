// hmi/frontend/__tests__/calibration.test.ts
import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  fetchCalibrationStatus,
  startCalibration,
  captureNeutral,
  finishSweep,
  saveCalibration,
  abortCalibration,
} from "../lib/calibration";

const okJson = (body: unknown) =>
  Promise.resolve(new Response(JSON.stringify(body), { status: 200 }));

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn());
});

describe("calibration client", () => {
  it("GETs /calibration/status", async () => {
    (fetch as any).mockReturnValue(okJson({
      arms: [{ id: "right", has_file: true, path: "/x", mtime: 1, in_session: false }],
      current_session: null,
    }));
    const r = await fetchCalibrationStatus("http://b");
    expect(fetch).toHaveBeenCalledWith("http://b/calibration/status",
      expect.objectContaining({ method: "GET" }));
    expect(r.arms[0].id).toBe("right");
  });

  it("POSTs start/capture/finish/save/abort", async () => {
    (fetch as any).mockReturnValue(okJson({ ok: true, state: "homing" }));
    await startCalibration("http://b", "right");
    expect(fetch).toHaveBeenCalledWith("http://b/calibration/right/start",
      expect.objectContaining({ method: "POST" }));

    (fetch as any).mockReturnValue(okJson({ ok: true, state: "sweeping", homing_offsets: {} }));
    await captureNeutral("http://b", "right");
    (fetch as any).mockReturnValue(okJson({ ok: true, state: "review", proposed: {}, current: null }));
    await finishSweep("http://b", "right");
    (fetch as any).mockReturnValue(okJson({ ok: true, state: "done", path: "/x", backup_path: null }));
    await saveCalibration("http://b", "right");
    (fetch as any).mockReturnValue(okJson({ ok: true, state: "aborted" }));
    await abortCalibration("http://b", "right");
  });

  it("throws on non-OK response with the detail message", async () => {
    (fetch as any).mockReturnValue(Promise.resolve(
      new Response(JSON.stringify({ detail: "session active for arm 'right'" }), { status: 409 })));
    await expect(startCalibration("http://b", "right"))
      .rejects.toThrow(/session active/);
  });
});
