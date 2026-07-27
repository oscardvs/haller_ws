// hmi/frontend/__tests__/BasePanel.test.tsx
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { BasePanel } from "../components/BasePanel";
import * as apiMod from "../lib/api";
import * as tele from "../lib/telemetry";

vi.mock("../lib/api");
vi.mock("../lib/telemetry");

const BASE_CAM = {
  id: "base_front",
  role: "base" as const,
  source: "opencv" as const,
  arm_id: null,
  active: true,
  width: 640,
  height: 480,
  fps: 30,
};

beforeEach(() => {
  vi.resetAllMocks();
  (tele.useTelemetry as any).mockImplementation((sel: any) =>
    sel({ lastFrame: undefined }),
  );
  // BasePanel posts cmd_vel on an interval and chains .catch() on the result.
  (apiMod.api as any).cmdVel = vi.fn().mockResolvedValue({
    ok: true, linear: 0, angular: 0,
  });
  (apiMod.api as any).cameras = vi.fn().mockResolvedValue({ cameras: [BASE_CAM] });
  (apiMod.cameraStreamUrl as any).mockImplementation(
    (id: string) => `http://desktop:8000/cameras/${id}/stream`,
  );
});

afterEach(() => {
  vi.useRealTimers();
});

describe("BasePanel base camera tile", () => {
  it("renders the live MJPEG stream when base_front is active", async () => {
    render(<BasePanel />);
    const img = await screen.findByAltText("base_front live feed");
    expect(img.getAttribute("src")).toBe(
      "http://desktop:8000/cameras/base_front/stream",
    );
  });

  it("shows the resolution reported by /cameras rather than placeholder dashes", async () => {
    render(<BasePanel />);
    expect(await screen.findByText("640×480 · 30 fps")).toBeInTheDocument();
  });

  it("falls back to 'no feed' when the base camera is inactive", async () => {
    (apiMod.api as any).cameras = vi
      .fn()
      .mockResolvedValue({ cameras: [{ ...BASE_CAM, active: false }] });
    render(<BasePanel />);
    expect(await screen.findByText("no feed")).toBeInTheDocument();
    expect(screen.queryByAltText("base_front live feed")).toBeNull();
  });
});
