// hmi/frontend/__tests__/EStopButton.test.tsx
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { EStopButton } from "../components/EStopButton";

describe("EStopButton", () => {
  it("calls api.estop on click", async () => {
    const mockEstop = vi.fn().mockResolvedValue({ ok: true });
    vi.doMock("@/lib/api", () => ({ api: { estop: mockEstop } }));
    // re-import after mock
    const { EStopButton: Reloaded } = await import("../components/EStopButton");
    render(<Reloaded />);
    fireEvent.click(screen.getByRole("button", { name: /emergency stop/i }));
    await Promise.resolve();
    expect(mockEstop).toHaveBeenCalled();
  });
});
