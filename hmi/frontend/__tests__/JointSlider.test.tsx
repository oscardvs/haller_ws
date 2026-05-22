// hmi/frontend/__tests__/JointSlider.test.tsx
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { JointSlider } from "../components/JointSlider";

describe("JointSlider", () => {
  it("calls onChange (debounced) when moved", async () => {
    const onChange = vi.fn();
    render(
      <JointSlider name="shoulder_pan" pos={0} min={-100} max={100}
                   onChange={onChange} disabled={false} />
    );
    expect(screen.getByText("shoulder_pan")).toBeTruthy();
    // Radix slider uses keyboard for tests:
    const slider = screen.getByRole("slider");
    fireEvent.keyDown(slider, { key: "ArrowRight" });
    await new Promise((r) => setTimeout(r, 80));
    expect(onChange).toHaveBeenCalled();
  });

  it("disables interaction when disabled prop is set", () => {
    render(
      <JointSlider name="gripper" pos={0} min={0} max={100}
                   onChange={() => {}} disabled />
    );
    const slider = screen.getByRole("slider");
    expect(slider.getAttribute("aria-disabled")).toBe("true");
  });
});
