// hmi/frontend/__tests__/JointSlider.test.tsx
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { JointSlider } from "../components/JointSlider";

function getRangeInput(): HTMLInputElement {
  // base-ui slider renders a hidden <input type="range"> that holds the value.
  // It's not exposed via role="slider", so we query directly.
  const input = document.querySelector('input[type="range"]');
  if (!(input instanceof HTMLInputElement)) {
    throw new Error("range input not found in JointSlider markup");
  }
  return input;
}

describe("JointSlider", () => {
  it("calls onChange (debounced) when the value moves", async () => {
    const onChange = vi.fn();
    render(
      <JointSlider name="shoulder_pan" pos={0} min={-100} max={100}
                   onChange={onChange} disabled={false} />
    );
    expect(screen.getByText("shoulder_pan")).toBeTruthy();
    const range = getRangeInput();
    // Programmatic value change → base-ui dispatches onValueChange.
    fireEvent.change(range, { target: { value: "5" } });
    await new Promise((r) => setTimeout(r, 80));
    expect(onChange).toHaveBeenCalled();
  });

  it("renders a disabled-marked root when disabled prop is set", () => {
    const { container } = render(
      <JointSlider name="gripper" pos={0} min={0} max={100}
                   onChange={() => {}} disabled />
    );
    // base-ui marks the slider root with data-disabled when disabled
    const root = container.querySelector('[data-disabled]');
    expect(root).not.toBeNull();
  });
});
