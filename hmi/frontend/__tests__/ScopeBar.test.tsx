import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { ScopeBar } from "../components/ScopeBar";

describe("ScopeBar", () => {
  it("renders the commanded value and limits", () => {
    const { container } = render(
      <ScopeBar label="pan" min={-90} max={90} commanded={30} intended={30} />
    );
    // The component must render `30.0` somewhere readable.
    expect(container.textContent).toMatch(/30\.0/);
    expect(container.textContent).toMatch(/pan/);
  });

  it("shows a ghost tick only when intended differs from commanded", () => {
    const { container, rerender } = render(
      <ScopeBar label="pan" min={-90} max={90} commanded={30} intended={30} />,
    );
    // Without divergence, the ghost element should not be present.
    expect(container.querySelector("[data-ghost]")).toBeNull();
    rerender(
      <ScopeBar label="pan" min={-90} max={90} commanded={30} intended={45} />,
    );
    expect(container.querySelector("[data-ghost]")).not.toBeNull();
  });

  it("reads empty at rest on an asymmetric range (real gripper limits)", () => {
    // min=-9.97, max=100.27: 0° sits at ~9% of the range, not the middle.
    // At commanded=0 the fill must have ~zero width, not render a ~half-full
    // bar anchored to the geometric center.
    const { container } = render(
      <ScopeBar label="gripper" min={-9.97} max={100.27} commanded={0} />
    );
    const fill = container.querySelector("[data-fill]") as HTMLElement;
    expect(fill).not.toBeNull();
    const width = parseFloat(fill.style.width);
    expect(width).toBeCloseTo(0, 1);
  });

  it("does not paint the fill past the zero tick for a negative commanded value", () => {
    // Symmetric range, so zero sits at 50%. A commanded=-45 fill must span
    // from 25% to 50% — never extend to the right of the zero tick.
    const { container } = render(
      <ScopeBar label="pan" min={-90} max={90} commanded={-45} />
    );
    const fill = container.querySelector("[data-fill]") as HTMLElement;
    expect(fill).not.toBeNull();
    const left = parseFloat(fill.style.left);
    const width = parseFloat(fill.style.width);
    expect(left).toBeCloseTo(25, 1);
    expect(width).toBeCloseTo(25, 1);
    expect(left + width).toBeCloseTo(50, 1); // does not cross the zero tick
  });
});
