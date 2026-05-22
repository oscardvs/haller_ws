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
});
