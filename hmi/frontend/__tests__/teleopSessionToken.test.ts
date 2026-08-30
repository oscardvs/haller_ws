import { afterEach, describe, expect, it, vi } from "vitest";

import {
  driverToken, forgetDriverToken, rememberDriverToken,
} from "../lib/teleopSessionToken";

const KEY = "haller.teleop.driverToken";

afterEach(() => {
  vi.restoreAllMocks();
  try { window.sessionStorage.clear(); } catch { /* nothing to clear */ }
});

describe("the driver's token", () => {
  it("round-trips through session storage", () => {
    expect(driverToken()).toBeNull();
    rememberDriverToken("tok-123");
    expect(driverToken()).toBe("tok-123");
    expect(window.sessionStorage.getItem(KEY)).toBe("tok-123");
  });

  it("is forgotten when the operator stops the session themselves", () => {
    // Keeping it would make the next mount ask the backend to hold open a
    // session that was deliberately ended.
    rememberDriverToken("tok-123");
    forgetDriverToken();
    expect(driverToken()).toBeNull();
  });

  it("uses SESSION storage, not local", () => {
    // The scope IS the policy: it survives a reload of this tab, which is the
    // whole point, and dies when the tab closes — a closed tab is an operator
    // who has genuinely left, and that session should stop.
    rememberDriverToken("tok-123");
    expect(window.localStorage.getItem(KEY)).toBeNull();
  });

  it("degrades to no token when storage throws", () => {
    // Private mode, or a browser set to block site data. This runs on the
    // socket's message handler where a throw is silent, and the fallback is
    // exactly the old behaviour: a reload costs the session, nothing breaks.
    vi.spyOn(window.sessionStorage.__proto__, "setItem")
      .mockImplementation(() => { throw new Error("blocked"); });
    vi.spyOn(window.sessionStorage.__proto__, "getItem")
      .mockImplementation(() => { throw new Error("blocked"); });
    expect(() => rememberDriverToken("tok-123")).not.toThrow();
    expect(driverToken()).toBeNull();
    expect(() => forgetDriverToken()).not.toThrow();
  });
});
