// hmi/frontend/__tests__/api.test.ts
import { describe, it, expect, vi, beforeEach } from "vitest";
import { postJson, getJson, api } from "../lib/api";

describe("postJson", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("posts JSON and parses response", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), { status: 200 })
    );
    const out = await postJson<{ ok: boolean }>("/foo", { a: 1 });
    expect(out).toEqual({ ok: true });
    const call = fetchSpy.mock.calls[0];
    expect((call[1] as RequestInit).method).toBe("POST");
    expect((call[1] as RequestInit).body).toBe(JSON.stringify({ a: 1 }));
  });

  it("throws on non-2xx", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ error: "bad" }), { status: 409 })
    );
    await expect(postJson("/foo", {})).rejects.toThrow(/bad|409/);
  });
});

describe("getJson", () => {
  it("returns parsed body", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ x: 1 }), { status: 200 })
    );
    expect(await getJson<{ x: number }>("/x")).toEqual({ x: 1 });
  });
});

describe("api human-teleop wrappers", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("humanTeleopStart posts the correct body", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ ok: true, state: "armed", running: true }),
                   { status: 200 })
    );
    await api.humanTeleopStart({ left_arm: "left", right_arm: "right", swap: false });
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/teleop/human/start"),
      expect.objectContaining({ method: "POST" })
    );
  });

  it("humanTeleopStop hits the stop endpoint", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ ok: true, running: false }), { status: 200 })
    );
    await api.humanTeleopStop();
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/teleop/human/stop"),
      expect.objectContaining({ method: "POST" })
    );
  });

  it("humanTeleopCalibrate posts pinch ranges", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ ok: true }), { status: 200 })
    );
    await api.humanTeleopCalibrate({
      left: { min_m: 0.02, max_m: 0.18 },
      right: { min_m: 0.022, max_m: 0.175 },
    });
    const call = fetchMock.mock.calls[0];
    expect(call[0]).toMatch("/teleop/human/calibrate");
    expect(JSON.parse((call[1] as RequestInit).body as string)).toEqual({
      left:  { min_m: 0.02,  max_m: 0.18 },
      right: { min_m: 0.022, max_m: 0.175 },
    });
  });
});
