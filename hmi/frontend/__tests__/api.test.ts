// hmi/frontend/__tests__/api.test.ts
import { describe, it, expect, vi, beforeEach } from "vitest";
import { postJson, getJson } from "../lib/api";

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
