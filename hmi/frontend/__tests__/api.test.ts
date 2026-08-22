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

  it("humanTeleopStart posts the pairing verbatim", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ ok: true, state: "armed", running: true }),
                   { status: 200 })
    );
    await api.humanTeleopStart({ left_arm: "left", right_arm: "right", hz: 60 });
    const call = fetchMock.mock.calls[0];
    expect(call[0]).toMatch("/teleop/human/start");
    expect((call[1] as RequestInit).method).toBe("POST");
    expect(JSON.parse((call[1] as RequestInit).body as string)).toEqual({
      left_arm: "left", right_arm: "right", hz: 60,
    });
  });

  it("starts a single-arm session with a null side", async () => {
    // The absent hand is ignored and nothing is ever written to that arm —
    // null is the whole mechanism, so it has to survive serialisation.
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ ok: true, running: true }), { status: 200 })
    );
    await api.humanTeleopStart({ left_arm: "left", right_arm: null });
    const body = JSON.parse(
      (fetchMock.mock.calls[0][1] as RequestInit).body as string,
    );
    expect(body).toEqual({ left_arm: "left", right_arm: null });
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

  it("humanTeleopCollision posts the switch", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ ok: true, collision: { enabled: false } }),
                   { status: 200 })
    );
    await api.humanTeleopCollision(false);
    const call = fetchMock.mock.calls[0];
    expect(call[0]).toMatch("/teleop/human/collision");
    expect(JSON.parse((call[1] as RequestInit).body as string))
      .toEqual({ enabled: false });
  });
});

describe("api dataset wrappers", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("omits repo_id entirely when none was picked", async () => {
    // The endpoints default to the recorder's current repo; sending
    // `?repo_id=` is a different ask, and would resolve to nothing.
    // A fresh Response per call: a body can only be read once.
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(
      async () => new Response(JSON.stringify({ episodes: [] }), { status: 200 })
    );
    await api.recordEpisodes();
    expect(fetchMock.mock.calls[0][0]).toMatch(/\/record\/episodes$/);
    await api.recordEpisodes(null);
    expect(fetchMock.mock.calls[1][0]).toMatch(/\/record\/episodes$/);
  });

  it("escapes the repo id, which carries a slash", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ episodes: [] }), { status: 200 })
    );
    await api.recordEpisodes("osrdvs/haller_pick_cube");
    expect(fetchMock.mock.calls[0][0]).toContain(
      "repo_id=osrdvs%2Fhaller_pick_cube",
    );
  });

  it("cameraRecord posts the runtime toggle for one camera", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ id: "mast", record: true }), { status: 200 })
    );
    await api.cameraRecord("mast", true);
    const call = fetchMock.mock.calls[0];
    expect(call[0]).toMatch("/cameras/mast/record");
    expect(JSON.parse((call[1] as RequestInit).body as string))
      .toEqual({ record: true });
  });

  it("surfaces a delete-last refusal as a status plus the backend's own words", async () => {
    // The pop refuses with 409 rather than leave a dataset lerobot can no
    // longer resume. The Dataset tab has to tell that from a real failure —
    // one is a state you can clear, the other is an error — and the detail
    // names which guard tripped, so it is shown verbatim.
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "cannot pop the only episode" }),
                   { status: 409 })
    );
    await expect(api.recordDeleteLastEpisode()).rejects.toMatchObject({
      status: 409,
      detail: "cannot pop the only episode",
    });
  });
});
