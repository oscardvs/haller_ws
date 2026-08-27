// hmi/frontend/__tests__/api.test.ts
import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  postJson, getJson, api, recordRateGate, recordRateOk, recordRateTolerance,
  recordRateFaithful, RECORD_RATE_GATE_FALLBACK, RATE_DECIMALS, formatHz,
  type RecordStatus,
} from "../lib/api";

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

describe("the record rate band", () => {
  // The recorder's gate became a FAITHFULNESS BOUND: `|measured − fps| / fps >
  // 0.005` refuses. That is a symmetric tolerance, not a floor, and the key was
  // renamed rather than revalued for one reason — publishing 0.005 under
  // `record_rate_gate` would make every existing caller compute
  // `declared * 0.005` and warn below half a percent of the declared rate. The
  // warning would not become wrong; it would stop firing.
  const st = (over: Partial<RecordStatus>): RecordStatus => ({
    recording: false, repo_id: null, task: null, episode_frames: 0,
    skipped_frames: 0, started_at: null, last_error: null, ...over,
  }) as RecordStatus;

  it("reads the tolerance the recorder publishes", () => {
    expect(recordRateTolerance(st({ record_rate_tolerance: 0.005 }))).toBe(0.005);
  });

  it("returns null — never a number — when no tolerance is published", () => {
    // THE decisive assertion. The tempting fallback is the one already in this
    // file, and `0.9` read as a tolerance means ±90%: a band no real rate can
    // fall outside, so the check could not fire in either direction. A
    // reassuring check that protects nothing is worse than an absent one, so
    // the absent case is named on screen instead.
    expect(recordRateTolerance(st({}))).toBeNull();
    expect(recordRateTolerance(null)).toBeNull();
    expect(recordRateTolerance(st({ record_rate_tolerance: 0 }))).toBeNull();
    // And specifically NOT the floor fallback wearing a tolerance's clothes.
    expect(recordRateTolerance(st({}))).not.toBe(RECORD_RATE_GATE_FALLBACK);
  });

  it("refuses a rate that is too FAST, which a floor never could", () => {
    // The half of the bound that `measured >= declared * g` cannot express at
    // any value of g. Timestamps synthesised from a rate the rig overshot are
    // as dishonest as ones it undershot.
    const fast = st({ record_rate_tolerance: 0.005, fps_declared: 30, fps_measured: 30.6 });
    expect(recordRateFaithful(fast)).toBe(false);
    // The old one-sided reading calls the same take fine, which is exactly why
    // the key had to be renamed rather than revalued.
    expect(recordRateOk({ ...fast, record_rate_gate: 0.9 })).toBe(true);
  });

  it("pins the boundary on both sides, not just the fixed value", () => {
    const at = (m: number) =>
      recordRateFaithful(st({
        record_rate_tolerance: 0.005, fps_declared: 30, fps_measured: m,
      }));
    expect(at(30)).toBe(true);
    expect(at(30 * 1.005)).toBe(true);   // exactly on the bound
    expect(at(30 * 0.995)).toBe(true);
    expect(at(30 * 1.006)).toBe(false);
    expect(at(30 * 0.994)).toBe(false);
  });

  it("says NOT ANSWERABLE rather than false for either unknown", () => {
    // Not measured yet, and no published band, are different unknowns and
    // neither is a warning. An operator shown a rate warning during the first
    // second after boot learns to ignore rate warnings.
    expect(recordRateFaithful(st({ record_rate_tolerance: 0.005, fps_declared: 30 }))).toBeNull();
    expect(recordRateFaithful(st({ fps_declared: 30, fps_measured: 30 }))).toBeNull();
  });

  it("leaves the OLD accessor meaning exactly what it meant", () => {
    // `recordRateGate` still has a caller in the headset client, which applies
    // it as `declared * gate`. Keeping the symbol and changing what it returns
    // would compile everywhere, warn nowhere, and move the silent window into
    // the migration rather than closing it — so its floor semantics are pinned
    // here until the key it reads is removed.
    expect(recordRateGate(st({ record_rate_gate: 0.95 }))).toBe(0.95);
    expect(recordRateGate(st({}))).toBe(RECORD_RATE_GATE_FALLBACK);
    // A tolerance on the wire must NOT leak into the floor reader.
    expect(recordRateGate(st({ record_rate_tolerance: 0.005 }))).toBe(RECORD_RATE_GATE_FALLBACK);
  });
});

describe("a rate readout shows the band it is judged against", () => {
  // A readout coloured as a warning while showing two numbers that look equal
  // teaches the operator that the warning is spurious. The decimal count is a
  // CADENCE-COUPLED CONSTANT: with `d` decimals a rate outside the tolerance
  // still renders as the declared one whenever `fps < 10^(2-d)`, so one
  // decimal is safe at 30 and silently broken at 5.
  const TOL = 0.005;
  const st = (declared: number, measured: number): RecordStatus => ({
    recording: false, repo_id: null, task: null, episode_frames: 0,
    skipped_frames: 0, started_at: null, last_error: null,
    record_rate_tolerance: TOL, fps_declared: declared, fps_measured: measured,
  }) as RecordStatus;

  // Every cadence reachable through `POST /teleop/human/start {hz}`, plus the
  // 4.8 Hz the kit's real rollout actually ran at.
  const CADENCES = [60, 30, 20, 10, 5, 4.8, 2];

  it("never renders a REFUSED rate as the declared one", () => {
    // Written against the real band function rather than a remembered
    // threshold: if the tolerance tightens, this fails rather than rotting.
    for (const fps of CADENCES) {
      const outside = fps + fps * TOL * 1.2;
      expect(recordRateFaithful(st(fps, outside))).toBe(false);
      expect(formatHz(outside)).not.toBe(formatHz(fps));
    }
  });

  it("would fail at one decimal, which is why it is not one", () => {
    // The instrument, not the assertion: this states WHERE the old resolution
    // broke, so the constant cannot be quietly lowered back. 5 Hz is inside
    // `fps < 10^(2-1)`; 30 Hz is not, which is why nobody saw it.
    const at = (fps: number, d: number) => {
      const outside = fps + fps * TOL * 1.2;
      return outside.toFixed(d) === fps.toFixed(d);
    };
    expect(at(5, 1)).toBe(true);    // collides at one decimal
    expect(at(5, RATE_DECIMALS)).toBe(false);
    expect(at(30, 1)).toBe(false);  // safe at 30 — how it survived
    expect(at(30, 0)).toBe(true);   // and the `RATE 30/30` defect at zero
  });
});
