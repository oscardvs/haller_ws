// hmi/frontend/__tests__/api.test.ts
import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  postJson, getJson, api, recordRateTolerance, recordRateFaithful,
  RATE_DECIMALS, formatHz, type RecordStatus, type RecordAlert,
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
  // the now-removed `record_rate_gate` would have made every caller compute
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
  });

  it("refuses a rate that is too FAST, which a floor never could", () => {
    // The half of the bound that `measured >= declared * g` cannot express at
    // any value of g. Timestamps synthesised from a rate the rig overshot are
    // as dishonest as ones it undershot.
    const fast = st({ record_rate_tolerance: 0.005, fps_declared: 30, fps_measured: 30.6 });
    expect(recordRateFaithful(fast)).toBe(false);
    // The floor this replaced calls the same take FINE. Written as arithmetic
    // rather than a call because the function is gone — the claim is about the
    // SHAPE, and it is the reason the key was renamed rather than revalued.
    expect(30.6 >= 30 * 0.9).toBe(true);
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

describe("RecordAlert against the wire it describes", () => {
  // Written from the PAYLOAD, not from the type. This is the literal object
  // `recorder.py::_rate_alerts()` builds — captured from a live
  // `GET /record/status` on a clean tree, then held here as the claim the type
  // has to satisfy.
  //
  // The type declared `code`, `detail?` and `since?` until 2026-08-27: one key
  // of eight, with two fields the backend has never sent. It type-checked
  // throughout, because both phantoms were optional and NOTHING READ THEM. An
  // unread type cannot render `undefined`, which is why this survived while
  // five noisier defects on the run surface were found and fixed.
  const WIRE = {
    level: "warn",
    code: "record_rate",
    source: "recorder",
    measured_hz: 28.4,
    fps: 30,
    tolerance: 0.005,
    held_s: 3.2,
    message:
      "tick rate has been outside 0.5% of fps 30 for 3s (measured 28.40 Hz); "
      + "timestamps in this take are being written as frame_index/30 regardless",
  } as const;

  // WHAT THIS SUITE CANNOT DETECT, established by mutation and not by argument:
  // reverting `RecordAlert` to its old 1-of-8 shape leaves every test in this
  // describe GREEN. It was mutation-checked twice — once naively, then again
  // after a deliberate attempt to make the runtime half load-bearing — and it
  // stayed green both times.
  //
  // The reason is not fixable by a better assertion: **TypeScript types are
  // erased before vitest ever runs.** `TYPE_KEYS` below is annotated
  // `Record<keyof RecordAlert, true>`, which does force exact key
  // correspondence in both directions — a dropped field and an invented one
  // are each a compile error — but at runtime it is only the object literal
  // written here. No expression vitest can evaluate knows what `RecordAlert`
  // declares.
  //
  // So: **`npx tsc --noEmit` is the guard for this contract, and it is the
  // only one.** Under the mutation it produced six errors, which is the whole
  // of the protection. This repo has NO CI typecheck — `tsc` is run by hand —
  // so a green `npm test` here says nothing about whether the type still
  // matches the wire. Run both, or the check that reassures you is the one
  // that cannot fire.
  //
  // The runtime assertions below are still worth their lines: they pin the
  // WIRE fixture, so a future edit that quietly reshapes `WIRE` to agree with
  // a wrong type has to do it in the open.
  const TYPE_KEYS: Record<keyof RecordAlert, true> = {
    level: true, code: true, source: true, message: true,
    measured_hz: true, fps: true, tolerance: true, held_s: true,
  };

  it("types every key the recorder emits, and invents none", () => {
    // Both directions, against the literal key list `_rate_alerts()` builds.
    // A field dropped from the type fails here; a field the wire never sends
    // fails here too.
    expect(Object.keys(TYPE_KEYS).sort()).toEqual([
      "code", "fps", "held_s", "level", "measured_hz", "message", "source",
      "tolerance",
    ]);
    // ...and the fixture really is that shape, so the list above is the wire's
    // and not just the type's.
    expect(Object.keys({ ...WIRE }).sort()).toEqual(Object.keys(TYPE_KEYS).sort());
  });

  it("carries the sentence in `message`, and a DURATION in `held_s`", () => {
    // The two phantoms, named. `detail` was the only text-shaped field the old
    // type offered, so the first consumer would have reached for it and drawn
    // an empty row. `since` invites `new Date(since * 1000)`; `held_s` is 3.2
    // seconds of elapsed breach, and read as a timestamp it dates the alert to
    // January 1970.
    const alert: RecordAlert = { ...WIRE };
    expect(alert.message).toContain("tick rate has been outside");
    expect(alert.held_s).toBe(3.2);
    // Against TYPE_KEYS, not against the fixture — `"detail" in {...WIRE}` is
    // false however the type is declared, which is the vacuity described above.
    expect(TYPE_KEYS).not.toHaveProperty("detail");
    expect(TYPE_KEYS).not.toHaveProperty("since");
  });

  it("keeps `fps` as the DECLARED rate, distinct from the measured one", () => {
    // Two rate-shaped numbers on one object, and the bound is a ratio against
    // the declared one. Reading `fps` as the measured rate makes the ratio 1.0
    // and the alert self-justifying.
    const alert: RecordAlert = { ...WIRE };
    expect(alert.fps).toBe(30);
    expect(alert.measured_hz).toBe(28.4);
    expect(alert.fps).not.toBe(alert.measured_hz);
  });
});
