// hmi/frontend/__tests__/labRuns.test.tsx
//
// The Train surface's contract with the backend it actually talks to.
//
// Every defect these pin was found by RENDERING the real `/lab/runs` routes
// against a real recorded ACT run, and every one of them type-checked. That is
// the point of the file: the Train and Compare panes had no tests at all, and
// the only prior evidence for them was a hand-written mock, which agreed with
// the types rather than with the server.
//
// The assertions below are written from the CLAIM and in the claim's own
// terms — the literal strings `runs.py` emits, the literal path shape
// `_checkpoint_wire` sends — rather than from the components, because a test
// written from the code inherits the code's blind spot. Three of these four
// bugs were a value the type said was a `number` and the wire said was
// something else; none of them threw, and each rendered a plausible wrong
// answer instead.
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

import { CheckpointList, checkpointName } from "@/components/lab/CheckpointList";
import { ComparePane, batchKeys, mergeSeries } from "@/components/lab/ComparePane";
import { MetricGrid } from "@/components/lab/MetricGrid";
import { RunDetail } from "@/components/lab/RunDetail";
import { RunList, epochSeconds, fullWhen, shortWhen } from "@/components/lab/RunList";
import {
  metricKeys, metricX, plottableMetricKeys,
  type Checkpoint, type MetricRow, type Run, type RunSummary,
} from "@/lib/lab";

/* ─── the wire, verbatim ──────────────────────────────────────────────────
 * Captured from `GET /lab/runs` on a real backend serving Oscar's real ACT
 * training run (46-episode so101_pick_cube, 60k steps). Kept as literals so
 * this file states the contract independently of anything that produces it. */

const STARTED = "2026-08-26T19:33:50+00:00";
const FINISHED = "2026-08-26T20:27:00+00:00";
/** 19:33:50 -> 20:27:00 is 53m10s. The one number here that is arithmetic
 *  rather than convention, so it is spelled out rather than computed. */
const RAN_SECONDS = 53 * 60 + 10;

const CK_PATH =
  "/home/odesha/robot-data/runs/train-20260826-213350-act_so101_pick_cube" +
  "/train/checkpoints/060000/pretrained_model";
const CK_LAST_PATH =
  "/home/odesha/robot-data/runs/train-20260826-213350-act_so101_pick_cube" +
  "/train/checkpoints/last/pretrained_model";

beforeEach(() => vi.restoreAllMocks());
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

/* ─── timestamps ──────────────────────────────────────────────────────── */

describe("run timestamps are ISO 8601 strings, not unix numbers", () => {
  // `runs.py::_now()` is `datetime.now(UTC).isoformat(timespec="seconds")`,
  // and the catalog sorts the field as a STRING (`r.get("started_at") or ""`).
  // The type said `number | null` and the formatters gated on
  // `typeof ts === "number"`, so every run in the list and the detail pane
  // rendered `—`. It never threw and it never failed a type-check.

  it("reads the stamp the backend actually sends", () => {
    expect(epochSeconds(STARTED)).toBe(Date.parse(STARTED) / 1000);
    // The assertion that would have caught the original bug, stated as the
    // claim rather than as a formatting preference: this must not be "—".
    expect(shortWhen(STARTED)).not.toBe("—");
    expect(fullWhen(STARTED)).toBeDefined();
  });

  it("still says — for the absent cases, which are real", () => {
    // A queued run has no start yet, and `TrainLauncher` mounts an optimistic
    // row with both stamps null before the backend has answered.
    for (const absent of [null, undefined, ""]) {
      expect(epochSeconds(absent)).toBeNull();
      expect(shortWhen(absent)).toBe("—");
      expect(fullWhen(absent)).toBeUndefined();
    }
  });

  it("refuses a unix number rather than rendering 1970", () => {
    // Pinning the boundary and not just the fixed value: the old code accepted
    // a number and multiplied it by 1000. If someone reintroduces that path,
    // a number must not quietly start working again — the backend does not
    // send one, so anything that does is a bug upstream and should read as
    // absent rather than as a date in 1970.
    // @ts-expect-error — the wire type is `string | null`; this is the shape
    // the field was WRONGLY declared as until 2026-08-27.
    expect(shortWhen(1756230830)).toBe("—");
  });

  it("computes elapsed from the two stamps", () => {
    // `RunDetail` gated elapsed on `typeof started_at === "number"`, so no run
    // ever reported how long it took. Measured against the real pair.
    const secs = epochSeconds(FINISHED)! - epochSeconds(STARTED)!;
    expect(secs).toBe(RAN_SECONDS);
  });
});

/* ─── the runs list is a log of experiments ───────────────────────────── */

describe("the runs list reads a row as an experiment, not a process", () => {
  const summary = (over: Partial<RunSummary>): RunSummary => ({
    id: "train-x", kind: "train", name: "cube baseline", status: "done",
    started_at: STARTED, finished_at: FINISHED,
    spec_summary:
      "train · local/so101_pick_cube · 35 of 46 episodes · act · 60000 steps",
    tags: ["baseline", "35ep"],
    ...over,
  });

  function mount(runs: RunSummary[], now: number) {
    return render(
      <RunList
        runs={runs}
        loading={false}
        error={null}
        selectedId={null}
        onSelect={() => {}}
        compare={new Set()}
        onToggleCompare={() => {}}
        now={now}
      />,
    );
  }

  it("measures a finished run against its own two stamps", () => {
    // 19:33:50 -> 20:27:00 is 53m10s — the stamp pair, not a wall clock.
    mount([summary({})], 0);
    expect(screen.getByText("53m 10s")).toBeTruthy();
  });

  it("measures a running run against the poll's clock, and no other", () => {
    // The list owns no timer: the pane hands down the moment of its last
    // read, and the row's elapsed moves exactly when the poll does.
    mount(
      [summary({ status: "running", finished_at: null })],
      epochSeconds(STARTED)! + 125,
    );
    expect(screen.getByText("2m 05s")).toBeTruthy();
  });

  it("invents nothing for a queued run", () => {
    // No start stamp and no clock: both cells read "—", like every other
    // absent reading on the page, rather than an elapsed counted from 1970.
    mount([summary({ status: "queued", started_at: null, finished_at: null })], 0);
    expect(screen.getAllByText("—")).toHaveLength(2);
  });

  it("prints the backend's spec summary verbatim, with the launch tags", () => {
    mount([summary({})], 0);
    expect(
      screen.getByText("train · local/so101_pick_cube · 35 of 46 episodes · act · 60000 steps"),
    ).toBeTruthy();
    expect(screen.getByText("baseline")).toBeTruthy();
    expect(screen.getByText("35ep")).toBeTruthy();
  });
});

/* ─── a metric chart fills the metrics panel on demand ────────────────── */

describe("MetricGrid — any chart can fill the metrics panel", () => {
  // Three logged steps of a real-looking ACT row: enough for a curve, few
  // enough that LTTB keeps them all.
  const ROWS: MetricRow[] = [
    { steps: 100, loss: 7.25, grad_norm: 159.6 },
    { steps: 200, loss: 3.1, grad_norm: 80.2 },
    { steps: 300, loss: 1.4, grad_norm: 40.0 },
  ];

  function mount() {
    // The overlay positions against the nearest relative ancestor — on the
    // page that is the metrics Panel; here a plain relative div plays it.
    return render(
      <div className="relative">
        <MetricGrid rows={ROWS} steps={null} />
      </div>,
    );
  }

  it("maximises one cell and restores it from the same toggle", () => {
    mount();
    fireEvent.click(screen.getByRole("button", { name: "maximize loss chart" }));
    expect(document.querySelector("[data-chart-zoom='loss']")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "restore loss chart" }));
    expect(document.querySelector("[data-chart-zoom]")).toBeNull();
  });

  it("Escape restores the grid", () => {
    mount();
    fireEvent.click(screen.getByRole("button", { name: "maximize grad_norm chart" }));
    expect(document.querySelector("[data-chart-zoom='grad_norm']")).toBeTruthy();

    fireEvent.keyDown(window, { key: "Escape" });
    expect(document.querySelector("[data-chart-zoom]")).toBeNull();
  });

  it("reads values at the mouse, with the EMA underlay out of the readout", () => {
    mount();
    // Smoothing draws the raw series underneath at low opacity. It is a
    // drawn guide, not a second metric — a probe that read it out would
    // answer "what is loss here" twice with two different numbers.
    fireEvent.change(screen.getByLabelText("ema smoothing weight"), {
      target: { value: "0.5" },
    });
    fireEvent.click(screen.getByRole("button", { name: "maximize loss chart" }));
    const zoomed = document.querySelector("[data-chart-zoom='loss']") as HTMLElement;

    const svg = within(zoomed).getByRole("img", { name: "loss against step" });
    fireEvent.pointerMove(svg, { clientX: 100, clientY: 10 });

    const readout = zoomed.querySelector("[data-probe-readout]")!;
    expect(readout).toHaveTextContent("loss");
    expect(readout.textContent).not.toContain("~raw");
  });
});

/* ─── checkpoint identity ─────────────────────────────────────────────── */

describe("a checkpoint is named by its step directory", () => {
  // `_checkpoint_wire` sends the path of the MODEL directory inside the
  // checkpoint, because that is what a rollout is pointed at. Taking the last
  // segment named all thirteen rows `pretrained_model`.

  it("names the checkpoint, not the model directory inside it", () => {
    expect(checkpointName(CK_PATH)).toBe("060000");
    expect(checkpointName(CK_LAST_PATH)).toBe("last");
  });

  it("gives every checkpoint in a real set a DISTINCT name", () => {
    // The sharper claim, and the one a per-item assertion cannot make: the
    // column exists to tell rows apart. Thirteen identical strings passed
    // every "does it render something" check.
    const paths = [5000, 10000, 15000, 60000]
      .map((s) => CK_PATH.replace("060000", String(s).padStart(6, "0")))
      .concat(CK_LAST_PATH);
    const names = paths.map(checkpointName);
    expect(new Set(names).size).toBe(paths.length);
  });

  it("falls back to the last segment for a path in another shape", () => {
    expect(checkpointName("/runs/x/train/checkpoints/060000")).toBe("060000");
  });
});

describe("CheckpointList renders lerobot's `last` symlink", () => {
  const ck = (over: Partial<Checkpoint>): Checkpoint => ({
    step: 60000, path: CK_PATH, has_model: true, ...over,
  });

  function mountWith(checkpoints: Checkpoint[]) {
    vi.spyOn(globalThis, "fetch").mockImplementation(
      async () => new Response(JSON.stringify({ checkpoints }), { status: 200 }),
    );
    return render(<CheckpointList runId="train-x" status="done" />);
  }

  it("shows — for the null step and keeps the row identifiable", async () => {
    // `step` is null for `last` ON PURPOSE — it is the only thing telling it
    // from the numbered checkpoint it points at. Typed `number`, it rendered
    // as an empty cell beside a name that was `pretrained_model` on every
    // row, so the one row that needed identifying had nothing to identify it.
    mountWith([ck({}), ck({ step: null, path: CK_LAST_PATH })]);
    await waitFor(() => expect(screen.getByText("last")).toBeTruthy());
    expect(screen.getByText("060000")).toBeTruthy();
    expect(screen.getByText("—")).toBeTruthy();
  });

  it("puts `latest` on the highest numbered step, never on the alias", async () => {
    // `last` POINTS AT the newest checkpoint; badging the pointer instead of
    // the thing would be a second answer to the same question. And with every
    // step null, an unguarded `c.step === latest` is `null === null` — a badge
    // on every row.
    mountWith([ck({}), ck({ step: null, path: CK_LAST_PATH })]);
    await waitFor(() => expect(screen.getByText("last")).toBeTruthy());
    const badges = screen.getAllByText("latest");
    expect(badges).toHaveLength(1);
    // The badge sits in the 060000 row, not the `last` row.
    expect(badges[0].closest("div")?.textContent).toContain("060000");
  });

  it("badges nothing when every step is null", async () => {
    // The case the guard actually exists for, and the one the test above
    // CANNOT see: with a numbered checkpoint present, `null === 60000` is
    // false either way, so that test passes with or without the null check.
    // Reachable when the numbered checkpoints have been cleaned up and only
    // lerobot's `last` symlink remains — then an unguarded comparison is
    // `null === null` and badges the alias as the newest thing on disk.
    mountWith([ck({ step: null, path: CK_LAST_PATH })]);
    await waitFor(() => expect(screen.getByText("last")).toBeTruthy());
    expect(screen.queryByText("latest")).toBeNull();
  });
});

/* ─── only chart what can be drawn ────────────────────────────────────── */

describe("a chart is offered only for a key that can sit on an axis", () => {
  // The first line of a real training `metrics.jsonl` is bookkeeping, not a
  // sample. Verbatim from the run this file is written against:
  const SPLIT: MetricRow = { kind: "split", train_episodes: 28, eval_episodes: 7 };
  const TRAIN: MetricRow = {
    kind: "train", steps: 200, loss: 7.25, grad_norm: 159.6, lr: 1e-5,
  };
  const EVAL: MetricRow = { kind: "eval", steps: 5000, eval_loss: 0.453 };

  it("drops keys that sit on NO axis", () => {
    const rows = [SPLIT, TRAIN, EVAL];
    // They ARE logged, and `metricKeys` is right to say so...
    expect(metricKeys(rows)).toContain("train_episodes");
    // ...but they have no step, epoch or wall, so they can never be drawn.
    // Two permanently empty cells labelled with metrics the run does not
    // measure over time.
    expect(plottableMetricKeys(rows)).not.toContain("train_episodes");
    expect(plottableMetricKeys(rows)).not.toContain("eval_episodes");
  });

  it("keeps every key that has a real sample", () => {
    // The other half of the claim, and the one that stops this from being a
    // filter that quietly eats data.
    const keys = plottableMetricKeys([SPLIT, TRAIN, EVAL]);
    expect(keys).toEqual(expect.arrayContaining(["loss", "grad_norm", "lr", "eval_loss"]));
  });

  it("keeps a key drawable on SOME axis but not the current one", () => {
    // The distinction the filter turns on. `eval_loss` here carries a step but
    // no epoch: on the epoch axis it draws nothing, and the grid says "no
    // samples on this axis" — which is TRUE and has a remedy. Filtering
    // per-axis instead would delete the chart and, with it, the only hint
    // that the metric exists at all.
    expect(plottableMetricKeys([EVAL])).toContain("eval_loss");
    expect(metricX(EVAL, "epoch")).toBeNull();
    expect(metricX(EVAL, "step")).toBe(5000);
  });
});

/* ─── tags on the detail ──────────────────────────────────────────────── */

describe("RunDetail shows the tags a run was launched with", () => {
  // Written only now, and the delay was the point. `GET /lab/runs/{id}` used
  // to add `tags` solely in its `if not detail:` branch, so the field never
  // reached this pane and `RunDetail`'s TagChips could not fire — while
  // `RunSummary.tags` being optional kept it type-checking. A test written
  // then would have handed the component a run WITH tags, passed, and
  // "proved" a feature the real backend never fed: the mock-agrees-with-the-
  // type failure that hid four defects this morning. The route was fixed at
  // `ff537da` and verified on the wire before this was written.
  it("renders them once the route actually sends the field", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(
      async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/metrics")) {
          return new Response(JSON.stringify({ offset: 0, rows: [] }), { status: 200 });
        }
        if (url.includes("/log")) {
          return new Response(JSON.stringify({ offset: 0, text: "" }), { status: 200 });
        }
        if (url.includes("/checkpoints")) {
          return new Response(JSON.stringify({ checkpoints: [] }), { status: 200 });
        }
        return new Response(JSON.stringify({
          id: "train-x", kind: "train", name: "tagged", status: "done",
          started_at: STARTED, finished_at: FINISHED,
          tags: ["baseline", "35ep"], spec: { policy_type: "act" },
        }), { status: 200 });
      },
    );
    render(<RunDetail runId="train-x" />);
    // Asserted case-INSENSITIVELY on purpose: the chips are uppercased by CSS,
    // and `innerText` reflects `text-transform`. Matching the literal lowercase
    // string reports a working feature as broken, which cost me two false
    // readings against the live page before I noticed.
    await waitFor(() => expect(screen.getByText(/baseline/i)).toBeTruthy());
    expect(screen.getByText(/35ep/i)).toBeTruthy();
  });
});

/* ─── the compare cap is READ, never copied ───────────────────────────── */

describe("compare batches to the cap the server publishes", () => {
  it("splits the key list at the cap", () => {
    expect(batchKeys(["a", "b", "c", "d", "e"], 2)).toEqual([["a", "b"], ["c", "d"], ["e"]]);
    expect(batchKeys(["a", "b"], 8)).toEqual([["a", "b"]]);
    expect(batchKeys([], 8)).toEqual([]);
    // A nonsense cap must not produce an infinite loop of empty requests.
    expect(batchKeys(["a", "b"], 0)).toEqual([["a"], ["b"]]);
  });

  it("unions the keys per run rather than replacing them", () => {
    // Batches carry DISJOINT keys for the SAME runs, so a top-level spread
    // would keep only the last batch's keys for every run — every chart but
    // the final batch's would silently vanish, which is indistinguishable
    // from a run that never logged them.
    const merged = mergeSeries([
      { r1: { loss: [[0, 1]] }, r2: { loss: [[0, 2]] } },
      { r1: { lr: [[0, 3]] }, r2: { lr: [[0, 4]] } },
    ]);
    expect(Object.keys(merged.r1).sort()).toEqual(["loss", "lr"]);
    expect(Object.keys(merged.r2).sort()).toEqual(["loss", "lr"]);
  });
});

describe("ComparePane reads the cap from the server", () => {
  /** Mount with `n` chartable keys and whatever `/lab/system` answers, and
   *  report the sizes of the cross-run metrics requests that resulted. */
  async function requestSizes(system: unknown, keyCount: number) {
    const row: Record<string, number> = { steps: 1 };
    for (let i = 0; i < keyCount; i += 1) row[`k${i}`] = i;
    const sizes: number[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(
      async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/lab/system")) {
          if (typeof system === "number") {
            return new Response(JSON.stringify({ detail: "no" }), { status: system });
          }
          return new Response(JSON.stringify(system), { status: 200 });
        }
        if (url.includes("/lab/runs/metrics")) {
          const keys = decodeURIComponent(url).match(/keys=([^&]*)/)?.[1] ?? "";
          sizes.push(keys.split(",").filter(Boolean).length);
          return new Response(JSON.stringify({ runs: {} }), { status: 200 });
        }
        if (url.includes("/metrics")) {
          return new Response(JSON.stringify({ offset: 0, rows: [row] }), { status: 200 });
        }
        if (url.includes("/lab/runs/")) {
          const id = decodeURIComponent(url.split("/lab/runs/")[1].split("?")[0]);
          return new Response(JSON.stringify({
            id, kind: "train", name: id, status: "done",
            started_at: STARTED, finished_at: FINISHED, tags: [], spec: {},
          }), { status: 200 });
        }
        return new Response(JSON.stringify({}), { status: 200 });
      },
    );
    render(<ComparePane runIds={["train-a", "train-b"]} />);
    await waitFor(() => expect(sizes.length).toBeGreaterThan(0));
    return sizes;
  }

  it("follows a published cap that is NOT the fallback", async () => {
    // The decisive test, and the only one that can tell a live read from a
    // hardcoded copy: `8` resolving to `8` proves nothing, because the
    // fallback is also 8. So the server publishes 3 and the request count has
    // to follow to ceil(10/3) = 4. Verified the same way against the real
    // backend by republishing `compare.MAX_KEYS` and watching [8,2] become
    // [3,3,3,1].
    const sizes = await requestSizes({ compare_max_keys: 3 }, 10);
    await waitFor(() => expect(sizes).toEqual([3, 3, 3, 1]));
  });

  it("falls back — visibly — when the backend publishes no cap", async () => {
    // The ABSENT case, pinned as its own test so it cannot silently become a
    // second source of truth. An older backend answers `/lab/system` without
    // the field; batching must still happen, at the named fallback of 8.
    const sizes = await requestSizes({ lerobot_home: "/x" }, 10);
    await waitFor(() => expect(sizes).toEqual([8, 2]));
  });

  it("falls back when /lab/system cannot be read at all", async () => {
    // A 404 is a backend with no Lab system route. It must cost nothing: an
    // unreadable cap is exactly what the fallback is for, and it must not
    // take the curves down with it.
    const sizes = await requestSizes(404, 10);
    await waitFor(() => expect(sizes).toEqual([8, 2]));
  });
});

/* ─── compare degrades to the curves, never the page ──────────────────── */

describe("ComparePane survives a refused metrics request", () => {
  const run = (id: string, over: Partial<Run> = {}): Run => ({
    id, kind: "train", name: id, status: "done",
    started_at: STARTED, finished_at: FINISHED, tags: [], spec_summary: "",
    spec: { repo_id: "local/so101_pick_cube", policy_type: "act" }, ...over,
  });

  it("keeps the runs on screen and names the refusal", async () => {
    // A real ACT run logs 12 numeric keys; `compare.py::MAX_KEYS` is 8. So the
    // ONLY kind of run worth comparing refuses by default, and this was fatal:
    // the 400 reached the outer catch, `state` stayed null, and the whole page
    // became the sentence "compare failed". The run list, the legend and the
    // hparam diff were all lost over a chart that could not be drawn.
    const REFUSAL = "too many keys: 12 — at most 8 per request";
    vi.spyOn(globalThis, "fetch").mockImplementation(
      async (input: RequestInfo | URL) => {
        const url = String(input);
        // The CROSS-RUN endpoint is `/lab/runs/metrics?ids=`, which is a
        // prefix-collision with the per-run `/lab/runs/<id>/metrics` — so it
        // is matched first and by its exact shape. Getting this order wrong
        // is how the test silently exercised the wrong route.
        if (url.includes("/lab/runs/metrics")) {
          return new Response(JSON.stringify({ detail: REFUSAL }), { status: 400 });
        }
        if (url.includes("/metrics")) {
          return new Response(
            JSON.stringify({ offset: 0, rows: [{ steps: 1, loss: 0.5 }] }),
            { status: 200 },
          );
        }
        if (url.includes("/lab/runs/")) {
          const id = decodeURIComponent(url.split("/lab/runs/")[1].split("?")[0]);
          return new Response(JSON.stringify(run(id)), { status: 200 });
        }
        return new Response(JSON.stringify({}), { status: 200 });
      },
    );

    render(<ComparePane runIds={["train-a", "train-b"]} />);

    // The refusal is reported in the backend's own words...
    await waitFor(() => expect(screen.getByText(/too many keys/)).toBeTruthy());
    // ...and the page is still a comparison: both runs survived it.
    expect(screen.getAllByText(/train-a|train-b/).length).toBeGreaterThan(0);
    // The sentence that used to replace the entire pane is gone.
    expect(screen.queryByText(/compare failed/)).toBeNull();
  });
});
