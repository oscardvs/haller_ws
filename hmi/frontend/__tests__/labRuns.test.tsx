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
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

import { CheckpointList, checkpointName } from "@/components/lab/CheckpointList";
import { ComparePane } from "@/components/lab/ComparePane";
import { epochSeconds, fullWhen, shortWhen } from "@/components/lab/RunList";
import type { Checkpoint, Run } from "@/lib/lab";

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
