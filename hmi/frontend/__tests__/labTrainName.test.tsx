// hmi/frontend/__tests__/labTrainName.test.tsx
//
// The launcher's job name: derived by default, editable, and never a lie.
//
// The bug this file exists for is not a crash. `job_name` was `act_<dataset>`,
// the field is STICKY, and Oscar typed `act_hilti_box_91` into it once — so
// three different runs (10000 steps, 10000 steps, 90000 steps, one of them
// dead) arrived in the run list under one identical name. Nothing threw and
// nothing looked wrong; the list was simply no longer a list of distinct
// things.
//
// So the claims below are about what the operator can TELL APART, and the
// last one is the one that keeps the feature honest: the name that goes on
// the wire is the name the box was showing.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

import { TrainLauncher } from "@/components/lab/TrainLauncher";
import { primeSticky } from "@/components/cockpit/lib";
import type { DatasetSummary } from "@/lib/lab";

const REPO = "local/so101_pick_cube";

const DATASET: DatasetSummary = {
  repo_id: REPO,
  task: "Pick up the battery and place it in the box",
  episodes: 113,
  frames: 120_000,
  duration_s: 4000,
  size_bytes: 1_000_000,
  marks: { keep: 91, reject: 22, unset: 0, train: 91 },
  is_backup: false,
  rig: "solo",
};

/** The launcher's own reads, routed by path. Ordered rather than keyed:
 *  `/lab/runs` is a prefix of `/lab/runs/train`, and a fragment match would
 *  answer the launch POST with the name list. */
function routeFetch(extra: [RegExp, unknown][] = []) {
  const calls: { url: string; init?: RequestInit }[] = [];
  const routes: [RegExp, unknown][] = [
    ...extra,
    [/\/lab\/runs\/train/, { id: "train-1" }],
    [/\/lab\/runs\?/, { runs: [] }],
    [/\/lab\/runs\//, { id: "train-1", kind: "train", name: "x", status: "queued" }],
    [/\/lab\/datasets\/split/, { train_episodes: [0, 1], eval_episodes: [2], order: [0, 1, 2] }],
    // Not fatal when it fails and not needed here: the episode list only
    // pins `episodes` into the spec. An empty one keeps the derived name on
    // the picker's `marks.train`, which is what an un-reviewed dataset does.
    [/\/lab\/datasets\/detail/, 404],
  ];
  const spy = vi.spyOn(globalThis, "fetch").mockImplementation(
    async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      calls.push({ url, init });
      for (const [pattern, body] of routes) {
        if (!pattern.test(url)) continue;
        if (typeof body === "number") {
          return new Response("{}", { status: body }) as Response;
        }
        return new Response(JSON.stringify(body), {
          status: 200, headers: { "content-type": "application/json" },
        }) as Response;
      }
      return new Response("{}", { status: 404 }) as Response;
    },
  );
  return { calls, spy };
}

/** The launcher's form is sticky in a module map that outlives a test, so
 *  every test states the whole form it means to render. */
function primeForm(over: Record<string, unknown> = {}) {
  primeSticky("lab.train.launcher", {
    policy: "act", steps: 90000, batch: 16, evalSplit: 0.1, seed: 42,
    mode: "random", evalEvery: 1000, saveEvery: 5000, workers: 4,
    device: "cuda", jobName: "", tags: [],
    ...over,
  });
}

function mount() {
  return render(
    <TrainLauncher
      datasets={[DATASET]}
      repoId={REPO}
      onRepoId={() => {}}
      onLaunched={() => {}}
    />,
  );
}

const nameBox = () => screen.getByLabelText(/job name/i) as HTMLInputElement;

beforeEach(() => vi.restoreAllMocks());
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("the job name the launcher offers", () => {
  it("names the dataset and the settings, without being typed into", async () => {
    routeFetch();
    primeForm();
    mount();
    await waitFor(() =>
      expect(nameBox().placeholder).toBe("act_so101_pick_cube_91ep_90k_b16_ev10_s42"),
    );
    // Empty box, derived name: what is shown is a DEFAULT and not an entry,
    // so changing a setting is still free.
    expect(nameBox().value).toBe("");
  });

  it("moves when a hyperparameter moves", async () => {
    routeFetch();
    primeForm();
    mount();
    const before = await waitFor(() => nameBox().placeholder);
    fireEvent.change(screen.getByLabelText(/^steps$/i), { target: { value: "10000" } });
    await waitFor(() => expect(nameBox().placeholder).not.toBe(before));
    expect(nameBox().placeholder).toContain("_10k_");
  });

  it("steps around a name a previous run already took", async () => {
    // The relaunch-after-a-crash case, which is the one in the screenshot.
    routeFetch([[
      /\/lab\/runs\?/,
      { runs: [{ id: "train-0", kind: "train", name: "act_so101_pick_cube_91ep_90k_b16_ev10_s42", status: "failed" }] },
    ]]);
    primeForm();
    mount();
    await waitFor(() =>
      expect(nameBox().placeholder).toBe("act_so101_pick_cube_91ep_90k_b16_ev10_s42_v2"),
    );
  });

  it("lets a typed name win, and gives it back", async () => {
    routeFetch();
    primeForm();
    mount();
    const derived = await waitFor(() => nameBox().placeholder);

    fireEvent.change(nameBox(), { target: { value: "hilti_demo" } });
    expect(nameBox().value).toBe("hilti_demo");

    // `auto` is the way back. Without it the sticky name outlives the run it
    // was typed for and silently re-labels every run after it — which is the
    // original defect, reintroduced by the fix.
    fireEvent.click(screen.getByRole("button", { name: /auto/i }));
    await waitFor(() => expect(nameBox().value).toBe(""));
    expect(nameBox().placeholder).toBe(derived);
  });

  it("sends the name the box was showing", async () => {
    const { calls } = routeFetch();
    primeForm();
    mount();
    const derived = await waitFor(() => nameBox().placeholder);

    fireEvent.click(screen.getByRole("button", { name: /start training/i }));
    const post = await waitFor(() => {
      const c = calls.find((x) => x.url.includes("/lab/runs/train"));
      if (!c) throw new Error("no launch");
      return c;
    });
    const spec = JSON.parse(String((post.init as RequestInit).body)).spec;
    expect(spec.job_name).toBe(derived);
  });
});
