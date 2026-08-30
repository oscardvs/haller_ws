// hmi/frontend/__tests__/labPolicyInputs.test.tsx
//
// Which dataset columns the policy is allowed to read.
//
// LeRobot builds a policy's observation space out of the DATASET — every
// `observation.*` column becomes an input, with no way to opt one out. On
// 2026-08-29 a schema migration added three columns to `local/so101_pick_cube`
// and every run launched afterwards fed them to ACT unasked, including
// `observation.wall_clock`: a per-episode clock a policy can fit instead of
// looking at the image.
//
// So the assertion that matters here is a NEGATIVE one — that the columns the
// operator did not tick are absent from the launch. A launcher that renders
// the chips correctly and still posts the whole set would look right and
// change nothing.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

import { TrainLauncher } from "@/components/lab/TrainLauncher";
import { primeSticky } from "@/components/cockpit/lib";
import type { DatasetDetail, DatasetSummary } from "@/lib/lab";

const REPO = "local/so101_pick_cube";

const DATASET: DatasetSummary = {
  repo_id: REPO,
  task: "Pick up the battery and place it in the box",
  episodes: 3,
  frames: 1800,
  duration_s: 60,
  size_bytes: 1_000_000,
  marks: { keep: 3, reject: 0, unset: 0, train: 3 },
  is_backup: false,
  rig: "solo",
};

/** The real dataset's columns after the 2026-08-29 migration: the two the
 *  policy should read, and the three it should not. */
const DETAIL = {
  repo_id: REPO,
  root: `/home/odesha/robot-data/lerobot/${REPO}`,
  fps: 30,
  robot_type: "so_follower",
  tasks: ["Pick up the battery and place it in the box"],
  video_keys: ["observation.images.top"],
  features: {
    action: { dtype: "float32", shape: [6], names: null },
    "observation.state": { dtype: "float32", shape: [6], names: null },
    "observation.images.top": {
      dtype: "video", shape: [480, 640, 3],
      names: ["height", "width", "channels"],
    },
    "observation.effort": { dtype: "float32", shape: [6], names: null },
    "observation.base": { dtype: "float32", shape: [2], names: ["v", "omega"] },
    "observation.wall_clock": { dtype: "float32", shape: [1], names: ["t"] },
  },
  policy_inputs_default: ["observation.state", "observation.images.top"],
  rig: "solo",
  episodes: [0, 1, 2].map((i) => ({
    episode_index: i, label: i + 1, mark: "keep", frames: 600, seconds: 20,
  })),
} as unknown as DatasetDetail;

function routeFetch(detail: unknown = DETAIL) {
  const calls: { url: string; init?: RequestInit }[] = [];
  const routes: [RegExp, unknown][] = [
    [/\/lab\/datasets\/detail/, detail],
    [/\/lab\/runs\/train/, { id: "train-1" }],
    [/\/lab\/runs\?/, { runs: [] }],
    [/\/lab\/runs\//, { id: "train-1", kind: "train", name: "x", status: "queued" }],
    [/\/lab\/datasets\/split/, {
      train_episodes: [0, 1], eval_episodes: [2], order: [0, 1, 2],
    }],
  ];
  vi.spyOn(globalThis, "fetch").mockImplementation(
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
  return { calls };
}

function primeForm() {
  primeSticky("lab.train.launcher", {
    policy: "act", steps: 90000, batch: 16, evalSplit: 0.15, seed: 42,
    mode: "random", evalEvery: 5000, saveEvery: 10000, workers: 8,
    device: "cuda", jobName: "", tags: [],
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

const chip = (name: string) =>
  screen.getByRole("button", { name: new RegExp(`^${name}`) });

/** The spec the launch POST actually carried. */
async function launchedSpec(calls: { url: string; init?: RequestInit }[]) {
  fireEvent.click(screen.getByRole("button", { name: /start training/i }));
  const post = await waitFor(() => {
    const c = calls.find((x) => x.url.includes("/lab/runs/train"));
    if (!c) throw new Error("no launch");
    return c;
  });
  return JSON.parse(String((post.init as RequestInit).body)).spec;
}

beforeEach(() => {
  vi.restoreAllMocks();
  primeForm();
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("the observation space the launcher offers", () => {
  it("offers every observation column and no other", async () => {
    routeFetch();
    mount();

    const group = await screen.findByRole("group", { name: /policy inputs/i });

    expect(group).toHaveTextContent("state");
    expect(group).toHaveTextContent("images.top");
    expect(group).toHaveTextContent("wall_clock");
    // `action` is the policy's OUTPUT. Offering it would be offering to train
    // a policy on its own labels.
    expect(group).not.toHaveTextContent("action");
  });

  it("ticks the state vector and the cameras, and nothing the migration added", async () => {
    routeFetch();
    mount();

    await screen.findByRole("group", { name: /policy inputs/i });

    expect(chip("state")).toHaveAttribute("aria-pressed", "true");
    expect(chip("images.top")).toHaveAttribute("aria-pressed", "true");
    expect(chip("effort")).toHaveAttribute("aria-pressed", "false");
    expect(chip("base")).toHaveAttribute("aria-pressed", "false");
    expect(chip("wall_clock")).toHaveAttribute("aria-pressed", "false");
  });

  it("takes the default from the SERVER rather than deriving its own", async () => {
    // The backend validates the choice on the way back in; a browser that
    // derived its own default would drift from that rule, and the drift would
    // be a policy trained on a space the form never showed.
    routeFetch({
      ...DETAIL,
      policy_inputs_default: ["observation.state", "observation.effort"],
    });
    mount();

    await screen.findByRole("group", { name: /policy inputs/i });

    expect(chip("effort")).toHaveAttribute("aria-pressed", "true");
    expect(chip("images.top")).toHaveAttribute("aria-pressed", "false");
  });

  it("posts only the ticked columns", async () => {
    const { calls } = routeFetch();
    mount();
    await screen.findByRole("group", { name: /policy inputs/i });

    const spec = await launchedSpec(calls);

    expect(spec.policy_inputs).toEqual([
      "observation.state", "observation.images.top",
    ]);
  });

  it("keeps the clock out of the launch — the whole point of the row", async () => {
    const { calls } = routeFetch();
    mount();
    await screen.findByRole("group", { name: /policy inputs/i });

    const spec = await launchedSpec(calls);

    expect(spec.policy_inputs).not.toContain("observation.wall_clock");
    expect(spec.policy_inputs).not.toContain("observation.effort");
    expect(spec.policy_inputs).not.toContain("observation.base");
  });

  it("adds a column back when it is ticked, in the dataset's own order", async () => {
    const { calls } = routeFetch();
    mount();
    await screen.findByRole("group", { name: /policy inputs/i });

    fireEvent.click(chip("effort"));
    const spec = await launchedSpec(calls);

    // Dataset order, not click order: the map handed to LeRobot is read back
    // as the run's record of what it trained on.
    expect(spec.policy_inputs).toEqual([
      "observation.state", "observation.images.top", "observation.effort",
    ]);
  });

  it("drops a column when it is unticked", async () => {
    const { calls } = routeFetch();
    mount();
    await screen.findByRole("group", { name: /policy inputs/i });

    fireEvent.click(chip("images.top"));
    const spec = await launchedSpec(calls);

    expect(spec.policy_inputs).toEqual(["observation.state"]);
  });

  it("says how many columns it is holding out", async () => {
    routeFetch();
    mount();

    const group = await screen.findByRole("group", { name: /policy inputs/i });

    expect(group).toHaveTextContent(/3 columns held out/i);
  });

  it("omits the field entirely when the dataset never loaded", async () => {
    // Absent means "LeRobot's own rule". Sending a list built from a dataset
    // this form could not read would pin the wrong space confidently.
    const { calls } = routeFetch(404);
    mount();
    await waitFor(() =>
      expect(screen.queryByRole("group", { name: /policy inputs/i })).toBeNull());

    const spec = await launchedSpec(calls);

    expect(spec).not.toHaveProperty("policy_inputs");
  });
});
