// hmi/frontend/__tests__/labClient.test.ts
//
// The Lab client's wire discipline and its derived readings.
//
// Two classes of thing are tested here, and both are things that fail
// SILENTLY in a browser. A query param that should have been omitted comes
// back as an empty result set that looks like an empty dataset. A video slice
// that is guessed rather than read plays the wrong episode at the operator,
// who then rejects a demonstration that was fine. Neither throws.
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

import { ApiError } from "@/lib/api";
import {
  armGroups, epLabel, isBusy, isDrawableTrace, isForbidden, isMissing,
  isRefused, trainableCount, trainJobName,
  isGripperChannel, lab, labVideoUrl, metricKeys, metricX, qs, reason,
  rigLabel, shortChannel, sliceFor, videoSrcKey,
  type LabEpisode, type MetricRow, type Trace,
} from "@/lib/lab";

/** Routes fetch by path so a call's boot requests need not be ordered.
 *  Mirrors __tests__/cockpitTabs.test.tsx so both read the same way. */
function routeFetch(routes: Record<string, unknown> = {}) {
  const calls: { url: string; init?: RequestInit }[] = [];
  const spy = vi.spyOn(globalThis, "fetch").mockImplementation(
    async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      calls.push({ url, init });
      for (const [fragment, body] of Object.entries(routes)) {
        if (!url.includes(fragment)) continue;
        if (typeof body === "number") {
          return new Response(JSON.stringify({ detail: "nope" }), { status: body });
        }
        return new Response(JSON.stringify(body), { status: 200 });
      }
      return new Response(JSON.stringify({}), { status: 200 });
    },
  );
  return { calls, spy };
}

const body = (c: { init?: RequestInit }) =>
  JSON.parse((c.init as RequestInit).body as string);

function ep(over: Partial<LabEpisode> = {}): LabEpisode {
  return {
    episode_index: 0, label: 1, frames: 855, duration_s: 28.5, share: 0.029,
    task: "Pick up the battery", verdict: "PASS", reasons: [],
    mark: "unset", note: null, tags: [], ...over,
  };
}

beforeEach(() => vi.restoreAllMocks());
afterEach(() => vi.restoreAllMocks());

describe("the refusal predicates say which refusal it was", () => {
  // Three different next moves, and the operator's is different in each: a
  // 404 hides the surface, a 403 says "do it on the machine", a 400 is a
  // decision the backend made about the request itself — the rollout rate
  // gate, which is a sentence to read and act on, not a fault.
  it("separates a 400 from a 404, a 403 and a 409", () => {
    expect(isRefused(new ApiError(400, "refusing to roll out at 25 Hz"))).toBe(true);
    for (const status of [403, 404, 409, 500, 501]) {
      expect(isRefused(new ApiError(status, "x"))).toBe(false);
    }
    expect(isRefused(new Error("network"))).toBe(false);
  });
});

describe("qs", () => {
  it("omits null, undefined and the empty string", () => {
    // An omitted filter and an empty one are different asks: `filter_mark=`
    // is a filter FOR the empty mark, and a server would be right to answer
    // it with nothing — which on screen is indistinguishable from a dataset
    // with no episodes.
    expect(qs({ a: 1, b: null, c: undefined, d: "", e: "x" })).toBe("?a=1&e=x");
  });

  it("returns an empty string rather than a bare ?", () => {
    expect(qs({ a: null })).toBe("");
  });

  it("escapes a repo id, which carries a slash", () => {
    expect(qs({ repo_id: "local/so101_pick_cube" }))
      .toBe("?repo_id=local%2Fso101_pick_cube");
  });

  it("keeps a zero and a false — both are real values", () => {
    expect(qs({ offset: 0, backup: false })).toBe("?offset=0&backup=false");
  });
});

describe("error classification", () => {
  it("separates a missing route from a refusal from a policy", () => {
    // The three mean different things to the operator: rebuild the backend,
    // clear the state, or do it from the machine. A UI that cannot tell them
    // apart has to call all three "failed".
    expect(isMissing(new ApiError(404, "no"))).toBe(true);
    expect(isMissing(new ApiError(501, "no"))).toBe(true);
    expect(isBusy(new ApiError(409, "a run holds it"))).toBe(true);
    expect(isForbidden(new ApiError(403, "loopback only"))).toBe(true);
    expect(isMissing(new ApiError(409, "x"))).toBe(false);
    expect(isBusy(new Error("network"))).toBe(false);
  });

  it("quotes the backend's own sentence, without the HTTP prefix", () => {
    expect(reason(new ApiError(409, "a run is using this dataset")))
      .toBe("a run is using this dataset");
    expect(reason(new Error("Failed to fetch"))).toBe("Failed to fetch");
  });
});

describe("episode numbering", () => {
  it("labels a stored index 1-based", () => {
    // Oscar numbers episodes 1-based in conversation and they are stored
    // 0-based. One implementation, because two is how a dialog offers to
    // delete the wrong demonstration.
    expect(epLabel(0)).toBe("Ep 1");
    expect(epLabel(45)).toBe("Ep 46");
  });
});

describe("video slices", () => {
  const packed = ep({
    episode_index: 2, label: 3, duration_s: 15.533,
    videos: {
      top: { chunk_index: 0, file_index: 1, from_timestamp: 0, to_timestamp: 15.533 },
    },
  });

  it("reads the declared slice", () => {
    expect(sliceFor(packed, "top")).toEqual({
      chunk_index: 0, file_index: 1, from_timestamp: 0, to_timestamp: 15.533,
    });
  });

  it("returns null rather than guessing when none was declared", () => {
    // The guess would be {0, duration_s}. On the real 46-episode dataset,
    // episode 1 lives at 28.5..45.93 of a file that starts with episode 0 —
    // so the guess opens the wrong take and plays it under the right label.
    expect(sliceFor(ep({ episode_index: 1 }), "top")).toBeNull();
    expect(sliceFor(packed, "left_wrist")).toBeNull();
    expect(sliceFor(packed, null)).toBeNull();
  });

  it("keys the loaded FILE, so a second episode in it is a seek", () => {
    const a = ep({
      episode_index: 0,
      videos: { top: { chunk_index: 0, file_index: 0, from_timestamp: 0, to_timestamp: 28.5 } },
    });
    const b = ep({
      episode_index: 1,
      videos: { top: { chunk_index: 0, file_index: 0, from_timestamp: 28.5, to_timestamp: 45.93 } },
    });
    // Same file: the src must not change, or every J/L keypress re-buffers.
    expect(videoSrcKey("r", a, "top")).toBe(videoSrcKey("r", b, "top"));
    // Different file: it must.
    expect(videoSrcKey("r", packed, "top")).not.toBe(videoSrcKey("r", a, "top"));
  });

  it("has no key at all without a slice", () => {
    expect(videoSrcKey("r", ep(), "top")).toBeNull();
  });

  it("builds a video URL carrying the episode, not a chunk path", () => {
    const url = labVideoUrl("local/so101_pick_cube", "top", 2);
    expect(url).toContain("/lab/datasets/video");
    expect(url).toContain("repo_id=local%2Fso101_pick_cube");
    expect(url).toContain("key=top");
    expect(url).toContain("episode=2");
  });
});

describe("rig-shaped trace readings", () => {
  const SOLO = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
  ];
  const BIMANUAL = [
    "left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
    "left_wrist_flex", "left_wrist_roll", "left_gripper",
    "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
    "right_wrist_flex", "right_wrist_roll", "right_gripper",
  ];

  it("groups a solo rig under one unprefixed side", () => {
    expect(armGroups(SOLO)).toEqual([
      { side: "arm", channels: [0, 1, 2, 3, 4, 5] },
    ]);
  });

  it("splits a bimanual rig at the side prefix", () => {
    // Both real datasets on disk, and the channel COUNT differs between them.
    // Anything that hardcodes five joints and a gripper is wrong on one.
    expect(armGroups(BIMANUAL)).toEqual([
      { side: "left", channels: [0, 1, 2, 3, 4, 5] },
      { side: "right", channels: [6, 7, 8, 9, 10, 11] },
    ]);
  });

  it("finds the gripper by NAME on both rigs", () => {
    // The kit took state[state.length - 1]: the gripper on a solo arm and the
    // RIGHT gripper on a bimanual one, so it silently charted one hand.
    expect(SOLO.filter(isGripperChannel)).toEqual(["gripper.pos"]);
    expect(BIMANUAL.filter(isGripperChannel))
      .toEqual(["left_gripper", "right_gripper"]);
  });

  it("shortens a channel to what the row group does not already say", () => {
    expect(shortChannel("left_shoulder_pan")).toBe("shoulder pan");
    expect(shortChannel("shoulder_pan.pos")).toBe("shoulder pan");
  });

  it("names every rig the backend can report", () => {
    expect(rigLabel("bimanual")).toBe("bimanual");
    expect(rigLabel("left")).toBe("solo left");
    expect(rigLabel("right")).toBe("solo right");
    expect(rigLabel("solo")).toBe("single arm");
    expect(rigLabel(null)).toBe("rig unknown");
  });
});

describe("gripper channels", () => {
  it("takes the thresholds from the trace's own channels", () => {
    // Measured on disk through the real backend: the kit's dataset grades at
    // 40.0 / 70.0, and the bimanual one — gripper calibrated in DEGREES over
    // [-9.97, 100.27] — at 34.13 / 67.20. The channel carries its own pair, so
    // the line and the guide under it come from one response and cannot
    // disagree. A hardcoded 40/70 would call every bimanual grasp a failure.
    const trace: Trace = {
      names: ["left_shoulder_pan", "left_gripper", "right_gripper"],
      t: [0, 0.033],
      state: [[1, 2], [90, 30], [88, 31]],
      action: [[1, 2], [90, 30], [88, 31]],
      gripper: [
        { side: "left", name: "left_gripper", index: 1,
          closed_below: 34.1254, open_above: 67.1965, values: [90, 30] },
        { side: "right", name: "right_gripper", index: 2,
          closed_below: 34.1254, open_above: 67.1965, values: [88, 31] },
      ],
    };
    expect(trace.gripper?.map((g) => g.name))
      .toEqual(["left_gripper", "right_gripper"]);
    expect(trace.gripper?.every((g) => g.closed_below === 34.1254)).toBe(true);
  });

  it("is drawable only with every array the charts read", () => {
    // A partial body arriving with a 200 makes `names.map` throw INSIDE a
    // render, which unmounts the pane rather than the chart.
    const full: Trace = { names: ["a"], t: [0], state: [[1]], action: [[1]] };
    expect(isDrawableTrace(full)).toBe(true);
    expect(isDrawableTrace(null)).toBe(false);
    expect(isDrawableTrace({} as Trace)).toBe(false);
    expect(isDrawableTrace({ names: ["a"], t: [0], state: [[1]] } as Trace)).toBe(false);
  });
});

describe("the trainable count", () => {
  it("counts unset episodes in, because unset is not reject", () => {
    // An unset episode is handed to the trainer: "I have not judged this" is
    // not "throw it away". `keep` alone understates the training set the
    // moment anything is unmarked, and that number is the answer to "how much
    // am I about to train on".
    expect(trainableCount({ keep: 35, reject: 11, unset: 0, train: 35 })).toBe(35);
    expect(trainableCount({ keep: 20, reject: 5, unset: 21, train: 41 })).toBe(41);
  });

  it("falls back to keep + unset when the backend does not publish it", () => {
    expect(trainableCount({ keep: 20, reject: 5, unset: 21 })).toBe(41);
    expect(trainableCount(null)).toBe(0);
  });

  it("prefers the backend's own sum over recomputing it", () => {
    // If the two ever disagree the backend is right — it owns what the run
    // will actually be handed. Recomputing would make the badge argue with
    // the runner.
    expect(trainableCount({ keep: 1, reject: 0, unset: 1, train: 99 })).toBe(99);
  });
});

describe("the derived job name", () => {
  // The launcher's name field is sticky, so whatever it says is what every
  // FUTURE run is called until someone retypes it. Oscar's real run list is
  // the failure this exists for: three `act_hilti_box_91` rows, at 10000,
  // 10000 and 90000 steps, one of them dead — one name over three different
  // runs, and no way to tell which was which without opening each.
  const BASE = {
    repoId: "local/so101_pick_cube",
    policy: "act" as const,
    episodes: 91,
    steps: 90000,
    batch: 16,
    evalSplit: 0.1,
    seed: 42,
    mode: "random" as const,
  };

  it("spells out the dataset and every hyperparameter that changes the result", () => {
    expect(trainJobName(BASE)).toBe("act_so101_pick_cube_91ep_90k_b16_ev10_s42");
  });

  it("moves when any of them moves — which is the whole point", () => {
    // Stated as inequality between the three runs from the screenshot rather
    // than as three literals: what matters is that they are DIFFERENT, not
    // what each one happens to spell.
    const names = new Set([
      trainJobName({ ...BASE, steps: 10000 }),
      trainJobName({ ...BASE, steps: 90000 }),
      trainJobName({ ...BASE, steps: 90000, batch: 8 }),
      trainJobName({ ...BASE, episodes: 46 }),
      trainJobName({ ...BASE, evalSplit: 0.2 }),
      trainJobName({ ...BASE, seed: 7 }),
      trainJobName({ ...BASE, policy: "smolvla" }),
      trainJobName({ ...BASE, repoId: "local/so101_box" }),
    ]);
    expect(names.size).toBe(8);
  });

  it("leaves out what does not change the outcome", () => {
    // The seed is only read under `random` (`split.py` shuffles there and
    // nowhere else). A name that carried it under `recent` would invite the
    // operator to reshuffle expecting a different holdout.
    const recent = trainJobName({ ...BASE, mode: "recent" });
    expect(recent).toContain("recent");
    expect(recent).not.toContain("s42");
    expect(trainJobName({ ...BASE, mode: "recent", seed: 7 })).toBe(recent);
  });

  it("says noeval rather than ev0, and drops the seed with it", () => {
    // eval_split 0 is a run with no val curve at all; `ev0` reads like a
    // fraction someone typed, and a seed on a split that was never drawn is
    // a number describing nothing.
    const none = trainJobName({ ...BASE, evalSplit: 0 });
    expect(none).toContain("noeval");
    expect(none).not.toMatch(/_s\d/);
  });

  it("numbers a repeat of an identical config instead of colliding", () => {
    // The case the screenshot is: a run that DIED, relaunched unchanged.
    const first = trainJobName(BASE);
    const second = trainJobName(BASE, new Set([first]));
    const third = trainJobName(BASE, new Set([first, second]));
    expect(second).toBe(`${first}_v2`);
    expect(third).toBe(`${first}_v3`);
    // A name taken by something else entirely must not push it either.
    expect(trainJobName(BASE, new Set(["act_hilti_box_91"]))).toBe(first);
  });

  it("keeps a debug run's odd step count spelled out", () => {
    expect(trainJobName({ ...BASE, steps: 1500 })).toContain("_1500_");
    expect(trainJobName({ ...BASE, steps: 250000 })).toContain("_250k_");
  });

  it("trims the DATASET to fit, never the half that tells runs apart", () => {
    const long = trainJobName({
      ...BASE,
      repoId: "local/a_very_long_dataset_name_that_will_not_fit_in_sixty_chars",
    });
    expect(long.length).toBeLessThanOrEqual(60);
    // The discriminating tail survives whole, and so does the `_vN` on top
    // of it — a truncation that ate either would hand out a duplicate.
    expect(long.endsWith("_91ep_90k_b16_ev10_s42")).toBe(true);
    const dup = trainJobName(
      { ...BASE, repoId: "local/a_very_long_dataset_name_that_will_not_fit_in_sixty_chars" },
      new Set([long]),
    );
    expect(dup.length).toBeLessThanOrEqual(60);
    expect(dup.endsWith("_91ep_90k_b16_ev10_s42_v2")).toBe(true);
    expect(dup).not.toBe(long);
  });

  it("keeps the name safe as a path segment", () => {
    // It reaches LeRobot as `--job_name`. A slash or a space in there is a
    // directory the operator did not ask for.
    const odd = trainJobName({ ...BASE, repoId: "local/so101 pick/cube v2!" });
    expect(odd).toMatch(/^[a-z0-9_]+$/);
  });

  it("still names something when no dataset is picked", () => {
    // The panel renders this into its header before anything is chosen.
    expect(trainJobName({ ...BASE, repoId: null })).toBe("act_91ep_90k_b16_ev10_s42");
  });
});

describe("metric stream readings", () => {
  const ROWS: MetricRow[] = [
    { step: 0, epochs: 0, wall_s: 0, loss: 7.25, lr: 1e-5, grad_norm: 4.2 },
    { step: 100, epochs: 0.25, wall_s: 5, loss: 3.1, lr: 1e-5, grad_norm: 3.9,
      eval_loss: 3.6, samples_per_s: 141.2, gpu_mem_gb: 9.4 },
    { step: 200, epochs: 0.5, wall_s: 10, loss: 0.068, lr: 9e-6, grad_norm: 2.2,
      note: "not a number", ok: true, missing: null },
  ];

  it("discovers every numeric key and never a hardcoded list", () => {
    // The kit charted `loss` and threw the rest of the same JSONL line away.
    expect(metricKeys(ROWS)).toEqual([
      "loss", "lr", "grad_norm", "eval_loss", "samples_per_s", "gpu_mem_gb",
    ]);
  });

  it("excludes ALL FOUR of MetricsTracker's counters", () => {
    // Read off lerobot 0.6.1: MetricsTracker.to_dict returns
    // {steps, samples, episodes, epochs, ...metrics}. All four climb
    // monotonically forever, so charting one against steps is a perfect
    // diagonal — a chart that can never say anything. `episodes` is the trap:
    // everywhere else in this UI the word means the dataset's episode list,
    // and here it counts episode-passes the trainer has consumed.
    const rows: MetricRow[] = [
      { steps: 100, samples: 800, episodes: 12, epochs: 0.25, loss: 3.1 },
    ];
    expect(metricKeys(rows)).toEqual(["loss"]);
  });

  it("excludes the axes — charting step against step teaches nothing", () => {
    const keys = metricKeys(ROWS);
    for (const axis of ["step", "steps", "epoch", "epochs", "wall_s", "t"]) {
      expect(keys).not.toContain(axis);
    }
  });

  it("ignores non-numeric and non-finite values", () => {
    expect(metricKeys(ROWS)).not.toContain("note");
    expect(metricKeys(ROWS)).not.toContain("ok");
    expect(metricKeys(ROWS)).not.toContain("missing");
    expect(metricKeys([{ loss: NaN }, { loss: Infinity }])).toEqual([]);
  });

  it("reads each axis, accepting either spelling", () => {
    expect(metricX(ROWS[1], "step")).toBe(100);
    expect(metricX(ROWS[1], "epoch")).toBe(0.25);
    expect(metricX(ROWS[1], "wall")).toBe(10 - 5);
    expect(metricX({ steps: 4 }, "step")).toBe(4);
    expect(metricX({ epoch: 2 }, "epoch")).toBe(2);
    expect(metricX({ elapsed_s: 90 }, "wall")).toBe(90);
  });

  it("reports null for a row that cannot sit on the chosen axis", () => {
    // An eval row logged without an epoch is not at epoch 0, and plotting it
    // there draws a spike at the origin that never happened.
    expect(metricX({ step: 500, eval_loss: 0.4 }, "epoch")).toBeNull();
    expect(metricX({ step: 500 }, "wall")).toBeNull();
  });
});

describe("lab client — what actually goes on the wire", () => {
  it("posts a mark without a note key when none was given", () => {
    // `note: undefined` serialises away, but `note: null` does not — and a
    // null would clear a note the operator only meant to leave alone.
    const { calls } = routeFetch({ "/lab/datasets/mark": { ok: true } });
    void lab.mark("r", 3, "reject");
    expect(body(calls[0])).toEqual({ repo_id: "r", episode: 3, status: "reject" });
  });

  it("posts a note when one was given, empty string included", () => {
    const { calls } = routeFetch({ "/lab/datasets/mark": { ok: true } });
    void lab.mark("r", 3, "keep", "");
    expect(body(calls[0]))
      .toEqual({ repo_id: "r", episode: 3, status: "keep", note: "" });
  });

  it("omits absent bulk fields rather than sending them as null", () => {
    // A bulk call that sent `status: null` would clear the marks of every
    // selected episode when the operator only added a tag.
    const { calls } = routeFetch({ "/lab/datasets/bulk": { updated: 2 } });
    void lab.bulk({ repo_id: "r", episodes: [1, 2], tags_add: ["blurry"] });
    expect(body(calls[0]))
      .toEqual({ repo_id: "r", episodes: [1, 2], tags_add: ["blurry"] });
  });

  it("previews autoclassify with a mode and params", () => {
    const { calls } = routeFetch({ "/autoclass/preview": { token: "t", diff: [] } });
    void lab.autoclassPreview("r", "knn", { k: 5, min_confidence: 0.6 });
    expect(body(calls[0])).toEqual({
      repo_id: "r", mode: "knn", params: { k: 5, min_confidence: 0.6 },
    });
  });

  it("applies autoclassify BY TOKEN, never by re-sending the diff", () => {
    // The token binds the diff to the dataset state it was computed against.
    // Re-sending a diff would apply decisions to a state the operator never
    // saw; the server recomputes the token and 409s instead.
    const { calls } = routeFetch({ "/autoclass/apply": { applied: 9, batch: "b" } });
    void lab.autoclassApply("r", "tok123");
    expect(body(calls[0])).toEqual({ repo_id: "r", token: "tok123" });
  });

  it("prunes with the expected episode set as a guard", () => {
    // The server refuses if the set moved between the dialog opening and the
    // click — exactly when a renumbering would delete a different take.
    const { calls } = routeFetch({ "/lab/datasets/prune": { run_id: "prune-1" } });
    void lab.prune("r", true, [2, 5, 9]);
    expect(body(calls[0]))
      .toEqual({ repo_id: "r", backup: true, expect_episodes: [2, 5, 9] });
  });

  it("puts the typed confirmation on the wire for a dataset delete", () => {
    // The gate is on the SERVER, not only in the dialog: `confirm` must equal
    // `repo_id` byte for byte or it is a 400. There is no undo and this box
    // has no backup of any kind.
    const { calls } = routeFetch({ "/lab/datasets": { repo_id: "r", freed_bytes: 1 } });
    void lab.deleteDataset("local/so101_pick_cube");
    expect(calls[0].url).toContain("repo_id=local%2Fso101_pick_cube");
    expect(calls[0].url).toContain("confirm=local%2Fso101_pick_cube");
    expect((calls[0].init as RequestInit).method).toBe("DELETE");
  });

  it("wraps a train spec in `spec`", () => {
    const { calls } = routeFetch({ "/lab/runs/train": { id: "train-1" } });
    void lab.train({
      repo_id: "r", policy_type: "act", steps: 20000, batch_size: 8,
      eval_split: 0.2, eval_seed: 42, eval_mode: "random", eval_steps: 1000,
      save_freq: 5000, num_workers: 4, device: "cuda", job_name: "act_r",
      tags: [], episodes: [0, 1, 2],
    });
    expect(body(calls[0]).spec.policy_type).toBe("act");
    expect(body(calls[0]).spec.episodes).toEqual([0, 1, 2]);
  });

  it("wraps a rollout spec in `spec`, and omits what was not chosen", () => {
    // The route reads both a flat body and a wrapped one (`_spec_of`), and
    // `train` wraps — one shape for both launch routes, so a reader of either
    // is reading the same thing.
    //
    // The OMISSION is the claim worth pinning: an absent `control_hz` is how
    // the operator asks for the rate the policy was trained at, and the server
    // stamps `control_hz_declared_by: "trained_fps"` because of it. A client
    // that sent `control_hz: undefined` would serialise nothing either — but
    // one that filled in a default would change what the run RECORDS about
    // itself, while running at exactly the same rate.
    const { calls } = routeFetch({ "/lab/runs/rollout": { id: "rollout-1" } });
    void lab.rollout({
      policy_path: "/runs/train-x/train/checkpoints/060000/pretrained_model",
      duration_s: 60, device: "cuda", side: "right",
    });
    expect(calls[0].url).toContain("/lab/runs/rollout");
    expect((calls[0].init as RequestInit).method).toBe("POST");
    const spec = body(calls[0]).spec;
    expect(spec.side).toBe("right");
    expect("control_hz" in spec).toBe(false);
  });

  it("asks for the split rather than computing one", () => {
    // Two implementations of "which episodes does the trainer not see" drift,
    // and when they do the val badges lie about which demonstrations the
    // policy has already learned.
    const { calls } = routeFetch({
      "/lab/datasets/split": { order: [], train_episodes: [], eval_episodes: [] },
    });
    void lab.split("r", 0.2, 42, "recent");
    expect(calls[0].url).toContain("eval_split=0.2");
    expect(calls[0].url).toContain("seed=42");
    expect(calls[0].url).toContain("mode=recent");
  });

  it("passes the metrics offset straight back, opaque", () => {
    const { calls } = routeFetch({ "/metrics": { offset: 900, rows: [] } });
    void lab.runMetrics("train-1", 512);
    expect(calls[0].url).toContain("/lab/runs/train-1/metrics?offset=512");
  });

  it("escapes a run id in the path", () => {
    const { calls } = routeFetch({ "/lab/runs": { id: "a b" } });
    void lab.run("a b");
    expect(calls[0].url).toContain("/lab/runs/a%20b");
  });

  it("caps compare points, because 200k rows do not fit in 600px", () => {
    const { calls } = routeFetch({ "/lab/runs/metrics": { runs: {} } });
    void lab.compareMetrics(["a", "b"], ["loss", "eval_loss"]);
    expect(calls[0].url).toContain("ids=a%2Cb");
    expect(calls[0].url).toContain("keys=loss%2Ceval_loss");
    expect(calls[0].url).toContain("max_points=600");
  });

  it("surfaces a 409 as an ApiError carrying the detail", async () => {
    routeFetch({ "/lab/datasets/prune": 409 });
    await expect(lab.prune("r", true, [1])).rejects.toMatchObject({
      status: 409, detail: "nope",
    });
  });
});
