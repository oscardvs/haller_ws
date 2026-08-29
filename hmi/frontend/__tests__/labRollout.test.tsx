// hmi/frontend/__tests__/labRollout.test.tsx
//
// The rollout launcher's contract with the one route that moves an arm.
//
// `POST /lab/runs/rollout` is not like the other launch routes: what it starts
// drives a servo bus through a policy nobody has watched yet. Its refusals are
// therefore load-bearing, and every assertion below is written from the
// SERVER's rule rather than from the component — the literal sentence
// `routes_runs.py` emits, the literal spec keys the route reads — because a
// test written from the dialog would agree with the dialog about a number the
// backend is the only authority on.
//
// The sharpest one is the first: the launcher must send NO `control_hz` by
// default. A dialog that helpfully forwarded the fps it had just read would
// type-check, launch, run at exactly the right rate, and stamp every run in
// the log book as a rate somebody chose — while routing the number through a
// second source that the server's own check would then agree with itself
// about. Nothing about that failure is visible on screen.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

import { CheckpointList } from "@/components/lab/CheckpointList";
import { RolloutDialog } from "@/components/lab/RolloutDialog";
import { RunDetail } from "@/components/lab/RunDetail";
import { type Checkpoint } from "@/lib/lab";

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() },
}));

/* ─── the wire, verbatim ──────────────────────────────────────────────────
 * Paths from Oscar's real ACT run over the kit's 46-episode pick-cube set —
 * a SOLO dataset, which is the shape that has to be asked about. */

const CK_PATH =
  "/home/odesha/robot-data/runs/train-20260826-213350-act_so101_pick_cube" +
  "/train/checkpoints/060000/pretrained_model";
const CK_LAST_PATH =
  "/home/odesha/robot-data/runs/train-20260826-213350-act_so101_pick_cube" +
  "/train/checkpoints/last/pretrained_model";
const CK_PARTIAL_PATH =
  "/home/odesha/robot-data/runs/train-20260826-213350-act_so101_pick_cube" +
  "/train/checkpoints/065000/pretrained_model";

const REPO = "local/so101_pick_cube";

/** `GET /lab/datasets/detail`, in `_detail_wire`'s spelling. Trimmed to the
 *  keys this dialog reads, plus enough around them to keep the shape
 *  recognisable. */
const soloDetail = {
  repo_id: REPO, root: `/home/odesha/robot-data/${REPO}`,
  fps: 30, robot_type: "so101_follower", rig: "solo",
  tasks: ["Pick up the cube"], video_keys: ["front", "wrist"], features: {},
  total_episodes: 46, episodes: [],
};

/** The rate gate's own sentence, for declared 25 against a trained 30. Copied
 *  from `routes_runs.py::post_rollout` rather than paraphrased: the assertion
 *  is that this reaches the operator's eyes UNCHANGED, and a paraphrase would
 *  pass while the screen said something shorter and less useful. */
const RATE_REFUSAL =
  "refusing to roll out at 25 Hz a policy trained at 30 Hz on " +
  "local/so101_pick_cube: that is a different dynamical system, not a faster " +
  "or slower one — the action deltas are sized for 33 ms steps and would be " +
  "applied over 40 ms. Declare 30 Hz, or leave control_hz out to get it. " +
  "Set allow_rate_mismatch to launch anyway.";

type Call = { url: string; init?: RequestInit };

/**
 * A backend for the two requests this dialog makes.
 *
 * The POST is matched by METHOD and not by path fragment: the run it creates
 * is read back at `/lab/runs/rollout-1`, which contains `/lab/runs/rollout`.
 * A fragment match answers the read-back with the launch response and the
 * test passes while asserting nothing.
 */
function backend(opts: {
  detail?: unknown | null;
  launch?: { id: string } | { status: number; detail: string };
} = {}) {
  const calls: Call[] = [];
  const detail = opts.detail === undefined ? soloDetail : opts.detail;
  const launch = opts.launch ?? { id: "rollout-1" };

  vi.spyOn(globalThis, "fetch").mockImplementation(
    async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      calls.push({ url, init });
      if (url.includes("/lab/datasets/detail")) {
        return detail === null
          ? new Response(JSON.stringify({ detail: "no such dataset" }), { status: 404 })
          : new Response(JSON.stringify(detail), { status: 200 });
      }
      if (init?.method === "POST" && url.includes("/lab/runs/rollout")) {
        return "status" in launch
          ? new Response(JSON.stringify({ detail: launch.detail }), { status: launch.status })
          : new Response(JSON.stringify(launch), { status: 200 });
      }
      return new Response(JSON.stringify({
        id: "rollout-1", kind: "rollout", name: REPO.split("/")[1],
        status: "queued", started_at: null, finished_at: null, spec: {},
      }), { status: 200 });
    },
  );
  return calls;
}

const post = (calls: Call[]) =>
  calls.find((c) => c.init?.method === "POST" && c.url.includes("/lab/runs/rollout"));

const sentSpec = (calls: Call[]) =>
  JSON.parse((post(calls)!.init!.body as string)).spec as Record<string, unknown>;

function ck(over: Partial<Checkpoint> = {}): Checkpoint {
  return { step: 60000, path: CK_PATH, has_model: true, ...over };
}

function mount(over: Partial<Parameters<typeof RolloutDialog>[0]> = {}) {
  const onClose = vi.fn();
  const onLaunched = vi.fn();
  render(
    <RolloutDialog
      checkpoint={ck()}
      repoId={REPO}
      onClose={onClose}
      onLaunched={onLaunched}
      {...over}
    />,
  );
  return { onClose, onLaunched };
}

const startButton = () => screen.getByRole("button", { name: /start rollout/i });
const armSelect = () => screen.getByLabelText("arm the policy drives");

/**
 * The solo fixture cannot launch until an arm is named, which is the point of
 * the fixture. Every test that is about something else answers first.
 *
 * Waits for the RIG to have landed, not for the field: the field is on screen
 * from the first render, because unknown is not "no". Waiting on the field
 * alone raced the dataset read and asserted against a rig nothing had answered
 * yet — which is a test measuring the scheduler, and it failed about one run
 * in six.
 */
const soloAsked = () => screen.getByText(/^required/i);

async function nameTheArm(side = "right") {
  await waitFor(() => expect(soloAsked()).toBeTruthy());
  fireEvent.change(armSelect(), { target: { value: side } });
}

beforeEach(() => vi.restoreAllMocks());
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

/* ─── the rate is the server's to choose ──────────────────────────────── */

describe("the launcher declares no rate unless somebody chose one", () => {
  it("sends a spec with no control_hz at all on the default path", async () => {
    // `post_rollout` defaults `control_hz` to the fps of the dataset the
    // CHECKPOINT records, and stamps `control_hz_declared_by` so a run says
    // later whether the rate was chosen or inherited. Sending the fps this
    // dialog happens to have read would make every run read as a choice, and
    // would compare the server's number against a copy of itself.
    const calls = backend();
    mount();
    await nameTheArm();
    fireEvent.click(startButton());

    await waitFor(() => expect(post(calls)).toBeTruthy());
    const spec = sentSpec(calls);
    expect("control_hz" in spec).toBe(false);
    expect("allow_rate_mismatch" in spec).toBe(false);
    // The rest of the default spec, pinned in the same breath: the path is the
    // MODEL directory verbatim, which is what the route loads.
    expect(spec.policy_path).toBe(CK_PATH);
    expect(spec.duration_s).toBe(60);
    expect(spec.device).toBe("cuda");
  });

  it("offers no rate override while the rate is the trained one", () => {
    // The override can only ever answer a gate that fires on a DECLARED rate.
    // Offered on the default path it would read as a second safety to turn
    // off, on a path where nothing is being overridden.
    backend();
    mount();
    expect(screen.queryByText(/launch even if this is not the rate/i)).toBeNull();
  });

  it("sends the declared rate, and the override only when it is ticked", async () => {
    const calls = backend();
    mount();
    await nameTheArm();

    fireEvent.change(screen.getByLabelText("control rate source"), {
      target: { value: "declared" },
    });
    fireEvent.change(screen.getByLabelText("control rate in hz"), {
      target: { value: "25" },
    });
    fireEvent.click(screen.getByLabelText(/launch even if this is not the rate/i));
    fireEvent.click(startButton());

    await waitFor(() => expect(post(calls)).toBeTruthy());
    expect(sentSpec(calls).control_hz).toBe(25);
    expect(sentSpec(calls).allow_rate_mismatch).toBe(true);
  });

  it("prefills the declared box from the dataset without ever sending it", async () => {
    // The number is shown because "declare a rate" with an empty box is a
    // question with no starting point. It becomes a REQUEST only by being left
    // there deliberately — which the run then records as the operator's, and
    // truthfully, because they saw it.
    const calls = backend();
    mount();
    // Said in two places once the dataset lands — beside the choice, and in
    // the sentence at the bottom — so this waits for "at least one", not "the".
    await waitFor(() => expect(screen.getAllByText(/30 Hz/).length).toBeGreaterThan(0));
    fireEvent.change(screen.getByLabelText("control rate source"), {
      target: { value: "declared" },
    });
    expect(screen.getByLabelText("control rate in hz")).toHaveValue(30);
    await nameTheArm();
    fireEvent.click(startButton());
    await waitFor(() => expect(post(calls)).toBeTruthy());
    expect(sentSpec(calls).control_hz).toBe(30);
  });
});

/* ─── which arm ───────────────────────────────────────────────────────── */

describe("an unprefixed policy names no arm, so the launcher will not guess", () => {
  it("will not launch a solo dataset until the arm is named", async () => {
    // `action_from_vector` refuses this rather than guessing, and the two arms
    // are 40 cm apart. Refusing at the door instead buys the operator the
    // question before the CUDA context, not after it.
    const calls = backend();
    mount();
    await waitFor(() => expect(soloAsked()).toBeTruthy());
    expect(startButton()).toBeDisabled();
    fireEvent.click(startButton());
    expect(post(calls)).toBeUndefined();

    fireEvent.change(armSelect(), { target: { value: "left" } });
    expect(startButton()).not.toBeDisabled();
    fireEvent.click(startButton());
    await waitFor(() => expect(post(calls)).toBeTruthy());
    expect(sentSpec(calls).side).toBe("left");
  });

  it("does not ask on a rig whose columns name their own side", async () => {
    // A bimanual dataset's columns carry `left_`/`right_`, so the child takes
    // the side from the data and ignores the spec's. Asking anyway would be a
    // question with no answer that changes anything — and a wrong answer that
    // looks like it did.
    const calls = backend({ detail: { ...soloDetail, rig: "bimanual" } });
    mount();
    await waitFor(() => expect(screen.queryByLabelText("arm the policy drives")).toBeNull());
    fireEvent.click(startButton());
    await waitFor(() => expect(post(calls)).toBeTruthy());
    expect("side" in sentSpec(calls)).toBe(false);
  });

  it("still asks when the dataset could not be read, and still launches", async () => {
    // Unknown is not "no". The rig read here only ever decides whether the
    // QUESTION is asked; the child re-derives it and refuses on its own, so a
    // failed read must not block a launch it has no opinion about.
    const calls = backend({ detail: null });
    mount();
    await waitFor(() => expect(screen.getByText(/could not be read/i)).toBeTruthy());
    expect(armSelect()).toBeTruthy();
    expect(startButton()).not.toBeDisabled();
    fireEvent.click(startButton());
    await waitFor(() => expect(post(calls)).toBeTruthy());
    expect("side" in sentSpec(calls)).toBe(false);
  });
});

/* ─── the bounds are the server's ─────────────────────────────────────── */

describe("the duration ceiling is the run's, not the spinner's", () => {
  it("cannot ask for longer than the server will accept", async () => {
    // `MAX_ROLLOUT_DURATION_S` is refused by the route AND by the child. A box
    // that put 5000 on the wire would buy a 400 for a number it offered.
    const calls = backend();
    mount();
    await nameTheArm();
    fireEvent.change(screen.getByLabelText("rollout duration in seconds"), {
      target: { value: "5000" },
    });
    fireEvent.click(startButton());
    await waitFor(() => expect(post(calls)).toBeTruthy());
    expect(sentSpec(calls).duration_s).toBe(900);
  });
});

/* ─── a refusal is the operator's next move ───────────────────────────── */

describe("a refused launch is quoted, and the dialog stays open", () => {
  it("shows the rate gate's own sentence, both numbers and all", async () => {
    // The sentence names what was asked for, what the policy was trained at,
    // the dataset it was trained on, and the two step lengths. A dialog that
    // summarised it to "bad rate" would throw away every part the operator
    // needs to decide what to do next.
    const calls = backend({ launch: { status: 400, detail: RATE_REFUSAL } });
    const { onClose, onLaunched } = mount();
    await nameTheArm();
    fireEvent.change(screen.getByLabelText("control rate source"), {
      target: { value: "declared" },
    });
    fireEvent.change(screen.getByLabelText("control rate in hz"), {
      target: { value: "25" },
    });
    fireEvent.click(startButton());

    await waitFor(() => expect(screen.getByText(RATE_REFUSAL)).toBeTruthy());
    expect(post(calls)).toBeTruthy();
    // Still open, still holding what was typed: the operator's next act is on
    // this form, and a dialog that closed would make them retype it to read
    // the sentence that told them why.
    expect(onLaunched).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getByLabelText("control rate in hz")).toHaveValue(25);
  });

  it("hands the launched run back and closes on success", async () => {
    const calls = backend();
    const { onClose, onLaunched } = mount();
    await nameTheArm();
    fireEvent.click(startButton());
    await waitFor(() => expect(onLaunched).toHaveBeenCalled());
    expect(onLaunched.mock.calls[0][0]).toMatchObject({ id: "rollout-1", kind: "rollout" });
    expect(onClose).toHaveBeenCalled();
    // The POST answers with an id, not a run — the record comes from a read.
    expect(calls.some((c) => c.url.endsWith("/lab/runs/rollout-1"))).toBe(true);
  });
});

/* ─── the button on the checkpoint row ────────────────────────────────── */

describe("CheckpointList offers a rollout only where one is possible", () => {
  function mountList(checkpoints: Checkpoint[], onRollout?: (c: Checkpoint) => void) {
    vi.spyOn(globalThis, "fetch").mockImplementation(
      async () => new Response(JSON.stringify({ checkpoints }), { status: 200 }),
    );
    return render(
      <CheckpointList runId="train-x" status="done" onRollout={onRollout} />,
    );
  }

  it("shows no button at all when the surface cannot launch one", async () => {
    mountList([ck()]);
    await waitFor(() => expect(screen.getByText("060000")).toBeTruthy());
    expect(screen.queryByRole("button", { name: /roll out/i })).toBeNull();
  });

  it("offers one per usable checkpoint, and none for a partial one", async () => {
    // A directory the runner created but never finished writing fails at load.
    // A disabled button would be a second way of saying what the `partial`
    // chip beside it already says.
    mountList(
      [ck(), ck({ step: null, path: CK_LAST_PATH }),
       ck({ step: 65000, path: CK_PARTIAL_PATH, has_model: false })],
      vi.fn(),
    );
    await waitFor(() => expect(screen.getByText("065000")).toBeTruthy());
    expect(screen.getAllByRole("button", { name: /roll out/i })).toHaveLength(2);
    expect(screen.queryByRole("button", { name: /roll out 065000/i })).toBeNull();
  });

  it("hands back the checkpoint, whose PATH is what the route loads", async () => {
    // The row is named by its step directory, and the route is pointed at the
    // model directory inside it. Handing back the name would be pointing a
    // rollout at a string that is not a path.
    const onRollout = vi.fn();
    mountList([ck({ step: null, path: CK_LAST_PATH })], onRollout);
    await waitFor(() => expect(screen.getByText("last")).toBeTruthy());
    fireEvent.click(screen.getByRole("button", { name: /roll out last/i }));
    expect(onRollout).toHaveBeenCalledWith(
      expect.objectContaining({ path: CK_LAST_PATH }),
    );
  });
});

/* ─── reading a rollout back ──────────────────────────────────────────── */

describe("RunDetail reads a rollout's spec, not a training run's", () => {
  /** The spec `post_rollout` STAMPS — not the one the launcher sent. The extra
   *  keys are the route's, written whether or not anything was overridden,
   *  which is what lets a run months later say where its rate came from. */
  const stamped = {
    policy_path: CK_PATH,
    control_hz: 30,
    control_hz_trained: 30,
    control_hz_declared_by: "trained_fps",
    control_hz_trained_repo_id: REPO,
    control_hz_trained_measured: true,
    control_hz_trained_measured_hz: 29.98,
    control_hz_mismatch_override: false,
    duration_s: 60,
    device: "cuda",
    side: "right",
    repo_id: REPO,
    allow_slow: false,
  };

  function mountRun(spec: Record<string, unknown>) {
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
          id: "rollout-1", kind: "rollout", name: "so101_pick_cube",
          status: "done", started_at: null, finished_at: null, spec,
        }), { status: 200 });
      },
    );
    return render(<RunDetail runId="rollout-1" />);
  }

  it("shows the rate it ran at beside the rate it was trained at", async () => {
    // One number cannot answer this. A run that inherited 30 Hz and one where
    // somebody typed 30 Hz over a policy trained at 25 read identically with
    // only the first — which is the whole reason the route stamps both.
    mountRun(stamped);
    await waitFor(() => expect(screen.getByText(/trained at/i)).toBeTruthy());
    expect(screen.getAllByText("30 Hz")).toHaveLength(2);
    expect(screen.getByText("60 s")).toBeTruthy();
    expect(screen.getByText("right")).toBeTruthy();
    // The checkpoint it loaded, which is the one thing "which policy was this"
    // has no other answer to.
    expect(screen.getByTitle(CK_PATH)).toBeTruthy();
  });

  it("does not render the training fields against it", async () => {
    // Before this branch existed, a rollout run showed a train run's stat row:
    // policy —, episodes —, steps —, batch —, eval split —. Five dashes read
    // as a run that recorded nothing, on a run whose spec is complete.
    mountRun(stamped);
    await waitFor(() => expect(screen.getByText(/trained at/i)).toBeTruthy());
    expect(screen.queryByText(/eval split/i)).toBeNull();
    expect(screen.queryByText(/batch/i)).toBeNull();
  });

  it("says a waived rate floor out loud", async () => {
    // `allow_slow` is a gate the operator turned off. A run that ran under its
    // own floor must not read like one that never approached it.
    mountRun({ ...stamped, allow_slow: true });
    await waitFor(() => expect(screen.getByText(/waived/i)).toBeTruthy());
  });
});
