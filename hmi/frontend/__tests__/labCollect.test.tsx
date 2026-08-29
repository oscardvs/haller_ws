// hmi/frontend/__tests__/labCollect.test.tsx
//
// The collect page's two framings of a take: which dataset it resumes, and
// which VR session demonstrates it.
//
// Both fail quietly when they are wrong. A resume that retypes the task
// string forks the dataset — LeRobot keys tasks by string, and the fork is
// invisible until two repos show up where one was meant. A preset list that
// offers a session the rig does not have is only found out by an arm that
// does not move. Neither throws; these pin the words and the wire bodies.
import {
  act, cleanup, fireEvent, render, screen, waitFor,
} from "@testing-library/react";
import { createRef } from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

import { CollectResumeCard } from "@/components/lab/CollectResumeCard";
import { CollectSessionCard } from "@/components/lab/CollectSessionCard";
import { TeleopTab } from "@/components/cockpit/TeleopTab";
import { RecordPopover } from "@/components/cockpit/RecordPopover";
import { DatasetTab } from "@/components/cockpit/DatasetTab";
import { effectiveRepoId, useRecorder } from "@/lib/recorder";
import { headsetOrigin } from "@/lib/config";
import { useTelemetry, type TelemetryFrame } from "@/lib/telemetry";
import type { HumanTeleopStatus } from "@/lib/api";

const VIEWPORT = { w: 1440, h: 900, compact: false, short: false };

/** The rig this work is for: one arm, id `left`. */
const SOLO_ARMS = [
  { id: "left", model: "so101_follower", port: "/dev/haller_arm_uart", mode: "auto" },
];

function frameWith(human: Partial<HumanTeleopStatus>): TelemetryFrame {
  return {
    t: 1,
    base: { linear: 0, angular: 0, odom: { x: 0, y: 0, yaw: 0 }, scan_min_range: null },
    arms: {},
    alerts: [],
    human_teleop: {
      running: false, state: "idle", left_arm: null, right_arm: null,
      started_at: null, last_error: null,
      tracking: {
        left: { age_ms: null, lost: true },
        right: { age_ms: null, lost: true },
      },
      ...human,
    },
  };
}

/** Routes fetch by path so a component's boot calls do not have to be ordered.
 *  Returns the calls for assertions. Mirrors __tests__/cockpitTabs.test.tsx. */
function routeFetch(routes: Record<string, unknown>) {
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

const REPO = "osrdvs/haller_pick_the_red_cube";
const REPOS = {
  "/record/repos": {
    repos: [{ repo_id: REPO, episodes: 3, frames: 900, size_bytes: 1024 }],
  },
  "/record/episodes": {
    repo_id: REPO, total_frames: 900, size_bytes: 1024,
    episodes: [
      { index: 0, frames: 300, task: "Pick the red cube", length_s: 10 },
      { index: 1, frames: 300, task: "Pick the red cube", length_s: 10 },
      { index: 2, frames: 300, task: "Pick the red cube", length_s: 10 },
    ],
  },
};

beforeEach(() => {
  localStorage.clear();
  useTelemetry.setState({ lastFrame: null, link: "live" });
  // The draft is module state shared with the recorder card and the headset —
  // reset it the way "new dataset" leaves it, pin included.
  useRecorder.setState({
    status: null, busy: false, task: "", hfUser: "", repoIdOverride: null,
  });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

/* ─── resume picker ───────────────────────────────────────────────────── */

describe("CollectResumeCard — resuming a dataset", () => {
  it("lists what is on disk with episode and frame counts", async () => {
    routeFetch(REPOS);
    render(<CollectResumeCard />);
    await screen.findByText(/osrdvs\/haller_pick_the_red_cube · 3 ep · 900 frames/);
  });

  it("writes the dataset's own task into the shared draft, pins the repo, and names the next episode", async () => {
    routeFetch(REPOS);
    render(<CollectResumeCard />);

    const select = await screen.findByLabelText("resume dataset");
    fireEvent.change(select, { target: { value: REPO } });

    // The draft the recorder card, the Record popover and the headset all
    // read. The string is the dataset's own, not a retype — a variant would
    // fork the dataset under LeRobot's string-keyed tasks — and the pin
    // carries the picked repo_id itself, so START RECORDING appends.
    await waitFor(() => {
      expect(useRecorder.getState().hfUser).toBe("osrdvs");
      expect(useRecorder.getState().task).toBe("Pick the red cube");
      expect(useRecorder.getState().repoIdOverride).toBe(REPO);
    });
    expect(effectiveRepoId(useRecorder.getState())).toBe(REPO);
    await screen.findByText(/next episode will be/);
    expect(screen.getByText("#3")).toBeInTheDocument();
    expect(screen.getByText(/task on disk:/)).toHaveTextContent(
      "task on disk: Pick the red cube",
    );
  });

  it("keeps 'new dataset' one gesture, and it clears the draft AND the pin", async () => {
    routeFetch(REPOS);
    useRecorder.setState({
      task: "Pick the red cube", hfUser: "osrdvs", repoIdOverride: REPO,
    });
    render(<CollectResumeCard />);

    // The draft resolves to an on-disk repo, so that is what the picker shows.
    const select = await screen.findByLabelText("resume dataset");
    await waitFor(() => expect(select).toHaveValue(REPO));

    fireEvent.change(select, { target: { value: "" } });
    expect(useRecorder.getState().task).toBe("");
    expect(useRecorder.getState().hfUser).toBe("");
    expect(useRecorder.getState().repoIdOverride).toBeNull();
  });

  it("follows a deliberate task edit back to 'new dataset' rather than disagreeing", async () => {
    routeFetch(REPOS);
    render(<CollectResumeCard />);

    const select = await screen.findByLabelText("resume dataset");
    fireEvent.change(select, { target: { value: REPO } });
    await waitFor(() => expect(select).toHaveValue(REPO));

    // Editing the task in the recorder is the deliberate way to leave a
    // dataset: setTask drops the pin, and the pick must flip, not keep
    // claiming the old repo.
    act(() => useRecorder.getState().setTask("Pick the red cube v2"));
    expect(useRecorder.getState().repoIdOverride).toBeNull();
    await waitFor(() => expect(select).toHaveValue(""));
  });

  it("pins the picked repo even when its name does not follow the slug rule", async () => {
    routeFetch({
      "/record/repos": {
        repos: [{ repo_id: "osrdvs/custom_ds", episodes: 1, size_bytes: 1 }],
      },
      "/record/episodes": {
        repo_id: "osrdvs/custom_ds", total_frames: 5, size_bytes: 1,
        episodes: [{ index: 0, frames: 5, task: "Do the thing", length_s: 1 }],
      },
    });
    render(<CollectResumeCard />);

    const select = await screen.findByLabelText("resume dataset");
    fireEvent.change(select, { target: { value: "osrdvs/custom_ds" } });

    // Composing from the resumed task would give osrdvs/haller_do_the_thing —
    // a NEW dataset. The pin carries the picked repo_id itself, so the take
    // appends; the warning is the guard for a draft that STILL disagrees with
    // its pin, and a clean pick must not trip it.
    await waitFor(() => {
      expect(useRecorder.getState().task).toBe("Do the thing");
      expect(useRecorder.getState().repoIdOverride).toBe("osrdvs/custom_ds");
    });
    expect(effectiveRepoId(useRecorder.getState())).toBe("osrdvs/custom_ds");
    expect(select).toHaveValue("osrdvs/custom_ds");
    expect(screen.queryByText(/pin was dropped/)).toBeNull();
  });

  it("freezes with the rest of the draft while a take is open", async () => {
    routeFetch(REPOS);
    useRecorder.setState({
      status: {
        recording: true, repo_id: REPO, task: "t", episode_frames: 3,
        skipped_frames: 0, started_at: 1, last_error: null,
      },
    });
    render(<CollectResumeCard />);
    const select = await screen.findByLabelText("resume dataset");
    expect(select).toBeDisabled();
  });
});

/* ─── VR session card ─────────────────────────────────────────────────── */

describe("CollectSessionCard — the session beside the recorder", () => {
  it("offers only the sessions a one-arm rig can start, with the reason for the rest", () => {
    routeFetch({});
    render(<CollectSessionCard arms={SOLO_ARMS} />);

    expect(screen.getByRole("radio", { name: /solo left/i })).toBeEnabled();
    expect(screen.getByRole("radio", { name: /dual/i })).toBeDisabled();
    expect(screen.getByText(/needs 2 enabled arms/)).toBeInTheDocument();
    // Presets come from the arms /config declares — a rig with no `right`
    // arm is never offered a solo-right session.
    expect(screen.queryByRole("radio", { name: /solo right/i })).toBeNull();
  });

  it("starts the solo preset with the same body the Teleop tab posts", async () => {
    const { calls } = routeFetch({ "/teleop/human/start": { ok: true, running: true } });
    render(<CollectSessionCard arms={SOLO_ARMS} />);

    fireEvent.click(screen.getByRole("button", { name: /start session/i }));
    await waitFor(() => {
      const start = calls.find((c) => c.url.includes("/teleop/human/start"));
      // Behind the arms (the default stance), the robot's left arm sits under
      // the operator's right hand — the same hand a dual session would give
      // it, and an explicit null on the absent side.
      expect(JSON.parse(start!.init!.body as string)).toEqual({
        left_arm: null, right_arm: "left", hz: 60,
      });
    });
  });

  it("shows who owns each side, and the URL the headset opens", () => {
    routeFetch({});
    useTelemetry.setState({
      lastFrame: frameWith({
        running: true, state: "driving", left_arm: null, right_arm: "left",
        acquire: {
          acquire_ms: 1500, match_dwell_ms: 250,
          left: { authority: "held", reason: "no_arm", remaining_ms: null, ramp: null },
          right: { authority: "driving", reason: "driving", remaining_ms: null, ramp: 1 },
        },
      }),
    });
    render(<CollectSessionCard arms={SOLO_ARMS} />);

    expect(screen.getByText("running · driving")).toBeInTheDocument();
    expect(
      screen.getByText(`${window.location.origin}/teleop/vr`),
    ).toBeInTheDocument();
  });

  it("stops a running session", async () => {
    const { calls } = routeFetch({ "/teleop/human/stop": { ok: true, running: false } });
    useTelemetry.setState({
      lastFrame: frameWith({ running: true, state: "driving", right_arm: "left" }),
    });
    render(<CollectSessionCard arms={SOLO_ARMS} />);

    fireEvent.click(screen.getByRole("button", { name: /stop session/i }));
    await waitFor(() => {
      expect(calls.some((c) => c.url.includes("/teleop/human/stop"))).toBe(true);
    });
  });
});

/* ─── the hand-off from the Teleop tab ────────────────────────────────── */

describe("TeleopTab — path back to collect", () => {
  it("offers 'record a dataset with this session' and fires the hand-off", () => {
    routeFetch({});
    const onOpenCollect = vi.fn();
    render(
      <TeleopTab
        arms={SOLO_ARMS}
        cameras={[]}
        viewport={VIEWPORT}
        onOpenCollect={onOpenCollect}
      />,
    );

    fireEvent.click(
      screen.getByRole("button", { name: /record a dataset with this session/i }),
    );
    expect(onOpenCollect).toHaveBeenCalledTimes(1);
  });
});

/* ─── one repo answer across every record-start surface ─────────────────── */

describe("record consumers — the pin or the composition, never a local recompute", () => {
  function openPopover() {
    return render(
      <RecordPopover
        onClose={() => {}}
        triggerRef={createRef<HTMLButtonElement>()}
        onTab={() => {}}
      />,
    );
  }

  it("RecordPopover composes from task/hfUser when no resume pin is set", async () => {
    const { calls } = routeFetch({ "/record/start": { ok: true, recording: true } });
    useRecorder.setState({ task: "Pick the red cube", hfUser: "osrdvs" });
    openPopover();

    expect(
      screen.getByText("osrdvs/haller_pick_the_red_cube"),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /start recording/i }));
    await waitFor(() => {
      const start = calls.find((c) => c.url.includes("/record/start"));
      expect(JSON.parse(start!.init!.body as string)).toEqual({
        repo_id: "osrdvs/haller_pick_the_red_cube", task: "Pick the red cube",
      });
    });
  });

  it("RecordPopover posts the pinned repo when a dataset was resumed", async () => {
    const { calls } = routeFetch({ "/record/start": { ok: true, recording: true } });
    useRecorder.setState({
      task: "Do the thing", hfUser: "osrdvs", repoIdOverride: "osrdvs/custom_ds",
    });
    openPopover();

    // The composed name would be osrdvs/haller_do_the_thing — a fork. The
    // pinned repo is the one that goes on the wire.
    fireEvent.click(screen.getByRole("button", { name: /start recording/i }));
    await waitFor(() => {
      const start = calls.find((c) => c.url.includes("/record/start"));
      expect(JSON.parse(start!.init!.body as string)).toEqual({
        repo_id: "osrdvs/custom_ds", task: "Do the thing",
      });
    });
  });

  it("effectiveRepoId — the headset's read path — composes only when unpinned", () => {
    // VRTeleopPanel's armAct reads the same store through this one helper, so
    // the desk and the headset cannot aim at different repos. Pinned here as a
    // unit because the WebXR panel itself has no headless render — its harness
    // is the pure-logic one in vrTeleopRecord.test.ts.
    expect(
      effectiveRepoId({ repoIdOverride: null, hfUser: "osrdvs", task: "Do the thing" }),
    ).toBe("osrdvs/haller_do_the_thing");
    expect(
      effectiveRepoId({ repoIdOverride: "osrdvs/custom_ds", hfUser: "osrdvs", task: "Do the thing" }),
    ).toBe("osrdvs/custom_ds");
    expect(
      effectiveRepoId({ repoIdOverride: null, hfUser: "", task: "Do the thing" }),
    ).toBe("local/haller_do_the_thing");
  });
});

/* ─── the headset origin ────────────────────────────────────────────────── */

describe("headsetOrigin — what the Quest actually opens", () => {
  it("strips /api off the baked single-HTTPS origin", () => {
    // up.sh bakes NEXT_PUBLIC_BACKEND_URL=…:8444/api; that origin minus the
    // suffix IS the cockpit the headset must open. window.location.origin
    // would print "localhost" — which, on a Quest, is the Quest itself.
    expect(headsetOrigin("http://localhost:3001", "https://192.168.0.191:8444/api"))
      .toBe("https://192.168.0.191:8444");
    expect(headsetOrigin("http://localhost:3001", "https://192.168.0.191:8444/api/"))
      .toBe("https://192.168.0.191:8444");
  });

  it("falls back to the page origin only for a loopback (dev) bundle", () => {
    expect(headsetOrigin("http://localhost:3001", "http://localhost:8000"))
      .toBe("http://localhost:3001");
    // Server render of a dev bundle: nothing trustworthy to print.
    expect(headsetOrigin(null, "http://localhost:8000")).toBeNull();
  });
});

/* ─── the dataset's write rate, steered on before record ──────────────── */

describe("CollectResumeCard — deriving the dataset's fps", () => {
  it("lifts the newest episode's rate on a pick, null on a new dataset", async () => {
    routeFetch(REPOS);
    const onDatasetFps = vi.fn();
    render(<CollectResumeCard onDatasetFps={onDatasetFps} />);

    // Nothing picked: a new dataset has no rate to match.
    await waitFor(() => expect(onDatasetFps).toHaveBeenLastCalledWith(null));

    fireEvent.change(
      await screen.findByLabelText("resume dataset"),
      { target: { value: REPO } },
    );
    // Every fixture episode is 300 frames over 10 s.
    await waitFor(() => expect(onDatasetFps).toHaveBeenLastCalledWith(30));
  });
});

describe("CollectSessionCard — steering the session rate before record", () => {
  it("warns on a rate mismatch, and the one-click fix settles it", () => {
    routeFetch({});
    render(<CollectSessionCard arms={SOLO_ARMS} datasetFps={30} />);

    expect(
      screen.getByText(/dataset is written at 30 fps — recording appends only/),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "use 30 Hz" }));
    expect(screen.getByLabelText("teleop rate in Hz")).toHaveValue("30");
    // Match reached — the steer withdraws.
    expect(screen.queryByText(/dataset is written at/)).toBeNull();
  });

  it("says nothing when the rates already match", () => {
    routeFetch({});
    render(<CollectSessionCard arms={SOLO_ARMS} datasetFps={60} />);
    expect(screen.queryByText(/appends only/)).toBeNull();
    expect(screen.queryByRole("button", { name: /use \d+ Hz/ })).toBeNull();
  });

  it("a running session at the wrong rate must be restarted — hz froze at start", () => {
    routeFetch({});
    useTelemetry.setState({
      lastFrame: frameWith({ running: true, state: "driving", right_arm: "left" }),
    });
    render(<CollectSessionCard arms={SOLO_ARMS} datasetFps={30} />);

    expect(
      screen.getByText(/stop and restart the session at that rate/),
    ).toBeInTheDocument();
    // No one-click fix while running: the rate input is the NEXT start's.
    expect(screen.queryByRole("button", { name: /use \d+ Hz/ })).toBeNull();
  });
});

describe("DatasetTab — the recorder warning names the append contract", () => {
  it("extends the no-session warning for a resumed fps-known dataset", async () => {
    routeFetch({
      "/record/repos": { repos: [] },
      "/record/episodes": { repo_id: "u/r", episodes: [], total_frames: 0, size_bytes: 0 },
    });
    useRecorder.setState({
      task: "Do the thing", hfUser: "osrdvs", repoIdOverride: "osrdvs/custom_ds",
    });
    render(<DatasetTab cameras={[]} onCameraRecord={vi.fn()} datasetFps={30} />);

    expect(
      await screen.findByText(/human teleop is not running/),
    ).toHaveTextContent(
      "osrdvs/custom_ds is a 30 fps dataset: takes append only from a session running at 30 Hz",
    );
  });

  it("stays the plain warning for a new dataset", async () => {
    routeFetch({
      "/record/repos": { repos: [] },
      "/record/episodes": { repo_id: "u/r", episodes: [], total_frames: 0, size_bytes: 0 },
    });
    render(<DatasetTab cameras={[]} onCameraRecord={vi.fn()} datasetFps={null} />);

    const warning = await screen.findByText(/human teleop is not running/);
    expect(warning).not.toHaveTextContent(/fps dataset/);
  });
});
