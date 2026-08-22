// hmi/frontend/__tests__/cockpitTabs.test.tsx
//
// The two tabs that gained controls rather than readouts. Both have states
// that only appear when the robot says something specific — a guard that
// cannot be enabled, a backend that cannot delete — and those are exactly the
// ones nobody exercises by hand.
import {
  render, screen, waitFor, cleanup, fireEvent,
} from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

import { TeleopTab } from "@/components/cockpit/TeleopTab";
import { DatasetTab } from "@/components/cockpit/DatasetTab";
import { useTelemetry, type TelemetryFrame } from "@/lib/telemetry";
import type { CameraInfo, HumanTeleopStatus } from "@/lib/api";

const VIEWPORT = { w: 1440, h: 900, compact: false, short: false };

const ARMS = [
  { id: "left", model: "so101_follower", port: "/dev/haller_arm_leader", mode: "auto" },
  { id: "right", model: "so101_follower", port: "/dev/haller_arm_uart", mode: "auto" },
];

function cam(id: string, over: Partial<CameraInfo> = {}): CameraInfo {
  return {
    id, role: "base", source: "opencv", active: true,
    width: 640, height: 480, fps: 30, ...over,
  };
}

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
 *  Returns the calls for assertions. */
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

beforeEach(() => {
  localStorage.clear();
  useTelemetry.setState({ lastFrame: null, link: "live" });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("TeleopTab — session launcher", () => {
  it("offers a preset per session shape and names the hand mapping", async () => {
    routeFetch({});
    render(<TeleopTab arms={ARMS} cameras={[]} viewport={VIEWPORT} />);

    expect(screen.getByRole("radio", { name: /dual/i })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /solo left/i })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /solo right/i })).toBeInTheDocument();
    // Default stance is "behind", which reverses the declared pair.
    expect(screen.getByText("L hand → right · R hand → left")).toBeInTheDocument();
  });

  it("re-maps the hands when the stance changes, before anything is started", async () => {
    routeFetch({});
    render(<TeleopTab arms={ARMS} cameras={[]} viewport={VIEWPORT} />);

    fireEvent.click(screen.getByRole("radio", { name: "front" }));
    expect(screen.getByText("L hand → left · R hand → right")).toBeInTheDocument();
  });

  it("starts the selected preset with the contract body", async () => {
    const { calls } = routeFetch({ "/teleop/human/start": { ok: true, running: true } });
    render(<TeleopTab arms={ARMS} cameras={[]} viewport={VIEWPORT} />);

    fireEvent.click(screen.getByRole("radio", { name: /solo left/i }));
    fireEvent.click(screen.getByRole("button", { name: /start session/i }));

    await waitFor(() => {
      const start = calls.find((c) => c.url.includes("/teleop/human/start"));
      expect(start).toBeTruthy();
      // Behind stance puts a solo arm under the left hand; the other side is
      // an explicit null, not a missing key.
      expect(JSON.parse(start!.init!.body as string)).toEqual({
        left_arm: "left", right_arm: null, hz: 60,
      });
    });
  });

  it("shows the headset entry URL on this origin", () => {
    routeFetch({});
    render(<TeleopTab arms={ARMS} cameras={[]} viewport={VIEWPORT} />);
    expect(
      screen.getByText(`${window.location.origin}/teleop/vr`),
    ).toBeInTheDocument();
  });

  it("reports a solo session's absent side instead of prompting for a grip", () => {
    routeFetch({});
    useTelemetry.setState({
      lastFrame: frameWith({
        running: true, state: "driving", left_arm: "left", right_arm: null,
      }),
    });
    render(<TeleopTab arms={ARMS} cameras={[]} viewport={VIEWPORT} />);
    expect(screen.getByText(/NOT IN SESSION — solo/i)).toBeInTheDocument();
    // The right panel's table area says what its emptiness means rather than
    // repeating the chip.
    expect(screen.getByText(/nothing written here/i)).toBeInTheDocument();
  });
});

describe("TeleopTab — collision guard", () => {
  it("shows the clearance while the guard is OFF, because off still measures", () => {
    routeFetch({});
    useTelemetry.setState({
      lastFrame: frameWith({
        collision: { enabled: false, available: true, slack_m: 0.128 },
      }),
    });
    render(<TeleopTab arms={ARMS} cameras={[]} viewport={VIEWPORT} />);

    expect(screen.getByText("GUARD OFF")).toBeInTheDocument();
    expect(screen.getByText("128 mm")).toBeInTheDocument();
    expect(screen.getByText(/off still MEASURES/i)).toBeInTheDocument();
  });

  it("posts the switch and says which way it went", async () => {
    const { calls } = routeFetch({
      "/teleop/human/collision": { ok: true, collision: { enabled: true } },
    });
    useTelemetry.setState({
      lastFrame: frameWith({ collision: { enabled: false, available: true } }),
    });
    render(<TeleopTab arms={ARMS} cameras={[]} viewport={VIEWPORT} />);

    fireEvent.click(screen.getByRole("button", { name: /GUARD OFF/ }));
    await waitFor(() => {
      const post = calls.find((c) => c.url.includes("/teleop/human/collision"));
      expect(JSON.parse(post!.init!.body as string)).toEqual({ enabled: true });
    });
  });

  it("refuses to offer a guard the rig cannot have, and says why", () => {
    // available:false is one-way — the backend 409s an enable. A live-looking
    // switch that always fails is worse than a disabled one with a reason.
    routeFetch({});
    useTelemetry.setState({
      lastFrame: frameWith({
        collision: { enabled: false, available: false, slack_m: 0.05 },
      }),
    });
    render(<TeleopTab arms={ARMS} cameras={[]} viewport={VIEWPORT} />);

    expect(screen.getByRole("button", { name: /GUARD OFF/ })).toBeDisabled();
    expect(screen.getByText(/no mount geometry/i)).toBeInTheDocument();
    // Still measuring: the number is the point of leaving it on screen.
    expect(screen.getByText("50 mm")).toBeInTheDocument();
  });
});

describe("DatasetTab — take composition", () => {
  const CAMS = [
    cam("mast", { record: true }),
    cam("wrist_left", { role: "wrist", arm_id: "left", record: false }),
  ];

  it("toggles one camera's membership and lifts the accepted answer", async () => {
    const { calls } = routeFetch({
      "/cameras/wrist_left/record": { id: "wrist_left", record: true },
      "/record/repos": { repos: [] },
      "/record/episodes": { repo_id: "u/r", episodes: [], total_frames: 0, size_bytes: 0 },
    });
    const onCameraRecord = vi.fn();
    render(<DatasetTab cameras={CAMS} onCameraRecord={onCameraRecord} />);

    fireEvent.click(screen.getByRole("switch", { name: /record wrist_left/i }));
    await waitFor(() => {
      const post = calls.find((c) => c.url.includes("/cameras/wrist_left/record"));
      expect(JSON.parse(post!.init!.body as string)).toEqual({ record: true });
      expect(onCameraRecord).toHaveBeenCalledWith("wrist_left", true);
    });
  });

  it("lists the episodes already on disk with their size", async () => {
    routeFetch({
      "/record/repos": { repos: [{ repo_id: "u/r", episodes: 2, size_bytes: 10 }] },
      "/record/episodes": {
        repo_id: "u/r", total_frames: 900, size_bytes: 5 * 1024 * 1024,
        episodes: [
          { index: 0, frames: 400, task: "pick the cube", length_s: 13.3 },
          { index: 1, frames: 500, task: "pick the cube", length_s: 16.7 },
        ],
      },
    });
    render(<DatasetTab cameras={CAMS} onCameraRecord={vi.fn()} />);

    await screen.findByText("2 ep · 900 frames · 5.0 MB");
    expect(screen.getAllByText("pick the cube")).toHaveLength(2);
    expect(screen.getByText("16.7s")).toBeInTheDocument();
  });

  it("names the newest episode on the confirm, not the last row", async () => {
    // "delete last" means the highest index. Naming the wrong one on a
    // confirm button is how an operator deletes a take they wanted.
    routeFetch({
      "/record/repos": { repos: [] },
      "/record/episodes": {
        repo_id: "u/r", total_frames: 9, size_bytes: 1,
        episodes: [
          { index: 2, frames: 5, task: "t", length_s: 0.2 },
          { index: 1, frames: 4, task: "t", length_s: 0.2 },
        ],
      },
    });
    render(<DatasetTab cameras={CAMS} onCameraRecord={vi.fn()} />);

    fireEvent.click(await screen.findByRole("button", { name: /delete last episode/i }));
    expect(
      await screen.findByRole("button", { name: /confirm · delete ep 2/i }),
    ).toBeInTheDocument();
  });

  it("shows a delete refusal in the backend's own words and keeps the button", async () => {
    // 409 is a refusal, not a failure: the pop declines rather than leave a
    // dataset lerobot cannot resume. Every state it names can be cleared, so
    // the control stays.
    routeFetch({
      "/record/repos": { repos: [] },
      "/record/episodes/last": 409,
      "/record/episodes": {
        repo_id: "u/r", total_frames: 4, size_bytes: 1,
        episodes: [{ index: 0, frames: 4, task: "t", length_s: 0.2 }],
      },
    });
    render(<DatasetTab cameras={CAMS} onCameraRecord={vi.fn()} />);

    fireEvent.click(await screen.findByRole("button", { name: /delete last episode/i }));
    fireEvent.click(
      await screen.findByRole("button", { name: /confirm · delete ep 0/i }),
    );

    expect(await screen.findByText(/refused: nope/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /delete last episode/i }),
    ).toBeInTheDocument();
  });

  it("stops offering delete-last on a backend that has no such route", async () => {
    // 404/501 is a property of the build, not of the dataset: a button that
    // can only ever error is worse than one line saying so.
    routeFetch({
      "/record/repos": { repos: [] },
      "/record/episodes/last": 501,
      "/record/episodes": {
        repo_id: "u/r", total_frames: 4, size_bytes: 1,
        episodes: [{ index: 0, frames: 4, task: "t", length_s: 0.2 }],
      },
    });
    render(<DatasetTab cameras={CAMS} onCameraRecord={vi.fn()} />);

    fireEvent.click(await screen.findByRole("button", { name: /delete last episode/i }));
    fireEvent.click(
      await screen.findByRole("button", { name: /confirm · delete ep 0/i }),
    );

    await screen.findByText(/cannot delete episodes/i);
    expect(
      screen.queryByRole("button", { name: /delete last episode/i }),
    ).not.toBeInTheDocument();
  });
});
