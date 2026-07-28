// hmi/frontend/lib/telemetry.ts
"use client";
import { create } from "zustand";
import { WS_URL } from "./config";
import type { HumanTeleopStatus } from "./api";

export type JointState = {
  pos: number;
  min: number;
  max: number;
  torque: boolean;
};

export type CalibrationTelemetryBlock = {
  state: "homing" | "sweeping" | "review" | "done" | "aborted";
  ticks?: Record<string, number>;
  min?: Record<string, number>;
  max?: Record<string, number>;
  error?: string;
};

export type ArmState = {
  mode: "auto" | "manual" | "stop";
  torque?: boolean;
  joints: Record<string, JointState>;
  calibration?: CalibrationTelemetryBlock;
};

export type BaseState = {
  linear: number;
  angular: number;
  odom: { x: number; y: number; yaw: number };
  scan_min_range: number | null;
};

export type TeleopFrameState = {
  running: boolean;
  leader?: string | null;
  follower?: string | null;
  hz?: number;
  tick_count?: number;
  last_error?: string | null;
  started_at?: number | null;
};

export type TelemetryFrame = {
  t: number;
  base: BaseState;
  arms: Record<string, ArmState>;
  alerts: { level: string; code: string; message: string; source: string }[];
  teleop?: TeleopFrameState;
  human_teleop?: HumanTeleopStatus;
};

/**
 * Three states, not two. `connected` alone cannot distinguish "the socket
 * dropped a moment ago and is about to come back" from "the robot host is
 * gone", and the operator needs to act differently on each: the first is worth
 * waiting out, the second means walk over and look at the machine.
 *
 *   live          — socket open AND a frame arrived within STALE_MS.
 *   reconnecting  — recoverable and being retried. Two ways in: the socket is
 *                   open but has gone quiet (backend wedged, not the network),
 *                   or it closed and we are inside the first few retries.
 *   disconnected  — retried past GRACE_ATTEMPTS and still nothing.
 *
 * Anything numeric downstream must render em-dashes unless this is "live":
 * a frozen last-known value drawn in the live style is a lie about the robot.
 */
export type LinkState = "live" | "reconnecting" | "disconnected";

/** No frame for this long, on an open socket, and the feed counts as stale.
 *  Telemetry runs at `telemetry.hz` (20 Hz → 50 ms), so this is ~10 missed
 *  frames: long enough not to trip on one dropped packet or a GC pause. */
const STALE_MS = 500;
/** Retries spent still calling it "reconnecting" before admitting defeat.
 *  Retry cadence is RETRY_MS, so this is ~3 s of grace. */
const GRACE_ATTEMPTS = 3;
const RETRY_MS = 1000;
/** How often the derived link state is recomputed. Frame age is a function of
 *  wall-clock, so nothing else would ever move it off "live" on a silent feed. */
const TICK_MS = 250;

type Store = {
  /** Retained: "the socket is open". Not the same as "telemetry is flowing" —
   *  prefer `link` for anything the operator reads. */
  connected: boolean;
  lastFrame: TelemetryFrame | null;
  /** performance-clock ms at which the last frame landed; null if none ever. */
  lastFrameAt: number | null;
  link: LinkState;
  /** One-line human explanation of `link`, safe to print in the rail. */
  linkDetail: string;
  /** Age of the newest frame, quantised to 50 ms so it is not a 20 Hz churn
   *  source for every component that wants to display it. */
  frameAgeMs: number | null;
  start: () => void;
  stop: () => void;
};

let socket: WebSocket | null = null;
let retryTimer: ReturnType<typeof setTimeout> | null = null;
let ticker: ReturnType<typeof setInterval> | null = null;
let attempts = 0;
/** Set once stop() has been called, so an in-flight close handler doesn't
 *  resurrect the socket the caller just asked us to drop. */
let stopped = false;

function describe(
  link: LinkState,
  ageMs: number | null,
  attemptCount: number,
  socketOpen: boolean,
): string {
  if (link === "live") {
    return `ws · connected · frame age ${ageMs ?? 0} ms`;
  }
  if (link === "reconnecting") {
    // An open-but-quiet socket and a closed socket are both recoverable, but
    // they point at different halves of the system, so they say so.
    if (socketOpen) {
      const secs = ageMs === null ? "?" : (ageMs / 1000).toFixed(1);
      return `socket open but no frame for ${secs} s — backend may be wedged`;
    }
    if (attemptCount === 0) return "connecting…";
    return `socket closed — retrying every ${RETRY_MS / 1000} s (attempt ${attemptCount})`;
  }
  return "websocket closed — check the robot host is up and on the same network";
}

export const useTelemetry = create<Store>((set, get) => ({
  connected: false,
  lastFrame: null,
  lastFrameAt: null,
  link: "reconnecting",
  linkDetail: "connecting…",
  frameAgeMs: null,

  start: () => {
    stopped = false;
    if (socket) return;

    // Derive link state on a clock. `set` is called only when something the UI
    // reads actually changed, so a healthy 20 Hz feed produces zero re-renders
    // from this timer.
    if (!ticker) {
      ticker = setInterval(() => {
        const s = get();
        const age =
          s.lastFrameAt === null
            ? null
            : Math.round((performance.now() - s.lastFrameAt) / 50) * 50;

        let link: LinkState;
        if (socket && s.connected && age !== null && age < STALE_MS) {
          link = "live";
        } else if (attempts <= GRACE_ATTEMPTS) {
          link = "reconnecting";
        } else {
          link = "disconnected";
        }

        const detail = describe(link, age, attempts, socket !== null && s.connected);
        if (link !== s.link || detail !== s.linkDetail || age !== s.frameAgeMs) {
          set({ link, linkDetail: detail, frameAgeMs: age });
        }
      }, TICK_MS);
    }

    const ws = new WebSocket(WS_URL);
    socket = ws;

    ws.addEventListener("open", () => {
      attempts = 0;
      set({ connected: true });
    });

    ws.addEventListener("close", () => {
      socket = null;
      set({ connected: false });
      if (stopped) return;
      attempts += 1;
      retryTimer = setTimeout(() => get().start(), RETRY_MS);
    });

    ws.addEventListener("message", (e) => {
      try {
        const frame = JSON.parse(e.data) as TelemetryFrame;
        // lastFrameAt is what makes staleness detectable. Frames carry their
        // own `t`, but that is the robot's clock — comparing it to ours would
        // measure clock skew, not link health.
        set({ lastFrame: frame, lastFrameAt: performance.now() });
      } catch {
        /* drop malformed frame */
      }
    });
  },

  stop: () => {
    stopped = true;
    if (retryTimer) {
      clearTimeout(retryTimer);
      retryTimer = null;
    }
    if (ticker) {
      clearInterval(ticker);
      ticker = null;
    }
    socket?.close();
    socket = null;
  },
}));
