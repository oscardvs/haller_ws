// hmi/frontend/lib/telemetry.ts
"use client";
import { create } from "zustand";
import { WS_URL } from "./config";

export type JointState = {
  pos: number;
  min: number;
  max: number;
  torque: boolean;
};

export type ArmState = {
  mode: "auto" | "manual" | "stop";
  torque?: boolean;
  joints: Record<string, JointState>;
};

export type BaseState = {
  linear: number;
  angular: number;
  odom: { x: number; y: number; yaw: number };
  scan_min_range: number | null;
};

export type TelemetryFrame = {
  t: number;
  base: BaseState;
  arms: Record<string, ArmState>;
  alerts: { level: string; code: string; message: string; source: string }[];
};

type Store = {
  connected: boolean;
  lastFrame: TelemetryFrame | null;
  start: () => void;
  stop: () => void;
};

let socket: WebSocket | null = null;

export const useTelemetry = create<Store>((set, get) => ({
  connected: false,
  lastFrame: null,
  start: () => {
    if (socket) return;
    const ws = new WebSocket(WS_URL);
    socket = ws;
    ws.addEventListener("open", () => set({ connected: true }));
    ws.addEventListener("close", () => {
      set({ connected: false });
      socket = null;
      // simple reconnect after 1 s
      setTimeout(() => get().start(), 1000);
    });
    ws.addEventListener("message", (e) => {
      try {
        const frame = JSON.parse(e.data) as TelemetryFrame;
        set({ lastFrame: frame });
      } catch {
        /* drop malformed frame */
      }
    });
  },
  stop: () => {
    socket?.close();
    socket = null;
  },
}));
