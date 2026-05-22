// hmi/frontend/lib/api.ts
import { BACKEND_URL } from "./config";

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const body = await res.json();
      detail = body.error ?? body.detail ?? detail;
    } catch {
      /* ignore parse error */
    }
    throw new Error(`HTTP ${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BACKEND_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return handle<T>(res);
}

export async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BACKEND_URL}${path}`);
  return handle<T>(res);
}

// Convenience wrappers
export type ArmGoal = Record<string, number>;

export type CameraInfo = {
  id: string;
  role: "wrist" | "base";
  source: "placeholder" | "opencv" | "mjpeg" | "webrtc";
  arm_id?: string | null;
  active: boolean;
  width: number;
  height: number;
  fps: number;
};

export const cameraSnapshotUrl = (id: string) =>
  `${BACKEND_URL}/cameras/${encodeURIComponent(id)}/snapshot`;
export const cameraStreamUrl = (id: string) =>
  `${BACKEND_URL}/cameras/${encodeURIComponent(id)}/stream`;

export const api = {
  health: () => getJson<{ status: string }>("/health"),
  config: () => getJson<{
    version: string;
    arms: { id: string; model: string; port: string; mode: string }[];
    cameras: { id: string; role: string; source: string; arm_id?: string }[];
  }>("/config"),
  cameras: () => getJson<{ cameras: CameraInfo[] }>("/cameras"),
  cmdVel: (linear: number, angular: number) =>
    postJson<{ ok: true; linear: number; angular: number }>("/base/cmd_vel", { linear, angular }),
  armGoal: (armId: string, goal: ArmGoal) =>
    postJson<{ ok: true; sent: ArmGoal }>(`/arm/${armId}/goal`, goal),
  armMode: (armId: string, mode: "auto" | "manual" | "stop") =>
    postJson<{ ok: true; mode: string }>(`/arm/${armId}/mode`, { mode }),
  armPreset: (armId: string, name: string) =>
    postJson<{ ok: true; sent: ArmGoal }>(`/arm/${armId}/preset`, { name }),
  armPresetRecord: (armId: string, name: string) =>
    postJson<{ ok: true; saved: ArmGoal }>(`/arm/${armId}/preset/record`, { name }),
  armPresetsList: (armId: string) =>
    getJson<{ names: string[] }>(`/arm/${armId}/presets`),
  armPresetDelete: async (armId: string, name: string) => {
    const res = await fetch(
      `${BACKEND_URL}/arm/${armId}/preset/${encodeURIComponent(name)}`,
      { method: "DELETE" },
    );
    if (!res.ok) {
      let detail = `${res.status}`;
      try {
        const body = await res.json();
        detail = body.error ?? body.detail ?? detail;
      } catch {
        /* noop */
      }
      throw new Error(`HTTP ${res.status}: ${detail}`);
    }
    return res.json() as Promise<{ ok: true }>;
  },
  armHome: (armId: string) =>
    postJson<{ ok: true; sent: ArmGoal }>(`/arm/${armId}/home`, {}),
  armTorque: (armId: string, enabled: boolean) =>
    postJson<{ ok: true; torque: boolean }>(`/arm/${armId}/torque`, { enabled }),
  teleopStatus: () =>
    getJson<TeleopStatus>("/teleop"),
  teleopStart: (leader: string, follower: string, hz = 60) =>
    postJson<{ ok: true } & TeleopStatus>("/teleop/start", { leader, follower, hz }),
  teleopStop: () =>
    postJson<{ ok: true } & TeleopStatus>("/teleop/stop", {}),
  estop: () => postJson<{ ok: true }>("/estop", {}),
};

export type TeleopStatus = {
  running: boolean;
  leader: string | null;
  follower: string | null;
  hz: number;
  tick_count?: number;
  last_error?: string | null;
  started_at?: number | null;
};
