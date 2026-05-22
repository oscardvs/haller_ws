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

export const api = {
  health: () => getJson<{ status: string }>("/health"),
  config: () => getJson<{
    version: string;
    arms: { id: string; model: string; port: string; mode: string }[];
    cameras: { id: string; role: string; source: string; arm_id?: string }[];
  }>("/config"),
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
  estop: () => postJson<{ ok: true }>("/estop", {}),
};
