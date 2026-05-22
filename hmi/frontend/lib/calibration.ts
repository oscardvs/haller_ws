// hmi/frontend/lib/calibration.ts

export type CalibrationState = "homing" | "sweeping" | "review" | "done" | "aborted";

export interface CalibrationArmStatus {
  id: string;
  has_file: boolean;
  path: string;
  mtime: number | null;
  in_session: boolean;
}

export interface JointCalibration {
  id: number;
  drive_mode: number;
  homing_offset: number;
  range_min: number;
  range_max: number;
}

export interface CalibrationCurrentSession {
  arm_id: string;
  state: CalibrationState;
  proposed?: Record<string, JointCalibration>;
  current?: Record<string, JointCalibration> | null;
}

export interface CalibrationStatusResponse {
  arms: CalibrationArmStatus[];
  current_session: CalibrationCurrentSession | null;
}

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

async function postNoBody<T>(url: string): Promise<T> {
  const res = await fetch(url, { method: "POST" });
  return handle<T>(res);
}

export async function fetchCalibrationStatus(
  base: string,
): Promise<CalibrationStatusResponse> {
  const res = await fetch(`${base}/calibration/status`, { method: "GET" });
  return handle<CalibrationStatusResponse>(res);
}

export const startCalibration = (base: string, id: string) =>
  postNoBody<{ ok: true; state: CalibrationState }>(`${base}/calibration/${id}/start`);

export const captureNeutral = (base: string, id: string) =>
  postNoBody<{
    ok: true;
    state: CalibrationState;
    homing_offsets: Record<string, number>;
  }>(`${base}/calibration/${id}/capture_neutral`);

export const finishSweep = (base: string, id: string) =>
  postNoBody<{
    ok: true;
    state: CalibrationState;
    proposed: Record<string, JointCalibration>;
    current: Record<string, JointCalibration> | null;
  }>(`${base}/calibration/${id}/finish_sweep`);

export const saveCalibration = (base: string, id: string) =>
  postNoBody<{
    ok: true;
    state: "done";
    path: string;
    backup_path: string | null;
  }>(`${base}/calibration/${id}/save`);

export const abortCalibration = (base: string, id: string) =>
  postNoBody<{ ok: true; state: "aborted" }>(`${base}/calibration/${id}/abort`);
