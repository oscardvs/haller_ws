// hmi/frontend/lib/config.ts
export const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export const WS_URL =
  BACKEND_URL.replace(/^http/, "ws") + "/ws/telemetry";
