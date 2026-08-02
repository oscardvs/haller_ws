import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
  output: "standalone",
  turbopack: {
    root: path.resolve(__dirname),
  },
  // The Quest reaches the dev server through the Caddy origin
  // (https://192.168.0.191:8444 → localhost:3001). Next 16 blocks dev-only
  // endpoints (incl. the HMR websocket) from origins other than the one the
  // server was started on, answering them with a bare "Unauthorized" — which
  // surfaces in the headset as a permanent dev-tools "Connecting..." badge
  // and 502s in the Caddy log. Hostnames only, no scheme/port.
  allowedDevOrigins: ["192.168.0.191"],
};

export default nextConfig;
