// hmi/frontend/components/TelemetryBar.tsx
"use client";
import { useEffect } from "react";
import { useTelemetry } from "@/lib/telemetry";

/**
 * Persistent telemetry rail beneath the app header.
 *
 *  - Two-tone connection lamp (emerald = live, red = disconnected).
 *  - Tight monospaced columns: t | v | ω | x | y.
 *  - Each column has a uppercase, tracked micro-label so meaning is glanceable.
 *  - Renders even when no frame has arrived yet (em-dash placeholders),
 *    so the page layout never shifts on first telemetry packet.
 */
export function TelemetryBar() {
  const { connected, lastFrame, start } = useTelemetry();
  useEffect(() => {
    start();
  }, [start]);

  const t = lastFrame?.t;
  const v = lastFrame?.base.linear;
  const w = lastFrame?.base.angular;
  const x = lastFrame?.base.odom.x;
  const y = lastFrame?.base.odom.y;
  const alerts = lastFrame?.alerts ?? [];

  const fmt = (n: number | undefined, d = 2) =>
    typeof n === "number" ? n.toFixed(d) : "—";

  const time = typeof t === "number" ? new Date(t * 1000) : null;
  const timeStr = time
    ? time.toLocaleTimeString([], { hour12: false }) +
      "." +
      String(time.getMilliseconds()).padStart(3, "0").slice(0, 2)
    : "—";

  return (
    <div className="flex items-stretch border-t border-border bg-card/40 text-foreground/90 font-mono text-[11px]">
      {/* Connection lamp */}
      <div className="flex items-center gap-2 px-3 py-1.5 border-r border-border min-w-[140px]">
        <span className="relative inline-flex h-1.5 w-1.5">
          {connected && (
            <span className="absolute inline-flex h-full w-full rounded-full bg-[var(--haller-live)] opacity-70 animate-haller-ping" />
          )}
          <span
            className={`relative inline-flex h-1.5 w-1.5 rounded-full ${
              connected ? "bg-[var(--haller-live)]" : "bg-[var(--haller-fault)]"
            }`}
          />
        </span>
        <span
          className={`label-micro ${
            connected ? "text-[var(--haller-live)]" : "text-[var(--haller-fault)]"
          }`}
        >
          {connected ? "Live" : "Disconnected"}
        </span>
      </div>

      <TelemetryCell label="t"  value={timeStr}  className="min-w-[160px]" />
      <TelemetryCell label="v"  value={`${fmt(v)} m/s`} />
      <TelemetryCell label="ω"  value={`${fmt(w)} rad/s`} />
      <TelemetryCell label="x"  value={fmt(x)} />
      <TelemetryCell label="y"  value={fmt(y)} />

      {/* Alerts column pinned right */}
      <div className="ml-auto flex items-center gap-2 px-3 py-1.5 border-l border-border">
        <span className="label-micro text-muted-foreground">Alerts</span>
        {alerts.length === 0 ? (
          <span className="text-muted-foreground">none</span>
        ) : (
          <span className="text-[var(--haller-warn)] flex items-center gap-1">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-[var(--haller-warn)]" />
            {alerts.length}
          </span>
        )}
      </div>
    </div>
  );
}

function TelemetryCell({
  label,
  value,
  className = "",
}: {
  label: string;
  value: string;
  className?: string;
}) {
  return (
    <div
      className={`flex items-baseline gap-2 px-3 py-1.5 border-r border-border ${className}`}
    >
      <span className="label-micro text-muted-foreground">{label}</span>
      <span data-num className="text-foreground">{value}</span>
    </div>
  );
}
