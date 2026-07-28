// hmi/frontend/components/TelemetryBar.tsx
"use client";
import { useEffect } from "react";
import { useTelemetry } from "@/lib/telemetry";

/**
 * Persistent telemetry rail beneath the deep-link header.
 *
 *  - Three-state connection lamp (see LINK_STYLE / lib/telemetry.ts).
 *  - Tight monospaced columns: t | v | ω | x | y.
 *  - Each column has a uppercase, tracked micro-label so meaning is glanceable.
 *  - Renders even when no frame has arrived yet (em-dash placeholders),
 *    so the page layout never shifts on first telemetry packet.
 *
 * Numeric cells fall back to em-dashes whenever the link is not `live`. The
 * last frame is still in the store and could be drawn — but drawing a frozen
 * odometry reading in the live style tells the operator the robot is where it
 * was when the link died, which is exactly the thing nobody can know.
 */
export const LINK_STYLE: Record<
  ReturnType<typeof useTelemetry.getState>["link"],
  { label: string; colour: string; ping: boolean }
> = {
  live: { label: "Live", colour: "var(--haller-live)", ping: true },
  reconnecting: { label: "Reconnecting", colour: "var(--haller-warn)", ping: false },
  disconnected: { label: "Disconnected", colour: "var(--haller-fault)", ping: false },
};

export function TelemetryBar() {
  const link = useTelemetry((s) => s.link);
  const linkDetail = useTelemetry((s) => s.linkDetail);
  const lastFrame = useTelemetry((s) => s.lastFrame);
  const start = useTelemetry((s) => s.start);
  useEffect(() => {
    start();
  }, [start]);

  const isLive = link === "live";
  const style = LINK_STYLE[link];
  const t = lastFrame?.t;
  const v = lastFrame?.base.linear;
  const w = lastFrame?.base.angular;
  const x = lastFrame?.base.odom.x;
  const y = lastFrame?.base.odom.y;
  const alerts = lastFrame?.alerts ?? [];

  const fmt = (n: number | undefined, d = 2) =>
    isLive && typeof n === "number" ? n.toFixed(d) : "—";

  const time = isLive && typeof t === "number" ? new Date(t * 1000) : null;
  const timeStr = time
    ? time.toLocaleTimeString([], { hour12: false }) +
      "." +
      String(time.getMilliseconds()).padStart(3, "0").slice(0, 2)
    : "—";

  return (
    <div className="flex items-stretch border-t border-border bg-card/40 text-foreground/90 font-mono text-[11px]">
      {/* Connection lamp */}
      <div
        className="flex items-center gap-2 px-3 py-1.5 border-r border-border min-w-[150px]"
        title={linkDetail}
      >
        <span className="relative inline-flex h-1.5 w-1.5">
          {style.ping && (
            <span
              className="absolute inline-flex h-full w-full rounded-full opacity-70 animate-haller-ping"
              style={{ backgroundColor: style.colour }}
            />
          )}
          <span
            className="relative inline-flex h-1.5 w-1.5 rounded-full"
            style={{ backgroundColor: style.colour }}
          />
        </span>
        <span className="label-micro" style={{ color: style.colour }}>
          {style.label}
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
