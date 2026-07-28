"use client";

/**
 * The 26px band under the header: link state, base telemetry, alert count.
 *
 * Every numeric cell reads from a primitive selector, so a 20 Hz frame
 * re-renders this strip and nothing above or below it. When the link is not
 * `live` the cells show em-dashes rather than the last frame's values — see
 * lib/telemetry.ts for why that is not merely cosmetic.
 */
import type { RefObject } from "react";

import { useTelemetry } from "@/lib/telemetry";
import { LINK_STYLE } from "@/components/TelemetryBar";
import { num } from "./lib";

export function TelemetryRail({
  alertsOpen,
  onToggleAlerts,
  alertsRef,
}: {
  alertsOpen: boolean;
  onToggleAlerts: () => void;
  alertsRef: RefObject<HTMLButtonElement | null>;
}) {
  const link = useTelemetry((s) => s.link);
  const linkDetail = useTelemetry((s) => s.linkDetail);
  const t = useTelemetry((s) => s.lastFrame?.t);
  const v = useTelemetry((s) => s.lastFrame?.base.linear);
  const w = useTelemetry((s) => s.lastFrame?.base.angular);
  const x = useTelemetry((s) => s.lastFrame?.base.odom.x);
  const y = useTelemetry((s) => s.lastFrame?.base.odom.y);
  const yaw = useTelemetry((s) => s.lastFrame?.base.odom.yaw);
  const scan = useTelemetry((s) => s.lastFrame?.base.scan_min_range);
  const alertCount = useTelemetry((s) => s.lastFrame?.alerts.length ?? 0);

  const live = link === "live";
  const style = LINK_STYLE[link];

  // The frame's own timestamp, not the browser clock: this is the number that
  // reveals a feed that has quietly stopped advancing.
  const stamp = live && typeof t === "number" ? new Date(t * 1000) : null;
  const timeStr = stamp
    ? stamp.toLocaleTimeString([], { hour12: false }) +
      "." +
      String(stamp.getMilliseconds()).padStart(3, "0").slice(0, 2)
    : "—";

  const cells: [string, string][] = [
    ["t", timeStr],
    ["v", `${num(v, live)} m/s`],
    ["ω", `${num(w, live)} rad/s`],
    ["x", num(x, live)],
    ["y", num(y, live)],
    ["ψ", `${num(yaw, live)} rad`],
    ["lidar", `${num(scan, live)} m`],
  ];

  return (
    <div className="flex items-stretch overflow-hidden border-b border-border bg-card font-mono text-[11px] tabular-nums">
      <div
        className="flex shrink-0 items-center gap-2 border-r border-border px-3"
        title={linkDetail}
      >
        <span className="relative inline-flex h-1.5 w-1.5">
          {style.ping && (
            <span
              className="absolute inline-flex h-full w-full animate-haller-ping rounded-full opacity-70"
              style={{ backgroundColor: style.colour }}
            />
          )}
          <span
            className="relative inline-flex h-1.5 w-1.5 rounded-full"
            style={{ backgroundColor: style.colour }}
          />
        </span>
        <span className="label-micro whitespace-nowrap" style={{ color: style.colour }}>
          {style.label}
        </span>
      </div>

      {/* When the link is down the reason belongs on screen, not in a tooltip
          nobody hovers during an incident. */}
      {!live && (
        <div className="flex min-w-0 items-center border-r border-border px-3">
          <span
            className="truncate font-mono text-[10px]"
            style={{ color: style.colour }}
          >
            {linkDetail}
          </span>
        </div>
      )}

      {cells.map(([label, value]) => (
        <div
          key={label}
          className="flex shrink-0 items-center gap-2 border-r border-border px-3"
        >
          <span className="label-micro text-muted-foreground">{label}</span>
          <span data-num>{value}</span>
        </div>
      ))}

      <button
        ref={alertsRef}
        type="button"
        onClick={onToggleAlerts}
        aria-expanded={alertsOpen}
        className={
          "ml-auto flex shrink-0 items-center gap-2 border-l border-border px-3 font-mono text-[11px] " +
          (alertsOpen ? "bg-muted" : "hover:bg-muted/50")
        }
      >
        <span className="label-micro text-muted-foreground">Alerts</span>
        <span
          className="inline-flex items-center gap-1.5"
          style={{
            color: alertCount === 0 ? "var(--muted-foreground)" : "var(--haller-warn)",
          }}
        >
          <span
            className="h-1.5 w-1.5 rounded-full"
            style={{
              backgroundColor:
                alertCount === 0 ? "var(--muted-foreground)" : "var(--haller-warn)",
            }}
          />
          {alertCount === 0 ? "none" : alertCount}
        </span>
        <span className="text-[9px] text-muted-foreground">▾</span>
      </button>
    </div>
  );
}
