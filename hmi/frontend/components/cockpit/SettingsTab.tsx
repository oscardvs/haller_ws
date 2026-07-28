"use client";

/**
 * Health, theme, and the read-only view of what config.yaml declared.
 *
 * The v3 design also carried a "Preview states" row — a set of buttons that
 * faked link-down, config-loading, arm-silence and a calibration bus error so
 * those states could be reviewed in the design tool. Every one of them is now
 * driven by a real signal (lib/telemetry.ts's link state, the boot fetch, a
 * missing `arms[id]`, and the calibration block disappearing), so the row is
 * gone rather than ported. A control that lies to the operator about the robot
 * has no business on a robot's settings page.
 */
import { useEffect, useState, useSyncExternalStore } from "react";
import { useTheme } from "next-themes";

import { api, type CameraInfo } from "@/lib/api";
import { useTelemetry } from "@/lib/telemetry";
import { LINK_STYLE } from "@/components/TelemetryBar";

type ConfigBody = Awaited<ReturnType<typeof api.config>>;

export function SettingsTab({
  cfg,
  cameras,
}: {
  cfg: ConfigBody;
  cameras: CameraInfo[];
}) {
  const link = useTelemetry((s) => s.link);
  const frameAge = useTelemetry((s) => s.frameAgeMs);
  const linkDetail = useTelemetry((s) => s.linkDetail);
  const armModes = useTelemetry((s) =>
    cfg.arms.map((a) => `${a.id}:${s.lastFrame?.arms?.[a.id]?.mode ?? "—"}`).join(" "),
  );
  const [health, setHealth] = useState<string | null>(null);
  const { theme, setTheme } = useTheme();
  // next-themes only knows the resolved theme after hydration, so the selected
  // segment has to wait for it — otherwise the server renders "dark" as active
  // for an operator whose stored preference is light. A store that reports
  // false on the server and true on the client is exactly this question, and
  // says it without a setState-in-effect.
  const mounted = useSyncExternalStore(subscribeNever, () => true, () => false);

  useEffect(() => {
    let cancelled = false;
    api
      .health()
      .then((h) => {
        if (!cancelled) setHealth(h.status);
      })
      .catch(() => {
        if (!cancelled) setHealth("unreachable");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const style = LINK_STYLE[link];
  const modeOf = Object.fromEntries(
    armModes.split(" ").filter(Boolean).map((p) => p.split(":") as [string, string]),
  );

  return (
    <div className="grid min-h-0 grid-cols-2 grid-rows-[auto_minmax(0,1fr)] gap-2 overflow-hidden p-2">
      <div className="col-span-2 flex h-9.5 items-center gap-3 rounded-lg bg-card px-3 shadow-[0_0_0_1px_var(--border)]">
        <span className="label-tracked shrink-0 text-muted-foreground">Health</span>
        {/* GET /health and the telemetry websocket are separate signals and can
            disagree — an HTTP-reachable backend with a dead publisher is a real
            and confusing failure, so both are shown. */}
        <span
          className="inline-flex shrink-0 items-center gap-1.5 font-mono text-[11px]"
          style={{
            color:
              health === "ok" ? "var(--haller-live)" : "var(--haller-fault)",
          }}
        >
          <span
            aria-hidden
            className="h-1.5 w-1.5 rounded-full"
            style={{
              backgroundColor:
                health === "ok" ? "var(--haller-live)" : "var(--haller-fault)",
            }}
          />
          GET /health · {health ?? "…"}
        </span>
        <span
          className="inline-flex min-w-0 items-center gap-1.5 font-mono text-[11px]"
          style={{ color: style.colour }}
          title={linkDetail}
        >
          <span
            aria-hidden
            className="h-1.5 w-1.5 shrink-0 rounded-full"
            style={{ backgroundColor: style.colour }}
          />
          <span className="truncate">
            ws · {style.label.toLowerCase()}
            {link === "live" && frameAge !== null
              ? ` · frame age ${frameAge} ms`
              : ""}
          </span>
        </span>

        <span aria-hidden className="h-px flex-1 bg-border" />

        <span className="label-tracked shrink-0 text-muted-foreground">Theme</span>
        <div className="inline-flex shrink-0 overflow-hidden rounded-sm border border-border">
          {(["dark", "light"] as const).map((k) => {
            const on = mounted && theme === k;
            return (
              <button
                key={k}
                type="button"
                onClick={() => setTheme(k)}
                aria-pressed={on}
                className={
                  "h-6.5 min-w-[60px] px-2.5 label-micro " +
                  (on
                    ? "bg-[var(--haller-live-soft)] text-[var(--haller-live)]"
                    : "text-muted-foreground hover:text-foreground")
                }
              >
                {k}
              </button>
            );
          })}
        </div>
      </div>

      <Table
        title="Arms"
        right={`${cfg.arms.length} configured`}
        columns="64px 1fr 1fr 62px"
        head={["id", "model", "port", "mode"]}
        rows={cfg.arms.map((a) => {
          const mode = modeOf[a.id] ?? "—";
          return {
            key: a.id,
            cells: [a.id, a.model, a.port, mode],
            // Live mode, from telemetry — config.yaml's `mode` is only the
            // boot default and drifts the moment anyone touches a mode toggle.
            lastColour:
              mode === "manual"
                ? "var(--haller-manual)"
                : mode === "auto"
                  ? "var(--haller-live)"
                  : mode === "stop"
                    ? "var(--haller-fault)"
                    : "var(--muted-foreground)",
          };
        })}
      />

      <Table
        title="Cameras"
        right={cameraCensus(cameras)}
        columns="1fr 60px 96px 88px"
        head={["id", "role", "source", "format"]}
        rows={cameras.map((c) => ({
          key: c.id,
          cells: [
            c.id,
            c.role,
            c.source,
            c.width && c.height && c.fps ? `${c.width}×${c.height}·${c.fps}` : "—",
          ],
        }))}
      />
    </div>
  );
}

/** Hydration is a one-way door: there is nothing to subscribe to. */
const subscribeNever = () => () => {};

function cameraCensus(cameras: CameraInfo[]): string {
  const sim = cameras.filter((c) => c.source === "sim_camera").length;
  const reserved = cameras.filter((c) => c.source === "placeholder").length;
  const hw = cameras.length - sim - reserved;
  return [
    hw ? `${hw} hardware` : null,
    reserved ? `${reserved} reserved` : null,
    sim ? `${sim} sim` : null,
  ]
    .filter(Boolean)
    .join(" · ") || "none";
}

function Table({
  title,
  right,
  columns,
  head,
  rows,
}: {
  title: string;
  right: string;
  columns: string;
  head: string[];
  rows: { key: string; cells: string[]; lastColour?: string }[];
}) {
  return (
    <div className="flex min-h-0 flex-col overflow-hidden rounded-lg bg-card shadow-[0_0_0_1px_var(--border)]">
      <div className="flex h-8.5 shrink-0 items-center justify-between border-b border-border px-3">
        <span className="font-mono text-[11px] font-semibold tracking-[0.14em] uppercase">
          {title}
        </span>
        <span className="label-micro text-muted-foreground">{right}</span>
      </div>
      <div className="min-h-0 overflow-y-auto font-mono text-[11px]">
        <div
          className="grid gap-2 border-b border-border px-3 py-1.5 label-micro text-muted-foreground"
          style={{ gridTemplateColumns: columns }}
        >
          {head.map((h) => (
            <span key={h}>{h}</span>
          ))}
        </div>
        {rows.map((r) => (
          <div
            key={r.key}
            className="grid gap-2 border-b border-border px-3 py-2 text-muted-foreground"
            style={{ gridTemplateColumns: columns }}
          >
            {r.cells.map((c, i) => (
              <span
                key={i}
                className="truncate"
                style={
                  i === 0
                    ? { color: "var(--foreground)" }
                    : i === r.cells.length - 1 && r.lastColour
                      ? { color: r.lastColour }
                      : undefined
                }
              >
                {c}
              </span>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
