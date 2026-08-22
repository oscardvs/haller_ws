"use client";

/**
 * Teleop: set a session up here, then put the headset on.
 *
 * The desktop is mission control, not an input device — every pose comes from
 * the Quest over its own socket. So this tab does the three things that are
 * easier with a keyboard and a big screen: choose the shape of the session
 * (which hands drive which arms), watch each side's authority while someone
 * else is wearing the headset, and hold the collision guard where it can be
 * read and flipped without taking the headset off.
 *
 * Every telemetry subscription here is a PRIMITIVE selector, and the two live
 * readouts that genuinely churn at frame rate — the per-side joint tables —
 * live in their own components, so a driving session re-renders those and
 * nothing else. See ArmCard for the same rule and the same reason.
 */
import {
  useCallback, useEffect, useState, useSyncExternalStore,
} from "react";
import { toast } from "sonner";

import { api, ApiError } from "@/lib/api";
import { useTelemetry, type TelemetryFrame } from "@/lib/telemetry";
import { STANCES, useStance } from "@/lib/stance";
import { DeadManIndicator } from "@/components/DeadManIndicator";
import { SimViewTile } from "@/components/SimViewTile";
import {
  presetsFor, simLeaderFor, type ConfigArm, type SessionPreset,
} from "./teleopPresets";
import type { Viewport } from "./lib";

const MIN_HZ = 1;
const MAX_HZ = 200;
const DEFAULT_HZ = "60";

/** The route the headset opens. Same origin as the cockpit — Caddy serves
 *  both — so the operator can read it off this screen and type it in. */
const HEADSET_PATH = "/teleop/vr";

export function TeleopTab({
  arms,
  cameras,
  viewport,
}: {
  arms: ConfigArm[];
  cameras: { source: string }[];
  viewport: Viewport;
}) {
  const running = useTelemetry((s) => s.lastFrame?.human_teleop?.running ?? false);
  const leftArm = useTelemetry((s) => s.lastFrame?.human_teleop?.left_arm ?? null);
  const rightArm = useTelemetry((s) => s.lastFrame?.human_teleop?.right_arm ?? null);
  const state = useTelemetry((s) => s.lastFrame?.human_teleop?.state ?? null);
  const lastError = useTelemetry((s) => s.lastFrame?.human_teleop?.last_error ?? null);
  const frameAge = useTelemetry((s) => s.lastFrame?.human_teleop?.frame_age_ms ?? null);

  const armIds = arms.map((a) => a.id);
  const hasSimCamera = cameras.some((c) => c.source === "sim_camera");

  return (
    <div
      className="grid min-h-0 gap-2 overflow-hidden p-2"
      style={{ gridTemplateColumns: "minmax(300px,352px) minmax(0,1fr)" }}
    >
      <div className="flex min-h-0 flex-col gap-2 overflow-y-auto">
        <SessionLauncher arms={arms} running={running} />
        <HeadsetEntry />
        <SimLeaderCard arms={arms} />
        {hasSimCamera && <SimViewTile placement="inline" />}
      </div>

      <div className="grid min-h-0 grid-rows-[auto_minmax(0,1fr)] gap-2 overflow-hidden">
        <CollisionGuardCard />

        <div className="flex min-h-0 flex-col overflow-hidden rounded-lg bg-card shadow-[0_0_0_1px_var(--border)]">
          <div className="flex h-8.5 shrink-0 items-center gap-2.5 border-b border-border px-3">
            <span className="label-tracked text-muted-foreground">Live session</span>
            <span aria-hidden className="h-px flex-1 bg-border" />
            <span className="font-mono text-[10px] whitespace-nowrap text-muted-foreground">
              state{" "}
              <span
                data-num
                style={{
                  color: running ? "var(--haller-live)" : "var(--muted-foreground)",
                }}
              >
                {state ?? "—"}
              </span>
            </span>
            {/* Frame age is the one number that says whether the HEADSET is
                still talking. A session can read "driving" with the operator's
                browser long since closed — the WS-disconnect grace is what
                ends it, and this is what shows it running out. */}
            <span className="font-mono text-[10px] whitespace-nowrap text-muted-foreground">
              frame{" "}
              <span data-num className="text-foreground">
                {running && frameAge != null ? `${Math.round(frameAge)} ms` : "—"}
              </span>
            </span>
          </div>

          {lastError && (
            <div className="mx-3 mt-2.5 shrink-0 rounded-md border border-[var(--haller-fault)] px-2.5 py-1.5 font-mono text-[10px] break-all text-[var(--haller-fault)]">
              session error: {lastError}
            </div>
          )}

          <div
            className="grid min-h-0 flex-1 gap-2 overflow-y-auto p-2.5"
            style={{
              gridTemplateColumns: viewport.compact
                ? "minmax(0,1fr)"
                : "repeat(2, minmax(0,1fr))",
            }}
          >
            <SidePanel side="left" armId={leftArm} running={running} />
            <SidePanel side="right" armId={rightArm} running={running} />
          </div>

          {!running && (
            <div className="shrink-0 border-t border-border px-3 py-2 text-[11px] text-pretty text-muted-foreground">
              No session. Pick a preset on the left, start it, then open{" "}
              <span className="font-mono">{HEADSET_PATH}</span> in the headset
              browser and squeeze a grip. Nothing moves until a side has
              acquired.
            </div>
          )}

          {running && (
            <div className="flex shrink-0 items-center gap-2 border-t border-border px-3 py-2">
              <span className="font-mono text-[10px] text-muted-foreground">
                {armIds.length} arm{armIds.length === 1 ? "" : "s"} configured ·
                session owns {[leftArm, rightArm].filter(Boolean).join(", ") || "none"}
              </span>
              <HomeButton />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ---- session launcher --------------------------------------------------- */

function SessionLauncher({
  arms,
  running,
}: {
  arms: ConfigArm[];
  running: boolean;
}) {
  // Not component state: the stance is remembered across reloads and shared
  // with the headset page in the same browser, so it lives in lib/stance.ts.
  const [stance, pickStance] = useStance();
  const [chosen, setChosen] = useState("dual");
  const [hz, setHz] = useState(DEFAULT_HZ);
  const [busy, setBusy] = useState(false);

  const armIds = arms.map((a) => a.id);
  const presets = presetsFor(armIds, stance);
  // Derived, not mirrored: the arm list comes from /config and the stance can
  // change under a selection, so the chosen preset is resolved at render and
  // falls back to the first one that this rig can actually start.
  const selected =
    presets.find((p) => p.id === chosen && !p.unavailable) ??
    presets.find((p) => !p.unavailable) ??
    null;

  const act = async () => {
    setBusy(true);
    try {
      if (running) {
        await api.humanTeleopStop();
        toast.message("teleop session stopped");
        return;
      }
      if (!selected) return;
      const rate = Number(hz);
      if (!Number.isFinite(rate) || rate < MIN_HZ || rate > MAX_HZ) {
        toast.error(`teleop start failed: rate must be ${MIN_HZ}–${MAX_HZ} Hz`);
        return;
      }
      await api.humanTeleopStart({ ...selected.pairing, hz: rate });
      toast.success(`session started · ${selected.detail}`);
    } catch (e) {
      toast.error(
        `teleop ${running ? "stop" : "start"} failed: ${(e as Error).message}`,
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex shrink-0 flex-col overflow-hidden rounded-lg bg-card shadow-[0_0_0_1px_var(--border)]">
      <div className="flex h-8.5 shrink-0 items-center justify-between border-b border-border px-3">
        <span className="label-tracked text-muted-foreground">Session</span>
        <span
          className="inline-flex h-5 items-center gap-1.5 rounded-full px-2 font-mono text-[10px]"
          style={{
            background: running ? "var(--haller-live-soft)" : "var(--secondary)",
            color: running ? "var(--haller-live)" : "var(--muted-foreground)",
          }}
        >
          <span
            aria-hidden
            className="h-1.5 w-1.5 rounded-full"
            style={{
              backgroundColor: running
                ? "var(--haller-live)"
                : "var(--muted-foreground)",
            }}
          />
          {running ? "running" : "idle"}
        </span>
      </div>

      <div className="flex flex-col gap-3 p-3">
        <div className="flex flex-col gap-1.5">
          <span className="label-tracked text-muted-foreground">Stance</span>
          {/* Where the operator is standing decides which physical arm the
              right hand drives — frozen at start, so it is disabled while a
              session runs rather than silently applying to the next one. */}
          <div
            role="radiogroup"
            aria-label="operator stance"
            className="flex items-center gap-1 rounded-md bg-muted p-1"
          >
            {STANCES.map((s) => (
              <button
                key={s.id}
                type="button"
                role="radio"
                aria-checked={stance === s.id}
                disabled={running}
                title={s.hint}
                onClick={() => pickStance(s.id)}
                className={
                  "h-6 flex-1 rounded-sm label-micro disabled:opacity-55 " +
                  (stance === s.id
                    ? "bg-card text-foreground"
                    : "text-muted-foreground hover:text-foreground")
                }
              >
                {s.label}
              </button>
            ))}
          </div>
          <span className="text-[11px] text-pretty text-muted-foreground">
            {STANCES.find((s) => s.id === stance)?.hint}
          </span>
        </div>

        <div className="flex flex-col gap-1.5">
          <span className="label-tracked text-muted-foreground">Preset</span>
          <div role="radiogroup" aria-label="session preset" className="flex flex-col gap-1.5">
            {presets.map((p) => (
              <PresetButton
                key={p.id}
                preset={p}
                active={selected?.id === p.id}
                disabled={running}
                onPick={() => setChosen(p.id)}
              />
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="label-micro shrink-0 text-muted-foreground">Rate</span>
          <input
            value={hz}
            onChange={(e) => setHz(e.target.value)}
            disabled={running}
            inputMode="numeric"
            aria-label="teleop rate in Hz"
            className="h-7 w-[58px] rounded-sm border border-input bg-background px-2 text-right font-mono text-[11px] disabled:opacity-50"
          />
          <span className="font-mono text-[11px] text-muted-foreground">Hz</span>
          <button
            type="button"
            disabled={busy || (!running && !selected)}
            onClick={act}
            className={
              "ml-auto h-8 rounded-md px-4 label-micro tracking-[0.12em] disabled:opacity-50 " +
              (running
                ? "border border-[var(--haller-fault)] bg-[oklch(0.62_0.245_27/0.2)] text-[var(--haller-fault)]"
                : "bg-primary text-primary-foreground")
            }
          >
            {running ? "stop session" : "start session"}
          </button>
        </div>
      </div>
    </div>
  );
}

function PresetButton({
  preset,
  active,
  disabled,
  onPick,
}: {
  preset: SessionPreset;
  active: boolean;
  disabled: boolean;
  onPick: () => void;
}) {
  const off = disabled || preset.unavailable !== null;
  return (
    <button
      type="button"
      role="radio"
      aria-checked={active}
      disabled={off}
      onClick={onPick}
      className={
        "flex flex-col items-start gap-0.5 rounded-md border px-2.5 py-1.5 text-left disabled:opacity-55 " +
        (active && !preset.unavailable
          ? "border-[var(--haller-live)] bg-[var(--haller-live-soft)]"
          : "border-border bg-[var(--haller-inset)] hover:border-[var(--haller-rail)]")
      }
    >
      <span
        className="label-micro"
        style={{
          color: active && !preset.unavailable
            ? "var(--haller-live)"
            : "var(--foreground)",
        }}
      >
        {preset.label}
      </span>
      <span className="font-mono text-[10px] text-muted-foreground">
        {preset.unavailable ?? preset.detail}
      </span>
    </button>
  );
}

/* ---- collision guard ---------------------------------------------------- */

/**
 * The guard, as a control rather than a status line.
 *
 * Two facts have to survive being read in a hurry: off still MEASURES, so the
 * clearance number means the same thing either way; and `available: false` is
 * one-way — a rig with no mount geometry for every arm cannot guard at all,
 * and enabling it there is refused by the backend, not merely ignored.
 */
function CollisionGuardCard() {
  const wired = useTelemetry((s) => s.lastFrame?.human_teleop?.collision != null);
  const enabled = useTelemetry(
    (s) => s.lastFrame?.human_teleop?.collision?.enabled ?? false,
  );
  const available = useTelemetry(
    (s) => s.lastFrame?.human_teleop?.collision?.available ?? true,
  );
  const slack = useTelemetry(
    (s) => s.lastFrame?.human_teleop?.collision?.slack_m ?? null,
  );
  const worst = useTelemetry(
    (s) => s.lastFrame?.human_teleop?.collision?.worst ?? null,
  );
  const limited = useTelemetry(
    (s) => s.lastFrame?.human_teleop?.collision?.limited ?? false,
  );
  const marginM = useTelemetry(
    (s) => s.lastFrame?.human_teleop?.collision?.margin_m ?? null,
  );
  const [busy, setBusy] = useState(false);

  const toggle = async () => {
    setBusy(true);
    const next = !enabled;
    try {
      await api.humanTeleopCollision(next);
      toast.success(`collision guard ${next ? "enabled" : "disabled"}`);
    } catch (e) {
      toast.error(`collision guard: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  // Colour ranks the same way the headset HUD does: actively clamping beats
  // inside-the-margin beats clear.
  const slackColour = limited
    ? "var(--haller-fault)"
    : slack !== null && slack < 0
      ? "var(--haller-warn)"
      : "var(--haller-live)";

  return (
    <div className="flex shrink-0 flex-col overflow-hidden rounded-lg bg-card shadow-[0_0_0_1px_var(--border)]">
      <div className="flex h-8.5 shrink-0 items-center justify-between border-b border-border px-3">
        <span className="label-tracked text-muted-foreground">Collision guard</span>
        <span className="font-mono text-[10px] text-muted-foreground">
          {!wired
            ? "not wired"
            : !available
              ? "unavailable on this rig"
              : `margin ${marginM !== null ? `${(marginM * 1000).toFixed(0)} mm` : "—"}`}
        </span>
      </div>

      <div className="flex items-stretch gap-3 p-3">
        <button
          type="button"
          disabled={busy || !wired || (!enabled && !available)}
          onClick={toggle}
          aria-pressed={enabled}
          className={
            "flex h-16 w-[168px] shrink-0 flex-col items-center justify-center gap-1 rounded-md border label-micro tracking-[0.14em] disabled:opacity-50 " +
            (enabled
              ? "border-[var(--haller-live)] bg-[var(--haller-live-soft)] text-[var(--haller-live)]"
              : "border-[var(--haller-warn)] bg-[oklch(0.82_0.16_78/0.14)] text-[var(--haller-warn)]")
          }
        >
          <span className="text-[15px] tracking-[0.18em]">
            {enabled ? "GUARD ON" : "GUARD OFF"}
          </span>
          <span className="label-micro opacity-70">
            {wired ? (enabled ? "click to disable" : "click to enable") : "no guard"}
          </span>
        </button>

        <div className="flex min-w-0 flex-1 flex-col justify-center gap-1">
          <div className="flex items-baseline gap-2">
            <span className="label-micro text-muted-foreground">
              {limited ? "collision hold" : "clearance"}
            </span>
            <span
              data-num
              className="font-mono text-[22px] leading-none"
              style={{ color: slackColour }}
            >
              {slack === null ? "—" : `${(slack * 1000).toFixed(0)} mm`}
            </span>
          </div>
          <span className="truncate font-mono text-[10px] text-muted-foreground">
            worst {worst ?? "—"}
          </span>
          <p className="text-[11px] text-pretty text-muted-foreground">
            {!wired
              ? "This backend reports no guard. The workspace floor, joint limits, rate caps and motion envelope are still on."
              : !available
                ? "This rig has no mount geometry for every arm, so the guard would pass every check it made. One-way: enabling it is refused."
                : "Off still MEASURES — the clearance above keeps updating, it just stops holding steps back. The workspace floor, joint limits, rate caps and motion envelope stay on either way."}
          </p>
        </div>
      </div>
    </div>
  );
}

/* ---- per-side authority ------------------------------------------------- */

function SidePanel({
  side,
  armId,
  running,
}: {
  side: "left" | "right";
  armId: string | null;
  running: boolean;
}) {
  const authority = useTelemetry(
    (s) => s.lastFrame?.human_teleop?.acquire?.[side]?.authority ?? null,
  );
  const reason = useTelemetry(
    (s) => s.lastFrame?.human_teleop?.acquire?.[side]?.reason ?? undefined,
  );
  const remainingMs = useTelemetry(
    (s) => s.lastFrame?.human_teleop?.acquire?.[side]?.remaining_ms ?? null,
  );
  const ramp = useTelemetry(
    (s) => s.lastFrame?.human_teleop?.acquire?.[side]?.ramp ?? null,
  );
  const gripHeld = useTelemetry((s) => {
    const c = s.lastFrame?.human_teleop?.clutch;
    if (!c) return false;
    // Per-side when the backend says which sides `engaged` covers; the single
    // boolean is the fallback, and it means both.
    return c.sides ? Boolean(c.sides[side]) : Boolean(c.engaged);
  });
  const lost = useTelemetry(
    (s) => s.lastFrame?.human_teleop?.tracking?.[side]?.lost ?? false,
  );
  const ageMs = useTelemetry(
    (s) => s.lastFrame?.human_teleop?.tracking?.[side]?.age_ms ?? null,
  );
  // One string, so this subscription is a stable primitive and the 20-30 Hz
  // churn of a driving session re-renders this panel and nothing above it.
  // Rows are "joint\tgoal\tmeasured", newline-separated. Same idea as
  // ArmCard's \0-joined joint-key list.
  const rows = useTelemetry((s) => jointRows(s.lastFrame, armId, side));

  const noArm = armId === null;

  return (
    <div className="flex min-h-0 flex-col overflow-hidden rounded-md border border-border bg-[var(--haller-inset)]">
      <div className="flex h-7.5 shrink-0 items-center gap-2 border-b border-border bg-[var(--haller-chrome)] px-2.5">
        <span className="label-micro shrink-0 text-muted-foreground">
          {side} hand
        </span>
        <span className="shrink-0 font-mono text-[11px]">{armId ?? "—"}</span>
        <span className="ml-auto shrink-0 font-mono text-[10px] text-muted-foreground">
          {noArm
            ? "unused"
            : authority === "driving" && ramp !== null
              ? `ramp ${(ramp * 100).toFixed(0)}%`
              : ageMs !== null
                ? `age ${Math.round(ageMs)} ms`
                : "—"}
        </span>
      </div>

      <div className="flex shrink-0 items-center gap-2 px-2.5 py-2">
        <DeadManIndicator
          source="vr_grip"
          held={gripHeld && authority === "driving"}
          acquiring={authority === "acquiring"}
          remainingMs={remainingMs}
          trackingLost={lost}
          reason={noArm ? "no_arm" : reason}
        />
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {!running || noArm || rows === "" ? (
          <div className="relative flex h-full min-h-[64px] items-center justify-center">
            <span className="scanlines absolute inset-0" aria-hidden />
            <span className="relative font-mono text-[10px] tracking-[0.14em] uppercase text-muted-foreground opacity-70">
              {/* The chip above already says "not in session" for an absent
                  side; this line says what the empty table means instead. */}
              {noArm ? "nothing written here" : running ? "no goals yet" : "idle"}
            </span>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-[minmax(0,1fr)_54px_54px_50px] gap-2 border-b border-border bg-muted px-2.5 py-1 label-micro text-muted-foreground">
              <span>joint</span>
              <span className="text-right">goal</span>
              <span className="text-right">meas</span>
              <span className="text-right">Δ</span>
            </div>
            {rows.split("\n").map((row) => {
              const [joint, goal, meas, delta] = row.split("\t");
              return (
                <div
                  key={joint}
                  className="grid grid-cols-[minmax(0,1fr)_54px_54px_50px] gap-2 border-b border-border px-2.5 py-0.5 font-mono text-[10px] tabular-nums"
                >
                  <span className="truncate">{joint}</span>
                  <span className="text-right text-muted-foreground">{goal}</span>
                  <span className="text-right">{meas}</span>
                  {/* The gap between what teleop asked for and where the servo
                      actually is. Large and persistent means the arm is
                      fighting something — a limit, the guard, or a stall. */}
                  <span className="text-right text-muted-foreground">{delta}</span>
                </div>
              );
            })}
          </>
        )}
      </div>
    </div>
  );
}

function fmt1(n: number | undefined): string {
  return typeof n === "number" && Number.isFinite(n) ? n.toFixed(1) : "—";
}

/** "joint\tgoal\tmeasured\tdelta" per line. Empty when the side is not in the
 *  session or nothing has been commanded yet. */
export function jointRows(
  frame: TelemetryFrame | null,
  armId: string | null,
  side: "left" | "right",
): string {
  if (!frame || !armId) return "";
  const goals = frame.human_teleop?.goal_deg?.[side] ?? {};
  const joints = frame.arms?.[armId]?.joints ?? {};
  // Measured joints order the table — that is the arm's own joint set. A goal
  // for a joint the arm does not report is still shown: it means the two
  // disagree about what this arm has, which is worth seeing.
  const keys = Object.keys(joints);
  for (const k of Object.keys(goals)) if (!keys.includes(k)) keys.push(k);
  if (keys.length === 0) return "";
  return keys
    .map((k) => {
      const g = goals[k];
      const m = joints[k]?.pos;
      const d =
        typeof g === "number" && typeof m === "number" ? fmt1(g - m) : "—";
      return `${k}\t${fmt1(g)}\t${fmt1(m)}\t${d}`;
    })
    .join("\n");
}

/* ---- the headset entry point -------------------------------------------- */

/** Nothing to subscribe to — the "store" is whether we are on the client. */
function subscribeNever(): () => void {
  return () => {};
}

function HeadsetEntry() {
  // There is no origin during the server render, and rendering one the client
  // then corrects is a hydration mismatch. Same idiom as SettingsTab's theme
  // segment: a store that is false on the server and true on the client.
  const mounted = useSyncExternalStore(subscribeNever, () => true, () => false);
  const origin = mounted ? window.location.origin : null;

  const url = origin === null ? null : `${origin}${HEADSET_PATH}`;
  // WebXR only runs in a secure context. Reading a plain-http URL off this
  // card and typing it into the Quest gets a page that loads and then refuses
  // to enter VR, which is a long way to walk to find out.
  const insecure =
    origin !== null &&
    !origin.startsWith("https://") &&
    !/^https?:\/\/(localhost|127\.0\.0\.1)(:|$)/.test(origin);

  const copy = useCallback(() => {
    if (!url) return;
    navigator.clipboard
      ?.writeText(url)
      .then(() => toast.success("headset URL copied"))
      .catch(() => toast.error("clipboard blocked — read it off the card"));
  }, [url]);

  return (
    <div className="flex shrink-0 flex-col overflow-hidden rounded-lg bg-card shadow-[0_0_0_1px_var(--border)]">
      <div className="flex h-8.5 shrink-0 items-center justify-between border-b border-border px-3">
        <span className="label-tracked text-muted-foreground">Open in the headset</span>
        <span className="font-mono text-[10px] text-muted-foreground">Quest browser</span>
      </div>
      <div className="flex flex-col gap-2 p-3">
        <div
          className="rounded-md border border-border bg-[var(--haller-inset)] px-2.5 py-2 font-mono text-[12px] break-all select-all"
          style={{ color: insecure ? "var(--haller-warn)" : "var(--haller-live)" }}
        >
          {url ?? "…"}
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={copy}
            className="h-7 rounded-md border border-border bg-secondary px-3 label-micro tracking-[0.12em]"
          >
            copy
          </button>
          <a
            href={HEADSET_PATH}
            target="_blank"
            rel="noreferrer"
            className="h-7 rounded-md border border-border px-3 leading-7 label-micro tracking-[0.12em] text-muted-foreground hover:text-foreground"
          >
            open here ↗
          </a>
        </div>
        {insecure ? (
          <p className="text-[11px] text-pretty text-[var(--haller-warn)]">
            This origin is not HTTPS. The page will load in the headset and then
            refuse to enter VR — WebXR needs a secure context. Serve the
            single HTTPS origin with{" "}
            <span className="font-mono">scripts/quest-teleop/up.sh</span>.
          </p>
        ) : (
          <p className="text-[11px] text-pretty text-muted-foreground">
            Same origin as this cockpit. Enter VR, then squeeze a grip to take a
            side; B/Y is E-STOP and A/X held toggles the recorder.
          </p>
        )}
      </div>
    </div>
  );
}

/* ---- sim leader (bring-up without a headset) ---------------------------- */

function SimLeaderCard({ arms }: { arms: ConfigArm[] }) {
  const pair = simLeaderFor(arms);
  const running = useTelemetry((s) => s.lastFrame?.teleop?.running ?? false);
  const [busy, setBusy] = useState(false);
  const [simRunning, setSimRunning] = useState(false);

  // Depends on the two ids, not on `pair` — simLeaderFor returns a fresh
  // object every render, so a `[pair]` dependency would re-poll on every one.
  const leaderId = pair?.leader;
  useEffect(() => {
    if (!leaderId) return;
    let cancelled = false;
    api.simTeleopStatus()
      .then((s) => { if (!cancelled) setSimRunning(s.running); })
      .catch(() => { /* no sim world — the card says so on the first click */ });
    return () => { cancelled = true; };
  }, [leaderId]);

  if (!pair) return null;

  const act = async () => {
    setBusy(true);
    try {
      const s = simRunning
        ? await api.simTeleopStop()
        : await api.simTeleopStart({
            follower: pair.follower,
            leader: { source: "mouse", arm_name: pair.leader },
          });
      setSimRunning(s.running);
      toast.message(`sim leader ${s.running ? "started" : "stopped"}`);
    } catch (e) {
      toast.error(`sim leader: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex shrink-0 flex-col overflow-hidden rounded-lg bg-card shadow-[0_0_0_1px_var(--border)]">
      <div className="flex h-8.5 shrink-0 items-center justify-between border-b border-border px-3">
        <span className="label-tracked text-muted-foreground">Sim leader</span>
        <span className="font-mono text-[10px] text-muted-foreground">
          {pair.leader} → {pair.follower}
        </span>
      </div>
      <div className="flex flex-col gap-2 p-3">
        <p className="text-[11px] text-pretty text-muted-foreground">
          Bring-up without a headset: drag{" "}
          <span className="font-mono">{pair.leader}</span>&apos;s joints in the
          native MuJoCo viewer (<span className="font-mono">MUJOCO_VIEWER=1</span>)
          and <span className="font-mono">{pair.follower}</span> tracks it.
        </p>
        <button
          type="button"
          disabled={busy || running}
          title={running ? "the leader→follower bridge is already running" : undefined}
          onClick={act}
          className={
            "h-7 rounded-md label-micro tracking-[0.12em] disabled:opacity-50 " +
            (simRunning
              ? "border border-[var(--haller-fault)] bg-[oklch(0.62_0.245_27/0.2)] text-[var(--haller-fault)]"
              : "border border-border bg-secondary")
          }
        >
          {simRunning ? "stop sim leader" : "start sim leader"}
        </button>
      </div>
    </div>
  );
}

/* ---- in-session home ---------------------------------------------------- */

/** The counterpart to the headset's left-stick hold. The discrete
 *  `/arm/{id}/home` is refused while a session owns the arms, so this is the
 *  only way to park them from the desk — and it rides the session's own LPF,
 *  rate caps and collision guard rather than going around them. */
function HomeButton() {
  const [busy, setBusy] = useState(false);
  return (
    <button
      type="button"
      disabled={busy}
      onClick={async () => {
        setBusy(true);
        try {
          const r = await api.humanTeleopHome();
          toast.message(
            r.sides.length
              ? `homing ${r.sides.join(", ")}`
              : "every side is driving — release the grips first",
          );
        } catch (e) {
          const msg =
            e instanceof ApiError && e.status === 409
              ? "no session running"
              : (e as Error).message;
          toast.error(`home failed: ${msg}`);
        } finally {
          setBusy(false);
        }
      }}
      className="ml-auto h-6 shrink-0 rounded-sm border border-border bg-secondary px-2.5 label-micro tracking-[0.12em] disabled:opacity-50"
    >
      park non-driving sides
    </button>
  );
}
