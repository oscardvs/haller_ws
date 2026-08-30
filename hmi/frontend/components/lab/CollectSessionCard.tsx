"use client";

/**
 * The collect page's slice of the Teleop tab: the session the take will be
 * demonstrated through, startable without leaving the recorder.
 *
 * Compact on purpose — the collision guard, the per-joint tables and the
 * solo-hand override stay on the Teleop tab, which is mission control for the
 * session itself. What lives here is what a collection run needs beside the
 * recorder: the presets this rig can actually start, drawn from the same
 * `presetsFor` the Teleop tab calls so the two surfaces can never offer
 * different sessions; start/stop; who owns each side right now; and the URL
 * the headset opens. Recording without a session stays allowed-and-warned —
 * that warning lives on the recorder card, not here.
 */
import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { toast } from "sonner";

import { api } from "@/lib/api";
import { headsetOrigin, isLoopback } from "@/lib/config";
import { useTelemetry } from "@/lib/telemetry";
import { STANCES, useSoloHand, useStance } from "@/lib/stance";
import {
  presetsFor, type ConfigArm, type SessionPreset,
} from "@/components/cockpit/teleopPresets";
import { Button, Panel, PanelHead } from "@/components/lab/ui";

const MIN_HZ = 1;
const MAX_HZ = 200;
const DEFAULT_HZ = "60";

/** The route the headset opens. Duplicated from TeleopTab rather than shared:
 *  two small constants that must agree with the backend route, not a module. */
const HEADSET_PATH = "/teleop/vr";

export function CollectSessionCard({
  arms,
  datasetFps = null,
}: {
  arms: ConfigArm[];
  /** The rate the resumed dataset was written at (CollectResumeCard's read
   *  off its newest episode), null for a new dataset. Appends are refused
   *  outside the recorder's fps band, so a mismatch is steered on BEFORE
   *  record, not after. */
  datasetFps?: number | null;
}) {
  const running = useTelemetry((s) => s.lastFrame?.human_teleop?.running ?? false);
  const state = useTelemetry((s) => s.lastFrame?.human_teleop?.state ?? null);
  const leftArm = useTelemetry((s) => s.lastFrame?.human_teleop?.left_arm ?? null);
  const rightArm = useTelemetry((s) => s.lastFrame?.human_teleop?.right_arm ?? null);
  const leftAuth = useTelemetry(
    (s) => s.lastFrame?.human_teleop?.acquire?.left?.authority ?? null,
  );
  const rightAuth = useTelemetry(
    (s) => s.lastFrame?.human_teleop?.acquire?.right?.authority ?? null,
  );

  // Why a session ended when nobody pressed stop. The backend's WS-grace
  // auto-stop used to be completely silent — INFO-level, and the app loggers
  // serve at WARNING — so an operator who reloaded and found the start button
  // back had nothing anywhere telling them what had happened.
  const stoppedReason = useTelemetry(
    (s) => s.lastFrame?.human_teleop?.stopped_reason ?? null,
  );
  const toldRef = useRef<string | null>(null);
  useEffect(() => {
    // Once per reason, not once per frame: this arrives at telemetry rate and
    // holds its value until the next session starts.
    if (running || !stoppedReason || toldRef.current === stoppedReason) return;
    toldRef.current = stoppedReason;
    toast.warning(`teleop session ended — ${stoppedReason}`);
  }, [running, stoppedReason]);
  useEffect(() => { if (running) toldRef.current = null; }, [running]);

  const [stance, pickStance] = useStance();
  const [soloHand] = useSoloHand();
  const [chosen, setChosen] = useState("dual");
  const [hz, setHz] = useState(DEFAULT_HZ);
  const [busy, setBusy] = useState(false);

  const presets = presetsFor(arms.map((a) => a.id), stance, soloHand);
  // Same rule as the Teleop launcher: the choice is resolved at render and
  // falls back to the first preset this rig can actually start, so a stance
  // change under a selection can never arm a session the rig does not have.
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
    <Panel>
      <PanelHead
        title="VR session"
        right={
          <span style={{ color: running ? "var(--haller-live)" : undefined }}>
            {running ? `running · ${state ?? "—"}` : "idle"}
          </span>
        }
      />

      <div className="flex flex-col gap-2.5 p-3">
        {/* Where the operator stands decides which physical arm the right hand
            drives; frozen at session start, so disabled while one runs. The
            choice is the shared one (lib/stance.ts) — set it here or on the
            Teleop tab, the headset reads the same value. */}
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
          <Button
            tone={running ? "danger" : "primary"}
            disabled={busy || (!running && !selected)}
            onClick={act}
            className="ml-auto h-8 px-4"
          >
            {running ? "stop session" : "start session"}
          </Button>
        </div>

        {/* The rest pose, said BEFORE the button rather than after the fault.
            2026-08-28: the arm sat sagged with torque off, elbow_flex read
            121 deg against a +/-92 limit, preflight failed and dropped
            torque — and starting a session cleared that stop and drove it
            anyway, until wrist_flex latched an overload. The servo's own
            protection was the thing that caught it. An arm parked in the L
            never presents that first reading. */}
        {!running && (
          <div className="rounded-md border border-input px-2.5 py-1.5 text-[10px] text-pretty text-muted-foreground">
            Park the arm in the <span className="text-foreground font-medium">L
            rest position</span> before starting — upper arm up, forearm level.
            A sagged arm reads outside its joint limits, fails preflight, and
            can trip a servo into overload on the first move.
          </div>
        )}

        {/* Appending to a resumed dataset is refused outside the recorder's
            fps band — and hz is frozen at session start, so the steer has to
            happen here, before record, not in the gate's refusal after it. */}
        {datasetFps !== null && Number(hz) !== datasetFps && (
          <div className="flex items-center gap-2 rounded-md border border-[var(--haller-warn)] px-2.5 py-1.5 text-[10px] text-pretty text-[var(--haller-warn)]">
            <span className="min-w-0 flex-1">
              {running
                ? `session is running at ${hz} Hz — this dataset appends only at ${datasetFps} Hz; stop and restart the session at that rate`
                : `dataset is written at ${datasetFps} fps — recording appends only if the session runs at that rate`}
            </span>
            {!running && (
              <button
                type="button"
                onClick={() => setHz(String(datasetFps))}
                className="h-6 shrink-0 rounded-sm border border-[var(--haller-warn)] px-2 label-micro tracking-[0.12em]"
              >
                use {datasetFps} Hz
              </button>
            )}
          </div>
        )}

        {/* Per-side authority, one line: who the session gave each hand and
            what that hand is doing with it. "—" for a side the session does
            not own, which is what a solo preset always produces. */}
        <div className="flex items-center gap-3 font-mono text-[10px] text-muted-foreground">
          <span>
            L → <span className="text-foreground">{leftArm ?? "—"}</span>
            {leftArm && leftAuth ? ` · ${leftAuth}` : ""}
          </span>
          <span>
            R → <span className="text-foreground">{rightArm ?? "—"}</span>
            {rightArm && rightAuth ? ` · ${rightAuth}` : ""}
          </span>
        </div>

        <HeadsetUrl />
      </div>
    </Panel>
  );
}

/** Same shape as the Teleop tab's preset button, local because the shared
 *  primitive file is not this card's to grow. Unavailable presets are drawn
 *  greyed WITH the reason — "there is no second arm" is a fact about the
 *  robot, and a picker that quietly drops the option makes the operator
 *  doubt their memory. */
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

/* ---- the headset entry point -------------------------------------------- */

/** Nothing to subscribe to — the "store" is whether we are on the client. */
function subscribeNever(): () => void {
  return () => {};
}

/** Compact read of the same card the Teleop tab draws: the single HTTPS
 *  origin the headset must open, and the one warning that matters — WebXR
 *  refuses a plain-http origin. The origin is the bundle's baked backend URL
 *  minus `/api`, NOT the page's own: on a Quest, "localhost" is the headset. */
function HeadsetUrl() {
  // There is no origin during the server render; the store idiom keeps the
  // first client render from disagreeing with it.
  const mounted = useSyncExternalStore(subscribeNever, () => true, () => false);
  const origin = headsetOrigin(mounted ? window.location.origin : null);
  const url = origin === null ? null : `${origin}${HEADSET_PATH}`;
  const insecure =
    origin !== null && !origin.startsWith("https://") && !isLoopback(origin);

  return (
    <div className="flex flex-col gap-1">
      <span className="label-micro text-muted-foreground">open in the headset</span>
      <div className="flex items-center gap-1.5">
        <span
          className="min-w-0 flex-1 truncate rounded-sm border border-border bg-[var(--haller-inset)] px-2 py-1 font-mono text-[10px] select-all"
          style={{ color: insecure ? "var(--haller-warn)" : "var(--haller-live)" }}
          title={insecure ? "not HTTPS — WebXR will refuse to enter VR" : url ?? undefined}
        >
          {url ?? "…"}
        </span>
        <Button
          tone="ghost"
          className="h-6 border-border"
          onClick={() => {
            if (!url) return;
            navigator.clipboard
              ?.writeText(url)
              .then(() => toast.success("headset URL copied"))
              .catch(() => toast.error("clipboard blocked — read it off the card"));
          }}
        >
          copy
        </Button>
      </div>
    </div>
  );
}
