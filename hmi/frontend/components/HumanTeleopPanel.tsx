"use client";

/**
 * Top-level orchestrator for the human-teleop page. Owns:
 *   - The camera stream and the MediaPipeRunner lifecycle.
 *   - The render loop that runs detection + WS publish at ~30 Hz.
 *   - The dead-man key state.
 *   - The pinch calibration state (persisted in localStorage).
 *   - Start/stop/swap session calls.
 */
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";

import { api, type HumanTeleopStatus, type JointDiag, type JointReason } from "@/lib/api";
import { BACKEND_URL } from "@/lib/config";
import { useTelemetry, type ArmState } from "@/lib/telemetry";
import {
  MediaPipeRunner, fuseLandmarkResults, buildOverlaySides,
  type KeypointFrame, type SideFrame,
} from "@/lib/mediapipe";
import { HumanTeleopClient } from "@/lib/humanTeleopClient";

import { CameraOverlay, type CameraOverlayHandle } from "./CameraOverlay";
import { ScopeBar } from "./ScopeBar";
import { DeadManIndicator } from "./DeadManIndicator";
import { PinchCalibrationStep, type PinchCalib } from "./PinchCalibrationStep";

const WS_URL = `${BACKEND_URL.replace(/^http/, "ws")}/ws/teleop/human/in`;

const JOINTS = [
  "shoulder_pan", "shoulder_lift", "elbow_flex",
  "wrist_flex", "wrist_roll", "gripper",
] as const;

const CALIB_LS_KEY = "haller.humanTeleop.pinchCalib.v1";

export function HumanTeleopPanel({ armIds }: { armIds: string[] }) {
  const status = useTelemetry((s) => s.lastFrame?.human_teleop);
  const armsState = useTelemetry((s) => s.lastFrame?.arms);

  const [leftArm, setLeftArm] = useState(armIds[0] ?? "");
  const [rightArm, setRightArm] = useState(armIds[1] ?? armIds[0] ?? "");
  const [swap, setSwap] = useState(false);

  const overlayRef = useRef<CameraOverlayHandle | null>(null);
  const runnerRef = useRef<MediaPipeRunner | null>(null);
  const clientRef = useRef<HumanTeleopClient | null>(null);
  const deadManRef = useRef(false);
  const statusRef = useRef<HumanTeleopStatus | undefined>(undefined);
  useEffect(() => { statusRef.current = status; }, [status]);

  const [calib, setCalib] = useState<{ left: PinchCalib; right: PinchCalib }>(() => {
    if (typeof window === "undefined") return defaultCalib();
    try {
      const raw = localStorage.getItem(CALIB_LS_KEY);
      if (raw) return JSON.parse(raw);
    } catch { /* ignore */ }
    return defaultCalib();
  });
  const [liveDistance, setLiveDistance] = useState<{ left: number | null; right: number | null }>({
    left: null, right: null,
  });
  const [liveConf, setLiveConf] = useState<{ left: number | null; right: number | null }>({
    left: null, right: null,
  });

  // Persist calib on change.
  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem(CALIB_LS_KEY, JSON.stringify(calib));
    }
  }, [calib]);

  // Bind dead-man key.
  useEffect(() => {
    const onDown = (e: KeyboardEvent) => {
      if (e.code === "Space" && !isInput(e.target)) {
        e.preventDefault();
        deadManRef.current = true;
      }
    };
    const onUp = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        deadManRef.current = false;
      }
    };
    window.addEventListener("keydown", onDown);
    window.addEventListener("keyup", onUp);
    return () => {
      window.removeEventListener("keydown", onDown);
      window.removeEventListener("keyup", onUp);
    };
  }, []);

  // One-shot: open camera + load models.
  useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay || !overlay.video) return;
    let cancelled = false;
    let stream: MediaStream | null = null;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 960, height: 540 }, audio: false,
        });
        if (cancelled) return;
        overlay.video!.srcObject = stream;
        await overlay.video!.play();
        runnerRef.current = new MediaPipeRunner();
        await runnerRef.current.load();
        clientRef.current = new HumanTeleopClient(WS_URL);
        clientRef.current.connect();
      } catch (e) {
        toast.error(`camera/MediaPipe init failed: ${(e as Error).message}`);
      }
    })();
    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
      runnerRef.current?.close();
      runnerRef.current = null;
      clientRef.current?.close();
      clientRef.current = null;
    };
  }, []);

  // Render loop: detect → overlay → publish.
  useEffect(() => {
    let raf = 0;
    let last_t = 0;
    const tick = (t: number) => {
      raf = requestAnimationFrame(tick);
      const runner = runnerRef.current;
      const overlay = overlayRef.current;
      const client = clientRef.current;
      const video = overlay?.video;
      if (!runner || !overlay || !client || !video || video.readyState < 2) return;
      // Cap to ~30 Hz; MediaPipe will internally throttle if GPU saturates.
      if (t - last_t < 33) return;
      last_t = t;
      const { hands, pose } = runner.detect(video, t);
      const fused = fuseLandmarkResults(pose, hands);
      const ld = liveThumbIndex(fused);

      overlay.draw(buildOverlaySides(pose, hands, {
        leftLost:  statusRef.current?.tracking?.left?.lost ?? false,
        rightLost: statusRef.current?.tracking?.right?.lost ?? false,
        leftPinch01:  pinch01For(ld.left, calib.left),
        rightPinch01: pinch01For(ld.right, calib.right),
      }));

      // Functional update with an identity bail-out: returning `prev` unchanged
      // makes React skip the re-render, so this needs no effect dependency.
      // Do NOT add `liveDistance` to the effect's dep array — that would tear
      // down and recreate the requestAnimationFrame loop on every frame.
      setLiveDistance((prev) =>
        prev.left === ld.left && prev.right === ld.right ? prev : ld,
      );

      // Functional update with an identity bail-out: returning `prev` unchanged
      // makes React skip the re-render, so this needs no effect dependency.
      // Do NOT add `liveConf` to the effect's dep array — that would tear down
      // and recreate the requestAnimationFrame loop on every confidence change.
      const lc = { left: fused.left?.confidence ?? null, right: fused.right?.confidence ?? null };
      setLiveConf((prev) =>
        prev.left === lc.left && prev.right === lc.right ? prev : lc,
      );

      const frame: KeypointFrame = {
        type: "keypoints",
        ts_ms: Math.floor(performance.now()),
        dead_man: deadManRef.current,
        pinch_calib: {
          left:  calib.left.min_m !== null && calib.left.max_m !== null
            ? { min_m: calib.left.min_m, max_m: calib.left.max_m } : undefined,
          right: calib.right.min_m !== null && calib.right.max_m !== null
            ? { min_m: calib.right.min_m, max_m: calib.right.max_m } : undefined,
        },
        left: fused.left, right: fused.right,
      };
      client.queueFrame(frame);
      client.tick();
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [calib]);

  const running = status?.running ?? false;
  const state = status?.state ?? "idle";

  const handleStart = async () => {
    try {
      // Push calibration first if both sides are complete.
      const cl = calib.left.min_m !== null && calib.left.max_m !== null
        ? { min_m: calib.left.min_m, max_m: calib.left.max_m } : undefined;
      const cr = calib.right.min_m !== null && calib.right.max_m !== null
        ? { min_m: calib.right.min_m, max_m: calib.right.max_m } : undefined;
      if (cl || cr) await api.humanTeleopCalibrate({ left: cl, right: cr });
      await api.humanTeleopStart({ left_arm: leftArm, right_arm: rightArm, swap });
      toast.success(`human teleop started`);
    } catch (e) {
      toast.error(`start failed: ${(e as Error).message}`);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-3">
      <div className="space-y-2">
        <CameraOverlay ref={overlayRef} aspectRatio="16/9" />
        <div className="flex items-center justify-between">
          <DeadManIndicator
            held={state === "driving"}
            trackingLost={!!status?.tracking?.left?.lost || !!status?.tracking?.right?.lost}
          />
          <div className="flex items-center gap-2 font-mono text-[12px]">
            <Badge variant={running ? "default" : "secondary"}>{state}</Badge>
            {running ? (
              <Button size="sm" variant="destructive"
                      className="h-7 px-3"
                      onClick={() => api.humanTeleopStop().catch(() => null)}>
                stop
              </Button>
            ) : (
              <Button size="sm" className="h-7 px-3" onClick={handleStart}
                      disabled={!leftArm || !rightArm || leftArm === rightArm}>
                start
              </Button>
            )}
          </div>
        </div>
        <Card className="p-3 flex flex-wrap items-center gap-2 font-mono text-[12px]">
          <span className="text-muted-foreground">assign</span>
          <NativeSelect ariaLabel="left arm" value={leftArm} onChange={setLeftArm}
                        options={armIds.map((id) => ({ value: id, label: id }))} />
          <Button size="sm" variant="outline" className="h-7"
                  onClick={() => { const t = leftArm; setLeftArm(rightArm); setRightArm(t); }}>
            ⇄
          </Button>
          <NativeSelect ariaLabel="right arm" value={rightArm} onChange={setRightArm}
                        options={armIds.filter((id) => id !== leftArm).map((id) => ({ value: id, label: id }))} />
          <Button size="sm" variant="outline" className="h-7"
                  onClick={() => { setSwap(!swap); api.humanTeleopSwap(!swap).catch(() => null); }}>
            mirror: {swap ? "off" : "on"}
          </Button>
        </Card>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <PinchCalibrationStep
            side="left" liveDistance={liveDistance.left} confidence={liveConf.left} value={calib.left}
            onChange={(next) => setCalib({ ...calib, left: next })}
          />
          <PinchCalibrationStep
            side="right" liveDistance={liveDistance.right} confidence={liveConf.right} value={calib.right}
            onChange={(next) => setCalib({ ...calib, right: next })}
          />
        </div>
      </div>
      <div className="space-y-2">
        <ArmScopePanel label={`arm: ${leftArm}`} goal={status?.goal_deg?.left}
                       limits={limitsFor(armsState, leftArm)}
                       diag={status?.joints?.left} />
        <ArmScopePanel label={`arm: ${rightArm}`} goal={status?.goal_deg?.right}
                       limits={limitsFor(armsState, rightArm)}
                       diag={status?.joints?.right} />
      </div>
    </div>
  );
}

// Only reasons worth flagging get a label. `ok` is the silent default and
// deliberately has none. Typed as Partial so the compiler — not a runtime
// `badge ?? ""` fallback — is what proves "two of four reasons intentionally
// have no label".
const REASON_LABEL: Partial<Record<JointReason, string>> = {
  clamped: "CLAMPED",
  rate_capped: "RATE-CAP",
  held: "HELD",
};

function ArmScopePanel({
  label, goal, limits, diag,
}: {
  label: string;
  goal?: Record<string, number>;
  limits?: Record<string, { min: number; max: number }>;
  diag?: Record<string, JointDiag>;
}) {
  return (
    <Card className="p-3">
      <div className="flex justify-between text-[12px] font-mono mb-2">
        <span>{label}</span>
      </div>
      <div className="space-y-1">
        {JOINTS.map((j) => {
          const d = diag?.[j];
          const badge = d ? REASON_LABEL[d.reason] : undefined;
          return (
            <div key={j} className="flex items-center gap-2">
              <div className="flex-1">
                <ScopeBar
                  label={j}
                  min={limits?.[j]?.min ?? -90}
                  max={limits?.[j]?.max ?? 90}
                  commanded={goal?.[j] ?? 0}
                  intended={d?.target ?? undefined}
                />
              </div>
              <span
                className="w-16 text-right font-mono text-[10px] text-[var(--instrument-warn,oklch(75%_0.16_70))]"
              >
                {badge ?? ""}
              </span>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

function NativeSelect({
  value, onChange, options, ariaLabel,
}: {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
  ariaLabel?: string;
}) {
  return (
    <select
      aria-label={ariaLabel}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="h-7 rounded-sm border border-border bg-background px-2 font-mono text-[12px]"
    >
      {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  );
}

function defaultCalib(): { left: PinchCalib; right: PinchCalib } {
  return {
    left:  { min_m: null, max_m: null },
    right: { min_m: null, max_m: null },
  };
}

function isInput(t: EventTarget | null): boolean {
  return t instanceof HTMLElement &&
    (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
}

function liveThumbIndex(
  fused: { left: SideFrame | null; right: SideFrame | null }
): { left: number | null; right: number | null } {
  const dist = (s: SideFrame | null): number | null => {
    if (!s) return null;
    const a = s.hand.thumb_tip, b = s.hand.index_tip;
    const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  };
  return { left: dist(fused.left), right: dist(fused.right) };
}

/** Map a raw thumb-index distance onto [0,1] using the captured calibration.
 *  Returns 0.5 (neutral) when that side isn't calibrated yet, so the overlay
 *  still draws rather than showing a permanently-dashed pinch line. */
function pinch01For(distance: number | null, calib: PinchCalib): number {
  if (distance === null || calib.min_m === null || calib.max_m === null) return 0.5;
  const span = calib.max_m - calib.min_m;
  if (span <= 0) return 0.5;
  return Math.max(0, Math.min(1, (distance - calib.min_m) / span));
}

/** Per-joint {min,max} in degrees for one arm, straight from calibration via
 *  telemetry. Returns undefined when that arm isn't reporting yet, in which
 *  case ScopeBar falls back to its own default range. */
function limitsFor(
  armsState: Record<string, ArmState> | undefined, armId: string,
): Record<string, { min: number; max: number }> | undefined {
  const joints = armsState?.[armId]?.joints;
  if (!joints) return undefined;
  return Object.fromEntries(
    Object.entries(joints).map(([j, s]) => [j, { min: s.min, max: s.max }]),
  );
}
