"use client";

/**
 * Top-level orchestrator for the human-teleop page. Owns:
 *   - The camera stream and the MediaPipeRunner lifecycle.
 *   - The render loop that runs detection + WS publish at ~30 Hz.
 *   - The dead-man key state.
 *   - Which source holds clutch authority (spacebar or mouth).
 *   - The pinch and mouth calibration state (persisted in localStorage).
 *   - Start/stop/swap session calls.
 *
 * It decides nothing about safety. `dead_man` is the raw spacebar key state
 * and `jaw_open` is the raw blendshape score; the backend applies both the
 * threshold and the hold timer.
 *
 * In the cockpit this panel stays MOUNTED (hidden) while a session is running
 * and the operator looks at another tab, because unmounting it would stop the
 * ~30 Hz publish loop that *is* the teleop input while the backend still
 * believed a session was live. The `active` prop is how it knows it is not the
 * visible tab: the dead-man key is bound only while active, and releases the
 * moment it stops being — the robot must not be drivable from a screen the
 * operator is not looking at.
 */
import { useEffect, useRef, useState, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

import { api, type HumanTeleopStatus, type JointDiag, type JointReason } from "@/lib/api";
import { BACKEND_URL } from "@/lib/config";
import { isEditableTarget } from "@/lib/keys";
import { useTelemetry, type ArmState } from "@/lib/telemetry";
import {
  MediaPipeRunner, fuseLandmarkResults, buildOverlaySides, extractJawOpen, FACE_EVERY_N,
  type KeypointFrame, type SideFrame,
} from "@/lib/mediapipe";
import { HumanTeleopClient } from "@/lib/humanTeleopClient";

import { CameraOverlay, type CameraOverlayHandle } from "./CameraOverlay";
import { ScopeBar } from "./ScopeBar";
import { DeadManIndicator, type ClutchSource } from "./DeadManIndicator";
import { PinchCalibrationStep, type PinchCalib } from "./PinchCalibrationStep";
import {
  MouthClutchCalibration, mouthCalibReady, MOUTH_MIN_SEPARATION, type MouthCalib,
} from "./MouthClutchCalibration";

const WS_URL = `${BACKEND_URL.replace(/^http/, "ws")}/ws/teleop/human/in`;

const JOINTS = [
  "shoulder_pan", "shoulder_lift", "elbow_flex",
  "wrist_flex", "wrist_roll", "gripper",
] as const;

const CALIB_LS_KEY = "haller.humanTeleop.pinchCalib.v1";
const MOUTH_LS_KEY = "haller.humanTeleop.mouthCalib.v1";

export function HumanTeleopPanel({
  armIds,
  active = true,
  fill = false,
  simTile,
}: {
  armIds: string[];
  /** False when this panel is mounted but not the visible tab. */
  active?: boolean;
  /** Fill a fixed-height container (the cockpit) rather than sizing to content. */
  fill?: boolean;
  /** Sim picture-in-picture, supplied by the caller so the panel does not have
   *  to know whether it is pinned over a page or in a column of a grid. */
  simTile?: ReactNode;
}) {
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

  const [clutchSource, setClutchSource] = useState<ClutchSource>("spacebar");
  const [mouthCalib, setMouthCalib] = useState<MouthCalib>(
    () => (typeof window === "undefined"
      ? defaultMouthCalib()
      : parseMouthCalib(localStorage.getItem(MOUTH_LS_KEY)) ?? defaultMouthCalib()),
  );
  const [liveJaw, setLiveJaw] = useState<number | null>(null);
  const faceTickRef = useRef(0);
  // Latches true after the face model has failed to load once, so the
  // "unavailable" toast fires exactly once instead of on every face tick
  // (~every 100 ms) for the rest of the session.
  const faceLoadWarnedRef = useRef(false);

  // Persist calib on change.
  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem(CALIB_LS_KEY, JSON.stringify(calib));
    }
  }, [calib]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem(MOUTH_LS_KEY, JSON.stringify(mouthCalib));
    }
  }, [mouthCalib]);

  // Bind dead-man key — only while this panel is the visible surface.
  //
  // The cleanup releases the dead-man rather than merely unbinding. Without
  // that, switching tabs mid-drive would leave `deadManRef` latched true and
  // the publish loop would keep sending an engaged clutch for a screen nobody
  // is watching. Releasing is the fail-safe direction: the operator stops
  // driving, exactly as if they had let go of the key.
  useEffect(() => {
    if (!active) {
      deadManRef.current = false;
      return;
    }
    const onDown = (e: KeyboardEvent) => {
      if (e.code === "Space" && !isEditableTarget(e.target)) {
        // Stops the page scrolling and stops a focused button re-firing.
        e.preventDefault();
        deadManRef.current = true;
      }
    };
    // Not gated on the target, and not on `active` either by the time it runs:
    // a key that went down must always be able to come up.
    const onUp = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        deadManRef.current = false;
      }
    };
    // Alt-tabbing away never delivers keyup.
    const onBlur = () => {
      deadManRef.current = false;
    };
    window.addEventListener("keydown", onDown);
    window.addEventListener("keyup", onUp);
    window.addEventListener("blur", onBlur);
    return () => {
      deadManRef.current = false;
      window.removeEventListener("keydown", onDown);
      window.removeEventListener("keyup", onUp);
      window.removeEventListener("blur", onBlur);
    };
  }, [active]);

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
        // Connect the keypoint WS independently of — and before — model
        // loading below. It used to run only after `runner.load()` resolved,
        // so a rejection there (the face model failing to load: see
        // MediaPipeRunner.load()'s doc comment) silently killed the whole
        // frame pipeline, including the pre-existing hand/pose spacebar path
        // that had already succeeded. The rAF tick loop bails while
        // `!client`/`!runner`, so connecting early is safe — nothing is
        // queued until both exist.
        clientRef.current = new HumanTeleopClient(WS_URL);
        clientRef.current.connect();
        // Publish the runner only once load() has RESOLVED.
        //
        // Assigning runnerRef before awaiting meant a failed load — offline,
        // a blocked model CDN, no GPU delegate — left a runner whose detect()
        // throws on every one of the ~30 frames a second the loop asks for,
        // forever, behind a toast that fires once. The tick loop already has
        // the right behaviour for "no runner" (it idles), so the fix is to let
        // it see none. This matters more now that the panel can stay mounted
        // while hidden.
        const runner = new MediaPipeRunner();
        await runner.load();
        if (cancelled) {
          runner.close();
          return;
        }
        runnerRef.current = runner;
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
      // Decimate the face model: one tracking tick in FACE_EVERY_N runs it,
      // and only when the mouth actually holds authority. In spacebar mode the
      // backend discards jaw_open outright, so running FaceLandmarker there is
      // a third model competing with Hand and Pose for the same GPU inside the
      // same 33 ms budget, for no consumer — and tracking-loss is judged on a
      // 300 ms budget, so that headroom is not free.
      //
      // The counter advances unconditionally so a mid-session switch into
      // mouth mode picks up cleanly at whatever phase it is on. It also sits
      // below the 30 Hz cap on purpose: it counts *processed* ticks, so a
      // browser painting at 120 Hz still samples the jaw every ~100 ms rather
      // than every ~25 ms.
      const runFace = clutchSource === "mouth" && faceTickRef.current === 0;
      faceTickRef.current = (faceTickRef.current + 1) % FACE_EVERY_N;
      if (runFace) {
        // Lazy, on-demand load: the face model is only ever requested once
        // mouth mode is actually selected, mirroring the decimation gate
        // above — a spacebar session never asks MediaPipeRunner for it, at
        // startup or ever. loadFace() is idempotent (safe to call on every
        // face tick) and non-fatal: a rejection leaves `runner.detect()`
        // returning `face: null` forever, which the backend reads as
        // staleness and fails the mouth clutch closed rather than crashing
        // tracking. Fire-and-forget — the toast is a side effect, not state,
        // so there is nothing here for this render to wait on.
        void runner.loadFace().then((ok) => {
          if (!ok && !faceLoadWarnedRef.current) {
            faceLoadWarnedRef.current = true;
            toast.error("mouth clutch unavailable: face model failed to load");
          }
        });
      }
      const { hands, pose, face } = runner.detect(video, t, { face: runFace });
      // Only carries a value on ticks that actually ran the model. Skipped
      // ticks send null and the backend keeps using its last sample until the
      // staleness budget expires — which is why that budget (250 ms) has to
      // sit above the ~100 ms decimation gap.
      const jaw = runFace ? extractJawOpen(face) : null;
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

      // Same identity bail-out as liveDistance/liveConf above, and the same
      // rule: do NOT add `liveJaw` to the effect's dep array, or the loop is
      // torn down and rebuilt on every face tick. Only face ticks publish —
      // the null of a skipped tick is for the backend, not for the readout,
      // which would otherwise flicker to "—" twice per real sample.
      if (runFace) setLiveJaw((prev) => (prev === jaw ? prev : jaw));

      const frame: KeypointFrame = {
        type: "keypoints",
        ts_ms: Math.floor(performance.now()),
        clutch_source: clutchSource,
        // Stays the RAW spacebar key state whichever source has authority.
        dead_man: deadManRef.current,
        jaw_open: jaw,
        mouth_calib: mouthCalibReady(mouthCalib)
          ? { talk_max: mouthCalib.talk_max!, open_min: mouthCalib.open_min! }
          : undefined,
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
    // `clutchSource` and `mouthCalib` belong here for the same reason `calib`
    // does: the frame literal closes over them, and they change on an operator
    // action — not per frame. Rebuilding the loop on a source toggle is fine;
    // rebuilding it on a jaw sample would not be.
  }, [calib, clutchSource, mouthCalib]);

  const running = status?.running ?? false;
  const state = status?.state ?? "idle";

  const handleStart = async () => {
    try {
      // Push calibration first if both sides are complete.
      const cl = calib.left.min_m !== null && calib.left.max_m !== null
        ? { min_m: calib.left.min_m, max_m: calib.left.max_m } : undefined;
      const cr = calib.right.min_m !== null && calib.right.max_m !== null
        ? { min_m: calib.right.min_m, max_m: calib.right.max_m } : undefined;
      // The backend refuses a mouth-mode start without an adequate separation
      // (400). Check here too so the operator gets the reason, not an HTTP code.
      if (clutchSource === "mouth" && !mouthCalibReady(mouthCalib)) {
        toast.error(
          `mouth clutch needs talk + open captures at least ${MOUTH_MIN_SEPARATION} apart`,
        );
        return;
      }
      const mouth = mouthCalibReady(mouthCalib)
        ? { talk_max: mouthCalib.talk_max!, open_min: mouthCalib.open_min! }
        : undefined;
      if (cl || cr || mouth) {
        await api.humanTeleopCalibrate({ left: cl, right: cr, mouth });
      }
      await api.humanTeleopStart({
        left_arm: leftArm, right_arm: rightArm, swap, clutch_source: clutchSource,
      });
      toast.success(`human teleop started`);
    } catch (e) {
      toast.error(`start failed: ${(e as Error).message}`);
    }
  };

  const trackingLost =
    !!status?.tracking?.left?.lost || !!status?.tracking?.right?.lost;
  const age = (ms: number | null | undefined) =>
    typeof ms === "number" ? `${Math.round(ms)} ms` : "—";

  return (
    <div
      className={
        "grid gap-2 " +
        (fill
          ? "min-h-0 grid-cols-[minmax(0,1fr)_300px] overflow-hidden"
          : "grid-cols-1 lg:grid-cols-[1fr_320px]")
      }
    >
      <div
        className={
          "grid gap-2 " +
          (fill ? "min-h-0 grid-rows-[minmax(120px,1fr)_auto_auto]" : "grid-rows-none")
        }
      >
        <div className="relative min-h-0 overflow-hidden rounded-lg bg-[var(--haller-inset)] shadow-[0_0_0_1px_var(--border)]">
          <CameraOverlay ref={overlayRef} aspectRatio="16/9" fill={fill} />
          {/* Per-side tracking age, over the picture where the operator is
              already looking. `lost` outranks the number — a stale side is not
              driving its arm, whatever its last age said. */}
          <div className="pointer-events-none absolute top-2 left-2.5 flex items-center gap-1.5">
            {(["left", "right"] as const).map((side) => (
              <span
                key={side}
                className="inline-flex h-5.5 items-center gap-1.5 rounded-sm border border-border bg-card px-2 font-mono text-[10px]"
              >
                <span
                  aria-hidden
                  className="h-1.5 w-1.5 rounded-full"
                  style={{
                    backgroundColor: status?.tracking?.[side]?.lost
                      ? "var(--haller-warn)"
                      : "var(--haller-live)",
                  }}
                />
                {side} {age(status?.tracking?.[side]?.age_ms)}
              </span>
            ))}
          </div>
          <span className="pointer-events-none absolute right-2.5 bottom-2 font-mono text-[10px] text-muted-foreground">
            frame age {running ? age(status?.frame_age_ms) : "—"}
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2 rounded-lg bg-card px-2.5 py-2 shadow-[0_0_0_1px_var(--border)]">
          <DeadManIndicator
            held={state === "driving"}
            trackingLost={trackingLost}
            source={clutchSource}
            reason={status?.clutch?.reason}
          />
          <span
            className="inline-flex h-6.5 items-center rounded-full px-2.5 label-micro tracking-[0.14em]"
            style={{
              background:
                state === "driving" ? "var(--haller-live-soft)" : "var(--secondary)",
              color:
                state === "driving" ? "var(--haller-live)" : "var(--muted-foreground)",
            }}
          >
            {state}
          </span>
          <span aria-hidden className="h-px flex-1 bg-border" />
          <span className="label-micro text-muted-foreground">assign</span>
          <NativeSelect ariaLabel="left arm" value={leftArm} onChange={setLeftArm}
                        options={armIds.map((id) => ({ value: id, label: id }))} />
          <Button size="sm" variant="outline" className="h-6.5" title="swap arms"
                  onClick={() => { const t = leftArm; setLeftArm(rightArm); setRightArm(t); }}>
            ⇄
          </Button>
          <NativeSelect ariaLabel="right arm" value={rightArm} onChange={setRightArm}
                        options={armIds.filter((id) => id !== leftArm).map((id) => ({ value: id, label: id }))} />
          <Button size="sm" variant="outline" className="h-6.5 label-micro"
                  onClick={() => { setSwap(!swap); api.humanTeleopSwap(!swap).catch(() => null); }}>
            mirror · {swap ? "off" : "on"}
          </Button>
          {/* Authority never changes mid-session: the backend forces one
              disengage on a source switch, and an operator who is mid-reach
              should not be able to trigger that from the page. */}
          <Button size="sm" variant="outline" className="h-6.5 label-micro"
                  disabled={running}
                  title={running
                    ? "authority cannot change mid-session"
                    : "switch clutch authority"}
                  onClick={() => setClutchSource(clutchSource === "mouth" ? "spacebar" : "mouth")}>
            clutch · {clutchSource}
          </Button>
          {running ? (
            <Button size="sm" variant="destructive" className="h-6.5 px-3.5 label-micro"
                    onClick={() => api.humanTeleopStop().catch(() => null)}>
              stop
            </Button>
          ) : (
            <Button size="sm" className="h-6.5 px-3.5 label-micro" onClick={handleStart}
                    disabled={!leftArm || !rightArm || leftArm === rightArm}>
              start
            </Button>
          )}
        </div>

        <div className="flex items-stretch gap-2">
          {simTile ? <div className="w-[232px] shrink-0">{simTile}</div> : null}
          <div
            className={
              "grid min-w-0 flex-1 gap-2 " +
              (clutchSource === "mouth" ? "grid-cols-3" : "grid-cols-2")
            }
          >
            <PinchCalibrationStep
              side="left" liveDistance={liveDistance.left} confidence={liveConf.left} value={calib.left}
              onChange={(next) => setCalib({ ...calib, left: next })}
            />
            <PinchCalibrationStep
              side="right" liveDistance={liveDistance.right} confidence={liveConf.right} value={calib.right}
              onChange={(next) => setCalib({ ...calib, right: next })}
            />
            {/* The mouth card is a third column of this row rather than a
                wrapped third child. As a wrapped child it landed bottom-left,
                which is exactly where the PINNED sim tile sits on the deep-link
                page (fixed bottom-16 left-4 z-40) — covering the capture
                buttons and the "too close" warning. Here the sim tile is a
                sibling to the left when it is inline, and still pinned
                elsewhere when it is not, so neither can reach this card. */}
            {clutchSource === "mouth" ? (
              <MouthClutchCalibration
                liveJawOpen={liveJaw}
                value={mouthCalib}
                onChange={setMouthCalib}
              />
            ) : null}
          </div>
        </div>
      </div>

      <div className={"flex flex-col gap-2 " + (fill ? "min-h-0 overflow-y-auto" : "")}>
        <ArmScopePanel label={leftArm} goal={status?.goal_deg?.left}
                       limits={limitsFor(armsState, leftArm)}
                       diag={status?.joints?.left} />
        <ArmScopePanel label={rightArm} goal={status?.goal_deg?.right}
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
    <div className="flex flex-1 flex-col gap-1.5 rounded-lg bg-card p-2.5 shadow-[0_0_0_1px_var(--border)]">
      <div className="flex items-center gap-2">
        <span className="label-micro text-muted-foreground">scope</span>
        <span className="font-mono text-[11px] font-semibold tracking-[0.1em] uppercase">
          {label}
        </span>
        <span aria-hidden className="h-px flex-1 bg-border" />
        <span className="font-mono text-[9px] text-muted-foreground">
          asked → committed
        </span>
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
              <span className="w-14 text-right font-mono text-[10px] text-[var(--haller-warn)]">
                {badge ?? ""}
              </span>
            </div>
          );
        })}
      </div>
    </div>
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
      className="h-6.5 rounded-sm border border-input bg-background px-2 font-mono text-[10px]"
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

function defaultMouthCalib(): MouthCalib {
  return { talk_max: null, open_min: null };
}

/** Restore a persisted mouth calibration, or null if there isn't a usable one.
 *  A bare `JSON.parse(raw) as MouthCalib` would hand MouthClutchCalibration
 *  whatever shape happened to be under the key and blow up on
 *  `value.talk_max.toFixed(2)` during render — a page that will not load, from
 *  data the page itself never validated. Checking is cheap; the key is also
 *  versioned, so a future shape change gets a new one instead of meeting this. */
function parseMouthCalib(raw: string | null): MouthCalib | null {
  if (!raw) return null;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) return null;
    const { talk_max, open_min } = parsed as Record<string, unknown>;
    const ok = (v: unknown) => v === null || (typeof v === "number" && Number.isFinite(v));
    if (!ok(talk_max) || !ok(open_min)) return null;
    return { talk_max: talk_max as number | null, open_min: open_min as number | null };
  } catch {
    return null;
  }
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
