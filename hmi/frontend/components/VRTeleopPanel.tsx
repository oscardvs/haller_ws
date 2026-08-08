"use client";

/**
 * Quest teleop. Owns the WebXR session, the ~30 Hz publish loop, and the
 * start/stop of the backend human-teleop session.
 *
 * It decides nothing about safety. Squeeze state and controller poses are
 * shipped unmodified; the backend applies the acquisition countdown, the
 * confidence floor, the staleness budget, every joint clamp and the bimanual
 * collision guard. Same contract the MediaPipe panel has — see
 * HumanTeleopPanel. The decisions this file DOES own are input-shaped, not
 * safety-shaped: B/Y on either controller fires POST /estop (one press, one
 * post); a half-second hold of A/X on either controller starts or stops the
 * dataset recorder (hold-gated so a thumb brush cannot toggle a take); and
 * while the session is not fully visible (Quest system menu open) every
 * published frame is forced disengaged — the page keeps receiving the last
 * input state in that menu, so a grip held when it opened would otherwise
 * stay held forever.
 *
 * The session asks for **passthrough AR first** (the operator must watch the
 * REAL arms, not a void) with the HUD div as a dom-overlay, falling back to
 * plain immersive-vr where AR is refused. The publish loop runs off
 * `XRSession.requestAnimationFrame`, NOT `window.requestAnimationFrame` —
 * only the XR callback is handed an `XRFrame`, which is the sole way to
 * resolve poses, and the window loop is throttled to near-nothing while an
 * immersive session holds the display.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

import {
  api, cameraStreamUrl, type CameraInfo, type HumanTeleopStatus,
  type RecordStatus,
} from "@/lib/api";
import { BACKEND_URL } from "@/lib/config";
import { HumanTeleopClient } from "@/lib/humanTeleopClient";
import { useRecorder } from "@/lib/recorder";
import {
  attachRenderScene, axPressed, CAM_TILE_SIZES, cycleIndex, disengagedFrame,
  estopPressed, hapticCues, holdToggle, holdToggleInit, paintHud, pulse,
  RECORD_HOLD_MS, requestTeleopSession, sampleVRFrame, thumbstickPressed,
  xrAvailableAtAll, xrSupported,
  type BodyOverride, type HoldToggleState, type SideAuthorityLike,
  type TeleopXRSession, type VRFrame, type VrMenuLike, type XRFrameLike,
  type XRSessionLike,
} from "@/lib/vrTeleop";
import { repoIdFor } from "./cockpit/CommandBar";
import { DeadManIndicator } from "./DeadManIndicator";

const WS_URL = `${BACKEND_URL.replace(/^http/, "ws")}/ws/teleop/vr/in`;

/** ~30 Hz, matching the MediaPipe panel. The Quest renders at 72–90 Hz, but the
 *  backend commit loop and the staleness budget are tuned around this rate, and
 *  publishing every display frame would trade real headroom for nothing. */
const PUBLISH_MS = 33;

const BODY_LS_KEY = "haller.vrTeleop.body.v1";
const MIRROR_LS_KEY = "haller.vrTeleop.mirror.v1";
const VRMODE_LS_KEY = "haller.vrTeleop.mode.v1";

const TILE_LS_KEY = "haller.vrTeleop.tile.v1";
const VIEW_LS_KEY = "haller.vrTeleop.view.v1";

function fmtMm(m: number | undefined): string {
  return m === undefined ? "—" : `${(m * 1000).toFixed(0)} mm`;
}

/** Menu label for a camera. Ids are backend-shaped (`threequarter_sim`); this
 *  is what an operator reads at arm's length through a headset, so it drops the
 *  `_sim` suffix and says which arm a wrist cam belongs to. */
export function viewLabel(c: CameraInfo): string {
  const base = c.id.replace(/_sim$/, "").replace(/_/g, " ");
  return c.arm_id ? `${base} (${c.arm_id})` : base;
}

export function VRTeleopPanel({ armIds }: { armIds: string[] }) {
  const [supported, setSupported] = useState<boolean | null>(null);
  const [inSession, setInSession] = useState(false);
  const [xrMode, setXrMode] = useState<TeleopXRSession["mode"] | null>(null);
  const [estopped, setEstopped] = useState(false);
  const [status, setStatus] = useState<HumanTeleopStatus | null>(null);
  const [body, setBody] = useState<BodyOverride>({});
  const [mirrorMode, setMirrorMode] = useState<"none" | "both">("none");
  const [vrMode, setVrMode] = useState<"pose" | "joints">("pose");
  // The workspace camera floated into the HUD. In passthrough the operator
  // normally watches the REAL arms — but against a sim backend there is
  // nothing physical to look at, so the MuJoCo camera IS the workspace view
  // and defaults on. On the real rig the same toggle shows the mast camera.
  const [baseCam, setBaseCam] = useState<CameraInfo | null>(null);
  const [showCam, setShowCam] = useState(false);
  // Recorder status, polled alongside teleop status so the HUD (in-scene or
  // dom-overlay) can show whether a take is rolling and how long it is.
  const [recStatus, setRecStatus] = useState<RecordStatus | null>(null);
  // Every camera the backend advertises, in menu order. Sim serves the MuJoCo
  // views; the real rig serves the mast and the egocentric gripper cams. The
  // menu is deliberately generic over whatever comes back.
  const [views, setViews] = useState<CameraInfo[]>([]);
  const [tileIdx, setTileIdx] = useState(0);

  const sessionRef = useRef<XRSessionLike | null>(null);
  const refSpaceRef = useRef<unknown>(null);
  const clientRef = useRef<HumanTeleopClient<VRFrame> | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const bodyRef = useRef<BodyOverride>({});
  const mirrorModeRef = useRef<"none" | "both">("none");
  const vrModeRef = useRef<"pose" | "joints">("pose");
  const lastPubRef = useRef(0);
  const estopDownRef = useRef(false);
  const estopInFlightRef = useRef(false);
  const prevAuthRef = useRef<Partial<Record<"left" | "right", SideAuthorityLike>>>({});
  const statusRef = useRef<HumanTeleopStatus | null>(null);
  const baseCamRef = useRef<CameraInfo | null>(null);
  const showCamRef = useRef(false);
  const camImgRef = useRef<HTMLImageElement | null>(null);
  const hudPaintRef = useRef(0);
  const recStatusRef = useRef<RecordStatus | null>(null);
  const axToggleRef = useRef<HoldToggleState>(holdToggleInit());
  const viewsRef = useRef<CameraInfo[]>([]);
  const tileIdxRef = useRef(0);
  const lStickDownRef = useRef(false);
  const rStickDownRef = useRef(false);
  const menuRef = useRef<VrMenuLike>(null);

  bodyRef.current = body;
  mirrorModeRef.current = mirrorMode;
  vrModeRef.current = vrMode;
  statusRef.current = status;
  baseCamRef.current = baseCam;
  showCamRef.current = showCam;
  recStatusRef.current = recStatus;
  viewsRef.current = views;
  tileIdxRef.current = tileIdx;
  menuRef.current = {
    views: views.map((c) => ({ id: c.id, label: viewLabel(c) })),
    activeViewId: baseCam?.id ?? null,
    tileSize: CAM_TILE_SIZES[tileIdx]?.name ?? "S",
  };

  useEffect(() => {
    void xrSupported().then(setSupported);
    api.cameras()
      .then(({ cameras }) => {
        // Base views first, then wrist: the menu should open on something you
        // can judge a whole pose from, with the close-ups after it.
        const active = cameras.filter((c) => c.active);
        const ordered = [...active.filter((c) => c.role === "base"),
                         ...active.filter((c) => c.role !== "base")];
        setViews(ordered);
        const base = ordered.filter((c) => c.role === "base");
        let remembered: CameraInfo | undefined;
        try {
          const id = localStorage.getItem(VIEW_LS_KEY);
          remembered = ordered.find((c) => c.id === id);
        } catch { /* private mode: fall through to the default pick */ }
        const cam = remembered
          ?? base.find((c) => c.source === "sim_camera") ?? base[0] ?? null;
        setBaseCam(cam);
        if (cam?.source === "sim_camera") setShowCam(true);
      })
      .catch(() => { /* no cameras is fine; the toggle just stays hidden */ });
    try {
      const raw = localStorage.getItem(BODY_LS_KEY);
      if (raw) setBody(JSON.parse(raw));
      const mm = localStorage.getItem(MIRROR_LS_KEY);
      if (mm === "both") setMirrorMode("both");
      if (localStorage.getItem(VRMODE_LS_KEY) === "joints") setVrMode("joints");
      const t = Number(localStorage.getItem(TILE_LS_KEY));
      if (Number.isInteger(t) && t >= 0 && t < CAM_TILE_SIZES.length) setTileIdx(t);
    } catch {
      /* a corrupt override must not block entering VR; defaults are fine */
    }
  }, []);

  // Status poll. Cheap, and it is what feeds the HUD — the only thing the
  // operator can see from inside the headset. Also the haptic driver: the
  // controller buzz on handover comes from status transitions seen here.
  useEffect(() => {
    let alive = true;
    const t = setInterval(() => {
      // Recorder status rides the same poll: the HUD's REC line is the only
      // way an operator inside the headset can see a take rolling.
      api.recordStatus()
        .then((rs) => { if (alive) setRecStatus(rs); })
        .catch(() => { /* recorder not ready; the badge just stays hidden */ });
      api.humanTeleopStatus()
        .then((s) => {
          if (!alive) return;
          setStatus(s);
          const session = sessionRef.current;
          if (session && s.acquire) {
            const next: Partial<Record<"left" | "right", SideAuthorityLike>> = {
              left: s.acquire.left.authority,
              right: s.acquire.right.authority,
            };
            for (const cue of hapticCues(prevAuthRef.current, next)) {
              pulse(session, cue.hand, cue.intensity, cue.durationMs);
            }
            // A guard actively holding a step back is worth feeling, not just
            // reading: a light buzz on every driving hand while it clamps.
            if (s.collision?.limited) {
              for (const hand of ["left", "right"] as const) {
                if (next[hand] === "driving") pulse(session, hand, 0.2, 60);
              }
            }
            prevAuthRef.current = next;
          } else {
            prevAuthRef.current = {};
          }
        })
        .catch(() => { /* transient; the next tick retries */ });
    }, 250);
    return () => { alive = false; clearInterval(t); };
  }, []);

  /** Next camera into the tile. Repoints the live MJPEG <img> in place — the
   *  in-scene HUD paints whatever that element currently holds, so swapping
   *  `src` is the whole switch. */
  const cycleView = useCallback(() => {
    const list = viewsRef.current;
    if (list.length < 2) return;
    const here = list.findIndex((c) => c.id === baseCamRef.current?.id);
    const next = list[cycleIndex(list.length, here < 0 ? -1 : here, 1)];
    if (!next) return;
    setBaseCam(next);
    baseCamRef.current = next;
    try { localStorage.setItem(VIEW_LS_KEY, next.id); } catch { /* fine */ }
    if (camImgRef.current) camImgRef.current.src = cameraStreamUrl(next.id);
    const s = sessionRef.current;
    if (s) pulse(s, "left", 0.15, 40);
  }, []);

  /** Next tile size. The quad rescales on the following rendered frame. */
  const cycleTile = useCallback(() => {
    const next = cycleIndex(CAM_TILE_SIZES.length, tileIdxRef.current, 1);
    setTileIdx(next);
    tileIdxRef.current = next;
    try { localStorage.setItem(TILE_LS_KEY, String(next)); } catch { /* fine */ }
    const s = sessionRef.current;
    if (s) pulse(s, "right", 0.15, 40);
  }, []);

  const teardown = useCallback(async (opts: { stopBackend: boolean }) => {
    const session = sessionRef.current;
    sessionRef.current = null;
    refSpaceRef.current = null;
    if (camImgRef.current) {
      // Detach the MJPEG stream or it keeps pulling frames forever.
      camImgRef.current.src = "";
      camImgRef.current = null;
    }
    const client = clientRef.current;
    clientRef.current = null;
    if (client) {
      // Parting shot: an explicit disengage releases both arms on THIS frame
      // instead of leaving the backend to age the last engaged frame past the
      // staleness budget.
      client.queueFrame(disengagedFrame(Date.now()));
      client.tick();
      client.close();
    }
    setInSession(false);
    setXrMode(null);
    if (session) {
      try { await session.end(); } catch { /* already ended */ }
    }
    if (opts.stopBackend) {
      // Stop even if the socket already dropped: the backend's WS grace window
      // would eventually catch this, but leaving a session ARMED with no
      // operator attached is exactly the state we do not want to rely on a
      // timeout to clear.
      try { await api.humanTeleopStop(); } catch { /* best effort */ }
    }
  }, []);

  const fireEstop = useCallback(async () => {
    if (estopInFlightRef.current) return;
    estopInFlightRef.current = true;
    const session = sessionRef.current;
    if (session) {
      pulse(session, "left", 1.0, 300);
      pulse(session, "right", 1.0, 300);
    }
    try {
      await api.estop();
      setEstopped(true);
      toast.warning("E-STOP: torque dropped on both arms");
    } catch (e) {
      toast.error(`E-STOP request failed: ${(e as Error).message}`);
    } finally {
      estopInFlightRef.current = false;
    }
    // Leave the headset session: after an E-STOP the operator deals with the
    // rig, and the 2D panel is where re-arming lives.
    await teardown({ stopBackend: false });
  }, [teardown]);

  const rearm = useCallback(async () => {
    try {
      for (const id of armIds) {
        await api.armMode(id, "manual");   // MANUAL also re-enables torque
      }
      setEstopped(false);
      toast.success("arms back in MANUAL with torque on");
    } catch (e) {
      toast.error(`re-arm failed: ${(e as Error).message}`);
    }
  }, [armIds]);

  /** Start or stop the dataset take. Invoked from inside the XR loop by a
   *  deliberate half-second A/X hold — see `holdToggle`. The task/HF-user
   *  draft comes from the cockpit's Dataset tab (persisted), so a solo
   *  operator never has to leave the headset between takes. */
  const toggleRecording = useCallback(async () => {
    const rec = useRecorder.getState();
    const session = sessionRef.current;
    try {
      if (rec.status?.recording) {
        const s = await rec.stop(true);   // always save; discard lives in the cockpit
        if (session) {
          pulse(session, "left", 0.5, 150);
          pulse(session, "right", 0.5, 150);
        }
        toast.success(`take saved — ${s.episode_frames} frames`);
      } else {
        const task = rec.task.trim();
        if (!task) {
          toast.error("no task drafted — set one in the cockpit's Dataset tab first");
          return;
        }
        const repoId = repoIdFor(rec.hfUser, task);
        await rec.start(repoId, task);
        if (session) {
          pulse(session, "left", 0.7, 200);
          pulse(session, "right", 0.7, 200);
        }
        toast.success(`recording → ${repoId}`);
      }
      await rec.refresh();
    } catch (e) {
      toast.error(`record toggle failed: ${(e as Error).message}`);
    }
  }, []);

  const enterVR = useCallback(async () => {
    if (armIds.length < 2) {
      // Checked here rather than letting the backend 400, because the reason is
      // not obvious from "left_arm and right_arm must be different". The session
      // commit loop drives both sides unconditionally, so it needs two distinct
      // arm handles; single-arm teleop is a real feature, not a config tweak.
      toast.error(
        "VR teleop needs 2 enabled arms — the session drives both sides",
      );
      return;
    }
    let xrs: TeleopXRSession;
    try {
      xrs = await requestTeleopSession(overlayRef.current);
    } catch (e) {
      toast.error(`could not start XR: ${(e as Error).message}`);
      return;
    }
    const session = xrs.session;
    sessionRef.current = session;
    setXrMode(xrs.mode);

    try {
      refSpaceRef.current = await session.requestReferenceSpace("local-floor");
    } catch {
      toast.error("headset would not grant a local-floor reference space");
      await teardown({ stopBackend: false });
      return;
    }

    // Backend session first, socket second. If start() is going to be refused
    // — another teleop already running, an unknown arm — we want to find out
    // before frames are in flight, not while the operator is already immersed
    // and squeezing a grip that does nothing.
    try {
      await api.humanTeleopStart({
        left_arm: armIds[0],
        right_arm: armIds[1],
        swap: false,
        clutch_source: "vr_grip",
      });
    } catch (e) {
      toast.error(`teleop start refused: ${(e as Error).message}`);
      await teardown({ stopBackend: false });
      return;
    }

    const client = new HumanTeleopClient<VRFrame>(WS_URL);
    client.connect();
    clientRef.current = client;
    setInSession(true);
    setEstopped(false);

    const onEnd = () => { void teardown({ stopBackend: true }); };
    session.addEventListener("end", onEnd);

    const scene = attachRenderScene(session, xrs.mode);
    estopDownRef.current = false;

    // The Meta Quest Browser refuses `dom-overlay` on-device (it only works
    // in Meta's desktop emulator), and the feature is optional by design —
    // so when the session comes back without it, the HUD moves INTO the
    // scene: a world-locked quad repainted from a canvas at ~10 Hz.
    const overlayActive = Boolean(
      (session as unknown as { domOverlayState?: unknown }).domOverlayState);
    let hudCanvas: HTMLCanvasElement | null = null;
    let hudCtx: CanvasRenderingContext2D | null = null;
    if (!overlayActive) {
      hudCanvas = document.createElement("canvas");
      hudCanvas.width = 1024;
      hudCanvas.height = 768;
      hudCtx = hudCanvas.getContext("2d");
      toast.info("browser has no dom-overlay — HUD is drawn inside the scene");
    }
    if (showCamRef.current && baseCamRef.current && !overlayActive) {
      const img = new Image();
      img.src = cameraStreamUrl(baseCamRef.current.id);
      camImgRef.current = img;
    }

    const onXRFrame = (t: number, frame: XRFrameLike) => {
      const live = sessionRef.current;
      if (!live) return;
      live.requestAnimationFrame(onXRFrame);
      // View menu, on the two inputs nothing else claims. Edge-detected: a
      // stick click is a discrete choice, and unlike the record toggle it is
      // harmless enough to fire on press rather than on a hold.
      const lStick = thumbstickPressed(live, "left");
      if (lStick && !lStickDownRef.current) cycleView();
      lStickDownRef.current = lStick;
      const rStick = thumbstickPressed(live, "right");
      if (rStick && !rStickDownRef.current) cycleTile();
      rStickDownRef.current = rStick;

      let hudDirty = false;
      if (hudCanvas && hudCtx && t - hudPaintRef.current > 100) {
        hudPaintRef.current = t;
        paintHud(hudCtx, statusRef.current, camImgRef.current,
                 recStatusRef.current, menuRef.current);
        hudDirty = true;
      }
      scene.render(frame, refSpaceRef.current, hudCanvas, hudDirty,
                   CAM_TILE_SIZES[tileIdxRef.current]?.widthM);

      // E-STOP scan runs at display rate, not publish rate: 33 ms of extra
      // latency on a stop button is 33 ms too many. Edge-detected so one
      // press is one POST.
      const down = estopPressed(live);
      if (down && !estopDownRef.current) void fireEstop();
      estopDownRef.current = down;

      // Record toggle: same display-rate scan, but hold-gated — a brush of
      // the thumb near A/X must not start or end a take.
      const ax = holdToggle(axToggleRef.current, axPressed(live), t, RECORD_HOLD_MS);
      axToggleRef.current = ax;
      if (ax.toggled) void toggleRecording();

      if (t - lastPubRef.current < PUBLISH_MS) return;
      lastPubRef.current = t;
      const blurred =
        live.visibilityState !== undefined && live.visibilityState !== "visible";
      const vrFrame = sampleVRFrame(live, frame, refSpaceRef.current, {
        tsMs: Date.now(),
        body: Object.keys(bodyRef.current).length ? bodyRef.current : undefined,
        forceDisengaged: blurred,
        mirrorMode: mirrorModeRef.current,
        vrMode: vrModeRef.current,
      });
      client.queueFrame(vrFrame);
      client.tick();
    };
    session.requestAnimationFrame(onXRFrame);
  }, [armIds, teardown, fireEstop, toggleRecording, cycleView, cycleTile]);

  // Unmount must release the arms. Unlike the MediaPipe panel this one cannot
  // be left mounted-but-hidden: an immersive session already owns the display,
  // so there is no "operator looked at another tab" case to preserve.
  useEffect(() => () => { void teardown({ stopBackend: true }); }, [teardown]);

  const clutch = status?.clutch;
  const running = Boolean(status?.running);
  const acquire = status?.acquire;
  const collision = status?.collision;

  const sideRow = (side: "left" | "right") => {
    const a = acquire?.[side];
    if (!a) return null;
    const grip = clutch?.sides?.[side] ?? clutch?.engaged ?? false;
    return (
      <div key={side} className="flex items-center gap-2">
        <span className="w-10 uppercase">{side}</span>
        <span className={grip ? "text-lime-400" : "text-neutral-400"}>
          {grip ? "grip ●" : "grip ○"}
        </span>
        <span
          className={
            a.authority === "driving" ? "text-lime-400 font-bold"
            : a.authority === "acquiring" ? "text-amber-400"
            : "text-neutral-400"
          }
        >
          {a.authority}
          {a.authority === "acquiring" && a.remaining_ms !== null
            ? ` ${(a.remaining_ms / 1000).toFixed(1)}s`
            : ""}
        </span>
        {a.authority === "acquiring" && a.blocking.length > 0 && (
          <span className="text-amber-300 truncate">
            match: {a.blocking.map((j) => {
              const e = a.error_deg?.[j];
              return e === undefined ? j
                : `${j} ${e > 0 ? "+" : ""}${e.toFixed(0)}°`;
            }).join(", ")}
          </span>
        )}
        {a.reason === "no_tracking" && (
          <span className="text-red-400">no tracking</span>
        )}
      </div>
    );
  };

  // The HUD div doubles as the dom-overlay root. In-session it floats over
  // the passthrough view; out of session the identical markup sits in the
  // page as the live status card. One element, one code path.
  const hud = (
    <div
      ref={overlayRef}
      className={
        inSession
          ? "fixed inset-x-0 bottom-0 z-50 flex flex-col items-center gap-2 p-6 pointer-events-none"
          : "rounded border p-2 space-y-1"
      }
    >
      {inSession && showCam && baseCam && (
        // MJPEG rides a plain <img>; the dom-overlay composites it into the
        // headset view. pointer-events-none so a stray controller ray can
        // never "click" the video instead of the E-STOP below it.
        <img
          src={cameraStreamUrl(baseCam.id)}
          alt={`${baseCam.id} live view`}
          className="rounded-lg border border-white/25 pointer-events-none"
          style={{ width: "min(85vw, 560px)" }}
        />
      )}
      <div
        className={
          "font-mono text-[13px] space-y-1 " +
          (inSession
            ? "rounded-lg bg-black/75 text-white px-4 py-3 min-w-[340px]"
            : "")
        }
      >
        <div className="flex items-center gap-3">
          <span className="uppercase font-bold">{status?.state ?? "—"}</span>
          {collision?.enabled ? (
            <span
              className={
                collision.limited
                  ? "text-red-400 font-bold"
                  : (collision.slack_m ?? 1) < 0
                    ? "text-amber-400"
                    : "text-neutral-300"
              }
            >
              {collision.limited ? "◉ COLLISION HOLD · " : "clearance "}
              {fmtMm(collision.slack_m)}
            </span>
          ) : (
            <span className="text-amber-400">no collision guard</span>
          )}
          {recStatus?.recording && (
            <span className="text-red-400 font-bold animate-haller-rec">
              ● REC {recStatus.episode_frames}
            </span>
          )}
        </div>
        {sideRow("left")}
        {sideRow("right")}
        {status?.last_error && (
          <div className="text-red-400 truncate">{status.last_error}</div>
        )}
        <div className="text-neutral-400">
          grip = drive · trigger = gripper · <b>B / Y = E-STOP</b> ·{" "}
          <b>A / X hold = record</b>
        </div>
      </div>
      {inSession && (
        <div className="flex gap-3 pointer-events-auto">
          <Button
            variant="destructive"
            size="lg"
            className="font-bold"
            onClick={() => void fireEstop()}
          >
            E-STOP
          </Button>
          {baseCam && (
            <Button variant="secondary" onClick={() => setShowCam((v) => !v)}>
              {showCam ? "Cam off" : "Cam on"}
            </Button>
          )}
          <Button
            variant="secondary"
            onClick={() => void teardown({ stopBackend: true })}
          >
            Exit
          </Button>
        </div>
      )}
    </div>
  );

  return (
    <div className="space-y-3 font-mono text-[12px]">
      <div className="flex items-center gap-3">
        {!inSession ? (
          <Button onClick={() => void enterVR()} disabled={supported !== true}>
            Enter Passthrough
          </Button>
        ) : (
          <Button variant="destructive" onClick={() => void teardown({ stopBackend: true })}>
            Exit VR
          </Button>
        )}
        <Button variant="outline" onClick={() => void rearm()}>
          Re-arm arms
        </Button>
        <DeadManIndicator
          held={Boolean(clutch?.engaged)}
          trackingLost={Boolean(status?.tracking?.left?.lost && status?.tracking?.right?.lost)}
          source="vr_grip"
          reason={clutch?.reason}
        />
        <span className="text-muted-foreground">
          state: {status?.state ?? "—"}{running ? "" : " (stopped)"}
          {xrMode === "immersive-vr" ? " · VR fallback (no passthrough!)" : ""}
        </span>
      </div>

      {estopped && (
        <div className="rounded border border-destructive/60 p-2 text-destructive">
          E-STOPPED — both arms are torque-off in STOP mode. Check the bench,
          then <b>Re-arm arms</b> to restore MANUAL + torque.
        </div>
      )}

      {supported === false && (
        <div className="rounded border border-destructive/40 p-2 text-destructive">
          {xrAvailableAtAll()
            ? "This browser has WebXR but refused an immersive session."
            : (
              <>
                <div>WebXR is not available on this page.</div>
                <div className="mt-1 text-muted-foreground">
                  `navigator.xr` is only exposed in a secure context. Open this
                  page over <b>HTTPS</b> (or via localhost) from inside the Quest
                  browser — a plain <code>http://</code> LAN address will always
                  report unsupported, with no permission prompt to explain why.
                </div>
              </>
            )}
        </div>
      )}

      {hud}

      <div className="text-muted-foreground">
        <div>
          Hold a <b>grip</b> to drive that side&apos;s arm — each grip is its own
          dead-man. Release to freeze that arm where it is.
        </div>
        <div><b>Trigger</b> is the gripper — analog, 0 open to 1 closed.</div>
        <div>
          <b>B or Y</b> (either controller) is the E-STOP: torque drops on both
          arms instantly.
        </div>
        <div>
          <b>Hold A or X</b> (either controller, ~0.5 s) starts or stops the
          dataset take. Draft the task in the cockpit&apos;s Dataset tab first —
          the recorder shows <span className="font-bold">● REC</span> in the HUD
          while it rolls.
        </div>
        {vrMode === "pose" ? (
          <div>
            Squeezing a grip <b>anchors your hand to the arm where it is</b> —
            hold still through the countdown, feel the buzz, then your hand&apos;s
            movement drives the gripper. Release, reposition your hand, squeeze
            again to ratchet across the workspace.
          </div>
        ) : (
          <div>
            Engagement runs the acquisition countdown: match the robot&apos;s pose
            (the HUD lists what&apos;s off), then authority transfers. The buzz on
            your controller is the handover.
          </div>
        )}
      </div>

      <label className="flex items-center gap-2 text-muted-foreground">
        <span>hand mapping</span>
        <select
          className="bg-transparent border rounded px-1 py-0.5"
          value={vrMode}
          onChange={(e) => {
            const v = e.target.value === "joints" ? "joints" : "pose";
            setVrMode(v);
            try { localStorage.setItem(VRMODE_LS_KEY, v); } catch { /* non-fatal */ }
          }}
        >
          <option value="pose">hand position (natural, default)</option>
          <option value="joints">body angles (legacy)</option>
        </select>
        <span>
          — position mode tracks your hand through IK; no pose matching, no
          limb-length calibration.
        </span>
      </label>

      <label className="flex items-center gap-2 text-muted-foreground">
        <span>arm mounting</span>
        <select
          className="bg-transparent border rounded px-1 py-0.5"
          value={mirrorMode}
          onChange={(e) => {
            const v = e.target.value === "both" ? "both" : "none";
            setMirrorMode(v);
            try { localStorage.setItem(MIRROR_LS_KEY, v); } catch { /* non-fatal */ }
          }}
        >
          <option value="none">identical, side by side (Haller tower)</option>
          <option value="both">mirrored pair</option>
        </select>
        <span>
          — if an arm drives <b>away</b> from where your hand goes, flip this.
        </span>
      </label>

      <details className="text-muted-foreground">
        <summary className="cursor-pointer">operator limb lengths (metres)</summary>
        <div className="mt-2 grid grid-cols-2 gap-2">
          {(["upper_arm", "fore_arm", "shoulder_drop", "shoulder_half_width"] as const).map((k) => (
            <label key={k} className="flex items-center gap-2">
              <span className="w-40">{k}</span>
              <input
                type="number" step="0.01" className="w-20 bg-transparent border rounded px-1"
                value={body[k] ?? ""}
                placeholder="default"
                onChange={(e) => {
                  const v = e.target.value === "" ? undefined : Number(e.target.value);
                  const next = { ...body };
                  if (v === undefined || Number.isNaN(v)) delete next[k];
                  else next[k] = v;
                  setBody(next);
                  try { localStorage.setItem(BODY_LS_KEY, JSON.stringify(next)); } catch { /* non-fatal */ }
                }}
              />
            </label>
          ))}
        </div>
        <p className="mt-2">
          These are not cosmetic. The elbow is synthesized from these lengths and
          the measured shoulder-to-controller distance, so a model longer than
          your real arm means you run out of reach before the robot&apos;s elbow
          ever straightens.
        </p>
      </details>
    </div>
  );
}
