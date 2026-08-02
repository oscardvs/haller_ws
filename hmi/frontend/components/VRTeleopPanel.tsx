"use client";

/**
 * Quest teleop. Owns the WebXR session, the ~30 Hz publish loop, and the
 * start/stop of the backend human-teleop session.
 *
 * It decides nothing about safety. Squeeze state and controller poses are
 * shipped unmodified; the backend applies the acquisition countdown, the
 * confidence floor, the staleness budget, every joint clamp and the bimanual
 * collision guard. Same contract the MediaPipe panel has — see
 * HumanTeleopPanel. The two decisions this file DOES own are input-shaped,
 * not safety-shaped: B/Y on either controller fires POST /estop (one press,
 * one post), and while the session is not fully visible (Quest system menu
 * open) every published frame is forced disengaged — the page keeps receiving
 * the last input state in that menu, so a grip held when it opened would
 * otherwise stay held forever.
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
} from "@/lib/api";
import { BACKEND_URL } from "@/lib/config";
import { HumanTeleopClient } from "@/lib/humanTeleopClient";
import {
  attachRenderLayer, disengagedFrame, estopPressed, hapticCues, pulse,
  requestTeleopSession, sampleVRFrame, xrAvailableAtAll, xrSupported,
  type BodyOverride, type SideAuthorityLike, type TeleopXRSession,
  type VRFrame, type XRFrameLike, type XRSessionLike,
} from "@/lib/vrTeleop";
import { DeadManIndicator } from "./DeadManIndicator";

const WS_URL = `${BACKEND_URL.replace(/^http/, "ws")}/ws/teleop/vr/in`;

/** ~30 Hz, matching the MediaPipe panel. The Quest renders at 72–90 Hz, but the
 *  backend commit loop and the staleness budget are tuned around this rate, and
 *  publishing every display frame would trade real headroom for nothing. */
const PUBLISH_MS = 33;

const BODY_LS_KEY = "haller.vrTeleop.body.v1";
const MIRROR_LS_KEY = "haller.vrTeleop.mirror.v1";

function fmtMm(m: number | undefined): string {
  return m === undefined ? "—" : `${(m * 1000).toFixed(0)} mm`;
}

export function VRTeleopPanel({ armIds }: { armIds: string[] }) {
  const [supported, setSupported] = useState<boolean | null>(null);
  const [inSession, setInSession] = useState(false);
  const [xrMode, setXrMode] = useState<TeleopXRSession["mode"] | null>(null);
  const [estopped, setEstopped] = useState(false);
  const [status, setStatus] = useState<HumanTeleopStatus | null>(null);
  const [body, setBody] = useState<BodyOverride>({});
  const [mirrorMode, setMirrorMode] = useState<"none" | "both">("none");
  // The workspace camera floated into the HUD. In passthrough the operator
  // normally watches the REAL arms — but against a sim backend there is
  // nothing physical to look at, so the MuJoCo camera IS the workspace view
  // and defaults on. On the real rig the same toggle shows the mast camera.
  const [baseCam, setBaseCam] = useState<CameraInfo | null>(null);
  const [showCam, setShowCam] = useState(false);

  const sessionRef = useRef<XRSessionLike | null>(null);
  const refSpaceRef = useRef<unknown>(null);
  const clientRef = useRef<HumanTeleopClient<VRFrame> | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const bodyRef = useRef<BodyOverride>({});
  const mirrorModeRef = useRef<"none" | "both">("none");
  const lastPubRef = useRef(0);
  const estopDownRef = useRef(false);
  const estopInFlightRef = useRef(false);
  const prevAuthRef = useRef<Partial<Record<"left" | "right", SideAuthorityLike>>>({});

  bodyRef.current = body;
  mirrorModeRef.current = mirrorMode;

  useEffect(() => {
    void xrSupported().then(setSupported);
    api.cameras()
      .then(({ cameras }) => {
        const base = cameras.filter((c) => c.role === "base" && c.active);
        const cam = base.find((c) => c.source === "sim_camera") ?? base[0] ?? null;
        setBaseCam(cam);
        if (cam?.source === "sim_camera") setShowCam(true);
      })
      .catch(() => { /* no cameras is fine; the toggle just stays hidden */ });
    try {
      const raw = localStorage.getItem(BODY_LS_KEY);
      if (raw) setBody(JSON.parse(raw));
      const mm = localStorage.getItem(MIRROR_LS_KEY);
      if (mm === "both") setMirrorMode("both");
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

  const teardown = useCallback(async (opts: { stopBackend: boolean }) => {
    const session = sessionRef.current;
    sessionRef.current = null;
    refSpaceRef.current = null;
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

    const clearFrame = attachRenderLayer(session, xrs.mode);
    estopDownRef.current = false;

    const onXRFrame = (t: number, frame: XRFrameLike) => {
      const live = sessionRef.current;
      if (!live) return;
      live.requestAnimationFrame(onXRFrame);
      clearFrame();

      // E-STOP scan runs at display rate, not publish rate: 33 ms of extra
      // latency on a stop button is 33 ms too many. Edge-detected so one
      // press is one POST.
      const down = estopPressed(live);
      if (down && !estopDownRef.current) void fireEstop();
      estopDownRef.current = down;

      if (t - lastPubRef.current < PUBLISH_MS) return;
      lastPubRef.current = t;
      const blurred =
        live.visibilityState !== undefined && live.visibilityState !== "visible";
      const vrFrame = sampleVRFrame(live, frame, refSpaceRef.current, {
        tsMs: Date.now(),
        body: Object.keys(bodyRef.current).length ? bodyRef.current : undefined,
        forceDisengaged: blurred,
        mirrorMode: mirrorModeRef.current,
      });
      client.queueFrame(vrFrame);
      client.tick();
    };
    session.requestAnimationFrame(onXRFrame);
  }, [armIds, teardown, fireEstop]);

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
            match: {a.blocking.join(", ")}
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
        </div>
        {sideRow("left")}
        {sideRow("right")}
        {status?.last_error && (
          <div className="text-red-400 truncate">{status.last_error}</div>
        )}
        <div className="text-neutral-400">
          grip = drive · trigger = gripper · <b>B / Y = E-STOP</b>
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
          Engagement runs the same acquisition countdown as the camera path:
          match the robot&apos;s pose, then authority transfers. The buzz on your
          controller is the handover.
        </div>
      </div>

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
