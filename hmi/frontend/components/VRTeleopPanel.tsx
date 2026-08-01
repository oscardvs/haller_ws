"use client";

/**
 * Quest teleop. Owns the WebXR session, the ~30 Hz publish loop, and the
 * start/stop of the backend human-teleop session.
 *
 * It decides nothing about safety. `dead_man` is the raw grip-button state and
 * the controller poses are shipped unmodified; the backend applies the
 * acquisition countdown, the confidence floor, the staleness budget and every
 * joint clamp. Same contract the MediaPipe panel has — see HumanTeleopPanel.
 *
 * Why the grip button rather than the mouth clutch this replaces: the mouth
 * clutch's usable band was ~0.12 of jaw range at the very top of one operator's
 * travel, sustained while both arms move, and a 22 s trace never once reached
 * `driving` (see hmi/HANDOVER-teleop-engagement.md). A squeeze button is
 * holdable for a whole session and cannot be closed by speech, so the safety
 * property that was untested there is structural here.
 *
 * The publish loop runs off `XRSession.requestAnimationFrame`, NOT
 * `window.requestAnimationFrame`. Only the XR callback is handed an `XRFrame`,
 * which is the sole way to resolve poses — and the window loop is throttled to
 * near-nothing while an immersive session holds the display.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

import { api, type HumanTeleopStatus } from "@/lib/api";
import { BACKEND_URL } from "@/lib/config";
import { HumanTeleopClient } from "@/lib/humanTeleopClient";
import {
  requestVRSession, sampleVRFrame, xrAvailableAtAll, xrSupported,
  type BodyOverride, type VRFrame, type XRFrameLike, type XRSessionLike,
} from "@/lib/vrTeleop";
import { DeadManIndicator } from "./DeadManIndicator";

const WS_URL = `${BACKEND_URL.replace(/^http/, "ws")}/ws/teleop/vr/in`;

/** ~30 Hz, matching the MediaPipe panel. The Quest renders at 72–90 Hz, but the
 *  backend commit loop and the staleness budget are tuned around this rate, and
 *  publishing every display frame would trade real headroom for nothing. */
const PUBLISH_MS = 33;

const BODY_LS_KEY = "haller.vrTeleop.body.v1";

export function VRTeleopPanel({ armIds }: { armIds: string[] }) {
  const [supported, setSupported] = useState<boolean | null>(null);
  const [inSession, setInSession] = useState(false);
  const [status, setStatus] = useState<HumanTeleopStatus | null>(null);
  const [body, setBody] = useState<BodyOverride>({});

  const sessionRef = useRef<XRSessionLike | null>(null);
  const refSpaceRef = useRef<unknown>(null);
  const clientRef = useRef<HumanTeleopClient<VRFrame> | null>(null);
  const bodyRef = useRef<BodyOverride>({});
  const lastPubRef = useRef(0);

  bodyRef.current = body;

  useEffect(() => {
    void xrSupported().then(setSupported);
    try {
      const raw = localStorage.getItem(BODY_LS_KEY);
      if (raw) setBody(JSON.parse(raw));
    } catch {
      /* a corrupt override must not block entering VR; defaults are fine */
    }
  }, []);

  // Status poll. Cheap, and it is the only thing that reports what the backend
  // actually did with our frames — the headset shows the operator nothing.
  useEffect(() => {
    let alive = true;
    const t = setInterval(() => {
      api.humanTeleopStatus()
        .then((s) => { if (alive) setStatus(s); })
        .catch(() => { /* transient; the next tick retries */ });
    }, 250);
    return () => { alive = false; clearInterval(t); };
  }, []);

  const teardown = useCallback(async (opts: { stopBackend: boolean }) => {
    const session = sessionRef.current;
    sessionRef.current = null;
    refSpaceRef.current = null;
    clientRef.current?.close();
    clientRef.current = null;
    setInSession(false);
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
    let session: XRSessionLike;
    try {
      session = await requestVRSession();
    } catch (e) {
      toast.error(`could not start VR: ${(e as Error).message}`);
      return;
    }
    sessionRef.current = session;

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

    const onEnd = () => { void teardown({ stopBackend: true }); };
    session.addEventListener("end", onEnd);

    const onXRFrame = (t: number, frame: XRFrameLike) => {
      const live = sessionRef.current;
      if (!live) return;
      live.requestAnimationFrame(onXRFrame);
      if (t - lastPubRef.current < PUBLISH_MS) return;
      lastPubRef.current = t;
      const vrFrame = sampleVRFrame(live, frame, refSpaceRef.current, {
        tsMs: Date.now(),
        body: Object.keys(bodyRef.current).length ? bodyRef.current : undefined,
      });
      client.queueFrame(vrFrame);
      client.tick();
    };
    session.requestAnimationFrame(onXRFrame);
  }, [armIds, teardown]);

  // Unmount must release the arms. Unlike the MediaPipe panel this one cannot
  // be left mounted-but-hidden: an immersive session already owns the display,
  // so there is no "operator looked at another tab" case to preserve.
  useEffect(() => () => { void teardown({ stopBackend: true }); }, [teardown]);

  const clutch = status?.clutch;
  const running = Boolean(status?.running);

  return (
    <div className="space-y-3 font-mono text-[12px]">
      <div className="flex items-center gap-3">
        {!inSession ? (
          <Button onClick={() => void enterVR()} disabled={supported !== true}>
            Enter VR
          </Button>
        ) : (
          <Button variant="destructive" onClick={() => void teardown({ stopBackend: true })}>
            Exit VR
          </Button>
        )}
        <DeadManIndicator
          held={Boolean(clutch?.engaged)}
          trackingLost={Boolean(status?.tracking?.left?.lost && status?.tracking?.right?.lost)}
          source="vr_grip"
          reason={clutch?.reason}
        />
        <span className="text-muted-foreground">
          state: {status?.state ?? "—"}{running ? "" : " (stopped)"}
        </span>
      </div>

      {supported === false && (
        <div className="rounded border border-destructive/40 p-2 text-destructive">
          {xrAvailableAtAll()
            ? "This browser has WebXR but refused an immersive-vr session."
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

      <div className="text-muted-foreground">
        <div>Hold either <b>grip</b> to drive. Release to freeze both arms.</div>
        <div><b>Trigger</b> is the gripper — analog, 0 open to 1 closed.</div>
        <div>
          Engagement runs the same acquisition countdown as the camera path:
          match the robot&apos;s pose, then authority transfers.
        </div>
      </div>

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
