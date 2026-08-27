"use client";

/**
 * Quest teleop. Owns the WebXR session, the ~30 Hz publish loop, and the
 * start/stop of the backend teleop session. The only VR client there is.
 *
 * It decides nothing about safety. Squeeze state and controller poses are
 * shipped unmodified; the backend applies the acquisition countdown, the
 * staleness budget, every joint clamp and the bimanual collision guard. The
 * decisions this file DOES own are input-shaped, not safety-shaped: B/Y on
 * either controller fires POST /estop (one press, one post); a half-second
 * hold of A/X on either controller starts a take or ends one (hold-gated so
 * a thumb brush cannot toggle a take, and an ended take asks save-or-discard
 * before it commits); and while the session is not fully visible (Quest
 * system menu open) every published frame is forced disengaged — the page
 * keeps receiving the last input state in that menu, so a grip held when it
 * opened would otherwise stay held forever.
 *
 * The socket is two-way. Frames go up at 30 Hz; the backend answers with its
 * own `ik_state` at 20 Hz — per-side conditioning, the wrist's orientation
 * deficit, and a trouble mix this panel turns into controller haptics — and
 * takes `config_update` back for live tuning, clamped server-side.
 *
 * The session asks for **passthrough AR first** (the operator must watch the
 * REAL arms, not a void) with the HUD div as a dom-overlay, falling back to
 * plain immersive-vr where AR is refused. The publish loop runs off
 * `XRSession.requestAnimationFrame`, NOT `window.requestAnimationFrame` —
 * only the XR callback is handed an `XRFrame`, which is the sole way to
 * resolve poses, and the window loop is throttled to near-nothing while an
 * immersive session holds the display.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

import {
  api, ApiError, cameraStreamUrl, recordRateFaithful, type CameraInfo,
  type HumanTeleopStatus, type RecordDrops, type RecordStatus,
} from "@/lib/api";
import { BACKEND_URL } from "@/lib/config";
import { HumanTeleopClient } from "@/lib/humanTeleopClient";
import { useRecorder } from "@/lib/recorder";
import { useSoloHand, useStance, type Pairing } from "@/lib/stance";
import {
  ALL_KNOBS, applyServerConfig, attachRenderScene, axPressed, CAM_TILE_SIZES,
  clampKnob, clusterLayout, controllerRays, cycleIndex, DEFAULT_HUD_ANCHOR,
  DEFAULT_WRIST_PIVOT_M, disengagedFrame, episodesTotal, estopPressed,
  formatKnob, hapticCues, holdToggle, holdToggleInit, ikHapticCues,
  ORIENT_DEFICIT, paintHud,
  parseVrSocketMessage, precisionHeld, pulse, RECORD_HOLD_MS, rayQuadHit,
  reconcileConfig, recorderHapticCue,
  requestTeleopSession, RESET_HOLD_MS, sampleVRFrame, stepTake,
  stepTuning, stickAxes, thumbstickPressed, WRIST_PIVOT_KEY,
  xrAvailableAtAll, xrSupported, yawTowardHead,
  type ArmSetLike, type DatasetTally, type HoldToggleState, type HudAnchor,
  type IkSides,
  type SideAuthorityLike, type TakeEvent, type TakeState,
  type TeleopXRSession, type TuningNav, type VRFrame,
  type VrMenuLike, type XRFrameLike, type XRSessionLike,
} from "@/lib/vrTeleop";
import { isSimArm, presetsFor, type ConfigArm } from "./cockpit/teleopPresets";
import { repoIdFor } from "./cockpit/CommandBar";
import { DeadManIndicator } from "./DeadManIndicator";

const WS_URL = `${BACKEND_URL.replace(/^http/, "ws")}/ws/teleop/vr/in`;

/** ~30 Hz, matching the MediaPipe panel. The Quest renders at 72–90 Hz, but the
 *  backend commit loop and the staleness budget are tuned around this rate, and
 *  publishing every display frame would trade real headroom for nothing. */
const PUBLISH_MS = 33;

/** How long the RIGHT stick must stay clicked before the tuning list opens
 *  or closes. Same short/hold split the left stick already uses, so a short
 *  click still means what it always did (next tile size). */
const TUNE_HOLD_MS = 500;

/** How long the HUD keeps saying that home was refused. Long enough to read
 *  with a headset on, short enough that it is gone before the next decision.
 *  A timestamp, not a timer: nothing in the XR loop may own a setTimeout. */
const HOME_REFUSED_MS = 1500;

const SOLO_LS_KEY = "haller.vrTeleop.soloArm.v1";
const WRIST_PIVOT_LS_KEY = "haller.vrTeleop.wristPivot.v1";

const TILE_LS_KEY = "haller.vrTeleop.tile.v1";
// v2, deliberately: v1 headsets remember a view from before the
// over-the-shoulder camera existed, and the natural default is that view
// paired with the "behind" stance. One forced re-default; the choice
// persists again from there.
const VIEW_LS_KEY = "haller.vrTeleop.view.v2";
const HUD_ANCHOR_LS_KEY = "haller.vrTeleop.hudAnchor.v1";

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

/** A backend that predates the start gate, told apart from a refusal by its
 *  status alone. 404/405 is "this recorder has no gate"; a 409 is the gate
 *  working — a colliding camera key, a measured rate under the floor — and
 *  its detail is a sentence the operator has nowhere else to read. */
function isMissingRoute(e: unknown): boolean {
  return e instanceof ApiError && (e.status === 404 || e.status === 405);
}

/** What to show the operator for a failed act. `ApiError.detail` is the
 *  backend's own reason without the `HTTP nnn:` prefix. */
function refusalText(e: unknown): string {
  return e instanceof ApiError ? e.detail : (e as Error).message;
}

/** The recorder's word for where a take is, in the take machine's vocabulary.
 *  `/record/status` says "recording"; the machine says "rolling", because
 *  "prompt" is a fourth state the recorder has no concept of and must never
 *  be handed. A backend that predates the gate reports only `recording`. */
function takeStateOf(rs: { state?: string; recording?: boolean } | null): TakeState {
  if (!rs) return "idle";
  if (rs.state === "armed") return "armed";
  return rs.state === "recording" || rs.recording ? "rolling" : "idle";
}

/** Where the recorder actually is, asked rather than assumed. A failed act
 *  must never leave the HUD claiming a state the recorder is not in. */
async function recorderTakeState(): Promise<TakeState> {
  try {
    return takeStateOf(await api.recordStatus());
  } catch {
    return "idle";
  }
}

/** The single worst drop source, reduced to one name — an operator needs a
 *  cable to go check, not a table. Reads both shapes the status has carried:
 *  a flat `{key: n}` and the `{cameras, arms}` split. Null when nothing has
 *  dropped, which is the common case and must not paint a line. */
export function worstDropSource(drops: RecordDrops | undefined): string | null {
  if (!drops) return null;
  let worst: string | null = null;
  let most = 0;
  // The kind is carried into the label, not discarded: arms are `left`/`right`
  // and nothing stops a camera being named for a side, so a bare "left" would
  // send the operator to the wrong end of the rig. That collision is the whole
  // reason this arrives as two maps instead of one.
  for (const [kind, map] of [["cam", drops.cameras], ["arm", drops.arms]] as const) {
    for (const [key, n] of Object.entries(map ?? {})) {
      if (n > most) {
        most = n;
        worst = `${kind} ${key}`;
      }
    }
  }
  return worst;
}

export function VRTeleopPanel({ arms }: { arms: ConfigArm[] }) {
  // The whole arm record, not just the ids: a session about to bank 46
  // episodes has to say on its own button whether this rig is the real one.
  // Memoised because half the callbacks below take it as a dependency, and a
  // fresh array every render would rebuild every one of them.
  const armIds = useMemo(() => arms.map((a) => a.id), [arms]);
  const [supported, setSupported] = useState<boolean | null>(null);
  const [inSession, setInSession] = useState(false);
  const [xrMode, setXrMode] = useState<TeleopXRSession["mode"] | null>(null);
  const [estopped, setEstopped] = useState(false);
  const [status, setStatus] = useState<HumanTeleopStatus | null>(null);
  // One stance across both surfaces. lib/stance.ts holds it in localStorage
  // behind a `storage`-backed external store, so the cockpit and this page
  // open side by side in one browser can never disagree about where the
  // operator is standing — and there is one pairing rule, not two.
  const [stance, setStance] = useStance();
  const [soloHand, setSoloHand] = useSoloHand();
  // Which arm a SINGLE-ARM session drives, or null for both. The session
  // accepts a null side, so this is the whole mechanism — the other hand's
  // controller is simply ignored and nothing is ever written to it. This is
  // the shape a first hardware run wants, and the only shape a rig with one
  // working servo board has.
  const [soloArm, setSoloArm] = useState<string | null>(null);
  // The backend's own view of the solver, off the socket at 20 Hz. Mirrored
  // into state at a quarter of that: the desktop readout does not need 20
  // re-renders a second, and the HUD reads the ref.
  const [ikSides, setIkSides] = useState<IkSides>({});
  // Live-tunable config, server-clamped. Seeded by the socket's first
  // `ik_state`; the wrist pivot is the one client-side entry (only the
  // client has a grip pose to apply it to) and persists here.
  const [tuneValues, setTuneValues] = useState<Record<string, number>>(
    { [WRIST_PIVOT_KEY]: DEFAULT_WRIST_PIVOT_M });
  const [precision, setPrecision] = useState(false);
  // Takes saved since the page loaded. The recorder reports frames, not an
  // episode index, and "how many good ones so far" is what an operator
  // mid-run actually asks.
  const [takes, setTakes] = useState(0);
  // Where the take is: idle, ARMED (loaded, nothing written), rolling, or the
  // decision. Mirrored into state for the desktop panel; the XR loop and the
  // status poll both read the ref.
  const [take, setTake] = useState<TakeState>("idle");
  // Mirrors gateServerRef for the desktop card. The in-scene HUD carries
  // "(local gate)" for as long as it is true; the toast that announced it is
  // gone in seconds, and a caveat the operator can only have seen once is a
  // caveat they do not have.
  const [localGate, setLocalGate] = useState(false);
  // The knobs the operator moved this page-load, mirrored out of the ref for
  // the desktop panel's ◆ markers.
  const [dirtyTune, setDirtyTune] = useState<string[]>([]);
  // Which hand drives which arm, from the very pairing this session was
  // STARTED with — the stance select stays live in-session and re-picking it
  // there cannot re-pair arms the backend already owns, so a HUD that
  // recomputed would describe a mapping nothing is running.
  const [armSet, setArmSet] = useState<ArmSetLike>(null);
  // What the dataset on disk holds, so a rolling take can be named by the
  // index it will actually land at rather than by a page-local ordinal.
  const [tally, setTally] = useState<DatasetTally | null>(null);
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
  const stanceRef = useRef(stance);
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
  const menuRef = useRef<VrMenuLike>(null);
  const ikSidesRef = useRef<IkSides>({});
  const prevIkRef = useRef<IkSides>({});
  const ikPushedAtRef = useRef(0);
  const tuneValuesRef = useRef<Record<string, number>>(
    { [WRIST_PIVOT_KEY]: DEFAULT_WRIST_PIVOT_M });
  const tuneOpenRef = useRef(false);
  const tuneNavRef = useRef<TuningNav>({ index: 0, lastStepMs: 0 });
  const precisionRef = useRef(false);
  const takesRef = useRef(0);
  const takeRef = useRef<TakeState>("idle");
  // Whether this backend has the start gate. Null until the first probe:
  // false is a claim (hold ARMED here instead), and an unprobed backend has
  // made no claim either way.
  const gateServerRef = useRef<boolean | null>(null);
  const gateNotedRef = useRef(false);
  // True while a gate call is in flight. The recorder goes on reporting the
  // state it is leaving until the call lands, and reconciling to that would
  // walk the HUD backwards — and fire the ROLL cue, the firmest in the
  // vocabulary, at an operator who is stopping a take.
  const actInFlightRef = useRef(false);
  // Bumped on every LOCAL take transition. A status read carries the epoch it
  // was ISSUED under; if the epoch has moved by the time it resolves, the read
  // describes a state the machine has already left. `actInFlightRef` cannot
  // catch that one — it is back to false by the time such a read lands.
  const takeEpochRef = useRef(0);
  // What ARM settled on, so ROLL writes to the same dataset the gate opened.
  const gatePairRef = useRef<{ repoId: string; task: string } | null>(null);
  // When the operator last asked for home during the prompt and was told no.
  const homeRefusedAtRef = useRef(-Infinity);
  const runTakeRef = useRef<((ev: TakeEvent) => void) | null>(null);
  const dirtyTuneRef = useRef<Set<string>>(new Set());
  // Set on every socket OPEN, cleared by the first config-bearing message:
  // the re-assert is one `config_update` per connection, never one per 20 Hz
  // push, which would flood the socket it is trying to correct.
  const reassertPendingRef = useRef(false);
  const tallyRef = useRef<DatasetTally | null>(null);
  // The pairing the SELECTED button describes. Posted verbatim rather than
  // recomputed at click time: a preset that shows one mapping and starts
  // another is invisible until an arm moves the wrong way.
  const pairingRef = useRef<Pairing | null>(null);
  // The right stick mirrors the left's short/hold split: short click cycles
  // the tile size, a hold opens the tuning list. So its action also fires on
  // RELEASE, and a hold cannot resize the tile on its way to the menu.
  const rStickDownAtRef = useRef<number | null>(null);
  const rStickFiredRef = useRef(false);
  // Where the HUD cluster hangs, and the in-flight grab if the operator is
  // currently dragging it. Refs, not state: both change inside the XR loop.
  const anchorRef = useRef<HudAnchor>({
    pos: [...DEFAULT_HUD_ANCHOR.pos], yawDeg: DEFAULT_HUD_ANCHOR.yawDeg });
  const grabRef = useRef<null | {
    hand: "left" | "right"; dist: number; offset: [number, number, number];
  }>(null);
  // Left-stick click doubles up: short click cycles the view, a long hold
  // resets the arms — so the click action fires on RELEASE, gated by how
  // long the stick was down.
  const lStickDownAtRef = useRef<number | null>(null);
  const lStickFiredRef = useRef(false);

  stanceRef.current = stance;
  // The launcher, from the cockpit's own helper — one list of sessions and one
  // pairing rule across both surfaces, so a preset cannot mean different things
  // depending on which screen you started it from. A remembered solo arm the
  // config no longer enables reads as "dual": sending it would earn a 400 from
  // a backend that has never heard of that arm.
  const presets = presetsFor(armIds, stance, soloHand);
  const soloSelected = soloArm && armIds.includes(soloArm) ? soloArm : null;
  const wanted = presets.find(
    (pr) => pr.id === (soloSelected ? `solo-${soloSelected}` : "dual"));
  // A preset this rig cannot offer is drawn, but never SELECTED — a one-armed
  // rig defaults to its solo session rather than sitting on a disabled "dual"
  // that Enter Passthrough would happily start anyway.
  const selected = (wanted && wanted.unavailable === null)
    ? wanted
    : (presets.find((pr) => pr.unavailable === null) ?? null);
  pairingRef.current = selected?.pairing ?? null;
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
    stance,
    armSet,
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
      const wp = Number(localStorage.getItem(WRIST_PIVOT_LS_KEY));
      if (Number.isFinite(wp) && wp >= 0 && wp <= 0.2) {
        tuneValuesRef.current = { ...tuneValuesRef.current, [WRIST_PIVOT_KEY]: wp };
        setTuneValues((v) => ({ ...v, [WRIST_PIVOT_KEY]: wp }));
      }
      const solo = localStorage.getItem(SOLO_LS_KEY);
      if (solo) setSoloArm(solo);
      try {
        const a = JSON.parse(localStorage.getItem(HUD_ANCHOR_LS_KEY) ?? "");
        if (Array.isArray(a?.pos) && a.pos.length === 3
            && typeof a?.yawDeg === "number") {
          anchorRef.current = { pos: a.pos, yawDeg: a.yawDeg };
        }
      } catch { /* no saved placement: spawn at the default */ }
      const t = Number(localStorage.getItem(TILE_LS_KEY));
      if (Number.isInteger(t) && t >= 0 && t < CAM_TILE_SIZES.length) setTileIdx(t);
    } catch {
      /* a corrupt override must not block entering VR; defaults are fine */
    }
  }, []);

  // The hand-to-arm pairing is chosen at session START from the stance
  // (see enterVR). Changing stance mid-session re-maps the deltas on the
  // next squeeze but cannot re-pair the arms — say so instead of silently
  // half-applying.
  const stancePrevRef = useRef(stance);
  useEffect(() => {
    const changed = stancePrevRef.current !== stance;
    stancePrevRef.current = stance;
    if (changed && inSession) {
      toast.info("stance changed — exit and re-enter VR to re-pair hands to arms");
    }
  }, [stance, inSession]);

  // Status poll. Cheap, and it is what feeds the HUD — the only thing the
  // operator can see from inside the headset. Also the haptic driver: the
  // controller buzz on handover comes from status transitions seen here.
  useEffect(() => {
    let alive = true;
    const t = setInterval(() => {
      // Recorder status rides the same poll: the HUD's REC line is the only
      // way an operator inside the headset can see a take rolling.
      const issuedAt = takeEpochRef.current;
      api.recordStatus()
        .then((rs) => {
          if (!alive) return;
          setRecStatus(rs);
          // A gate call in flight is a state the recorder has not moved to
          // yet: it still reports the one it is leaving, and reconciling to
          // that walks the HUD backwards.
          if (actInFlightRef.current) return;
          // Issued before a transition and resolved after it: stale by exactly
          // one move. Reconciling to it walks the machine backwards, and the
          // operator's next A/X hold re-arms instead of rolling — they hold
          // twice and are still not recording.
          if (takeEpochRef.current !== issuedAt) return;
          // Otherwise the recorder's own state outranks the client's guess: a
          // take that ended some other way — a recorder fault, the cockpit
          // stopping it — takes the decision with it, and a gate that was
          // invalidated must not keep saying ARMED. `stepTake` is what keeps
          // a recorder still legitimately rolling from slamming the prompt.
          const gate = rs;
          const backend = takeStateOf(gate);
          // A locally held gate IS idle on the backend — that is exactly what
          // "nothing is written before you roll" means without a gate to hold
          // it — so its own idle report must not drop it out of ARMED.
          if (gateServerRef.current === false && takeRef.current === "armed"
              && backend === "idle") {
            return;
          }
          runTakeRef.current?.({
            kind: "recorder", state: backend,
            invalidated: Boolean(gate.invalidated_reason),
          });
        })
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
    // Whatever was on the HUD dies with the session: an open tuning list or
    // an unanswered save/discard has no controller left to answer it.
    tuneOpenRef.current = false;
    precisionRef.current = false;
    setPrecision(false);
    // The take machine's `abort`, written out: teardown must not fire REST.
    // Exiting VR stops teleop, which invalidates an armed gate on the
    // backend's own rule, so leaving disarms without a command.
    takeRef.current = "idle";
    takeEpochRef.current += 1;
    setTake("idle");
    setArmSet(null);
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

  /** Re-read what the dataset holds on disk.
   *
   *  Called on mount and on every take boundary, never on the status poll: it
   *  reads the dataset meta off disk, which is not something to do four times
   *  a second. A dataset that does not exist yet, or an endpoint that refuses,
   *  leaves the tally null and the HUD falls back to its own take counter —
   *  the episode index is a convenience, and nothing about a take depends on
   *  it. The baseline is re-taken whenever the repo changes, so the floor in
   *  `episodesTotal` always counts takes into the dataset it is describing.
   */
  const refreshEpisodes = useCallback(async (repoId?: string | null) => {
    try {
      const r = await api.recordEpisodes(repoId ?? undefined);
      const onDisk = r.episodes.length;
      const prev = tallyRef.current;
      const next: DatasetTally = prev && prev.repoId === r.repo_id
        ? { ...prev, onDisk }
        : { repoId: r.repo_id, onDisk,
            baselineOnDisk: onDisk, baselineTakes: takesRef.current };
      tallyRef.current = next;
      setTally(next);
    } catch {
      tallyRef.current = null;
      setTally(null);
    }
  }, []);

  useEffect(() => { void refreshEpisodes(); }, [refreshEpisodes]);

  /** Buzz both hands with one cue. Every recorder transition gets one, at a
   *  distinguishable weight — inside a headset the haptic is the fastest
   *  channel there is, and "did that take actually start" is exactly the
   *  question an operator should never have to guess at. */
  const buzzBoth = useCallback((intensity: number, durationMs: number) => {
    const session = sessionRef.current;
    if (!session) return;
    pulse(session, "left", intensity, durationMs);
    pulse(session, "right", intensity, durationMs);
  }, []);

  /** Put the panel back where the recorder actually is, with the cue that
   *  move earns. A refused or failed act must never leave the HUD claiming a
   *  state the recorder is not in — an armed gate that only exists on this
   *  page is the silent un-arming the gate was built to prevent. */
  const settleTake = useCallback((state: TakeState) => {
    const from = takeRef.current;
    const to = stepTake(from, { kind: "recorder", state }).state;
    const cue = recorderHapticCue(from, to, null);
    if (cue) buzzBoth(cue.intensity, cue.durationMs);
    takeRef.current = to;
    takeEpochRef.current += 1;
    setTake(to);
  }, [buzzBoth]);

  /** ARM: open the dataset, freeze its schema, camera set, arm set and
   *  measured fps, resolve the episode index — and write NOT ONE FRAME.
   *
   *  The task/HF-user draft comes from the cockpit's Dataset tab (persisted),
   *  so a solo operator never has to leave the headset between takes. */
  const armAct = useCallback(async () => {
    const rec = useRecorder.getState();
    const task = rec.task.trim();
    if (!task) {
      toast.error("no task drafted — set one in the cockpit's Dataset tab first");
      buzzBoth(0.2, 60);
      settleTake("idle");
      return;
    }
    const repoId = repoIdFor(rec.hfUser, task);
    gatePairRef.current = { repoId, task };
    try {
      const st = await api.recordArm(repoId, task);
      gateServerRef.current = true;
      setLocalGate(false);
      recStatusRef.current = st;
      setRecStatus(st);
      toast.success(`armed → ${repoId} — nothing written until you roll`);
      // The repo is settled here rather than at ROLL: the take in flight
      // lands at exactly the index this counts.
      void refreshEpisodes(st.repo_id ?? repoId);
    } catch (e) {
      if (isMissingRoute(e)) {
        // A backend that predates the gate. Hold ARMED here instead: nothing
        // is written before ROLL either way — the operator-facing half — but
        // the schema is not frozen, a colliding camera key 409s three seconds
        // into the take instead of now, and the episode index is a guess. It
        // upgrades silently the first time /record/arm answers 200.
        gateServerRef.current = false;
        setLocalGate(true);
        if (!gateNotedRef.current) {
          gateNotedRef.current = true;
          toast.info("no start gate on this backend — armed locally: nothing is "
            + "written before you roll, but the schema is not frozen");
        }
        return;
      }
      // Every refusal the gate exists to move earlier lands here: colliding
      // camera keys, an unknown repo, a measured rate under the threshold.
      toast.error(`arm refused: ${refusalText(e)}`);
      settleTake("idle");
    }
  }, [buzzBoth, refreshEpisodes, settleTake]);

  /** ROLL: frames start landing. Under a local gate there is nothing to roll
   *  into, so this is the plain /record/start the page held back — which is
   *  what "nothing is written before you roll" means without a real gate. */
  const rollAct = useCallback(async () => {
    const rec = useRecorder.getState();
    const pair = gatePairRef.current;
    try {
      let st: RecordStatus | null = null;
      if (gateServerRef.current) {
        st = await api.recordRoll();
      } else {
        if (!pair) {
          toast.error("nothing armed to roll — hold A/X again");
          settleTake("idle");
          return;
        }
        st = await rec.start(pair.repoId, pair.task);
      }
      if (st) {
        recStatusRef.current = st;
        setRecStatus(st);
      }
      toast.success(`rolling → ${st?.repo_id ?? pair?.repoId ?? "dataset"}`);
      void refreshEpisodes(st?.repo_id ?? pair?.repoId);
      await rec.refresh();
    } catch (e) {
      toast.error(`roll refused: ${refusalText(e)}`);
      settleTake(await recorderTakeState());
    }
  }, [refreshEpisodes, settleTake]);

  /** The decision, committed.
   *
   *  `POST /record/stop` takes the save decision AT stop time — there is no
   *  way to end the episode first and choose afterwards — so the choice runs
   *  while the recorder is still rolling and the tail of the episode is a
   *  second of the operator holding still. That is the honest trade, and the
   *  HUD says so rather than pretending the take already ended.
   *
   *  `rearm` is what keeps a session in ARMED, where it costs nothing and the
   *  next take is always the expected next thing. */
  const stopAct = useCallback(async (act: { save: boolean; rearm: boolean }) => {
    const rec = useRecorder.getState();
    const before = recStatusRef.current?.episode_frames ?? 0;
    try {
      let st: RecordStatus | null = null;
      if (gateServerRef.current) {
        st = await api.recordStop(act.save, act.rearm);
      } else {
        // No `rearm` on a backend without the gate — the re-arm is done by
        // hand below, and sending a field it has never heard of risks a 422
        // on the one call that must not fail: the one that saves the take.
        st = await rec.stop(act.save);
      }
      if (st) {
        recStatusRef.current = st;
        setRecStatus(st);
      }
      // The reply's count where it has one, else the last poll's: a recorder
      // that zeroes the counter on stop must not report a 0-frame take.
      const frames = st?.episode_frames || before;
      // What the recorder DID, never what we asked it for.
      //
      // `{rearm: true}` whose save lands and whose re-arm is refused is a
      // 200 carrying `state:"idle"` and a reason — deliberately not a 409,
      // because a 409 would report a banked take as a failure. So the ask is
      // no longer evidence of the outcome, and a toast built from `act.rearm`
      // would announce a gate that is DOWN at the one moment the operator is
      // deciding whether to keep driving. The poll reconciles the state a
      // quarter-second later and the HUD paints `GATE DROPPED — <reason>`,
      // but the toast is what gets read at the decision, and it would have
      // substituted a wrong conclusion rather than withholding one.
      //
      // Under a LOCAL gate there is no backend verdict to read: `rearm` was
      // never sent, and the re-arm below is this page's own. Reading `state`
      // there would invent a refusal that did not happen.
      const rearmed = gateServerRef.current ? st?.state === "armed" : act.rearm;
      const refused = act.rearm && !rearmed;
      const why = st?.invalidated_reason ? ` — ${st.invalidated_reason}` : "";
      if (act.save) {
        takesRef.current += 1;
        setTakes(takesRef.current);
        // The take is SAFE either way, and the copy says so first: a refused
        // re-arm must not read as a lost take.
        const saved = `take ${takesRef.current} saved — ${frames} frames`;
        if (refused) toast.error(`${saved} · NOT re-armed${why}`);
        else toast.success(saved + (rearmed ? " · armed for the next" : ""));
      } else {
        // Never touches disk: an episode buffer that is never saved is
        // dropped and the index does not advance.
        if (refused) toast.error(`take discarded · NOT re-armed${why}`);
        else toast.info("take discarded"
          + (rearmed ? " — armed again, same episode" : ""));
      }
      void refreshEpisodes(st?.repo_id);
      await rec.refresh();
      // A local gate has no server-side rearm — hold ARMED here again, which
      // re-probes /record/arm and upgrades the moment it exists.
      if (!gateServerRef.current && act.rearm) await armAct();
    } catch (e) {
      toast.error(`stop refused: ${refusalText(e)}`);
      settleTake(await recorderTakeState());
    }
  }, [armAct, refreshEpisodes, settleTake]);

  /** One step of the take machine: the pure transition, the cue it earns,
   *  then the REST act it demands. The sticks, the A/X hold, the desktop
   *  buttons and the status poll all come through here — where the take is
   *  cannot be two different answers if only one place decides it. */
  const runTake = useCallback(async (ev: TakeEvent) => {
    const from = takeRef.current;
    const tr = stepTake(from, ev);
    const cue = recorderHapticCue(
      from, tr.state, ev.kind === "choose" ? ev.choice : null);
    if (cue) buzzBoth(cue.intensity, cue.durationMs);
    takeRef.current = tr.state;
    takeEpochRef.current += 1;
    setTake(tr.state);
    // The prompt is modal — both sticks are its own — so an open tuning list
    // has to go, or the right stick walks a knob list and answers the prompt
    // with the same click.
    if (tr.state === "prompt") tuneOpenRef.current = false;
    const act = tr.act;
    if (!act) return;
    actInFlightRef.current = true;
    try {
      if (act.do === "arm") await armAct();
      else if (act.do === "roll") await rollAct();
      else await stopAct({ save: act.save, rearm: act.rearm });
    } finally {
      actInFlightRef.current = false;
    }
  }, [armAct, buzzBoth, rollAct, stopAct]);
  // The status poll fires take events at it, and that interval is declared
  // above the machine — through a ref, so a new `runTake` never tears the
  // 250 ms interval down and rebuilds it.
  useEffect(() => { runTakeRef.current = (ev) => { void runTake(ev); }; }, [runTake]);

  /** The A/X hold, from inside the XR loop. One gesture, the whole ladder:
   *  ARM, ROLL, raise the decision — or withdraw it, so a hold that was a
   *  mistake costs nothing. */
  const onRecordHold = useCallback(() => {
    void runTake({ kind: "ax_hold" });
  }, [runTake]);

  /** Write one tuning knob.
   *
   *  The wrist pivot never leaves the client: it moves the read-out point on
   *  the controller, which the backend cannot see. Everything else goes up as
   *  a `config_update` and comes back CLAMPED in `config_applied` — the
   *  optimistic local write below is what makes the stick feel instant, and
   *  the echo is what makes it honest. */
  const setTuning = useCallback((key: string, value: number) => {
    const next = { ...tuneValuesRef.current, [key]: value };
    tuneValuesRef.current = next;
    setTuneValues(next);
    if (key === WRIST_PIVOT_KEY) {
      try { localStorage.setItem(WRIST_PIVOT_LS_KEY, String(value)); }
      catch { /* private mode: the pivot just won't persist */ }
      return;
    }
    // Moved by the operator, so the page owns it from here: the teleoperator's
    // config lives PER CONNECTION and HumanTeleopClient reconnects after
    // 50 ms, and a gain that quietly halves mid-take feels exactly like an arm
    // that has started lagging. The local pivot returns above and is never in
    // this set — it never leaves the client to be reverted.
    if (!dirtyTuneRef.current.has(key)) {
      dirtyTuneRef.current.add(key);
      setDirtyTune([...dirtyTuneRef.current]);
    }
    clientRef.current?.send({ type: "config_update", config: { [key]: value } });
  }, [setDirtyTune]);

  /** One server → client message off the teleop socket. */
  const onSocketMessage = useCallback((data: string) => {
    const msg = parseVrSocketMessage(data);
    if (!msg) return;
    if (msg.kind === "config_applied") {
      // The clamped echo always wins: whatever the robot took IS the value,
      // and re-asserting the unclamped ask would only fight it.
      const applied = applyServerConfig(tuneValuesRef.current, msg.config);
      tuneValuesRef.current = applied;
      setTuneValues(applied);
      return;
    }
    if (msg.config) {
      // The server owns every knob the operator has not touched; the page owns
      // the ones it has. The re-assert goes out ONCE per connection — this
      // message also arrives at 20 Hz, and answering each one would flood the
      // socket it is correcting.
      const { values, reassert } = reconcileConfig(
        tuneValuesRef.current, [...dirtyTuneRef.current], msg.config);
      tuneValuesRef.current = values;
      setTuneValues(values);
      if (reassertPendingRef.current) {
        reassertPendingRef.current = false;
        const keys = Object.keys(reassert);
        if (keys.length) {
          clientRef.current?.send({ type: "config_update", config: reassert });
          // Said out loud: a value that survives a blip while its neighbours
          // revert to the server's defaults is otherwise witchcraft.
          toast.info(`re-asserted your tuning: ${keys.join(", ")}`);
        }
      }
    }
    if (msg.kind !== "ik_state") return;
    ikSidesRef.current = msg.sides;
    const session = sessionRef.current;
    if (session) {
      for (const cue of ikHapticCues(prevIkRef.current, msg.sides)) {
        pulse(session, cue.hand, cue.intensity, cue.durationMs);
      }
    }
    prevIkRef.current = msg.sides;
    // 20 Hz on the wire, 4 Hz into React: the HUD reads the ref every frame,
    // and the desktop readout does not need twenty re-renders a second.
    const now = performance.now();
    if (now - ikPushedAtRef.current >= 250) {
      ikPushedAtRef.current = now;
      setIkSides(msg.sides);
    }
  }, []);

  /** Reset both arms to their home pose from inside the headset (hold the
   *  left stick ~0.8 s). Gated on no side driving: home is a discrete move
   *  and must never fight a live teleop stream — with the grips open the
   *  arms are frozen, so the move owns them cleanly. */
  const resetArms = useCallback(async () => {
    const st = statusRef.current;
    const driving = (["left", "right"] as const).some(
      (side) => st?.acquire?.[side]?.authority === "driving");
    const session = sessionRef.current;
    if (driving) {
      // Refused, and the buzz says so: one weak tick instead of the firm
      // double pulse of a completed reset.
      if (session) pulse(session, "left", 0.2, 60);
      return;
    }
    try {
      // In-session home: the discrete /arm/{id}/home is refused while the
      // session owns the arms, and rightly so — this one slews home through
      // the session's own ramp and collision guard.
      const { sides } = await api.humanTeleopHome();
      if (session && sides.length) {
        pulse(session, "left", 0.6, 180);
        pulse(session, "right", 0.6, 180);
      }
      toast.success(`arms resetting to home (${sides.join(", ") || "none"})`);
    } catch (e) {
      toast.error(`arm reset failed: ${(e as Error).message}`);
    }
  }, []);

  const enterVR = useCallback(async () => {
    if (!armIds.length) {
      toast.error("no arms enabled in the backend config — nothing to drive");
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
    const pairing = pairingRef.current;
    if (!pairing || (!pairing.left_arm && !pairing.right_arm)) {
      toast.error("no arm resolved for this preset — pick another");
      await teardown({ stopBackend: false });
      return;
    }
    try {
      await api.humanTeleopStart(pairing);
      setArmSet({ left: pairing.left_arm, right: pairing.right_arm });
    } catch (e) {
      toast.error(`teleop start refused: ${(e as Error).message}`);
      await teardown({ stopBackend: false });
      return;
    }

    const client = new HumanTeleopClient<VRFrame>(WS_URL, {
      // Every open, reconnects included: the teleoperator's config lives per
      // connection, so a reconnected client whose list still shows the old
      // numbers is describing something that no longer exists. The reply is
      // also where the operator's own knobs go back up — armed here so it
      // happens once per connection rather than on every 20 Hz push.
      onOpen: () => {
        reassertPendingRef.current = true;
        client.send({ type: "request_settings" });
      },
      onMessage: onSocketMessage,
    });
    client.connect();
    clientRef.current = client;
    setInSession(true);
    setEstopped(false);

    const onEnd = () => { void teardown({ stopBackend: true }); };
    session.addEventListener("end", onEnd);
    prevIkRef.current = {};
    ikSidesRef.current = {};
    tuneOpenRef.current = false;
    precisionRef.current = false;

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
      hudCanvas.height = 440;
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
      // Both sticks share one idiom: the short click acts on RELEASE, so a
      // hold cannot first do the short thing on its way to the long one.
      // Left: click = next view, ~0.8 s hold = reset arms. Right: click =
      // next tile size, 0.5 s hold = the tuning list. While the end-of-take
      // decision is open it takes both clicks instead — keep left, redo right
      // — because nothing else may be one click away from banking or binning
      // an episode.
      const prompt = takeRef.current === "prompt";
      const lStick = thumbstickPressed(live, "left");
      if (lStick) {
        if (lStickDownAtRef.current === null) {
          lStickDownAtRef.current = t;
          lStickFiredRef.current = false;
        } else if (!lStickFiredRef.current
                   && t - lStickDownAtRef.current >= RESET_HOLD_MS) {
          lStickFiredRef.current = true;
          if (prompt) {
            // Home is refused through the tail of a take the operator may be
            // about to keep — homing would corrupt it. Answered rather than
            // silently dropped: the weak tick `resetArms` already uses for a
            // refusal, plus a line on the HUD. The fired flag is what stops
            // the release falling through to KEEP.
            homeRefusedAtRef.current = t;
            pulse(live, "left", 0.2, 60);
          } else {
            void resetArms();
          }
        }
      } else {
        if (lStickDownAtRef.current !== null && !lStickFiredRef.current
            && t - lStickDownAtRef.current < 350) {
          if (prompt) void runTake({ kind: "choose", choice: "keep" });
          else cycleView();
        }
        lStickDownAtRef.current = null;
      }
      const rStick = thumbstickPressed(live, "right");
      if (rStick) {
        if (rStickDownAtRef.current === null) {
          rStickDownAtRef.current = t;
          rStickFiredRef.current = false;
        } else if (!rStickFiredRef.current && !prompt
                   && t - rStickDownAtRef.current >= TUNE_HOLD_MS) {
          rStickFiredRef.current = true;
          tuneOpenRef.current = !tuneOpenRef.current;
          tuneNavRef.current = { ...tuneNavRef.current, lastStepMs: t };
          pulse(live, "right", 0.3, 60);
        }
      } else {
        if (rStickDownAtRef.current !== null && !rStickFiredRef.current
            && t - rStickDownAtRef.current < 350) {
          if (prompt) void runTake({ kind: "choose", choice: "redo" });
          else cycleTile();
        }
        rStickDownAtRef.current = null;
      }

      // The tuning list walks and adjusts on the RIGHT stick's axes, and only
      // while it is open and the prompt is not: an accidental nudge mid-take
      // must not move a gain, and a stick that both walks a knob list and
      // answers the decision answers it by accident.
      if (tuneOpenRef.current && !prompt) {
        const step = stepTuning(tuneNavRef.current, stickAxes(live, "right"),
                                t, tuneValuesRef.current);
        tuneNavRef.current = step.nav;
        if (step.patch) setTuning(step.patch.key, step.patch.value);
      }

      // Precision: the LEFT stick pushed away and held — see `precisionHeld`
      // for why it could not stay on A/X. Edge-buzzed, because gains that
      // silently halve feel exactly like an arm that has started lagging.
      const fine = precisionHeld(live);
      if (fine !== precisionRef.current) {
        precisionRef.current = fine;
        setPrecision(fine);
        pulse(live, "left", fine ? 0.3 : 0.15, 50);
      }

      // Grab-to-move: point at the HUD cluster and hold the trigger while
      // that hand's arm is NOT driving (while driving, the trigger is the
      // gripper and the HUD refuses to move). Quest-window semantics: the
      // cluster follows the ray at its grab distance and keeps facing you.
      const anchor = anchorRef.current;
      if (hudCanvas) {
        const rays = controllerRays(live, frame, refSpaceRef.current);
        const camW = CAM_TILE_SIZES[tileIdxRef.current]?.widthM ?? 1.1;
        const layout = clusterLayout(camW, 0.75, hudCanvas.height / hudCanvas.width,
                                     Boolean(camImgRef.current));
        const grab = grabRef.current;
        if (grab) {
          const ray = rays[grab.hand];
          if (!ray || !ray.trigger) {
            grabRef.current = null;
            try {
              localStorage.setItem(HUD_ANCHOR_LS_KEY, JSON.stringify(anchor));
            } catch { /* private mode: placement just won't persist */ }
          } else {
            for (let i = 0; i < 3; i++) {
              anchor.pos[i] = ray.origin[i] + ray.dir[i] * grab.dist
                - grab.offset[i];
            }
            const head = frame.getViewerPose(refSpaceRef.current)
              ?.transform?.position;
            if (head) {
              anchor.yawDeg = yawTowardHead(anchor.pos, [head.x, head.y, head.z]);
            }
          }
        } else {
          for (const hand of ["left", "right"] as const) {
            const ray = rays[hand];
            if (!ray?.trigger) continue;
            const authority = statusRef.current?.acquire?.[hand]?.authority;
            if (authority === "driving") continue;
            const tCam = rayQuadHit(ray.origin, ray.dir, anchor, 0,
                                    camW, layout.camH || 0.2);
            const tPanel = rayQuadHit(ray.origin, ray.dir, anchor,
                                      layout.panelYOff, layout.panelW,
                                      layout.panelH);
            const tHit = tCam ?? tPanel;
            if (tHit === null) continue;
            const offset: [number, number, number] = [
              ray.origin[0] + ray.dir[0] * tHit - anchor.pos[0],
              ray.origin[1] + ray.dir[1] * tHit - anchor.pos[1],
              ray.origin[2] + ray.dir[2] * tHit - anchor.pos[2],
            ];
            grabRef.current = { hand, dist: tHit, offset };
            pulse(live, hand, 0.3, 40);
            break;
          }
        }
      }

      let hudDirty = false;
      if (hudCanvas && hudCtx && t - hudPaintRef.current > 100) {
        hudPaintRef.current = t;
        const rs = recStatusRef.current;
        paintHud(
          hudCtx, statusRef.current,
          { state: takeRef.current,
            recording: Boolean(rs?.recording),
            episode_frames: rs?.episode_frames ?? 0,
            skipped_frames: rs?.skipped_frames,
            worstDrop: worstDropSource(rs?.drops),
            fpsMeasured: rs?.fps_measured ?? null,
            fpsDeclared: rs?.fps_declared ?? null,
            // The recorder publishes the gate it is actually refusing at, so the
            // HUD reads it instead of holding a second copy that can drift.
            rateFaithful: recordRateFaithful(rs),
            invalidatedReason: rs?.invalidated_reason ?? null,
            localGate: gateServerRef.current === false,
            takes: takesRef.current,
            // Two fields because they are two facts. The gate's index names
            // the take in hand and is null unless one is armed; the count is
            // the idle menu's, and is still a guess past lerobot's RAM buffer.
            // The fallback that used to sit between them is GONE: a backend
            // with no gate now reads `take N`, which is true, instead of
            // `ep N` off a count that was right only by coincidence.
            episodeIndex: rs?.episode_index ?? null,
            datasetCount: episodesTotal(tallyRef.current, takesRef.current) },
          menuRef.current && {
            ...menuRef.current,
            tuning: { open: tuneOpenRef.current,
                      index: tuneNavRef.current.index,
                      values: tuneValuesRef.current,
                      dirty: [...dirtyTuneRef.current] },
            precision: precisionRef.current,
            endPrompt: takeRef.current === "prompt",
            homeRefused: t - homeRefusedAtRef.current < HOME_REFUSED_MS,
          },
          ikSidesRef.current);
        hudDirty = true;
      }
      scene.render(frame, refSpaceRef.current, {
        panel: hudCanvas,
        panelDirty: hudDirty,
        cam: camImgRef.current,
        camMirrored: baseCamRef.current?.facing === "operator",
        camWidthM: CAM_TILE_SIZES[tileIdxRef.current]?.widthM,
        anchor,
      });

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
      if (ax.toggled) onRecordHold();

      if (t - lastPubRef.current < PUBLISH_MS) return;
      lastPubRef.current = t;
      const blurred =
        live.visibilityState !== undefined && live.visibilityState !== "visible";
      const vrFrame = sampleVRFrame(live, frame, refSpaceRef.current, {
        tsMs: Date.now(),
        forceDisengaged: blurred,
        stance: stanceRef.current,
        precision: precisionRef.current,
        wristPivotM: tuneValuesRef.current[WRIST_PIVOT_KEY],
      });
      client.queueFrame(vrFrame);
      client.tick();
    };
    session.requestAnimationFrame(onXRFrame);
  }, [armIds, teardown, fireEstop, onRecordHold, runTake, setTuning,
      onSocketMessage, resetArms, cycleView, cycleTile]);

  // Unmount must release the arms. Unlike the MediaPipe panel this one cannot
  // be left mounted-but-hidden: an immersive session already owns the display,
  // so there is no "operator looked at another tab" case to preserve.
  useEffect(() => () => { void teardown({ stopBackend: true }); }, [teardown]);

  // The COUNT, and it stays: `N in dataset` below is the size of the dataset,
  // which no gate index reports. It is still guessed past lerobot's RAM
  // buffer, so it still stalls at 7 while the operator banks their tenth.
  const datasetEpisodes = episodesTotal(tally, takes);
  // How a take is named on both surfaces: the index the gate resolved, and
  // nothing else. "ep 34" is what an operator reconciles against the dataset
  // browser afterwards; "take 3" is only ever true of this page-load, and is
  // what an ungated backend honestly gets. The count is NOT a stand-in — it
  // renumbers differently and is null at different moments.
  const episodeIdx = recStatus?.episode_index ?? null;
  const takeName = episodeIdx === null ? `take ${takes + 1}` : `ep ${episodeIdx}`;
  const clutch = status?.clutch;
  const running = Boolean(status?.running);
  const acquire = status?.acquire;
  const collision = status?.collision;

  const sideRow = (side: "left" | "right") => {
    const a = acquire?.[side];
    if (!a) return null;
    const grip = clutch?.sides?.[side] ?? clutch?.engaged ?? false;
    const d = ikSides[side];
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
        {a.authority === "driving" && typeof d?.sigma_min === "number" && (
          <span
            className={d.sigma_min < 0.02 ? "text-amber-400" : "text-neutral-400"}
            title="smallest singular value of the position Jacobian, m/rad — how far the tool moves in its worst direction per radian of joint travel"
          >
            σ {d.sigma_min.toFixed(3)}
          </span>
        )}
        {a.authority === "driving" && (d?.orient_residual ?? 0) > ORIENT_DEFICIT && (
          <span className="text-amber-300 truncate">
            wrist is out of twist — move your hand
          </span>
        )}
        {a.reason === "no_tracking" && (
          <span className="text-red-400">no tracking</span>
        )}
        {a.reason === "no_arm" && (
          <span className="text-neutral-500">no arm this side</span>
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
          {take === "armed" && (
            // ARMED must never read like REC: it is loaded and writing
            // nothing, and an operator who cannot tell them apart has been
            // given the gate's cost without its benefit.
            <span className="text-haller-warn font-bold">
              ◆ ARMED {takeName}
              {localGate && (
                <span className="font-normal"> (local gate — schema not frozen)</span>
              )}
            </span>
          )}
          {recStatus?.recording && (
            <span className="text-red-400 font-bold animate-haller-rec">
              ● REC {takeName} · {recStatus.episode_frames}
            </span>
          )}
          {!recStatus?.recording && (takes > 0 || datasetEpisodes !== null) && (
            <span className="text-neutral-400">
              {takes > 0 && `${takes} take${takes === 1 ? "" : "s"} this run`}
              {takes > 0 && datasetEpisodes !== null && " · "}
              {datasetEpisodes !== null && `${datasetEpisodes} in dataset`}
            </span>
          )}
          {precision && (
            <span className="text-sky-400 font-bold">◆ PRECISION</span>
          )}
        </div>
        {sideRow("left")}
        {sideRow("right")}
        {status?.last_error && (
          <div className="text-red-400 truncate">{status.last_error}</div>
        )}
        <div className="text-neutral-400">
          grip = drive · trigger = gripper · <b>B / Y = E-STOP</b> ·{" "}
          <b>A / X hold = arm · roll · end</b>
        </div>
        {take === "prompt" && (
          // Mirrors the in-scene prompt so the decision exists in both HUD
          // paths — dom-overlay headsets and the plain 2D page get the same
          // choices, and the stick clicks drive these same handlers. Four
          // here against the controller's two: standing a session down is the
          // desktop's, where a button costs nothing and collides with no
          // gesture the operator has trained.
          <div className="rounded border border-red-400/70 p-2 space-y-1">
            <div className="text-red-400 font-bold">
              take ended · {recStatus?.episode_frames ?? 0} frames — still
              rolling until you pick
            </div>
            <div className="flex gap-2 flex-wrap pointer-events-auto">
              <Button
                size="sm"
                onClick={() => void runTake({ kind: "choose", choice: "keep" })}
              >
                Keep (L click)
              </Button>
              <Button
                size="sm"
                variant="secondary"
                onClick={() => void runTake({ kind: "choose", choice: "redo" })}
              >
                Redo (R click)
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => void runTake({ kind: "choose", choice: "keep_stop" })}
              >
                Keep &amp; stop
              </Button>
              <Button
                size="sm"
                variant="destructive"
                onClick={() => void runTake({ kind: "choose", choice: "drop" })}
              >
                Discard &amp; stop
              </Button>
            </div>
            <div className="text-neutral-400">
              the first two arm the next take · the last two stand down to idle
            </div>
          </div>
        )}
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
          <b>Hold A or X</b> (either controller, ~0.5 s) <b>ARMS</b> a take: the
          dataset opens and its schema freezes, and{" "}
          <b>nothing is written until you roll</b>. A second hold <b>ROLLS</b>{" "}
          it — that is when frames start landing — and a third opens the
          decision: <b>keep</b> (left stick click) or <b>redo</b> (right stick
          click), both of which leave you armed for the next take. The take is
          still rolling while you pick, because{" "}
          <code>/record/stop</code> takes the save decision at stop time. Draft
          the task in the cockpit&apos;s Dataset tab first; the HUD shows{" "}
          <span className="font-bold">◆ ARMED</span>, then{" "}
          <span className="font-bold">● REC</span> while it rolls.
        </div>
        <div>
          Squeezing a grip <b>anchors your hand to the arm where it is</b> —
          hold still through the countdown, feel the buzz, then your hand&apos;s
          movement drives the gripper. Release, reposition your hand, squeeze
          again to ratchet across the workspace.
        </div>
        <div>
          <b>Push the left stick away and hold</b> for precision: both mapping
          gains scale by the precision factor for fine work. The HUD says
          PRECISION the whole time. <b>Hold the right stick</b> (~0.5 s) opens
          the tuning list; its stick then walks and adjusts it.
        </div>
      </div>

      <div className="space-y-1">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-muted-foreground">session</span>
          {presets.map((preset) => {
            const solo = preset.id === "dual" ? null : preset.id.slice("solo-".length);
            // On the button itself: an operator about to bank 46 episodes has
            // to know whether this is a sim take. `dual` earns the mark only
            // when EVERY arm is sim — a mixed session still moves a real arm.
            const sim = solo
              ? arms.some((a) => a.id === solo && isSimArm(a))
              : arms.length > 0 && arms.every(isSimArm);
            return (
              <Button
                key={preset.id}
                size="sm"
                variant={preset.id === selected?.id ? "default" : "outline"}
                disabled={inSession || preset.unavailable !== null}
                title={preset.unavailable ?? preset.detail}
                onClick={() => {
                  setSoloArm(solo);
                  try {
                    if (solo) localStorage.setItem(SOLO_LS_KEY, solo);
                    else localStorage.removeItem(SOLO_LS_KEY);
                  } catch { /* non-fatal */ }
                }}
              >
                {preset.label}{sim ? " (sim)" : ""}
              </Button>
            );
          })}
        </div>
        {selected && selected.id !== "dual" && (
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-muted-foreground">hand</span>
            {([
              [null, "auto (stance)"],
              ["left", "L hand"],
              ["right", "R hand"],
            ] as const).map(([hand, label]) => (
              <Button
                key={label}
                size="sm"
                variant={soloHand === hand ? "default" : "outline"}
                disabled={inSession}
                title={hand === null
                  ? "the stance decides which hand drives the solo arm"
                  : `the ${hand} controller drives it, whatever the stance says`}
                onClick={() => setSoloHand(hand)}
              >
                {label}
              </Button>
            ))}
          </div>
        )}
        <div className="text-muted-foreground">
          — {selected
              ? selected.detail
              : "no arms enabled in the backend config"}. A solo session ignores
          the other hand entirely: nothing is ever written to the arm it has
          none for. Half as much that can go wrong, which is what a first
          hardware run wants.
        </div>
      </div>

      <label className="flex items-center gap-2 text-muted-foreground">
        <span>collision guard</span>
        <select
          className="bg-transparent border rounded px-1 py-0.5"
          value={status?.collision?.enabled ? "on" : "off"}
          disabled={status?.collision?.available === false}
          onChange={(e) => {
            const enabled = e.target.value === "on";
            void api.humanTeleopCollision(enabled)
              .then(() => toast.info(
                `collision guard ${enabled ? "enabled" : "disabled"}`))
              .catch((err) => toast.error(
                `guard toggle refused: ${(err as Error).message}`));
          }}
        >
          <option value="on">on — clamp steps at the margin</option>
          <option value="off">off — measure only, never clamp</option>
        </select>
        <span>
          {status?.collision?.available === false
            ? "— unavailable: no mounts configured for every arm, so the guard has no geometry to reason about."
            : `— off still MEASURES (slack ${fmtMm(status?.collision?.slack_m)}), it just stops holding steps back. The workspace floor, joint limits, rate caps and motion envelope stay on either way.`}
        </span>
      </label>

      <label className="flex items-center gap-2 text-muted-foreground">
        <span>operator stance</span>
        <select
          className="bg-transparent border rounded px-1 py-0.5"
          value={stance}
          onChange={(e) => {
            const v = e.target.value === "front" ? "front"
              : e.target.value === "mirror" ? "mirror" : "behind";
            setStance(v);   // lib/stance.ts persists it and tells the cockpit
          }}
        >
          <option value="behind">egocentric — arms as your own (default)</option>
          <option value="mirror">facing the arms — mirror</option>
          <option value="front">facing the arms — match the camera tile</option>
        </select>
        <span>
          — how your hand maps to the gripper. Egocentric (pairs with the
          over-shoulder view, the default): the replica arm moves exactly
          like your own — push forward and it goes deeper into the scene.
          Mirror: face-to-face, the arm is your reflection. Camera tile:
          motion matches the front view&apos;s screen axes.
        </span>
      </label>

      <details className="text-muted-foreground" open={inSession}>
        <summary className="cursor-pointer">
          live tuning{inSession ? "" : " (needs a running session)"}
        </summary>
        <p className="mt-2">
          The same knobs the in-headset list walks, with the ones you set once
          against a bench measurement rather than mid-take. Every value is
          <b> clamped by the backend</b> and echoed back, so a box that snaps
          to a different number is the robot telling you what it took. They
          live on the socket, not in the config file: this section is live only
          while a session is running, and the values reset with it.
          {" "}The wrist pivot is the exception — it moves the read-out point on
          your controller, which only this client can do, and it persists here.
          {" "}A knob you move is <b>yours</b> until the page reloads and is
          marked <b>◆</b>: the teleoperator&apos;s config lives per connection
          and the client reconnects after 50 ms, so without the page re-asserting
          it a socket blip would silently revert it mid-take.
        </p>
        <Button
          variant="outline"
          size="sm"
          className="mt-2"
          disabled={!dirtyTune.length}
          title="hand every knob back to the robot's own defaults"
          onClick={() => {
            dirtyTuneRef.current = new Set();
            setDirtyTune([]);
            clientRef.current?.send({ type: "request_settings" });
          }}
        >
          reset to robot defaults
        </Button>
        <div className="mt-2 grid grid-cols-2 gap-2">
          {ALL_KNOBS.map((knob) => {
            const local = Boolean(knob.local);
            const mine = dirtyTune.includes(knob.key);
            return (
              <label key={knob.key} className="flex items-center gap-2">
                <span
                  className="w-44 truncate"
                  title={mine
                    ? "yours — re-sent over the server's per-connection default on reconnect"
                    : knob.key}
                >
                  {mine ? "◆ " : ""}{knob.label}
                </span>
                <input
                  type="number"
                  step={knob.step}
                  min={knob.min}
                  max={knob.max}
                  disabled={!inSession && !local}
                  className="w-24 bg-transparent border rounded px-1 disabled:opacity-40"
                  value={tuneValues[knob.key] ?? ""}
                  placeholder={formatKnob(undefined)}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    if (e.target.value === "" || Number.isNaN(v)) return;
                    setTuning(knob.key, clampKnob(knob, v));
                  }}
                />
              </label>
            );
          })}
        </div>
      </details>
    </div>
  );
}
