// hmi/frontend/lib/useCalibrationSession.ts
"use client";

/**
 * The calibration state machine, shared by the two surfaces that drive it:
 * the cockpit's Calibrate tab (in-band) and CalibrationWizard (the sheet the
 * deep-linked /settings page still opens).
 *
 * It is one hook rather than two copies because everything interesting here is
 * a safety rule, and safety rules that exist twice drift:
 *
 *   - Re-attach, don't restart. If the backend already has a session for this
 *     arm, adopt its phase; starting a second one would take torque off an arm
 *     mid-sweep.
 *   - A session belonging to a DIFFERENT arm is an error, not something to
 *     quietly take over.
 *   - The calibration block vanishing mid-session is distinct from an error
 *     string: it means the backend dropped the session (bus error, arm
 *     unplugged), nothing was written, and no amount of retrying the step will
 *     help — the fix is the cable.
 *   - Steps 2 and 3 hold work that only exists in the operator's hands (a
 *     hand-made sweep, a proposed calibration), so cancelling them confirms
 *     first. Step 1 has nothing to lose and closes instantly.
 *
 * `abortOnUnmount` is the one genuine difference between the two callers. The
 * sheet is modal: closing it IS ending the session. The cockpit tab is not —
 * an operator who glances at Cameras mid-sweep must not come back to an arm
 * that silently reverted, so there the session outlives the view and is
 * re-attached on return.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { useTelemetry } from "./telemetry";
import {
  startCalibration, captureNeutral, finishSweep, saveCalibration, abortCalibration,
  fetchCalibrationStatus,
  type CalibrationState, type JointCalibration,
} from "./calibration";
import { BACKEND_URL } from "./config";

export type ProposedMap = Record<string, JointCalibration>;

export type CalibrationSession = {
  /** null when no session is in progress on this surface. */
  armId: string | null;
  phase: CalibrationState;
  busy: boolean;
  error: string | null;
  /** Bus error reported inside the live calibration block. */
  blockError: string | null;
  /** The backend dropped the session out from under us. */
  sessionAborted: boolean;
  ticks: Record<string, number>;
  mins: Record<string, number>;
  maxes: Record<string, number>;
  joints: string[];
  proposed: ProposedMap | null;
  current: ProposedMap | null;
  confirmOpen: boolean;
  needsConfirm: boolean;
  start: (armId: string) => void;
  advance: () => void;
  attemptClose: () => void;
  keep: () => void;
  discard: () => void;
  /** Clear a backend-ended session from the view once acknowledged. */
  acknowledgeAborted: () => void;
};

export function useCalibrationSession({
  armId,
  autoStart = false,
  abortOnUnmount = false,
  onClose,
}: {
  /** Which arm this surface is driving. `null` means "no session". */
  armId: string | null;
  /** Open a session for `armId` as soon as the hook mounts (the sheet does). */
  autoStart?: boolean;
  abortOnUnmount?: boolean;
  onClose?: () => void;
}): CalibrationSession {
  const [activeArm, setActiveArm] = useState<string | null>(autoStart ? armId : null);
  const [localPhase, setLocalPhase] = useState<CalibrationState>("homing");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [proposed, setProposed] = useState<ProposedMap | null>(null);
  const [current, setCurrent] = useState<ProposedMap | null>(null);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [sawBlock, setSawBlock] = useState(false);
  const [acknowledged, setAcknowledged] = useState(false);
  const isDoneRef = useRef(false);

  const calBlock = useTelemetry((s) =>
    activeArm ? s.lastFrame?.arms?.[activeArm]?.calibration : undefined,
  );
  const stateFromTele = (calBlock?.state ?? null) as CalibrationState | null;

  // Phase only ever moves forward, taking whichever of telemetry and our own
  // optimistic advance is further along.
  //
  // Both directions of "just trust the other one" are wrong. Trusting only the
  // local phase means a session someone else advanced, or one adopted on
  // re-attach, shows the wrong step. Trusting only telemetry means the ~50 ms
  // between capture_neutral returning and the next frame carrying "sweeping"
  // redraws step 1 over the operator — a wizard that flickers backwards after
  // a confirmed action.
  const teleRank = stateFromTele ? PHASE_RANK[stateFromTele] : undefined;
  const phase: CalibrationState =
    teleRank !== undefined && teleRank > (PHASE_RANK[localPhase] ?? 0)
      ? (stateFromTele as CalibrationState)
      : localPhase;

  // Adjusted during render rather than in an effect — this only ever latches
  // false -> true, and deriving sessionAborted from it keeps the "backend
  // dropped us" signal out of a second piece of state that could disagree.
  if (calBlock && !sawBlock) setSawBlock(true);
  if (calBlock && acknowledged) setAcknowledged(false);

  const sessionAborted =
    activeArm !== null &&
    sawBlock &&
    !calBlock &&
    !acknowledged &&
    (phase === "homing" || phase === "sweeping");

  // Bootstrap: adopt an existing session, or open one.
  useEffect(() => {
    if (!autoStart || !armId) return;
    let cancelled = false;
    void (async () => {
      try {
        const status = await fetchCalibrationStatus(BACKEND_URL);
        if (cancelled) return;
        const active = status.current_session;
        if (active && active.arm_id === armId) {
          setLocalPhase(active.state);
          if (active.state === "review") {
            setProposed(active.proposed ?? null);
            setCurrent(active.current ?? null);
          }
        } else if (!active) {
          await startCalibration(BACKEND_URL, armId);
          if (!cancelled) setLocalPhase("homing");
        } else {
          setError(`Another session is active for arm '${active.arm_id}'.`);
        }
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [armId, autoStart]);

  // Modal surfaces abort on the way out; in-band ones leave the session with
  // the backend that owns it.
  const unmountArm = autoStart ? armId : null;
  useEffect(() => {
    if (!abortOnUnmount || !unmountArm) return;
    return () => {
      if (!isDoneRef.current) {
        abortCalibration(BACKEND_URL, unmountArm).catch(() => {});
      }
    };
  }, [abortOnUnmount, unmountArm]);

  const guarded = useCallback(async (fn: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, []);

  const start = useCallback((id: string) => {
    setActiveArm(id);
    setLocalPhase("homing");
    setError(null);
    setProposed(null);
    setCurrent(null);
    setSawBlock(false);
    setAcknowledged(false);
    isDoneRef.current = false;
    void (async () => {
      try {
        await startCalibration(BACKEND_URL, id);
      } catch (e) {
        setError((e as Error).message);
      }
    })();
  }, []);

  const reset = useCallback(() => {
    setActiveArm(null);
    setLocalPhase("homing");
    setProposed(null);
    setCurrent(null);
    setSawBlock(false);
    setConfirmOpen(false);
    setError(null);
  }, []);

  const advance = useCallback(() => {
    const target = activeArm ?? armId;
    if (!target) return;
    void guarded(async () => {
      if (phase === "homing") {
        await captureNeutral(BACKEND_URL, target);
        setLocalPhase("sweeping");
      } else if (phase === "sweeping") {
        const res = await finishSweep(BACKEND_URL, target);
        setProposed(res.proposed);
        setCurrent(res.current);
        setLocalPhase("review");
      } else if (phase === "review") {
        await saveCalibration(BACKEND_URL, target);
        isDoneRef.current = true;
        if (!autoStart) reset();
        onClose?.();
      }
    });
  }, [activeArm, armId, phase, guarded, onClose, autoStart, reset]);

  const needsConfirm = phase === "sweeping" || phase === "review";

  const attemptClose = useCallback(() => {
    if (needsConfirm) {
      setConfirmOpen(true);
      return;
    }
    // Step 1 holds nothing worth a dialog.
    if (!autoStart) {
      const target = activeArm;
      if (target) abortCalibration(BACKEND_URL, target).catch(() => {});
      reset();
    }
    onClose?.();
  }, [needsConfirm, onClose, autoStart, activeArm, reset]);

  const discard = useCallback(() => {
    setConfirmOpen(false);
    if (!autoStart) {
      const target = activeArm;
      if (target) abortCalibration(BACKEND_URL, target).catch(() => {});
      reset();
    }
    onClose?.();
  }, [onClose, autoStart, activeArm, reset]);

  const acknowledgeAborted = useCallback(() => {
    setAcknowledged(true);
    if (!autoStart) reset();
  }, [autoStart, reset]);

  const ticks = (calBlock?.ticks ?? EMPTY_NUM) as Record<string, number>;
  const mins = (calBlock?.min ?? EMPTY_NUM) as Record<string, number>;
  const maxes = (calBlock?.max ?? EMPTY_NUM) as Record<string, number>;
  const joints = Object.keys(ticks).length
    ? Object.keys(ticks)
    : Object.keys(proposed ?? current ?? {});

  return {
    armId: autoStart ? armId : activeArm,
    phase,
    busy,
    error,
    blockError: calBlock?.error ?? null,
    sessionAborted,
    ticks,
    mins,
    maxes,
    joints,
    proposed,
    current,
    confirmOpen,
    needsConfirm,
    start,
    advance,
    attemptClose,
    keep: () => setConfirmOpen(false),
    discard,
    acknowledgeAborted,
  };
}

const EMPTY_NUM: Record<string, number> = {};

/** Only the three phases a session actually walks through are ranked. `done`
 *  and `aborted` are terminal and handled by the session ending, not by
 *  advancing the step display. */
const PHASE_RANK: Partial<Record<CalibrationState, number>> = {
  homing: 0,
  sweeping: 1,
  review: 2,
};

/** Copy for the confirm dialog. Naming what is lost is the whole point of
 *  asking, so the two steps say different things. */
export function confirmCopy(phase: CalibrationState) {
  return {
    title:
      phase === "sweeping"
        ? "Discard the range-of-motion sweep?"
        : "Discard the proposed calibration?",
    body: "Closing now aborts the calibration session. The arm reverts to manual and torque is restored. Nothing is saved.",
    keep: phase === "sweeping" ? "Keep going" : "Keep reviewing",
  };
}

/** Why calibration cannot start right now, or undefined if it can. */
export function calibrationBlockedReason(
  allManual: boolean,
  sessionInProgress: boolean,
): string | undefined {
  if (!allManual) return "Put every arm in manual first";
  if (sessionInProgress) return "Another calibration is in progress";
  return undefined;
}
