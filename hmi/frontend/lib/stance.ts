// hmi/frontend/lib/stance.ts

/**
 * Operator stance, and the one rule that turns it into a hand↔arm pairing.
 *
 * Where the operator is standing relative to the arms is not a display
 * preference: it decides which physical arm the right hand drives. Getting it
 * wrong does not look like a setting that is off, it looks like the controls
 * are inverted — so there is exactly one implementation of the rule, here,
 * and both the cockpit launcher and the in-headset panel read it.
 *
 * The stance itself is also sent per-frame on the teleop socket, where the
 * backend uses it to pick the matching hand rotation. This module owns only
 * the desktop-side half: the persisted choice and the pairing.
 */

import { useSyncExternalStore } from "react";

export type Stance = "behind" | "mirror" | "front";

/** Shared with the in-headset panel: both surfaces remember one stance, so an
 *  operator who picks it in the cockpit and then opens the headset page in the
 *  same browser does not get asked twice. (Different browser — the Quest —
 *  keeps its own; the stance is a property of where the operator stands.) */
export const STANCE_LS_KEY = "haller.vrTeleop.stance.v1";

export const STANCES: readonly { id: Stance; label: string; hint: string }[] = [
  {
    id: "behind",
    label: "behind",
    hint: "egocentric — you stand behind the arms and they reach the way you do",
  },
  {
    id: "mirror",
    label: "mirror",
    hint: "face to face — the arm moves as your reflection",
  },
  {
    id: "front",
    label: "front",
    hint: "face to face, no reflection — the arm reaches toward you",
  },
];

export function isStance(v: unknown): v is Stance {
  return v === "behind" || v === "mirror" || v === "front";
}

/** Defaults to `behind`, which is also the backend's default for a frame that
 *  carries no stance — the two must not disagree. */
export function readStance(): Stance {
  if (typeof window === "undefined") return "behind";
  try {
    const raw = window.localStorage.getItem(STANCE_LS_KEY);
    return isStance(raw) ? raw : "behind";
  } catch {
    return "behind";
  }
}

export function writeStance(s: Stance): void {
  try {
    window.localStorage.setItem(STANCE_LS_KEY, s);
  } catch {
    /* private mode — the choice just won't survive a reload */
  }
  for (const notify of listeners) notify();
}

/* --- the stance as an external store -------------------------------------
 *
 * localStorage is exactly the kind of thing useSyncExternalStore exists for:
 * it is outside React, it is not readable during the server render, and a
 * second tab can change it. Subscribing to `storage` means the headset page
 * and the cockpit, open side by side in one browser, never disagree about
 * where the operator is standing.
 *
 * `getSnapshot` returning a fresh read is safe here because a Stance is a
 * string — equal snapshots are Object.is-equal, so React sees no change.
 */
const listeners = new Set<() => void>();

function subscribe(notify: () => void): () => void {
  listeners.add(notify);
  window.addEventListener("storage", notify);
  return () => {
    listeners.delete(notify);
    window.removeEventListener("storage", notify);
  };
}

/** "behind" on the server: the same default readStance() falls back to, so
 *  hydration never swaps a stance the operator did not choose. */
function serverSnapshot(): Stance {
  return "behind";
}

export function useStance(): [Stance, (s: Stance) => void] {
  const stance = useSyncExternalStore(subscribe, readStance, serverSnapshot);
  return [stance, writeStance];
}

/* --- which hand drives a solo arm ----------------------------------------
 *
 * By default the stance decides (see pairingFor below) — but on a rig with
 * one arm the operator holds whichever controller is charged, and a rule
 * that cannot be overridden reads as a rig that cannot be configured. The
 * override is also the field remedy for a wrong side-identity guess: flip a
 * toggle instead of renaming arms in YAML at the bench.
 *
 * Same store shape as the stance and for the same reason: cockpit and
 * headset page in one browser must agree. null = "auto, let the stance
 * decide", which is byte-for-byte the pre-override behavior.
 */
export type SoloHand = "left" | "right" | null;

export const SOLO_HAND_LS_KEY = "haller.vrTeleop.soloHand.v1";

export function readSoloHand(): SoloHand {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(SOLO_HAND_LS_KEY);
    return raw === "left" || raw === "right" ? raw : null;
  } catch {
    return null;
  }
}

export function writeSoloHand(h: SoloHand): void {
  try {
    if (h === null) window.localStorage.removeItem(SOLO_HAND_LS_KEY);
    else window.localStorage.setItem(SOLO_HAND_LS_KEY, h);
  } catch {
    /* private mode — the choice just won't survive a reload */
  }
  for (const notify of listeners) notify();
}

function soloHandServerSnapshot(): SoloHand {
  return null;
}

export function useSoloHand(): [SoloHand, (h: SoloHand) => void] {
  const hand = useSyncExternalStore(subscribe, readSoloHand, soloHandServerSnapshot);
  return [hand, writeSoloHand];
}

/** The two sides of a start body: which arm each HAND drives, null for a hand
 *  that drives nothing. */
export type Pairing = { left_arm: string | null; right_arm: string | null };

/**
 * The robot's own left and right, resolved from the arm ids.
 *
 * Identity first: an id containing "left" or "right" names the side the arm is
 * bolted to, whatever order the config happens to declare it in. This is not
 * pedantry — `config.yaml` declares `[right, left]` while every sim config
 * declares `[left, right]`, so a rule that reads position instead of identity
 * makes the same stance mean opposite things on the two.
 *
 * Ids that name no side fall back to declaration order, first = the robot's
 * left. The left slot is filled first, so a lone unnamed arm is the left one.
 */
export function sidesOf(arms: readonly string[]): {
  robotLeft: string | null;
  robotRight: string | null;
} {
  let robotLeft = arms.find((a) => /left/i.test(a)) ?? null;
  let robotRight =
    arms.find((a) => /right/i.test(a) && a !== robotLeft) ?? null;
  const spare = arms.filter((a) => a !== robotLeft && a !== robotRight);
  if (robotLeft === null) robotLeft = spare.shift() ?? null;
  if (robotRight === null) robotRight = spare.shift() ?? null;
  return { robotLeft, robotRight };
}

/** Behind the arms the operator faces the way the arms reach, so the robot's
 *  LEFT arm is the one on their right — the sides cross. Facing the arms
 *  (mirror / front) they line up. Either way "my right hand drives the arm on
 *  my right" stays true, which is the property worth preserving. */
function forStance(
  stance: Stance,
  sides: { robotLeft: string | null; robotRight: string | null },
): Pairing {
  return stance === "behind"
    ? { left_arm: sides.robotRight, right_arm: sides.robotLeft }
    : { left_arm: sides.robotLeft, right_arm: sides.robotRight };
}

/**
 * Hand↔arm pairing for a stance.
 *
 * A solo session is the dual pairing with the other hand emptied, deliberately:
 * "my right hand drives the arm I picked" then means the same thing whether
 * one arm is in the session or two. Picking the robot's left arm in the behind
 * stance puts it under the RIGHT hand, exactly as it would bimanually.
 *
 * `hand` overrides that inheritance for a solo session: "this hand, whatever
 * the stance thinks". Ignored (deliberately) for dual — a dual session with
 * both arms on one hand is not a thing this can express.
 */
export function pairingFor(
  stance: Stance,
  arms: readonly string[],
  solo: string | null = null,
  hand: SoloHand = null,
): Pairing {
  if (solo && hand) {
    return hand === "left"
      ? { left_arm: solo, right_arm: null }
      : { left_arm: null, right_arm: solo };
  }
  const dual = forStance(stance, sidesOf(arms));
  if (!solo) return dual;
  if (dual.left_arm === solo) return { left_arm: solo, right_arm: null };
  if (dual.right_arm === solo) return { left_arm: null, right_arm: solo };
  // An arm the resolved pair does not cover — a third arm, or ids the identity
  // rule could not place. Resolve it as a rig of its own rather than hand the
  // backend two nulls, which it refuses.
  return forStance(stance, sidesOf([solo]));
}
