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

/** The two sides of a start body: which arm each HAND drives, null for a hand
 *  that drives nothing. */
export type Pairing = { left_arm: string | null; right_arm: string | null };

/**
 * Hand↔arm pairing for a stance.
 *
 * Standing behind the arms, the operator faces the way the arms reach, so the
 * arm under their right hand is the one on frame-right of the over-shoulder
 * view — and the declared pair is taken in reverse. Face to face (mirror /
 * front) it is taken as declared. Either way "my right hand drives the arm on
 * my right" stays true, which is the property worth preserving.
 *
 * `arms` is the arm list **in config declaration order**, and the behind-stance
 * swap is positional on it — the same order both callers get from `/config`.
 * A single-arm session sends its one arm on the side of the hand that should
 * drive it, by the same rule, and null on the other.
 */
export function pairingFor(
  stance: Stance,
  arms: readonly string[],
  solo: string | null = null,
): Pairing {
  const behind = stance === "behind";
  if (solo) {
    return behind
      ? { left_arm: solo, right_arm: null }
      : { left_arm: null, right_arm: solo };
  }
  return {
    left_arm: (behind ? arms[1] : arms[0]) ?? null,
    right_arm: (behind ? arms[0] : arms[1]) ?? null,
  };
}
