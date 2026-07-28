"use client";

/** Shared vocabulary for the cockpit shell. Everything here is presentation
 *  glue — no fetching, no robot semantics. */
import { useCallback, useEffect, useState } from "react";

export type TabId =
  | "operate"
  | "human"
  | "calibrate"
  | "cameras"
  | "dataset"
  | "settings";

/** The nav, in order. Declared separately from TabId so the set of tabs the
 *  header offers can grow without every exhaustive switch in the shell having
 *  to be rewritten in the same commit. */
export const TABS: readonly { id: TabId; label: string }[] = [
  { id: "operate", label: "Operate" },
  { id: "cameras", label: "Cameras" },
  { id: "settings", label: "Settings" },
];

/** Popover slot. The shell allows exactly one open at a time — the design
 *  models this as a single nullable slot and it is worth keeping: two stacked
 *  popovers over a control surface is how you click the wrong thing. */
export type PopId = "teleop" | "rec" | "alerts" | `pose-${string}` | null;

/**
 * True when a key event came from somewhere the operator is typing.
 *
 * This is the fix for a live bug: BasePanel bound WASD to `window` with no
 * target check while RecordingPanel put a task-description input on the same
 * page, so typing "a red cube" drove the base sideways. `select` is in the
 * list because a focused <select> consumes letter keys for type-ahead — the
 * arm-assignment dropdowns on the teleop tab are exactly that.
 */
export function isEditableTarget(t: EventTarget | null): boolean {
  if (!(t instanceof HTMLElement)) return false;
  const tag = t.tagName;
  return (
    tag === "INPUT" ||
    tag === "TEXTAREA" ||
    tag === "SELECT" ||
    t.isContentEditable
  );
}

/** Breakpoints the design calls out by number, named once here.
 *  - compact: the Operate tab drops to a single arm with a picker.
 *  - short:   wrist camera tiles collapse to their label strip so the joint
 *             stack and the action row keep their height. */
export const COMPACT_W = 1180;
export const SHORT_H = 720;

export type Viewport = { w: number; h: number; compact: boolean; short: boolean };

/** Viewport size as state. Seeded with a desktop guess so the server render and
 *  the first client render agree; the real measurement lands in the effect,
 *  which runs well before the cockpit stops showing its boot screen. */
export function useViewport(): Viewport {
  const [size, setSize] = useState({ w: 1440, h: 900 });
  useEffect(() => {
    const read = () => setSize({ w: window.innerWidth, h: window.innerHeight });
    read();
    window.addEventListener("resize", read);
    return () => window.removeEventListener("resize", read);
  }, []);
  return {
    ...size,
    compact: size.w < COMPACT_W,
    short: size.h < SHORT_H,
  };
}

/** Format a number, or an em-dash when it is missing or the link is not live.
 *  Freezing the last known value in the live style would tell the operator
 *  where the robot was, dressed as where it is. */
export function num(
  n: number | null | undefined,
  live: boolean,
  digits = 2,
): string {
  if (!live || typeof n !== "number" || !Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}

/**
 * Component-local state that survives the component being unmounted.
 *
 * Tabs mount and unmount — that is what stops the Operate tab's 10 Hz cmd_vel
 * interval when you are not looking at it. But it also means the speed
 * governor and the selected camera would snap back to their defaults every
 * time the operator glanced at Cameras and came back, which is the kind of
 * small betrayal that makes a control surface feel untrustworthy.
 *
 * Hoisting these to the cockpit root would fix it and make every governor drag
 * re-render all six tabs' worth of tree. This keeps the state where it is used
 * and parks the last value in a module map, which lives exactly as long as the
 * page — a reload is a fresh session and should reset.
 */
const stickyValues = new Map<string, unknown>();

export function useSticky<T>(key: string, initial: T): [T, (v: T) => void] {
  const [value, setValue] = useState<T>(() =>
    stickyValues.has(key) ? (stickyValues.get(key) as T) : initial,
  );
  const set = useCallback(
    (next: T) => {
      stickyValues.set(key, next);
      setValue(next);
    },
    [key],
  );
  return [value, set];
}

/** Dataset repo slug. Mirrors RecordingPanel.slugify so the two agree on what
 *  a take is called, whichever surface started it. */
export function slugify(s: string): string {
  return (
    s.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 60) ||
    "task"
  );
}
