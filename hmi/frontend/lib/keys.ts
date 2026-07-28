// hmi/frontend/lib/keys.ts

/**
 * True when a key event came from somewhere the operator is typing.
 *
 * This is the fix for a live bug: BasePanel bound WASD to `window` with no
 * target check while RecordingPanel put a task-description input on the same
 * page, so typing "a red cube" drove the base sideways.
 *
 * `select` is on the list because a focused <select> consumes letter keys for
 * type-ahead — the arm-assignment dropdowns on the teleop tab are exactly that,
 * and "s" there should pick an option, not command a robot.
 *
 * Use this on key-DOWN only. Key-up must fire unconditionally: if a key goes
 * down on the page and comes up after focus has moved into a field, the key
 * still has to lift, or the drive latches on.
 */
export function isEditableTarget(t: EventTarget | null): boolean {
  if (!(t instanceof HTMLElement)) return false;
  const tag = t.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
  // `=== true`, not a bare truthiness check: HTMLElement.isContentEditable is
  // typed boolean but is not universally implemented (jsdom leaves it
  // undefined), and returning undefined from a function a keydown handler
  // branches on is how you end up with a guard that is neither on nor off.
  return t.isContentEditable === true;
}
