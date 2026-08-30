"use client";

/**
 * The child process's stdout, as a tail.
 *
 * PURE PRESENTATION: this component owns no timer and fetches nothing. The
 * only poll on the Train page lives in `RunDetail`, which appends to the
 * string handed down here. One component polling and another rendering is
 * what keeps "how often do we ask" a question with a single answer.
 *
 * The sticky-scroll rule is the whole design. A tail that always jumps to the
 * bottom is unreadable the moment something goes wrong, because the traceback
 * the operator scrolled up to read is yanked away every two seconds. So the
 * stickiness is sampled from the operator's own scrolling — within 30px of the
 * bottom means "following", anything else means "reading" — and is honoured
 * across the text change.
 */
import { useLayoutEffect, useRef, useState } from "react";

import { Button } from "@/components/lab/ui";

/** How close to the bottom still counts as following. A couple of lines of
 *  slack: an exact comparison loses the tail to a sub-pixel scroll height. */
const STICK_PX = 30;

/** A finished 200k-step run's log is megabytes. The tail is a tail — the head
 *  of it is on disk and this pane was never where you read it. */
const MAX_CHARS = 200_000;

export function RunLogTail({
  text,
  height = 240,
  fill = false,
}: {
  text: string;
  height?: number;
  /** Take the whole height the parent gives instead of a fixed strip. For a
   *  run whose log IS the reading — a rollout logs no metrics and writes no
   *  checkpoints, so its stdout is the only account of what happened — the
   *  240px strip put a traceback behind a scrollbar on a full-height column. */
  fill?: boolean;
}) {
  const ref = useRef<HTMLPreElement>(null);
  /** Sampled on scroll, so at the moment the text changes it still holds the
   *  answer for BEFORE the change. That ordering is the point. */
  const stuck = useRef(true);
  const [atEnd, setAtEnd] = useState(true);

  const trimmed = text.length > MAX_CHARS;
  const body = trimmed ? text.slice(text.length - MAX_CHARS) : text;

  // After the commit and before the paint: the operator never sees the tail
  // sitting one screen short of the new output.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || !stuck.current) return;
    el.scrollTop = el.scrollHeight;
  }, [body]);

  const onScroll = () => {
    const el = ref.current;
    if (!el) return;
    const near = el.scrollHeight - el.scrollTop - el.clientHeight <= STICK_PX;
    stuck.current = near;
    setAtEnd((v) => (v === near ? v : near));
  };

  const jump = () => {
    const el = ref.current;
    if (!el) return;
    stuck.current = true;
    setAtEnd(true);
    el.scrollTop = el.scrollHeight;
  };

  return (
    <div className={"relative " + (fill ? "min-h-0 flex-1" : "")}>
      <pre
        ref={ref}
        onScroll={onScroll}
        tabIndex={0}
        aria-label="run log tail"
        style={fill ? undefined : { height }}
        className={
          "overflow-y-auto rounded-md border border-border bg-[var(--haller-inset)] " +
          "p-2 font-mono text-[10px] leading-[1.5] break-all whitespace-pre-wrap " +
          (fill ? "h-full " : "") +
          (body ? "" : "text-muted-foreground")
        }
      >
        {trimmed && (
          <span className="text-muted-foreground opacity-70">
            … earlier output trimmed{"\n"}
          </span>
        )}
        {body || "no output yet"}
      </pre>

      {/* Only offered when the operator has actually scrolled away — a button
          that is always there is a button that always looks like a warning. */}
      {!atEnd && (
        <Button
          onClick={jump}
          aria-label="scroll the log to the newest output"
          className="absolute right-3 bottom-2 shadow-[0_2px_10px_oklch(0_0_0/0.35)]"
        >
          jump to end
        </Button>
      )}
    </div>
  );
}
