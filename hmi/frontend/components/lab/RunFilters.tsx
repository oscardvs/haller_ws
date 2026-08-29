"use client";

/**
 * Filters over the run list: one kind, one status, and a search.
 *
 * Both chip rows are single-select and clicking the active chip clears it,
 * because that is what the endpoint takes — `GET /lab/runs?kind=&status=`
 * accepts one of each, and offering a multi-select here would be a control
 * the server cannot honour.
 *
 * The status row carries four chips, not seven. `queued`, `died` and
 * `launch_failed` are real statuses and stay visible in the unfiltered list;
 * they are not triage buttons, and a row of seven chips buys nothing over
 * reading the four the operator actually sorts by.
 */
import { useEffect, useRef, useState } from "react";

import type { RunKind, RunStatus } from "@/lib/lab";
import { Chip, TextInput } from "@/components/lab/ui";
import { STATUS_COLOR } from "@/components/lab/RunList";

export type RunFilterState = { kind: RunKind | null; status: RunStatus | null; q: string };

export const DEFAULT_RUN_FILTERS: RunFilterState = { kind: null, status: null, q: "" };

const KINDS: readonly RunKind[] = ["train", "export", "prune", "rollout", "record"];
const STATUSES: readonly RunStatus[] = ["running", "done", "failed", "stopped"];

/** Typing re-queries the server, so the box waits for a pause rather than
 *  firing a request per keystroke. */
const DEBOUNCE_MS = 250;

export function RunFilters({
  value,
  onChange,
}: {
  value: RunFilterState;
  onChange: (v: RunFilterState) => void;
}) {
  const [text, setText] = useState(value.q);
  /** The last `q` this component put on the wire. It is the seam between the
   *  box and the prop: without it, an in-flight debounce and an external reset
   *  fight over the field and the operator watches their own typing vanish. */
  const emitted = useRef(value.q);
  const latest = useRef({ value, onChange });
  useEffect(() => {
    latest.current = { value, onChange };
  });

  // An outside reset ("clear filters") lands in the box; an echo of what we
  // just sent does not.
  useEffect(() => {
    if (value.q !== emitted.current) {
      emitted.current = value.q;
      setText(value.q);
    }
  }, [value.q]);

  useEffect(() => {
    if (text === emitted.current) return;
    const t = setTimeout(() => {
      emitted.current = text;
      latest.current.onChange({ ...latest.current.value, q: text });
    }, DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [text]);

  const setKind = (k: RunKind) =>
    onChange({ ...value, kind: value.kind === k ? null : k });
  const setStatus = (s: RunStatus) =>
    onChange({ ...value, status: value.status === s ? null : s });

  return (
    <div className="flex shrink-0 flex-col gap-1.5 border-b border-border px-2.5 py-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="label-micro shrink-0 text-muted-foreground">kind</span>
        {KINDS.map((k) => (
          <Chip
            key={k}
            on={value.kind === k}
            onClick={() => setKind(k)}
            title={value.kind === k ? "show every kind" : `only ${k} runs`}
          >
            {k}
          </Chip>
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <span className="label-micro shrink-0 text-muted-foreground">status</span>
        {STATUSES.map((s) => (
          <Chip
            key={s}
            on={value.status === s}
            colour={STATUS_COLOR[s]}
            onClick={() => setStatus(s)}
            title={value.status === s ? "show every status" : `only ${s} runs`}
          >
            {s}
          </Chip>
        ))}
      </div>

      <TextInput
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="name or id"
        aria-label="search runs by name or id"
        className="h-6.5"
      />
    </div>
  );
}
