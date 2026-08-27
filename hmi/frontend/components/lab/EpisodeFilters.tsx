"use client";

/**
 * What the episode list asks the server for.
 *
 * THE SORT IS SERVER-SIDE — this component reports state and never orders an
 * array. A 70-seed campaign is thousands of episodes and the list is paged, so
 * a browser sort would order one page and call it the answer.
 *
 * Mark and verdict are single-select, not sets: clicking the chip that is
 * already on clears it. `filter_mark` takes one value on the wire, and a chip
 * row that looks like a set but sends one value teaches the wrong model of the
 * filter.
 */
import { useEffect, useRef, useState } from "react";

import type { EpisodeSort, Mark, SortOrder, Verdict } from "@/lib/lab";
import { Button, Chip, MARK_COLOR, Select, TextInput, VERDICT_COLOR } from "./ui";

export type EpisodeFilterState = {
  sort: EpisodeSort;
  order: SortOrder;
  mark: Mark | null;
  verdict: Verdict | null;
  tag: string | null;
  q: string;
};

export const DEFAULT_FILTERS: EpisodeFilterState = {
  sort: "index",
  order: "asc",
  mark: null,
  verdict: null,
  tag: null,
  q: "",
};

const MARKS: readonly Mark[] = ["keep", "reject", "unset"];
const VERDICTS: readonly Verdict[] = ["PASS", "SUSPECT", "FAIL"];
const SORTS: readonly EpisodeSort[] = [
  "index", "duration", "frames", "share", "verdict", "mark", "task",
];

/** One request per keystroke over a USB tether is a list that lags the box it
 *  is typed into. Long enough to swallow a word, short enough to feel typed. */
const Q_DEBOUNCE_MS = 250;

export function EpisodeFilters({
  value,
  onChange,
  tags,
  counts = null,
}: {
  value: EpisodeFilterState;
  onChange: (v: EpisodeFilterState) => void;
  tags: string[];
  counts?: { keep: number; reject: number; unset: number } | null;
}) {
  // The box is local so typing is instant; the filter is what is debounced.
  const [q, setQ] = useState(value.q);

  // Read through a ref inside the timer: the effect must restart on the typed
  // text and on nothing else, or a parent re-render mid-word resets the clock.
  const latest = useRef({ value, onChange });
  useEffect(() => {
    latest.current = { value, onChange };
  });

  // An outside reset (clear filters, a different dataset) has to land in the
  // box. Compared during render rather than synced from an effect: an effect
  // paints the cleared filter's text for one frame, and the debounce below is
  // keyed off `q` — a frame of the old text is a frame that can fire a request
  // re-asking for the filter that was just cleared.
  const [outside, setOutside] = useState(value.q);
  if (outside !== value.q) {
    setOutside(value.q);
    setQ(value.q);
  }

  useEffect(() => {
    const id = setTimeout(() => {
      const { value: v, onChange: fire } = latest.current;
      if (q !== v.q) fire({ ...v, q });
    }, Q_DEBOUNCE_MS);
    return () => clearTimeout(id);
  }, [q]);

  const asc = value.order === "asc";

  return (
    <div className="flex shrink-0 flex-col gap-2 border-b border-border px-2.5 py-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="label-micro w-16 shrink-0 text-muted-foreground">mark</span>
        {MARKS.map((m) => (
          <Chip
            key={m}
            on={value.mark === m}
            colour={MARK_COLOR[m]}
            count={counts ? counts[m] : undefined}
            title={value.mark === m ? "show every mark" : `show only ${m}`}
            onClick={() => onChange({ ...value, mark: value.mark === m ? null : m })}
          >
            {m}
          </Chip>
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <span className="label-micro w-16 shrink-0 text-muted-foreground">verdict</span>
        {VERDICTS.map((v) => (
          <Chip
            key={v}
            on={value.verdict === v}
            colour={VERDICT_COLOR[v]}
            title={value.verdict === v ? "show every verdict" : `show only ${v}`}
            onClick={() => onChange({ ...value, verdict: value.verdict === v ? null : v })}
          >
            {v}
          </Chip>
        ))}
      </div>

      <TextInput
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="search task or note"
        aria-label="search task or note"
      />

      <div className="flex items-center gap-1.5">
        <Select
          aria-label="filter by tag"
          value={value.tag ?? ""}
          onChange={(e) => onChange({ ...value, tag: e.target.value || null })}
          className="min-w-0 flex-1"
        >
          <option value="">any tag</option>
          {tags.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </Select>

        <Select
          aria-label="sort by"
          value={value.sort}
          onChange={(e) => onChange({ ...value, sort: e.target.value as EpisodeSort })}
          className="min-w-0 flex-1"
        >
          {SORTS.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </Select>

        <Button
          aria-label={asc ? "sort ascending" : "sort descending"}
          title={asc ? "sort ascending" : "sort descending"}
          onClick={() => onChange({ ...value, order: asc ? "desc" : "asc" })}
        >
          <span aria-hidden>{asc ? "↑" : "↓"}</span>
        </Button>
      </div>
    </div>
  );
}
