"use client";

/**
 * One take in the triage list.
 *
 * Two lines, three when the grader had something to say. The list is scrolled
 * through 70 episodes at a time, so every element here has to earn its height:
 * what it is called, what the grader thought, whether the trainer will ever see
 * it, how long it ran, and the two decisions the operator makes about it.
 */
import { useState } from "react";

import type { LabEpisode, Mark } from "@/lib/lab";
import { Chip, MARK_COLOR, VerdictTag } from "./ui";
import { TagChips } from "./TagChips";

/** `12.4s · 3%` — how long it ran and how much of the corpus it is. A take
 *  that is a third of the frames dominates whatever is trained on it. */
function reading(ep: LabEpisode): string {
  const d = Number.isFinite(ep.duration_s) ? `${ep.duration_s.toFixed(1)}s` : "—";
  const s = Number.isFinite(ep.share) ? `${(ep.share * 100).toFixed(0)}%` : "—";
  return `${d} · ${s}`;
}

export function EpisodeRow({
  episode,
  selected,
  checked,
  inEval,
  onSelect,
  onToggleSelect,
  onMark,
  onNote,
}: {
  episode: LabEpisode;
  selected: boolean;
  checked: boolean;
  inEval: boolean;
  onSelect: () => void;
  onToggleSelect: (shiftKey: boolean) => void;
  onMark: (m: Mark) => void;
  onNote: (note: string) => void;
}) {
  const stored = episode.note ?? "";
  const [note, setNote] = useState(stored);
  // The server owns the note; a re-read after a bulk edit has to land in the
  // box. Compared during render rather than synced from an effect: an effect
  // paints the superseded text for one frame first, and this box is being
  // typed into.
  const [shown, setShown] = useState({ index: episode.index, note: stored });
  if (shown.index !== episode.index || shown.note !== stored) {
    setShown({ index: episode.index, note: stored });
    setNote(stored);
  }

  const rejected = episode.mark === "reject";
  // Struck through, never dimmed away: a rejected take is still the one whose
  // note explains why, and its controls stay at full contrast so the decision
  // can be taken back.
  const struck = rejected ? "line-through opacity-55" : "";
  const reasons = (episode.reasons ?? []).join(" · ");

  const commitNote = () => {
    const next = note.trim();
    if (next !== (episode.note ?? "")) onNote(next);
  };

  return (
    <div
      onClick={onSelect}
      data-selected={selected || undefined}
      className={
        "grid grid-cols-[18px_minmax(0,1fr)] items-start gap-2 border-b border-border " +
        "px-2.5 py-1.5 " +
        (selected
          ? "bg-secondary shadow-[inset_3px_0_0_var(--haller-live)]"
          : "hover:bg-muted")
      }
    >
      {/* The row only reports the modifier — the LIST owns the range maths,
          because a range is a fact about the order the server returned. */}
      <input
        type="checkbox"
        checked={checked}
        readOnly
        aria-label={`select episode ${episode.label}`}
        onClick={(e) => {
          e.stopPropagation();
          onToggleSelect(e.shiftKey);
        }}
        className="mt-1 h-3.5 w-3.5 shrink-0 accent-[var(--haller-live)]"
      />

      <div className="flex min-w-0 flex-col gap-1">
        <div className="grid grid-cols-[92px_minmax(0,1fr)_88px] items-center gap-2">
          {/* Oscar counts episodes from 1 in conversation and they are stored
              from 0. That off-by-one is how the wrong demo gets deleted, so
              both numbers are on screen at all times — the label leads, the
              stored index sits next to it. */}
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onSelect();
            }}
            aria-current={selected ? "true" : undefined}
            title={`episode ${episode.label} · stored index ${episode.index}`}
            className="flex min-w-0 items-baseline gap-1 text-left"
          >
            <span className={"font-mono text-[11px] font-semibold " + struck}>
              Ep {episode.label}
            </span>
            <span className="font-mono text-[9px] text-muted-foreground">
              idx {episode.index}
            </span>
          </button>

          <div className="flex min-w-0 items-center gap-1.5">
            <VerdictTag verdict={episode.verdict} />
            {inEval && (
              <Chip
                aria-pressed={undefined}
                tabIndex={-1}
                colour="var(--haller-manual)"
                on
                title="held out for eval loss — the policy never sees this one"
              >
                val
              </Chip>
            )}
            <span
              className={"min-w-0 truncate text-[10px] text-muted-foreground " + struck}
              title={episode.task ?? undefined}
            >
              {episode.task ?? "—"}
            </span>
          </div>

          <span
            data-num
            className="text-right font-mono text-[10px] tabular-nums text-muted-foreground"
          >
            {reading(episode)}
          </span>
        </div>

        <div className="flex min-w-0 items-center gap-1.5">
          {/* Clicking the mark that is already on returns the take to `unset` —
              "I have not judged this" is a state the operator can get back to. */}
          <Chip
            on={episode.mark === "keep"}
            colour={MARK_COLOR.keep}
            title="keep this take"
            onClick={() => onMark(episode.mark === "keep" ? "unset" : "keep")}
          >
            keep
          </Chip>
          <Chip
            on={episode.mark === "reject"}
            colour={MARK_COLOR.reject}
            title="reject this take"
            onClick={() => onMark(episode.mark === "reject" ? "unset" : "reject")}
          >
            reject
          </Chip>

          <TagChips tags={episode.tags ?? []} />

          <input
            value={note}
            onChange={(e) => setNote(e.target.value)}
            onBlur={commitNote}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                e.currentTarget.blur();
              }
            }}
            // Typing must not re-select the row out from under the player.
            onClick={(e) => e.stopPropagation()}
            placeholder="note"
            aria-label={`note for episode ${episode.label}`}
            className={
              "h-5.5 min-w-0 flex-1 rounded-sm border border-input bg-background px-1.5 " +
              "font-mono text-[10px]"
            }
          />
        </div>

        {reasons && (
          <span
            className={"truncate font-mono text-[10px] text-muted-foreground " + struck}
            title={reasons}
          >
            {reasons}
          </span>
        )}
      </div>
    </div>
  );
}
