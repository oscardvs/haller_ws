"use client";

/**
 * Tags, as chips.
 *
 * One component for two jobs, and the handlers are what separate them: in an
 * episode row it is given neither `onAdd` nor `onRemove` and is a label; in the
 * bulk bar it is given both and becomes the tag editor for the whole selection.
 * Read-only chips are taken out of the tab order — a button that does nothing
 * is a stop on the way to the controls that do.
 *
 * Tags are lowercased and trimmed on commit. `Grasp ` and `grasp` filtering as
 * two different tags is a split corpus that looks like a typo and reads like a
 * missing dataset.
 */
import { useId, useRef, useState } from "react";

import { Chip } from "./ui";

export function TagChips({
  tags,
  onRemove,
  onAdd,
  suggestions = [],
  disabled = false,
  className = "",
}: {
  tags: string[];
  onRemove?: (t: string) => void;
  onAdd?: (t: string) => void;
  suggestions?: string[];
  disabled?: boolean;
  className?: string;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  /** Escape has to beat the blur it causes. Without this flag the cancel path
   *  commits the text the operator just abandoned. */
  const cancelled = useRef(false);
  // useId is punctuated (React 19 wraps its ids in guillemets); strip it so the
  // value is a plain id a `list=` attribute can point at.
  const listId = "tags-" + useId().replace(/[^a-zA-Z0-9_-]/g, "");

  const list = tags ?? [];

  const commit = (raw: string) => {
    const tag = raw.trim().toLowerCase();
    setDraft("");
    setEditing(false);
    if (tag) onAdd?.(tag);
  };
  const cancel = () => {
    cancelled.current = true;
    setDraft("");
    setEditing(false);
  };

  if (list.length === 0 && !onAdd) return null;

  return (
    <span className={"flex min-w-0 flex-wrap items-center gap-1 " + className}>
      {list.map((t) =>
        onRemove ? (
          <Chip
            key={t}
            aria-pressed={undefined}
            aria-label={`remove tag ${t}`}
            title={`remove tag ${t}`}
            disabled={disabled}
            onClick={(e) => {
              e.stopPropagation();
              onRemove(t);
            }}
          >
            {t}
            <span aria-hidden className="opacity-60">
              ×
            </span>
          </Chip>
        ) : (
          <Chip key={t} aria-pressed={undefined} tabIndex={-1} title={t}>
            {t}
          </Chip>
        ),
      )}

      {onAdd &&
        (editing ? (
          <>
            <input
              autoFocus
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onClick={(e) => e.stopPropagation()}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  commit(draft);
                } else if (e.key === "Escape") {
                  e.preventDefault();
                  e.stopPropagation();
                  cancel();
                }
              }}
              onBlur={() => {
                if (cancelled.current) {
                  cancelled.current = false;
                  return;
                }
                commit(draft);
              }}
              aria-label="new tag"
              placeholder="tag"
              list={suggestions.length > 0 ? listId : undefined}
              disabled={disabled}
              className={
                "h-5.5 w-24 min-w-0 rounded-full border border-input bg-background px-2 " +
                "font-mono text-[10px] disabled:opacity-50"
              }
            />
            {suggestions.length > 0 && (
              <datalist id={listId}>
                {suggestions.map((s) => (
                  <option key={s} value={s} />
                ))}
              </datalist>
            )}
          </>
        ) : (
          <Chip
            aria-pressed={undefined}
            aria-label="add tag"
            title="add tag"
            disabled={disabled}
            onClick={(e) => {
              e.stopPropagation();
              setEditing(true);
            }}
          >
            + tag
          </Chip>
        ))}
    </span>
  );
}
