"use client";

/**
 * What a multi-select is for.
 *
 * A pinned strip on `--haller-chrome`, so it reads as chrome the selection put
 * there rather than as another row of the list. It exists only while something
 * is selected: a permanently docked bar showing "0 selected" is a control that
 * spends most of its life explaining that it does nothing.
 *
 * Every control goes dead while `busy` — a bulk write is one request over a
 * tether, and a second mark sent on top of an unfinished one is how two
 * different answers race to become the stored one.
 */
import type { Mark } from "@/lib/lab";
import { Button, MARK_COLOR } from "./ui";
import { TagChips } from "./TagChips";

export function BulkBar({
  count,
  busy = false,
  knownTags,
  onMark,
  onTag,
  onUntag,
  onClear,
  onSelectAll,
}: {
  count: number;
  busy?: boolean;
  knownTags: string[];
  onMark: (m: Mark) => void;
  onTag: (t: string) => void;
  onUntag: (t: string) => void;
  onClear: () => void;
  onSelectAll?: () => void;
}) {
  if (count <= 0) return null;

  return (
    <div
      className={
        "flex shrink-0 flex-wrap items-center gap-2 border-t border-border " +
        "bg-[var(--haller-chrome)] px-2.5 py-2"
      }
    >
      <span className="label-micro shrink-0 text-muted-foreground">
        <span data-num className="font-mono tabular-nums text-foreground">
          {count}
        </span>{" "}
        selected
      </span>

      <Button
        disabled={busy}
        style={{ color: MARK_COLOR.keep }}
        onClick={() => onMark("keep")}
      >
        keep
      </Button>
      <Button
        disabled={busy}
        style={{ color: MARK_COLOR.reject }}
        onClick={() => onMark("reject")}
      >
        reject
      </Button>
      <Button disabled={busy} onClick={() => onMark("unset")}>
        unset
      </Button>

      {/* These chips are the dataset's tag vocabulary, not the selection's own
          tags: `+ tag` adds one to every selected take, `×` takes it off them. */}
      <TagChips
        tags={knownTags}
        suggestions={knownTags}
        disabled={busy}
        onAdd={onTag}
        onRemove={onUntag}
        className="flex-1"
      />

      {onSelectAll && (
        <Button disabled={busy} onClick={onSelectAll}>
          select all
        </Button>
      )}
      <Button tone="ghost" disabled={busy} onClick={onClear}>
        clear
      </Button>
    </div>
  );
}
