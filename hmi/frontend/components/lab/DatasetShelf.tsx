"use client";

/**
 * The campaign view of what is on disk.
 *
 * `/lab/datasets` is the only call here, and it is the one place the operator
 * sees every recorded corpus side by side rather than one repo at a time
 * behind a `<select>`. A card is a shortcut into review, so the whole card is
 * the button — a card whose only keyboard path is a nested "open" link is a
 * card the keyboard cannot reach.
 *
 * Nothing on a card is derived from a channel or a camera count: `rig` is a
 * fact the backend reads out of `info.json`, and the two datasets on this disk
 * (6-channel solo, 12-channel bimanual) differ in every other number.
 *
 * Backups are shown, not hidden. A prune leaves the pre-prune copy behind, and
 * an operator who cannot see it cannot delete it — but training on it silently
 * undoes the prune, so the card says so and wears the warn colour.
 *
 * A card also says when the backend cannot tell what unit the numbers in a
 * corpus are in. It carries three scalars for that and not the reasons
 * (`catalog._units_summary`, because this endpoint is polled), and it is on
 * the card rather than one page deeper because a foreign dataset and one of
 * ours are indistinguishable by inspection: the warning has to be readable
 * before the dataset is opened, marked, or trained on.
 */
import { useCallback, useEffect, useState } from "react";

import {
  isMissing,
  lab,
  reason,
  rigLabel,
  unitsAlert,
  type DatasetSummary,
} from "@/lib/lab";
import {
  Button,
  Chip,
  Empty,
  MarkBar,
  Panel,
  PanelHead,
  Refusal,
  Stat,
} from "@/components/lab/ui";
import { fmtBytes, fmtDuration } from "@/components/lab/charts/svg";

const BACKUP_NOTE = "backup left by a prune — not a training source";

export function DatasetShelf({
  onOpen,
  selected = null,
  layout = "grid",
  refreshKey = 0,
}: {
  onOpen: (repoId: string) => void;
  selected?: string | null;
  layout?: "strip" | "grid";
  /** Bumped by whatever just changed the disk — a prune, a delete, a take
   *  that stopped. The shelf has no way to know from here. */
  refreshKey?: number;
}) {
  const strip = layout === "strip";
  const [list, setList] = useState<DatasetSummary[] | null>(null);
  /** The build predates the Lab. Not an error and not empty — a third state,
   *  because "no /lab routes" and "nothing recorded" are different fixes. */
  const [missing, setMissing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [epoch, setEpoch] = useState(0);

  const refresh = useCallback(() => setEpoch((n) => n + 1), []);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const r = await lab.datasets();
        if (cancelled) return;
        setList(r.datasets);
        setMissing(false);
        setError(null);
      } catch (e) {
        if (cancelled) return;
        setList([]);
        setMissing(isMissing(e));
        setError(isMissing(e) ? null : reason(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [epoch, refreshKey]);

  const cards = list ?? [];

  return (
    <Panel className={strip ? "h-full" : ""}>
      <PanelHead
        title="datasets on disk"
        right={list === null ? "reading…" : `${cards.length} on disk`}
      >
        <Button
          tone="ghost"
          onClick={refresh}
          aria-label="refresh dataset list"
          title="re-read /lab/datasets"
          className="ml-auto"
        >
          ↻
        </Button>
      </PanelHead>

      {missing ? (
        <Empty>this backend has no lab — dataset browsing needs the /lab routes</Empty>
      ) : error ? (
        <div className="p-2">
          <Refusal>{error}</Refusal>
        </div>
      ) : list === null ? (
        <Empty>reading…</Empty>
      ) : cards.length === 0 ? (
        <Empty>nothing recorded yet — start a take in collect</Empty>
      ) : (
        <div
          className={
            "min-h-0 flex-1 p-1.5 " +
            (strip
              ? "flex gap-1.5 overflow-x-auto"
              : "grid content-start gap-2 overflow-y-auto " +
                "[grid-template-columns:repeat(auto-fill,minmax(19rem,1fr))]")
          }
        >
          {cards.map((d) => (
            <DatasetCard
              key={d.repo_id}
              d={d}
              strip={strip}
              selected={d.repo_id === selected}
              onOpen={onOpen}
            />
          ))}
        </div>
      )}
    </Panel>
  );
}

/* ---- one dataset -------------------------------------------------------- */

function DatasetCard({
  d,
  strip,
  selected,
  onOpen,
}: {
  d: DatasetSummary;
  strip: boolean;
  selected: boolean;
  onOpen: (repoId: string) => void;
}) {
  const rig = rigLabel(d.rig);
  // A card is polled, so it gets the three scalars and not the sentence: the
  // label is the whole warning here, and the note it hovers is this file's own
  // copy. What it is warning about is that this corpus's numbers may not be
  // this robot's degrees, and every threshold and verdict drawn from them
  // moves silently if they are read as degrees anyway.
  const units = unitsAlert(d.units);
  return (
    <button
      type="button"
      onClick={() => onOpen(d.repo_id)}
      aria-current={selected ? "true" : undefined}
      aria-label={
        `open ${d.repo_id} — ${d.episodes} episode${d.episodes === 1 ? "" : "s"}, ` +
        rig + (d.is_backup ? ", " + BACKUP_NOTE : "") +
        (units ? ", " + units.label : "")
      }
      title={
        // Two lines rather than one: the chips inside the card are
        // pointer-events-none so the card owns every tooltip, and a units
        // warning must not cost the repo-id the operator hovers for.
        [d.is_backup ? BACKUP_NOTE : d.repo_id, units?.note]
          .filter(Boolean).join("\n")
      }
      className={
        "flex min-w-0 flex-col gap-1 overflow-hidden rounded-md border border-border " +
        "bg-[var(--haller-inset)] p-1.5 text-left transition-colors " +
        "hover:border-[var(--haller-rail)] " +
        (strip ? "h-full w-[19rem] shrink-0 " : "") +
        (d.is_backup ? "opacity-50 " : "") +
        (selected ? "shadow-[0_0_0_1px_var(--haller-live)] " : "")
      }
    >
      <div className="flex min-w-0 items-center gap-1.5">
        <span
          className={
            "min-w-0 flex-1 font-mono text-[11px] leading-[1.2] font-semibold break-all " +
            (strip ? "line-clamp-1" : "line-clamp-2")
          }
        >
          {d.repo_id}
        </span>
        {/* Decorative inside the card button: pointer-events pass through to
            the card and the chips are out of the tab order, so the card stays
            one control rather than three. */}
        {d.is_backup && (
          <Chip
            on
            colour="var(--haller-warn)"
            tabIndex={-1}
            className="pointer-events-none"
          >
            backup
          </Chip>
        )}
        {units && (
          <Chip
            on
            colour="var(--haller-warn)"
            tabIndex={-1}
            className="pointer-events-none"
            data-units-alert
          >
            {units.label}
          </Chip>
        )}
        <Chip tabIndex={-1} className="pointer-events-none">
          {rig}
        </Chip>
      </div>

      {/* A backup's task is its source's task; the consequence outranks it. */}
      <span
        className={
          "truncate text-[10px] leading-[1.2] " +
          (d.is_backup ? "text-[var(--haller-warn)]" : "text-muted-foreground")
        }
        title={d.task ?? undefined}
      >
        {d.is_backup ? BACKUP_NOTE : (d.task ?? "no task recorded")}
      </span>

      <div className="mt-auto flex flex-wrap items-baseline gap-x-2.5 gap-y-0.5">
        <Stat label="ep" value={d.episodes} />
        <Stat label="len" value={fmtDuration(d.duration_s)} />
        <Stat label="size" value={fmtBytes(d.size_bytes)} />
      </div>

      <div className="flex flex-col gap-0.5">
        <MarkBar marks={d.marks} />
        <span className="label-micro text-muted-foreground">
          {d.marks.keep} keep · {d.marks.reject} reject · {d.marks.unset} unset
        </span>
      </div>
    </button>
  );
}
