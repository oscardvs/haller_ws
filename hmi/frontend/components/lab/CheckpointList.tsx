"use client";

/**
 * What this run has actually written to disk.
 *
 * Re-read whenever `status` changes, because that is exactly when the set
 * moves: a run that has just gone `done` wrote its last checkpoint on the way
 * out, and a `failed` one is worth checking for a usable earlier step.
 *
 * A backend that has no checkpoints route renders NOTHING — not an empty
 * panel. An old build cannot answer the question, and a permanently blank card
 * saying so is noise on every run for the life of the deployment.
 */
import { useEffect, useState } from "react";

import { lab, isMissing, reason, type Checkpoint, type RunStatus } from "@/lib/lab";
import { Chip, Empty, HeadRow, Panel, PanelHead, Refusal } from "@/components/lab/ui";

const COLS = "minmax(0,1fr) 56px 84px";

/** The directory name. The full path is a title attribute — it is what you
 *  paste into a rollout, and it is far too long to be a column. */
function baseName(path: string): string {
  const parts = path.replace(/\/+$/, "").split("/");
  return parts[parts.length - 1] || path;
}

export function CheckpointList({
  runId,
  status,
}: {
  runId: string;
  status: RunStatus;
}) {
  const [list, setList] = useState<Checkpoint[] | null>(null);
  const [missing, setMissing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    // The previous error clears when the answer lands, not before it is asked
    // for: clearing up here is a setState in the effect body, and a card that
    // goes blank for the length of a request says less than the refusal it
    // just wiped.
    lab
      .checkpoints(runId)
      .then((r) => {
        if (cancelled) return;
        setList(r.checkpoints);
        setMissing(false);
        setError(null);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        if (isMissing(e)) {
          setMissing(true);
          setList(null);
          setError(null);
          return;
        }
        setError(reason(e));
      });
    return () => {
      cancelled = true;
    };
  }, [runId, status]);

  if (missing) return null;

  const rows = list ?? [];
  // `is_link` left the frozen contract, so "latest" is derived from the
  // highest step rather than from a symlink the backend no longer reports.
  // Highest step IS the newest checkpoint; nothing else about the set is
  // being guessed.
  const latest = rows.reduce<number | null>(
    (best, c) => (best === null || c.step > best ? c.step : best),
    null,
  );

  return (
    <Panel className="shrink-0">
      <PanelHead
        title="checkpoints"
        right={rows.length > 0 ? `${rows.length} on disk` : undefined}
      />

      {error && (
        <div className="p-2.5">
          <Refusal>{error}</Refusal>
        </div>
      )}

      {rows.length === 0 ? (
        <Empty>
          {list === null && !error
            ? "reading…"
            : status === "running"
              ? "no checkpoint written yet"
              : "no checkpoints"}
        </Empty>
      ) : (
        <div className="max-h-[168px] min-h-0 overflow-y-auto">
          <HeadRow
            style={{ gridTemplateColumns: COLS }}
            cols={[
              { key: "name", label: "name" },
              { key: "step", label: "step", align: "right" },
              { key: "state", label: "state", align: "right" },
            ]}
          />
          {rows.map((c) => (
            <div
              key={c.path}
              title={c.path}
              className="grid items-center gap-2 border-b border-border px-2.5 py-1"
              style={{ gridTemplateColumns: COLS }}
            >
              <span className="truncate font-mono text-[10px]">{baseName(c.path)}</span>
              <span data-num className="text-right font-mono text-[10px] tabular-nums">
                {c.step}
              </span>
              <span className="flex items-center justify-end gap-1">
                {/* A directory the runner created but never finished writing
                    would fail at load, so it is called out rather than listed
                    as if it were a usable rollout source. */}
                {!c.has_model && (
                  <Chip on colour="var(--haller-warn)" tabIndex={-1} title="no model file — this checkpoint will not load">
                    partial
                  </Chip>
                )}
                {c.step === latest && c.has_model && (
                  <Chip on tabIndex={-1} title="the newest checkpoint this run wrote">
                    latest
                  </Chip>
                )}
              </span>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}
