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
import { Button, Chip, HeadRow, Panel, PanelHead, Refusal } from "@/components/lab/ui";

/** With the roll-out column and without it. A caller that cannot run a policy
 *  — anything but the Train tab's detail pane today — gets the three-column
 *  table it always had rather than an empty gutter. */
const COLS = "minmax(0,1fr) 56px 84px";
const COLS_ROLLOUT = "minmax(0,1fr) 56px 84px 72px";

/** The CHECKPOINT directory's name — `060000`, or `last` for the symlink.
 *
 *  The backend sends the path of the model directory INSIDE it
 *  (`.../checkpoints/060000/pretrained_model`), because that is what a rollout
 *  is pointed at. So the last segment is `pretrained_model` on every row and
 *  taking it named all thirteen identically; the step directory is its parent.
 *  Falls back to the last segment for any path not in that shape rather than
 *  rendering empty. */
export function checkpointName(path: string): string {
  const parts = path.replace(/\/+$/, "").split("/").filter(Boolean);
  const last = parts[parts.length - 1] ?? path;
  if (last !== "pretrained_model") return last;
  return parts[parts.length - 2] ?? last;
}

/** `060000 · step 60000`, or just `last` for the symlink, whose step is null
 *  on purpose. Beside `checkpointName` because it is the same question asked
 *  one level up — how a checkpoint is SPOKEN OF — and the launcher and its
 *  button must not answer it two different ways. */
export function stepLabel(c: Checkpoint): string {
  const name = checkpointName(c.path);
  return c.step !== null ? `${name} · step ${c.step}` : name;
}

export function CheckpointList({
  runId,
  status,
  onRollout,
}: {
  runId: string;
  status: RunStatus;
  /**
   * Run this checkpoint on the arms. Absent means no column: this list is
   * also a plain readout of what is on disk, and a surface that cannot launch
   * must not show a button that does nothing.
   *
   * Raised rather than handled here — the launcher needs the RUN's dataset to
   * know whether it has to ask which arm, and this component knows only the
   * run id. The owner has the run.
   */
  onRollout?: (checkpoint: Checkpoint) => void;
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
  const cols = onRollout ? COLS_ROLLOUT : COLS;
  // `is_link` left the frozen contract, so "latest" is derived from the
  // highest step rather than from a symlink the backend no longer reports.
  // Highest step IS the newest checkpoint; nothing else about the set is
  // being guessed.
  // `last` carries a null step, so it is skipped rather than compared: it is
  // a POINTER at the newest numbered checkpoint, and letting it win would put
  // the badge on the alias instead of the thing it aliases — and with every
  // step null, `c.step === latest` would be `null === null` on every row.
  const latest = rows.reduce<number | null>(
    (best, c) =>
      c.step !== null && (best === null || c.step > best) ? c.step : best,
    null,
  );

  return (
    <Panel className="shrink-0">
      <PanelHead
        title="checkpoints"
        right={
          rows.length > 0
            ? `${rows.length} on disk`
            : /* The empty state IS the head. A run 6% in has written no
                 checkpoint yet and that is not news worth 90px of a column the
                 charts are queueing for — it was `Empty`'s full scanline
                 treatment under a live run, every run, for the first hour. */
              list === null && !error
                ? "reading…"
                : status === "running"
                  ? "none yet"
                  : "none"
        }
      />

      {error && (
        <div className="p-2.5">
          <Refusal>{error}</Refusal>
        </div>
      )}

      {rows.length === 0 ? null : (
        /* Viewport-relative rather than 168px: on a train run this list is the
           thing you came for — it is what a rollout is launched from — and a
           fixed cap showed four of thirteen under an empty log panel that had
           taken the rest of the column. The log still gets whatever is left,
           because this panel takes only what its rows need. */
        <div className="max-h-[34vh] min-h-0 overflow-y-auto">
          <HeadRow
            style={{ gridTemplateColumns: cols }}
            cols={[
              { key: "name", label: "name" },
              { key: "step", label: "step", align: "right" },
              { key: "state", label: "state", align: "right" },
              ...(onRollout ? [{ key: "run", label: "run", align: "right" as const }] : []),
            ]}
          />
          {rows.map((c) => (
            <div
              key={c.path}
              title={c.path}
              className="grid items-center gap-2 border-b border-border px-2.5 py-1"
              style={{ gridTemplateColumns: cols }}
            >
              <span className="truncate font-mono text-[10px]">{checkpointName(c.path)}</span>
              <span data-num className="text-right font-mono text-[10px] tabular-nums">
                {c.step ?? "—"}
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
                {c.step !== null && c.step === latest && c.has_model && (
                  <Chip on tabIndex={-1} title="the newest checkpoint this run wrote">
                    latest
                  </Chip>
                )}
              </span>
              {/* A partial checkpoint gets no button at all rather than a
                  disabled one: there is nothing to enable it, and the `partial`
                  chip beside it already says why. */}
              {onRollout && (
                <span className="flex justify-end">
                  {c.has_model && (
                    <Button
                      tone="ghost"
                      onClick={() => onRollout(c)}
                      title={`run ${checkpointName(c.path)} on the arms`}
                      aria-label={`roll out ${checkpointName(c.path)}`}
                    >
                      roll out
                    </Button>
                  )}
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}
