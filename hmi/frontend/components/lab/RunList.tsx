"use client";

/**
 * The run list: one row per job the backend has ever launched.
 *
 * Two things make this a list of EXPERIMENTS rather than a list of processes.
 * The first is `spec_summary`, printed verbatim under the name — the backend
 * words what a run was asked to do, and the list re-wording it is how two
 * surfaces end up describing the same run differently. The second is the
 * compare checkbox: the set it builds is what `/lab/compare` deep-links to, so
 * picking two runs to overlay is one click from the row you are reading, not a
 * separate mode.
 *
 * `runs` is typed `RunSummary[]` and not `Run[]` on purpose. `GET /lab/runs`
 * returns summaries; `Run` is the detail shape and is assignable to this, so a
 * caller holding either can pass it.
 */
import type { RunStatus, RunSummary } from "@/lib/lab";
import { Empty, HeadRow, Refusal } from "@/components/lab/ui";

/** Status → the cockpit's existing signal palette. `running` is MANUAL blue,
 *  not live green: green here would read as "this finished well", and a run
 *  that is still burning the GPU has not finished at all. Everything that
 *  ended badly — including the two that never got off the ground — is fault
 *  red, and the two waiting states are warn amber. */
export const STATUS_COLOR: Record<RunStatus, string> = {
  running: "var(--haller-manual)",
  done: "var(--haller-live)",
  failed: "var(--haller-fault)",
  died: "var(--haller-fault)",
  launch_failed: "var(--haller-fault)",
  stopped: "var(--haller-warn)",
  queued: "var(--haller-warn)",
};

/** `launch_failed` is the one status wider than the column it has to sit in.
 *  It is abbreviated here and spelled out in the pill's title and in the
 *  detail header, so the exact word is never more than a hover away. */
const STATUS_LABEL: Record<RunStatus, string> = {
  running: "running",
  done: "done",
  failed: "failed",
  died: "died",
  launch_failed: "launch",
  stopped: "stopped",
  queued: "queued",
};

/** The status, as a pill. Shared with `RunDetail` so a run wears the same
 *  badge in the list and in its own header. */
export function StatusPill({ status }: { status: RunStatus }) {
  const colour = STATUS_COLOR[status];
  return (
    <span
      title={status}
      className="inline-flex h-4.5 min-w-0 items-center gap-1 rounded-[3px] px-1 label-micro"
      style={{
        color: colour,
        background: "color-mix(in oklch, " + colour + " 16%, transparent)",
      }}
    >
      {/* The blink is decoration, never the only carrier: the pill is already
          blue and already says "running". It stops under reduced-motion. */}
      <span
        aria-hidden
        className={
          "h-1 w-1 shrink-0 rounded-full " +
          (status === "running" ? "animate-haller-rec" : "")
        }
        style={{ backgroundColor: colour }}
      />
      <span className="truncate">{STATUS_LABEL[status]}</span>
    </span>
  );
}

/**
 * Epoch SECONDS from the backend's stamp, or null if there is nothing to read.
 *
 * The stamp is an ISO 8601 UTC string — `runs.py::_now()` returns
 * `datetime.now(UTC).isoformat(timespec="seconds")`, e.g.
 * `"2026-08-26T19:33:50+00:00"`. It is NOT `time.time()`, and it never was:
 * the kit this was ported from writes the same string. Seconds rather than
 * milliseconds because `RunDetail` measures elapsed against a `Date.now()/1000`
 * poll clock, and a mixed pair there reads as a run that ran for 50 years.
 */
export function epochSeconds(ts: string | null | undefined): number | null {
  if (typeof ts !== "string" || ts === "") return null;
  const ms = Date.parse(ts);
  return Number.isFinite(ms) ? ms / 1000 : null;
}

/**
 * A started-at short enough for a 42px column: clock time for a run started
 * today, month-day for anything older.
 */
export function shortWhen(ts: string | null | undefined): string {
  const secs = epochSeconds(ts);
  if (secs === null) return "—";
  const d = new Date(secs * 1000);
  const now = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  const sameDay =
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate();
  return sameDay
    ? `${p(d.getHours())}:${p(d.getMinutes())}`
    : `${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

/** The full stamp, for a title attribute. */
export function fullWhen(ts: string | null | undefined): string | undefined {
  const secs = epochSeconds(ts);
  if (secs === null) return undefined;
  return new Date(secs * 1000).toLocaleString();
}

/** A run's display name. The id is the fallback rather than the primary: a
 *  job name is what the operator typed, and an id is what the filesystem
 *  needed. */
export function runLabel(r: { id: string; name: string | null }): string {
  return r.name && r.name.trim() !== "" ? r.name : r.id;
}

/** compare gutter · status · name · kind · started. One template shared by the
 *  header and every row, so the columns cannot drift apart. */
const COLS = "14px 74px minmax(0,1fr) 52px 42px";

export function RunList({
  runs,
  loading,
  error,
  selectedId,
  onSelect,
  compare,
  onToggleCompare,
}: {
  runs: RunSummary[];
  loading: boolean;
  error: string | null;
  selectedId: string | null;
  onSelect: (id: string) => void;
  /** The compare set, owned by the pane above — this list only reports the
   *  toggles. */
  compare: Set<string>;
  onToggleCompare: (id: string) => void;
}) {
  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
      {error && (
        <div className="shrink-0 p-2.5">
          <Refusal>{error}</Refusal>
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto">
        {runs.length > 0 && (
          <HeadRow
            style={{ gridTemplateColumns: COLS }}
            cols={[
              { key: "cmp", label: "" },
              { key: "status", label: "status" },
              { key: "run", label: "run" },
              { key: "kind", label: "kind" },
              { key: "at", label: "at", align: "right" },
            ]}
          />
        )}

        {runs.length === 0 ? (
          <Empty>{loading ? "reading…" : error ? "nothing to list" : "no runs yet"}</Empty>
        ) : (
          runs.map((r) => {
            const label = runLabel(r);
            const selected = r.id === selectedId;
            return (
              <div key={r.id} className="relative border-b border-border">
                <button
                  type="button"
                  onClick={() => onSelect(r.id)}
                  aria-current={selected ? "true" : undefined}
                  title={r.spec_summary ?? label}
                  className={
                    "grid w-full items-center gap-x-2 px-2.5 py-1 text-left " +
                    "transition-colors " +
                    (selected
                      ? "bg-secondary shadow-[inset_3px_0_0_var(--haller-live)]"
                      : "hover:bg-muted")
                  }
                  style={{ gridTemplateColumns: COLS }}
                >
                  {/* The compare checkbox floats over this cell — a checkbox
                      inside the row button would be interactive content nested
                      in interactive content, which no screen reader forgives. */}
                  <span aria-hidden />
                  <StatusPill status={r.status} />
                  <span className="truncate font-mono text-[11px]">{label}</span>
                  <span className="truncate label-micro text-muted-foreground">
                    {r.kind}
                  </span>
                  <span
                    data-num
                    title={fullWhen(r.started_at)}
                    className="text-right font-mono text-[10px] tabular-nums text-muted-foreground"
                  >
                    {shortWhen(r.started_at)}
                  </span>

                  {/* The backend's own one-line rendering of the spec, verbatim.
                      Re-deriving it here would let the list and the detail view
                      describe the same run in two different sentences. */}
                  {r.spec_summary && (
                    <span className="col-span-3 col-start-3 truncate font-mono text-[9px] text-muted-foreground">
                      {r.spec_summary}
                    </span>
                  )}
                </button>

                <input
                  type="checkbox"
                  checked={compare.has(r.id)}
                  onChange={() => onToggleCompare(r.id)}
                  aria-label={`compare ${label}`}
                  title="overlay this run in compare"
                  className="absolute top-[7px] left-2.5 h-3 w-3 shrink-0 cursor-pointer accent-[var(--haller-live)]"
                />
              </div>
            );
          })
        )}

        {loading && runs.length > 0 && (
          <div className="px-2.5 py-1 font-mono text-[10px] text-muted-foreground">
            reading…
          </div>
        )}
      </div>
    </div>
  );
}
