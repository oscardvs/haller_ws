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
 * A row is two lines: the verdict line (status pill, name, kind, duration,
 * started-at) and the detail line (the spec summary, then the launch tags).
 * Duration is measured from the run's own stamps when it has finished, and
 * against the pane's poll clock while it is running — this component owns no
 * timer, and a queued run gets no invented elapsed.
 *
 * `runs` is typed `RunSummary[]` and not `Run[]` on purpose. `GET /lab/runs`
 * returns summaries; `Run` is the detail shape and is assignable to this, so a
 * caller holding either can pass it.
 */
import type { RunStatus, RunSummary } from "@/lib/lab";
import { Empty, Refusal } from "@/components/lab/ui";
import { fmtDuration } from "@/components/lab/charts/svg";

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
 * A started-at short enough for a dense list row: clock time for a run
 * started today, month-day for anything older.
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

export function RunList({
  runs,
  loading,
  error,
  selectedId,
  onSelect,
  compare,
  onToggleCompare,
  now,
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
  /** Epoch seconds of the pane's last list read — the clock a RUNNING row's
   *  duration is measured against. It arrives with the poll, so the duration
   *  ticks without this component owning a timer of its own. */
  now: number;
}) {
  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
      {error && (
        <div className="shrink-0 p-2.5">
          <Refusal>{error}</Refusal>
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto">
        {runs.length === 0 ? (
          <Empty>{loading ? "reading…" : error ? "nothing to list" : "no runs yet"}</Empty>
        ) : (
          runs.map((r) => {
            const label = runLabel(r);
            const selected = r.id === selectedId;
            /* A finished run measures against its own stamp; a running one
               against the poll's clock. A queued run has neither, and gets a
               "—" rather than an elapsed counted from 1970. */
            const until =
              epochSeconds(r.finished_at) ??
              (r.status === "running" && now > 0 ? now : null);
            const startedSecs = epochSeconds(r.started_at);
            const ran =
              startedSecs !== null && until !== null ? until - startedSecs : null;
            return (
              <div key={r.id} className="relative border-b border-border">
                <button
                  type="button"
                  onClick={() => onSelect(r.id)}
                  aria-current={selected ? "true" : undefined}
                  title={r.spec_summary ?? label}
                  className={
                    "flex w-full flex-col gap-0.5 py-1.5 pr-2.5 pl-7 text-left " +
                    "transition-colors " +
                    (selected
                      ? "bg-secondary shadow-[inset_3px_0_0_var(--haller-live)]"
                      : "hover:bg-muted")
                  }
                >
                  <span className="flex min-w-0 items-center gap-2">
                    <StatusPill status={r.status} />
                    <span className="min-w-0 flex-1 truncate font-mono text-[11px]">
                      {label}
                    </span>
                    <span className="label-micro shrink-0 text-muted-foreground">
                      {r.kind}
                    </span>
                    <span
                      data-num
                      title={ran !== null ? "how long the run took" : undefined}
                      className="w-12 shrink-0 text-right font-mono text-[10px] tabular-nums text-muted-foreground"
                    >
                      {fmtDuration(ran)}
                    </span>
                    <span
                      data-num
                      title={fullWhen(r.started_at)}
                      className="w-10 shrink-0 text-right font-mono text-[10px] tabular-nums text-muted-foreground"
                    >
                      {shortWhen(r.started_at)}
                    </span>
                  </span>

                  {/* The backend's own one-line rendering of the spec,
                      verbatim. Re-deriving it here would let the list and the
                      detail view describe the same run in two different
                      sentences. */}
                  {(r.spec_summary || (r.tags && r.tags.length > 0)) && (
                    <span className="flex min-w-0 items-baseline gap-2">
                      {r.spec_summary && (
                        <span className="min-w-0 flex-1 truncate font-mono text-[9px] text-muted-foreground">
                          {r.spec_summary}
                        </span>
                      )}
                      {r.tags?.map((t) => (
                        <span
                          key={t}
                          className="label-micro shrink-0 text-muted-foreground opacity-70"
                        >
                          {t}
                        </span>
                      ))}
                    </span>
                  )}
                </button>

                {/* The compare checkbox floats over this row — a checkbox
                    inside the row button would be interactive content nested
                    in interactive content, which no screen reader forgives. */}
                <input
                  type="checkbox"
                  checked={compare.has(r.id)}
                  onChange={() => onToggleCompare(r.id)}
                  aria-label={`compare ${label}`}
                  title="overlay this run in compare"
                  className="absolute top-[9px] left-2.5 h-3 w-3 shrink-0 cursor-pointer accent-[var(--haller-live)]"
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
