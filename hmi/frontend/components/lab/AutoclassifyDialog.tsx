"use client";

/**
 * Autoclassify — preview → diff → apply → revert.
 *
 * The diff table is the dialog. The classifier reads the traces and proposes
 * marks; nothing is written until the operator has seen which episodes move,
 * which way, why, and how sure it is. An apply that can happen without the
 * table on screen is a button that rewrites a review, and the review is the
 * only record of which demonstrations are worth training on.
 *
 * All four modes render through this one table, which is why `confidence`
 * exists on a deterministic mode where it is always 1.0. `policy-loss` is the
 * exception that proves it: it is a SORT ORDER, its diff is empty by
 * construction, and it gets a ranking list and no apply button.
 *
 * `from` and `to` are drawn in their own mark colours rather than as plain
 * words — the direction is the thing being approved, and `keep → reject` and
 * `reject → keep` are two characters apart in text.
 *
 * Revert is one click with no confirmation of its own. It is the SAFE
 * direction — it restores marks that existed a moment ago — and friction on
 * the way back is what makes an operator live with a batch they disagree with.
 */
import { useCallback, useEffect, useState } from "react";

import {
  epLabel,
  isBusy,
  isForbidden,
  isMissing,
  lab,
  reason,
  REMOTE_REFUSED,
  type AutoclassApplied,
  type AutoclassChange,
  type AutoclassMode,
  type AutoclassParams,
  type AutoclassPreview,
} from "@/lib/lab";
import {
  Button, Dialog, Empty, Field, HeadRow, MARK_COLOR, Note, NumberInput,
  Refusal, Segmented, TextInput,
} from "@/components/lab/ui";

/* ---- failures ----------------------------------------------------------- */

/** Every failure here persists until the operator does something about it, so
 *  it gets a panel and never a toast. A 409 and a 403 both carry a sentence
 *  that is the operator's next move, and both are quoted rather than
 *  paraphrased. */
type Failure = { text: string; tone: "warn" | "fault" };

function failureOf(e: unknown): Failure {
  if (isMissing(e)) return { text: "this backend has no lab", tone: "warn" };
  if (isForbidden(e)) return { text: REMOTE_REFUSED, tone: "warn" };
  return { text: reason(e), tone: isBusy(e) ? "warn" : "fault" };
}

/* ---- modes -------------------------------------------------------------- */

const MODES: readonly { value: AutoclassMode; label: string; hint: string }[] = [
  { value: "grade", label: "grade", hint: "the rule ladder: FAIL→reject, PASS→keep, SUSPECT left alone" },
  { value: "rules", label: "rules", hint: "an expression you write, evaluated per episode" },
  { value: "knn", label: "knn", hint: "propagate the marks already made to their nearest neighbours" },
  { value: "policy-loss", label: "loss", hint: "a sort order — hardest to fit first. never a mark" },
];

/** True when the mode needs the operator to author something before a preview
 *  means anything. Those two do not auto-run on selection. */
const NEEDS_PARAMS = (m: AutoclassMode) => m === "rules" || m === "knn";

type Editor = { rejectIf: string; keepIf: string; k: number; minConf: number };

const EDITOR: Editor = { rejectIf: "", keepIf: "", k: 5, minConf: 0.6 };

function paramsFor(mode: AutoclassMode, ed: Editor): AutoclassParams {
  if (mode === "rules") {
    return {
      reject_if: ed.rejectIf.trim() || undefined,
      keep_if: ed.keepIf.trim() || undefined,
    };
  }
  if (mode === "knn") return { k: ed.k, min_confidence: ed.minConf };
  return {};
}

/** Confidence banding. Shown, never thresholded silently — the colour reads
 *  the number printed beside it, it does not filter anything behind it. */
function confidenceColour(c: number): string {
  if (!Number.isFinite(c)) return "var(--haller-rail)";
  if (c > 0.8) return "var(--haller-live)";
  if (c > 0.5) return "var(--haller-warn)";
  return "var(--haller-rail)";
}

const GRID = "grid-cols-[118px_128px_minmax(0,1fr)_96px]";

const COLS = [
  { key: "ep", label: "ep" },
  { key: "move", label: "from → to" },
  { key: "why", label: "why" },
  { key: "conf", label: "confidence", align: "right" as const },
];

/* ---- the dialog --------------------------------------------------------- */

type Request = { mode: AutoclassMode; params: AutoclassParams };

export function AutoclassifyDialog({
  repoId,
  onClose,
  onApplied,
}: {
  repoId: string;
  onClose: () => void;
  onApplied: () => void;
}) {
  const [mode, setMode] = useState<AutoclassMode>("grade");
  const [ed, setEd] = useState<Editor>(EDITOR);
  // What has actually been asked for. Held apart from `mode` so that typing an
  // expression does not fire a preview per keystroke, and so a mode that needs
  // authoring can sit unrun without pretending it found nothing.
  const [request, setRequest] = useState<Request | null>({ mode: "grade", params: {} });
  /** One preview, TAGGED with the ask it answers — by identity, because
   *  `setRequest` mints a new object per ask and re-running the same
   *  expression IS a new ask. Tagged rather than dropped at the top of the
   *  effect: a table left under the wrong mode's editor is a diff the operator
   *  could approve without it being the one that mode would produce, and the
   *  drop itself was a setState in the effect body. */
  const [read, setRead] = useState<{
    request: Request;
    preview: AutoclassPreview | null;
    failure: Failure | null;
  } | null>(null);
  const [applied, setApplied] = useState<AutoclassApplied | null>(null);
  const [reverted, setReverted] = useState<number | null>(null);
  /** A refusal from apply or revert. Held apart from the read's own failure: a
   *  write is a different question from the preview it was taken from. */
  const [writeFailure, setWriteFailure] = useState<Failure | null>(null);
  const [busy, setBusy] = useState(false);

  const answer = read !== null && read.request === request ? read : null;
  const preview = answer?.preview ?? null;
  const failure = writeFailure ?? answer?.failure ?? null;
  // No ask yet — a mode that needs authoring — is not loading, and never
  // pretends the classifier found nothing.
  const loading = request !== null && answer === null;

  useEffect(() => {
    if (!request) return;
    let cancelled = false;
    (async () => {
      try {
        const p = await lab.autoclassPreview(repoId, request.mode, request.params);
        if (!cancelled) setRead({ request, preview: p, failure: null });
      } catch (e) {
        if (!cancelled) setRead({ request, preview: null, failure: failureOf(e) });
      }
    })();
    return () => { cancelled = true; };
  }, [repoId, request]);

  const pickMode = useCallback((m: AutoclassMode) => {
    setMode(m);
    // grade and policy-loss take no authoring, so selecting them IS the ask.
    setRequest(NEEDS_PARAMS(m) ? null : { mode: m, params: {} });
  }, []);

  const runPreview = useCallback(() => {
    setRequest({ mode, params: paramsFor(mode, ed) });
  }, [mode, ed]);

  const doApply = useCallback(async () => {
    if (!preview) return;
    setBusy(true);
    setWriteFailure(null);
    try {
      // By token: the server recomputes it and 409s if the dataset moved under
      // the dialog, so what lands is the diff that was read. A 409 here is
      // never retried behind the operator — they re-read the traces by hand.
      const a = await lab.autoclassApply(repoId, preview.token);
      setApplied(a);
      onApplied();
    } catch (e) {
      setWriteFailure(failureOf(e));
      // A 409 means the server recomputed the token and the dataset had moved,
      // so THIS token can only ever 409 again — leaving the button live would
      // offer a retry that cannot succeed. Drop the read instead: the operator
      // is back at "preview", which is the only thing that can help, and the
      // diff they were about to approve is no longer on screen to be approved.
      if (isBusy(e)) setRead(null);
    } finally {
      setBusy(false);
    }
  }, [repoId, preview, onApplied]);

  const doRevert = useCallback(async () => {
    if (!applied?.batch) return;
    setBusy(true);
    setWriteFailure(null);
    try {
      const r = await lab.autoclassRevert(repoId, applied.batch);
      setReverted(r.reverted);
      onApplied();
    } catch (e) {
      setWriteFailure(failureOf(e));
    } finally {
      setBusy(false);
    }
  }, [repoId, applied, onApplied]);

  const diff = preview?.diff ?? [];
  const n = diff.length;
  const ranking = preview?.ranking ?? [];
  const gated = preview?.available === false;
  const isRanking = mode === "policy-loss";
  // The diff is on screen and non-empty, the mode can write, and nothing has
  // been written yet. Every one of those is a precondition for apply.
  const canApply =
    applied === null && !loading && !gated && !isRanking && n > 0 && preview !== null;

  const footer = applied ? (
    <Button onClick={onClose}>close</Button>
  ) : canApply ? (
    <>
      <Button onClick={onClose} disabled={busy}>cancel</Button>
      <Button tone="primary" onClick={doApply} disabled={busy}>
        {busy ? "applying…" : `apply ${n} change${n === 1 ? "" : "s"}`}
      </Button>
    </>
  ) : (
    <Button onClick={onClose}>close</Button>
  );

  return (
    <Dialog title="autoclassify" onClose={onClose} footer={footer} wide>
      <span className="font-mono text-[10px] break-all text-muted-foreground">
        {repoId}
      </span>

      {applied === null && (
        <Segmented
          options={MODES}
          value={mode}
          onChange={pickMode}
          label="autoclassify mode"
          disabled={busy}
        />
      )}

      {applied === null && mode === "rules" && (
        <div className="flex flex-col gap-2.5">
          <Field
            label="reject if"
            hint="Evaluated first — an episode it matches is never considered for keep."
          >
            <TextInput
              value={ed.rejectIf}
              onChange={(e) => setEd({ ...ed, rejectIf: e.target.value })}
              placeholder="duration_s < 2 or closes == 0"
              spellCheck={false}
              autoComplete="off"
            />
          </Field>
          <Field label="keep if">
            <TextInput
              value={ed.keepIf}
              onChange={(e) => setEd({ ...ed, keepIf: e.target.value })}
              placeholder="verdict == PASS and tracking < 8"
              spellCheck={false}
              autoComplete="off"
            />
          </Field>
          <Button onClick={runPreview} disabled={busy || loading}>preview</Button>
        </div>
      )}

      {applied === null && mode === "knn" && (
        <div className="flex flex-col gap-2.5">
          <div className="grid grid-cols-2 gap-2.5">
            <Field label="k" hint="Neighbours voting on each unmarked episode.">
              <NumberInput
                value={ed.k}
                onChange={(v) => setEd({ ...ed, k: v })}
                min={1}
                max={50}
                fallback={5}
              />
            </Field>
            <Field label="min confidence" hint="Below this the vote is not proposed at all.">
              <NumberInput
                value={ed.minConf}
                onChange={(v) => setEd({ ...ed, minConf: v })}
                min={0}
                max={1}
                step={0.05}
                fallback={0.6}
              />
            </Field>
          </div>
          <Button onClick={runPreview} disabled={busy || loading}>preview</Button>
        </div>
      )}

      {failure && <Refusal tone={failure.tone}>{failure.text}</Refusal>}

      {loading && <Empty>reading the traces…</Empty>}

      {!loading && request === null && (
        <Empty>nothing previewed yet</Empty>
      )}

      {!loading && !failure && gated && (
        <Note>
          {preview?.reason ??
            "this mode needs a run that logged per-episode loss, and there is none."}
        </Note>
      )}

      {/* ---- ranking: a sort order, and deliberately not a diff ---------- */}
      {!loading && !failure && !gated && isRanking && applied === null && (
        <>
          <Note>
            A sort order, never a mark. High loss is as often a rare-but-correct
            demonstration as a bad one, so this ranks and leaves the deciding to
            you.
          </Note>
          {ranking.length === 0 ? (
            <Empty>no ranking returned</Empty>
          ) : (
            <div className="max-h-[46vh] min-h-0 overflow-y-auto rounded-md border border-border">
              <HeadRow
                cols={[
                  { key: "rank", label: "rank" },
                  { key: "ep", label: "ep" },
                  { key: "score", label: "score", align: "right" },
                ]}
                className="grid-cols-[64px_minmax(0,1fr)_96px]"
              />
              {ranking.map((r) => (
                <div
                  key={r.episode}
                  className="grid grid-cols-[64px_minmax(0,1fr)_96px] gap-2 border-b border-border px-2.5 py-1 font-mono text-[10px] last:border-b-0"
                >
                  <span data-num className="tabular-nums text-muted-foreground">
                    {r.rank}
                  </span>
                  <EpCell index={r.episode} />
                  <span data-num className="text-right tabular-nums">
                    {Number.isFinite(r.score) ? r.score.toFixed(4) : "—"}
                  </span>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* ---- the diff ---------------------------------------------------- */}
      {!loading && !failure && !gated && !isRanking && applied === null && preview && n === 0 && (
        <Note>The classifier agrees with every mark already set.</Note>
      )}

      {!loading && !failure && !gated && !isRanking && applied === null && n > 0 && (
        <>
          <div className="flex items-baseline gap-1.5 font-mono text-[10px] text-muted-foreground">
            <span data-num className="tabular-nums text-foreground">{n}</span>
            <span>change{n === 1 ? "" : "s"} proposed · everything not listed is left alone</span>
          </div>

          <div className="max-h-[46vh] min-h-0 overflow-y-auto rounded-md border border-border">
            <HeadRow cols={COLS} className={GRID} />
            {diff.map((c) => (
              <ChangeRow key={c.episode} change={c} />
            ))}
          </div>

          <Note>
            Applying writes these marks and nothing else. The batch can be
            reverted in one click on the next screen.
          </Note>
        </>
      )}

      {/* ---- applied ----------------------------------------------------- */}
      {applied && (
        <>
          <Note>
            {applied.applied} mark{applied.applied === 1 ? "" : "s"} rewritten on{" "}
            <span className="font-mono">{repoId}</span> by{" "}
            <span className="font-mono">{request?.mode ?? mode}</span>. The rest
            of the review is untouched.
          </Note>

          {reverted !== null ? (
            <Note>
              Batch reverted · {reverted} mark{reverted === 1 ? "" : "s"} restored
              to what they were before.
            </Note>
          ) : applied.batch ? (
            <Button tone="danger" onClick={doRevert} disabled={busy}>
              {busy ? "reverting…" : "revert this batch"}
            </Button>
          ) : (
            <Note>
              This backend cannot revert a batch — undo by hand from the episode
              list.
            </Note>
          )}
        </>
      )}
    </Dialog>
  );
}

/* ---- cells -------------------------------------------------------------- */

/** Both spellings, always together. Episodes are stored 0-based and talked
 *  about 1-based, and that off-by-one is how the wrong demonstration gets
 *  marked. `epLabel` owns the mapping. */
function EpCell({ index }: { index: number }) {
  return (
    <span className="whitespace-nowrap">
      <span data-num className="tabular-nums">{epLabel(index)}</span>{" "}
      <span className="text-muted-foreground">(idx {index})</span>
    </span>
  );
}

function ChangeRow({ change: c }: { change: AutoclassChange }) {
  const pct = Number.isFinite(c.confidence)
    ? Math.round(Math.max(0, Math.min(1, c.confidence)) * 100)
    : null;

  return (
    <div
      className={
        "grid items-start gap-2 border-b border-border px-2.5 py-1.5 " +
        "font-mono text-[10px] last:border-b-0 " + GRID
      }
    >
      <EpCell index={c.episode} />

      <span className="inline-flex items-baseline gap-1.5 whitespace-nowrap">
        <span style={{ color: MARK_COLOR[c.from] }}>{c.from}</span>
        <span aria-hidden className="text-muted-foreground">→</span>
        <span className="sr-only">to</span>
        <span style={{ color: MARK_COLOR[c.to] }}>{c.to}</span>
      </span>

      <span className="min-w-0 text-pretty text-muted-foreground">{c.why}</span>

      <span className="flex items-center justify-end gap-2">
        <span data-num className="tabular-nums">
          {pct === null ? "—" : `${pct}%`}
        </span>
        {/* The number and the bar say the same thing; the bar is what makes a
            column of them comparable without reading each one. */}
        <span
          aria-hidden
          className="h-[3px] w-9 shrink-0 overflow-hidden rounded-[1px] bg-muted"
        >
          <span
            className="block h-full"
            style={{
              width: `${pct ?? 0}%`,
              backgroundColor: confidenceColour(c.confidence),
            }}
          />
        </span>
      </span>
    </div>
  );
}
