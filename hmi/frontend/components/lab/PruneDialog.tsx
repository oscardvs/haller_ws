"use client";

/**
 * Prune — permanently removes the episodes marked reject.
 *
 * Every dropped episode is NAMED, not counted. A count is something you agree
 * to; a list is something you read, and the difference is a demonstration that
 * cannot be re-recorded. The list is the tallest thing in this dialog on
 * purpose.
 *
 * The backup copy is the undo, and the confirmation is sized to match it: with
 * a backup there is nothing to type, because there is something to go back to.
 * Without one the operator types DELETE, because there is not.
 */
import { useCallback, useMemo, useState } from "react";
import { toast } from "sonner";

import {
  isBusy, isForbidden, isMissing, lab, reason, REMOTE_REFUSED,
  type LabEpisode,
} from "@/lib/lab";
import {
  Button, Dialog, Field, Note, Refusal, TextInput, WarnBox,
} from "@/components/lab/ui";

/** Persists until the operator acts on it, so it is a panel and not a toast.
 *  409 and 403 both carry the backend's own sentence and are quoted. */
type Failure = { text: string; tone: "warn" | "fault" };

function failureOf(e: unknown): Failure {
  if (isMissing(e)) return { text: "this backend cannot prune a dataset", tone: "warn" };
  if (isForbidden(e)) return { text: REMOTE_REFUSED, tone: "warn" };
  return { text: reason(e), tone: isBusy(e) ? "warn" : "fault" };
}

/** What the take was rejected for. Empty when the mark was set by hand and the
 *  grader had nothing to say — which is a fact, not a blank. */
function whyDropped(ep: LabEpisode): string {
  if (ep.reasons.length > 0) return ep.reasons.join(" · ");
  return ep.note?.trim() || "no reason recorded";
}

function secs(s: number): string {
  return Number.isFinite(s) ? `${s.toFixed(1)}s` : "—";
}

export function PruneDialog({
  repoId,
  episodes,
  onClose,
  onPruned,
}: {
  repoId: string;
  episodes: LabEpisode[];
  onClose: () => void;
  onPruned: () => void;
}) {
  const [backup, setBackup] = useState(true);
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<Failure | null>(null);

  const drop = useMemo(() => episodes.filter((e) => e.mark === "reject"), [episodes]);
  const keep = episodes.length - drop.length;

  const nothingToDo = drop.length === 0;
  const wouldEmpty = episodes.length > 0 && drop.length === episodes.length;
  const blocked = nothingToDo || wouldEmpty;
  // Typing is required only when there is no copy to go back to.
  // Trimmed, matching DeleteDatasetDialog — one rule for both typed gates,
  // because it is the same operator at the same keyboard. Trimming is safe in
  // both: no wrong value trims to a right one, and a pasted trailing space
  // blocking someone who typed the right word is friction that teaches nothing.
  const gateOpen = backup || typed.trim() === "DELETE";

  const confirm = useCallback(async () => {
    setBusy(true);
    setFailure(null);
    try {
      // `expect_episodes` is the guard, not a convenience: the server refuses
      // if the set moved between this dialog opening and this click, which is
      // exactly the window in which a renumbering would make these indices
      // name different takes.
      const r = await lab.prune(repoId, backup, drop.map((e) => e.index));
      toast.success(
        `prune started · ${drop.length} episode${drop.length === 1 ? "" : "s"} ` +
        `queued for removal · run ${r.run_id}`,
      );
      onPruned();
      onClose();
    } catch (e) {
      setFailure(failureOf(e));
    } finally {
      setBusy(false);
    }
  }, [repoId, backup, drop, onPruned, onClose]);

  const footer = blocked ? (
    <Button onClick={onClose}>close</Button>
  ) : (
    <>
      <Button onClick={onClose} disabled={busy}>cancel</Button>
      <Button tone="danger" onClick={confirm} disabled={busy || !gateOpen}>
        {busy
          ? "starting…"
          : `remove ${drop.length} episode${drop.length === 1 ? "" : "s"}`}
      </Button>
    </>
  );

  return (
    <Dialog title="prune rejected" onClose={onClose} footer={footer} wide>
      <span className="font-mono text-[10px] break-all text-muted-foreground">
        {repoId}
      </span>

      {failure && <Refusal tone={failure.tone}>{failure.text}</Refusal>}

      {nothingToDo && (
        <Note>
          No episodes are marked reject, so there is nothing to remove. Mark the
          failures first.
        </Note>
      )}

      {wouldEmpty && (
        <Note>
          Every episode is marked reject. A dataset cannot be emptied this way —
          delete the whole dataset instead.
        </Note>
      )}

      {!blocked && (
        <>
          <WarnBox tone="fault">
            <span className="label-micro">
              removing {drop.length} episode{drop.length === 1 ? "" : "s"} ·{" "}
              {keep} kept
            </span>
            {/* Named one per line. This is the part that has to be read. */}
            <div className="mt-2 max-h-[38vh] min-h-0 overflow-y-auto font-mono text-[10px] leading-[1.7]">
              {drop.map((ep) => (
                <div key={ep.index} className="text-pretty">
                  <span data-num className="tabular-nums">Ep {ep.label}</span>{" "}
                  <span className="opacity-70">(idx {ep.index})</span>
                  {" — "}
                  <span data-num className="tabular-nums">{secs(ep.duration_s)}</span>
                  {" — "}
                  <span className="opacity-70">{whyDropped(ep)}</span>
                </div>
              ))}
            </div>
          </WarnBox>

          <Note>
            The {keep} kept episode{keep === 1 ? "" : "s"} are renumbered 0…
            {Math.max(0, keep - 1)} and the video is re-encoded, so this runs as
            a background job rather than finishing when you click. Review marks
            are cleared afterwards: a mark names an index, and after a renumber
            those indices name different takes.
          </Note>

          <label className="flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              checked={backup}
              onChange={(e) => {
                setBackup(e.target.checked);
                setTyped("");
              }}
              disabled={busy}
              className="h-3.5 w-3.5 shrink-0 accent-[var(--haller-live)]"
            />
            <span className="label-micro text-foreground">
              keep the previous version as a backup
            </span>
          </label>

          {backup ? (
            <Note>
              The pre-prune copy stays on disk under its own repo id. It is never
              offered as a training source — training on it would silently undo
              this prune.
            </Note>
          ) : (
            <Field
              label="type DELETE to confirm"
              hint="No backup will be kept. These takes leave the disk and there is nothing to go back to."
            >
              <TextInput
                value={typed}
                onChange={(e) => setTyped(e.target.value)}
                placeholder="DELETE"
                aria-label="type DELETE to confirm"
                spellCheck={false}
                autoComplete="off"
                disabled={busy}
              />
            </Field>
          )}
        </>
      )}
    </Dialog>
  );
}
