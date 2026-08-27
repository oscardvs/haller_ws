"use client";

/**
 * Delete a whole dataset.
 *
 * Typing the name is the gate, deliberately, instead of a checkbox or a
 * countdown. A checkbox confirms the ACT, and the mistake this guards against
 * is not "I did not mean to delete" — it is "I did not notice which dataset
 * was selected". Only the name makes you read that, and the name is the one
 * thing a mis-click cannot supply.
 *
 * The size and the duration are here for the same reason: hours of
 * demonstration is what is actually being destroyed, and a byte count alone
 * reads as housekeeping.
 */
import { useCallback, useState } from "react";
import { toast } from "sonner";

import {
  isBusy, isForbidden, isMissing, lab, reason, REMOTE_REFUSED,
} from "@/lib/lab";
import { fmtBytes, fmtDuration } from "@/components/lab/charts/svg";
import {
  Button, Dialog, Field, Note, Refusal, TextInput, WarnBox,
} from "@/components/lab/ui";

/** Persists until the operator acts on it, so it is a panel and not a toast. */
type Failure = { text: string; tone: "warn" | "fault" };

function failureOf(e: unknown): Failure {
  // Whole-dataset delete is not in the frozen route list — a backend without
  // it 404s, and saying so is more use than a generic failure.
  if (isMissing(e)) {
    return {
      text:
        "this backend cannot delete a dataset — remove the directory on the " +
        "machine instead",
      tone: "warn",
    };
  }
  if (isForbidden(e)) return { text: REMOTE_REFUSED, tone: "warn" };
  return { text: reason(e), tone: isBusy(e) ? "warn" : "fault" };
}

export function DeleteDatasetDialog({
  repoId,
  episodes,
  durationS,
  sizeBytes,
  onClose,
  onDeleted,
}: {
  repoId: string;
  episodes: number;
  durationS: number;
  sizeBytes: number;
  onClose: () => void;
  onDeleted: () => void;
}) {
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<Failure | null>(null);

  // Exact, after trimming only. A near-match is a different dataset.
  const matches = typed.trim() === repoId;

  const confirm = useCallback(async () => {
    setBusy(true);
    setFailure(null);
    try {
      const r = await lab.deleteDataset(repoId);
      toast.success(`deleted ${r.repo_id} · ${fmtBytes(r.freed_bytes)} freed`);
      onDeleted();
      onClose();
    } catch (e) {
      setFailure(failureOf(e));
    } finally {
      setBusy(false);
    }
  }, [repoId, onDeleted, onClose]);

  return (
    <Dialog
      title="delete dataset"
      onClose={onClose}
      footer={
        <>
          <Button onClick={onClose} disabled={busy}>cancel</Button>
          <Button tone="danger" onClick={confirm} disabled={busy || !matches}>
            {busy ? "deleting…" : "delete this dataset"}
          </Button>
        </>
      }
    >
      <WarnBox tone="fault">
        <span className="font-mono text-[11px] break-all">{repoId}</span>
        <div className="mt-1.5 font-mono text-[10px]">
          <span data-num className="tabular-nums">{episodes}</span>{" "}
          episode{episodes === 1 ? "" : "s"} ·{" "}
          <span data-num className="tabular-nums">{fmtDuration(durationS)}</span>{" "}
          of demonstration ·{" "}
          <span data-num className="tabular-nums">{fmtBytes(sizeBytes)}</span> on
          disk
        </div>
        <p className="mt-2 text-pretty">
          There is no undo. Nothing is moved to a backup, the videos and the
          review go with it, and every one of those takes has to be re-recorded
          to get it back.
        </p>
      </WarnBox>

      {failure && <Refusal tone={failure.tone}>{failure.text}</Refusal>}

      <Field
        label="type the dataset name to confirm"
        hint="It has to match exactly. This is the only thing that makes you read which dataset is selected."
      >
        <TextInput
          value={typed}
          onChange={(e) => setTyped(e.target.value)}
          placeholder={repoId}
          aria-label="type the dataset name to confirm"
          spellCheck={false}
          autoComplete="off"
          autoCorrect="off"
          disabled={busy}
        />
      </Field>

      {/* Runs that trained on this dataset keep their own record of which
          episodes they saw; the episodes themselves will be gone. */}
      <Note>
        Training runs launched from this dataset are not touched, but their
        source will no longer exist — a run cannot be reproduced after this.
      </Note>
    </Dialog>
  );
}
