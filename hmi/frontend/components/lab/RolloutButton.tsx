"use client";

/**
 * "That run won — run it." The launch path for a surface that has no
 * checkpoint list.
 *
 * The Train tab reaches a rollout through `CheckpointList`: the operator is
 * already looking at what a run wrote, and picks a row. The compare view is
 * the other half of the same afternoon — three runs overlaid, one of them
 * clearly the best — and until now the only way to act on that reading was to
 * go back to the cockpit, find the run again in the Train tab's list, and
 * scroll to its checkpoints. So this is the same launch with the list folded
 * into it: read the run's checkpoints, open on the newest, and let the dialog
 * offer the rest.
 *
 * WHAT IT REFUSES TO OFFER, and why each refusal is a different shape:
 *
 *  - not a training run — no button. A rollout, an export and a prune write no
 *    checkpoints, and there is nothing here to be unavailable.
 *  - no `/checkpoints` route — no button. An older backend cannot answer the
 *    question, and a permanently dead control on every row is worse than
 *    silence. Same rule `CheckpointList` follows for the same 404.
 *  - the route answered, and the run has nothing loadable yet — a DISABLED
 *    button that says why. This is the ordinary state of a run that started
 *    twenty minutes ago, it changes on its own, and hiding it would mean the
 *    operator watching that run never learns the control exists.
 *
 * The read is per-run and its own, rather than hoisted into the pane's effect:
 * a comparison is two to five runs, the request is a directory listing, and
 * threading five more results through a tagged read that already juggles runs,
 * key sets and batched series would buy nothing but coupling.
 */
import { useEffect, useState } from "react";

import { isMissing, lab, reason, type Checkpoint, type Run } from "@/lib/lab";
import { Button } from "@/components/lab/ui";
import { checkpointName } from "@/components/lab/CheckpointList";
import { RolloutDialog } from "@/components/lab/RolloutDialog";

export function RolloutButton({
  run,
  onLaunched,
}: {
  run: Run;
  /** The rollout that was started — a DIFFERENT run than the one this button
   *  sits on. The owner decides what to do with it; this component's job ends
   *  at the launch. */
  onLaunched: (rollout: Run) => void;
}) {
  const [list, setList] = useState<Checkpoint[] | null>(null);
  const [missing, setMissing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  const trains = run.kind === "train";

  useEffect(() => {
    if (!trains) return;
    let cancelled = false;
    lab
      .checkpoints(run.id)
      .then((r) => {
        if (cancelled) return;
        // Normalised rather than trusted: everything downstream counts this
        // list, and a backend that answers the shape without the key would
        // otherwise crash the row it was meant to decorate.
        setList(Array.isArray(r.checkpoints) ? r.checkpoints : []);
        setError(null);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        if (isMissing(e)) setMissing(true);
        else setError(reason(e));
      });
    return () => {
      cancelled = true;
    };
    // `status` is in here for the same reason `CheckpointList` has it: a run
    // that has just gone `done` wrote its last checkpoint on the way out.
  }, [run.id, run.status, trains]);

  if (!trains || missing) return null;

  const usable = usableCheckpoints(list);
  const newest = usable[0] ?? null;
  /** The ID and not the name. Two runs of the same sweep carry the SAME
   *  `job_name` — which is the ordinary case for a comparison, and it made
   *  both rows read "roll out act_hilti_box_91" to a screen reader, on a
   *  control that starts an arm. The row's own title says the id for the same
   *  reason. */
  const label = run.id;

  const why =
    error !== null
      ? error
      : list === null
        ? "reading this run's checkpoints…"
        : newest === null
          ? run.status === "running"
            ? "no checkpoint written yet — this run is still training"
            : "this run wrote no loadable checkpoint"
          : `run ${label} on the arms — opens on ${checkpointName(newest.path)}`;

  return (
    <>
      <Button
        tone="ghost"
        disabled={newest === null}
        onClick={() => setOpen(true)}
        title={why}
        aria-label={`roll out ${label}`}
      >
        roll out
      </Button>

      {open && newest !== null && (
        <RolloutDialog
          checkpoint={newest}
          checkpoints={usable}
          // The dataset THIS run trained on, off its own spec. The rollout
          // resolves its own from the checkpoint; this only decides whether
          // the dialog has to ask which arm.
          repoId={typeof run.spec?.repo_id === "string" ? run.spec.repo_id : null}
          onClose={() => setOpen(false)}
          onLaunched={(r) => {
            setOpen(false);
            onLaunched(r);
          }}
        />
      )}
    </>
  );
}

/** The loadable checkpoints, newest first.
 *
 *  Partial directories are dropped rather than disabled — the dialog would
 *  only refuse them, and a select whose options cannot all be chosen is a
 *  worse list than a shorter one. `last` carries a null step ON PURPOSE (it is
 *  the symlink, not a step), so it sorts to the end rather than comparing as
 *  zero and pretending to be the oldest checkpoint on disk. */
function usableCheckpoints(list: Checkpoint[] | null): Checkpoint[] {
  return (list ?? [])
    .filter((c) => c.has_model)
    .slice()
    .sort((a, b) => {
      if (a.step === b.step) return 0;
      if (a.step === null) return 1;
      if (b.step === null) return -1;
      return b.step - a.step;
    });
}

