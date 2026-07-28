"use client";

/**
 * The one implementation of start / stop / discard for a take, shared by the
 * command-bar popover and the Dataset tab.
 *
 * On the human-teleop precondition: the v3 design refused to start a take at
 * all unless human teleop was running, on the grounds that the recorder logs
 * the teleop's commanded joint targets as `action` and there would be nothing
 * to log without it. That reasoning is right, but it is a warning here, not a
 * block — decided with Oscar, 2026-07-28:
 *
 *   - The backend does not enforce it. `Recorder.start_episode()` guards only
 *     against a double-start, so a UI-side block is a guard rail that a curl
 *     walks straight past — it would look like an invariant without being one.
 *   - It forecloses takes that are legitimately not human-driven: a scripted
 *     pose sequence, or a bring-up recording made to check the schema.
 *
 * So the operator is told exactly what the `action` column will contain, and
 * is left holding the decision.
 */
import { toast } from "sonner";

import { useRecorder } from "@/lib/recorder";

export const NO_TELEOP_WARNING =
  "human teleop is not running — the action column will log the arms' last commanded targets, not a demonstration";

export async function startTake(repoId: string, task: string, teleopRunning: boolean) {
  try {
    await useRecorder.getState().start(repoId, task);
    if (teleopRunning) {
      toast.success(`recording → ${repoId}`);
    } else {
      toast.warning(`recording → ${repoId} · ${NO_TELEOP_WARNING}`);
    }
  } catch (e) {
    toast.error(`recording failed: ${e instanceof Error ? e.message : String(e)}`);
  }
}

export async function stopTake(save: boolean) {
  try {
    const s = await useRecorder.getState().stop(save);
    if (save) toast.success(`saved ${s.episode_frames} frames`);
    else toast.message("take discarded");
  } catch (e) {
    toast.error(
      `${save ? "save" : "discard"} failed: ${e instanceof Error ? e.message : String(e)}`,
    );
  }
}
