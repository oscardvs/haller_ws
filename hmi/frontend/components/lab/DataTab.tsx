"use client";

/**
 * The LAB workspace: one tab, four sub-views, one selected dataset.
 *
 * Collect · review · train · runs is the order the work actually happens in,
 * and the repo id is the thread between the first three — opening a dataset
 * from the collect shelf or from the train launcher lands in review with that
 * repo already chosen. Runs sits after train because it reads what train
 * produced: the trained models, overlaid. Both the sub-view and the repo are
 * sticky, because tabs mount and unmount and losing your place in a
 * 46-episode review because you glanced at Cameras is the kind of small
 * betrayal that makes an operator stop trusting the surface.
 *
 * Sticky, not persisted: a reload is a fresh session. See `useSticky`.
 */
import { useCallback } from "react";

import { SubNav } from "@/components/lab/ui";
import { useSticky } from "@/components/cockpit/lib";
import { CollectPane } from "@/components/lab/CollectPane";
import { ReviewPane } from "@/components/lab/ReviewPane";
import { TrainPane } from "@/components/lab/TrainPane";
import { RunsPane } from "@/components/lab/RunsPane";
import type { ConfigArm } from "@/components/cockpit/teleopPresets";
import type { CameraInfo } from "@/lib/api";

type SubView = "collect" | "review" | "train" | "runs";

const VIEWS = [
  { id: "collect", label: "collect", hint: "compose the take and record it" },
  { id: "review", label: "review", hint: "watch, mark keep/reject, tag" },
  { id: "train", label: "train", hint: "launch a run on the kept set" },
  { id: "runs", label: "runs", hint: "compare trained models side by side" },
] as const satisfies readonly { id: SubView; label: string; hint: string }[];

export function DataTab({
  cameras,
  arms,
  onCameraRecord,
}: {
  cameras: CameraInfo[];
  /** Off /config — collect's session card offers only the presets these arms
   *  can actually run. */
  arms: ConfigArm[];
  onCameraRecord: (id: string, record: boolean) => void;
}) {
  const [view, setView] = useSticky<SubView>("lab.subview", "collect");
  const [repo, setRepo] = useSticky<string | null>("lab.repo", null);

  const openInReview = useCallback(
    (r: string) => {
      setRepo(r);
      setView("review");
    },
    [setRepo, setView],
  );

  return (
    <div className="grid min-h-0 grid-rows-[auto_minmax(0,1fr)] overflow-hidden">
      <SubNav items={VIEWS} value={view} onChange={setView} label="lab sub-view">
        <span
          className="min-w-0 flex-1 truncate font-mono text-[10px] text-muted-foreground"
          title={repo ?? undefined}
        >
          {repo ?? "no dataset selected"}
        </span>
        {/* The old "compare runs ↗" link lived here; the Runs sub-view is that
            surface in the cockpit now, and it carries its own ↗ deep link for
            the shareable form. */}
      </SubNav>

      {/* `grid` rather than `block` so the pane stretches to the row instead of
          sizing to its content — every pane below owns its own scrolling. */}
      <div className="grid min-h-0 overflow-hidden">
        {view === "collect" && (
          <CollectPane
            cameras={cameras}
            arms={arms}
            onCameraRecord={onCameraRecord}
            onOpenDataset={openInReview}
          />
        )}
        {/* A different dataset is a different review, so it gets a different
            component. The `key` is the whole reset: filters are dropped rather
            than carried across, because a tag or a task query from the
            bimanual corpus returns nothing on the solo one, and an empty list
            that looks like a missing dataset is worse than a filter the
            operator re-sets. A remount clears every piece of that at once —
            there is no list of resets left to keep in sync. Sticky values
            (`lab.review.*`) live in a module map and survive by design: the
            camera key and the trace overlay are the operator's, not the
            dataset's. */}
        {view === "review" && (
          <ReviewPane key={repo ?? "none"} repoId={repo} onPickDataset={setRepo} />
        )}
        {view === "train" && <TrainPane repoId={repo} onOpenDataset={openInReview} />}
        {view === "runs" && <RunsPane />}
      </div>
    </div>
  );
}
