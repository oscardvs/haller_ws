"use client";

/**
 * Collect: compose the next take, record it, and see where it landed.
 *
 * The top region is the cockpit's Dataset tab, mounted verbatim. The camera
 * take-composition grid, the recorder card and the on-disk episode browser
 * already answer "what will be in this take" and "did it save"; rebuilding
 * them here would give the operator two recorders that disagree.
 *
 * The shelf underneath is the only new thing: it is what turns a recorder into
 * a campaign, and clicking a dataset is the one hand-off from collect to
 * review. It is a fixed row rather than a share of the split — the recorder
 * above it must not shrink when a 12th dataset appears on disk.
 */
import { DatasetTab } from "@/components/cockpit/DatasetTab";
import { DatasetShelf } from "@/components/lab/DatasetShelf";
import type { CameraInfo } from "@/lib/api";

export function CollectPane({
  cameras,
  onCameraRecord,
  onOpenDataset,
}: {
  cameras: CameraInfo[];
  /** Lifts the accepted toggle back into the cockpit's one camera list. */
  onCameraRecord: (id: string, record: boolean) => void;
  onOpenDataset: (repoId: string) => void;
}) {
  return (
    <div className="grid min-h-0 grid-rows-[minmax(0,1fr)_auto] gap-2 overflow-hidden">
      <DatasetTab cameras={cameras} onCameraRecord={onCameraRecord} />
      {/* 144px is measured, not chosen: a 34px panel head, 12px of body
          padding, and a card that has to hold an id, a task, a stats row and
          the mark bar with its counts under it. */}
      <div className="h-[144px] shrink-0 px-2 pb-2">
        <DatasetShelf layout="strip" onOpen={onOpenDataset} />
      </div>
    </div>
  );
}
