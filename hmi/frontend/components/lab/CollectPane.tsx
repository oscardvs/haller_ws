"use client";

/**
 * Collect: pick the dataset, run the VR session, record the take.
 *
 * The left rail holds the two decisions that frame a take — WHICH dataset it
 * extends (CollectResumeCard) and WHAT session demonstrates it
 * (CollectSessionCard) — beside the cockpit's Dataset tab, mounted verbatim:
 * its camera take-composition grid, recorder card and on-disk episode browser
 * already answer "what will be in this take" and "did it save", and rebuilding
 * them here would give the operator two recorders that disagree.
 *
 * The shelf underneath is what turns a recorder into a campaign, and clicking
 * a dataset is the one hand-off from collect to review. It stays a fixed row
 * rather than a share of the split — the recorder above it must not shrink
 * when a 12th dataset appears on disk.
 */
import { useCallback, useState } from "react";

import { DatasetTab } from "@/components/cockpit/DatasetTab";
import type { ConfigArm } from "@/components/cockpit/teleopPresets";
import { CollectResumeCard } from "@/components/lab/CollectResumeCard";
import { CollectSessionCard } from "@/components/lab/CollectSessionCard";
import { DatasetShelf } from "@/components/lab/DatasetShelf";
import type { CameraInfo } from "@/lib/api";

export function CollectPane({
  cameras,
  arms,
  onCameraRecord,
  onOpenDataset,
}: {
  cameras: CameraInfo[];
  /** The rig's configured arms, off /config — the session card computes its
   *  presets from these, never from a hardcoded pair. */
  arms: ConfigArm[];
  /** Lifts the accepted toggle back into the cockpit's one camera list. */
  onCameraRecord: (id: string, record: boolean) => void;
  onOpenDataset: (repoId: string) => void;
}) {
  /** The resumed dataset's write rate, lifted out of the resume card so the
   *  session card and the recorder can both steer on it BEFORE a take is
   *  refused by the append gate. Null while the draft is a new dataset. */
  const [datasetFps, setDatasetFps] = useState<number | null>(null);
  const onDatasetFps = useCallback(
    (fps: number | null) => setDatasetFps(fps),
    [],
  );

  return (
    <div className="grid min-h-0 grid-rows-[minmax(0,1fr)_auto] gap-2 overflow-hidden">
      <div className="grid min-h-0 grid-cols-[300px_minmax(0,1fr)] gap-2 overflow-hidden">
        {/* The rail scrolls itself; the recorder beside it must not move when
            the repo list or the preset list grows. */}
        <div className="flex min-h-0 flex-col gap-2 overflow-y-auto py-2 pl-2">
          <CollectResumeCard onDatasetFps={onDatasetFps} />
          <CollectSessionCard arms={arms} datasetFps={datasetFps} />
        </div>
        <DatasetTab
          cameras={cameras}
          onCameraRecord={onCameraRecord}
          datasetFps={datasetFps}
        />
      </div>
      {/* 144px is measured, not chosen: a 34px panel head, 12px of body
          padding, and a card that has to hold an id, a task, a stats row and
          the mark bar with its counts under it. */}
      <div className="h-[144px] shrink-0 px-2 pb-2">
        <DatasetShelf layout="strip" onOpen={onOpenDataset} />
      </div>
    </div>
  );
}
