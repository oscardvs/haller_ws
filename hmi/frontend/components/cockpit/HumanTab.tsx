"use client";

/**
 * Human-pose teleop, in the cockpit.
 *
 * Thin on purpose: HumanTeleopPanel owns the webcam, the three MediaPipe
 * models, the ~30 Hz publish loop and the calibration state, and none of that
 * should be duplicated or re-parented for a layout change. This supplies the
 * cockpit's sizing and the inline sim tile, and forwards whether the tab is
 * the visible one so the panel can scope the dead-man key.
 */
import { HumanTeleopPanel } from "@/components/HumanTeleopPanel";
import { SimViewTile } from "@/components/SimViewTile";

export function HumanTab({
  armIds,
  active,
}: {
  armIds: string[];
  active: boolean;
}) {
  if (armIds.length < 2) {
    return (
      <div className="flex min-h-0 items-center justify-center p-2">
        <span className="font-mono text-[11px] text-muted-foreground">
          human teleop needs ≥2 enabled arms in hmi/backend/config.yaml
        </span>
      </div>
    );
  }

  return (
    <div className="min-h-0 overflow-hidden p-2">
      <HumanTeleopPanel
        armIds={armIds}
        active={active}
        fill
        // In-flow rather than pinned. SimViewTile picks the three-quarter
        // camera (overhead flattens away shoulder_lift and elbow_flex, the
        // exact joints you watch while teleoping) and renders nothing at all
        // when the running config has no sim camera — i.e. on real hardware.
        simTile={<SimViewTile placement="inline" />}
      />
    </div>
  );
}
