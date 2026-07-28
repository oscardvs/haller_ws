"use client";

/**
 * Operate: the everyday screen. Primary camera + drive console on the left,
 * one card per arm on the right.
 *
 * Below COMPACT_W the two arm columns collapse to one with a picker above it —
 * two 226px arm cards side by side stop being readable long before the browser
 * stops being able to draw them.
 */
import type { CameraInfo } from "@/lib/api";
import { ArmCard } from "./ArmCard";
import { DriveConsole } from "./DriveConsole";
import { PrimaryCameraPanel } from "./PrimaryCameraPanel";
import { useSticky, type PopId, type Viewport } from "./lib";

export function OperateTab({
  armIds,
  cameras,
  viewport,
  pop,
  setPop,
}: {
  armIds: string[];
  cameras: CameraInfo[];
  viewport: Viewport;
  pop: PopId;
  setPop: (p: PopId) => void;
}) {
  const { compact, short } = viewport;
  const [shownArm, setShownArm] = useSticky("operate.shownArm", armIds[0] ?? "");
  const visible =
    compact && armIds.length > 1
      ? [armIds.includes(shownArm) ? shownArm : armIds[0]]
      : armIds;

  return (
    <div
      className="grid min-h-0 gap-2 overflow-hidden p-2"
      style={{
        gridTemplateColumns: compact
          ? "minmax(320px,1fr) minmax(280px,340px)"
          : `minmax(360px,1.45fr) repeat(${Math.max(1, armIds.length)}, minmax(226px,330px))`,
      }}
    >
      <div className="grid min-h-0 grid-rows-[minmax(140px,1fr)_auto] gap-2">
        <PrimaryCameraPanel cameras={cameras} />
        <DriveConsole />
      </div>

      {compact && armIds.length > 1 ? (
        <div className="flex min-h-0 flex-col gap-2">
          <div className="flex shrink-0 items-center gap-1.5 rounded-md bg-muted p-1">
            <span className="pl-1.5 label-micro text-muted-foreground">Arm</span>
            {armIds.map((id) => (
              <button
                key={id}
                type="button"
                onClick={() => setShownArm(id)}
                aria-pressed={visible[0] === id}
                className={
                  "h-6 flex-1 rounded-sm label-micro " +
                  (visible[0] === id
                    ? "bg-card text-foreground"
                    : "text-muted-foreground hover:text-foreground")
                }
              >
                {id}
              </button>
            ))}
          </div>
          {visible.map((id) => (
            <ArmCard
              key={id}
              armId={id}
              cameras={cameras}
              short={short}
              poseOpen={pop === `pose-${id}`}
              onTogglePose={() =>
                setPop(pop === `pose-${id}` ? null : (`pose-${id}` as PopId))
              }
            />
          ))}
        </div>
      ) : (
        visible.map((id) => (
          <div key={id} className="flex min-h-0 flex-col">
            <ArmCard
              armId={id}
              cameras={cameras}
              short={short}
              poseOpen={pop === `pose-${id}`}
              onTogglePose={() =>
                setPop(pop === `pose-${id}` ? null : (`pose-${id}` as PopId))
              }
            />
          </div>
        ))
      )}
    </div>
  );
}
