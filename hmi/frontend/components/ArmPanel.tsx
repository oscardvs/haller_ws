// hmi/frontend/components/ArmPanel.tsx
"use client";
import { useMemo, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";

import { useTelemetry } from "@/lib/telemetry";
import { api } from "@/lib/api";
import { JointSlider } from "./JointSlider";
import { ModeToggle } from "./ModeToggle";
import { CameraTile } from "./CameraTile";

/**
 * ArmPanel — per-arm supervisory pane.
 *
 *  Header strip: arm id + mode toggle (the most-used row).
 *  Body:         wrist-camera tile + dense joint stack.
 *  Footer:       preset save/load — quiet utility row.
 *
 * Mode determines whether joints are operator-editable.
 */
export function ArmPanel({ armId }: { armId: string }) {
  const arm = useTelemetry((s) => s.lastFrame?.arms[armId]);
  const [presetName, setPresetName] = useState("");
  const joints = useMemo(() => Object.entries(arm?.joints ?? {}), [arm]);

  if (!arm) {
    return (
      <Card className="border-dashed">
        <CardContent className="p-3 font-mono text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
          arm · {armId} · awaiting telemetry…
        </CardContent>
      </Card>
    );
  }

  const disabled = arm.mode !== "manual";
  const torqueOn = joints.some(([, j]) => j.torque);

  return (
    <Card className="overflow-hidden p-0">
      {/* Header strip — flush, no padding. */}
      <div className="flex items-center justify-between gap-2 px-3 h-9 border-b border-border bg-card">
        <div className="flex items-center gap-2">
          <span className="label-micro text-muted-foreground">Arm</span>
          <span className="font-mono text-[12px] font-semibold tracking-[0.12em] uppercase text-foreground">
            {armId}
          </span>
          <span className="ml-2 inline-flex items-center gap-1.5 label-micro text-muted-foreground">
            <span
              className={`inline-block h-1.5 w-1.5 rounded-[1px] ${
                torqueOn ? "bg-[var(--haller-live)]" : "bg-muted-foreground/50"
              }`}
            />
            {torqueOn ? "torque" : "free"}
          </span>
        </div>
        <ModeToggle armId={armId} mode={arm.mode} />
      </div>

      <CardContent className="p-3 space-y-3">
        <CameraTile id={`${armId}_wrist`} role="wrist" />

        {/* Joint stack header */}
        <div className="flex items-center gap-2 pb-1 border-b border-border/70">
          <span className="label-tracked text-muted-foreground">Joints</span>
          <span className="h-px flex-1 bg-border/70" />
          <span className="label-micro text-muted-foreground">
            {joints.length} dof
          </span>
        </div>

        <div className="space-y-2.5">
          {joints.map(([name, j]) => (
            <JointSlider
              key={name}
              name={name}
              pos={j.pos}
              min={j.min}
              max={j.max}
              disabled={disabled}
              onChange={async (v) => {
                try {
                  await api.armGoal(armId, { [name]: v });
                } catch (e) {
                  toast.error(`${name} goal failed: ${(e as Error).message}`);
                }
              }}
            />
          ))}
        </div>

        {/* Preset row — quieter, separated. */}
        <div className="flex items-center gap-2 pt-2 border-t border-border/70">
          <span className="label-tracked text-muted-foreground shrink-0">
            Pose
          </span>
          <Input
            placeholder="preset name"
            value={presetName}
            onChange={(e) => setPresetName(e.target.value)}
            className="h-7 max-w-[140px] font-mono text-[12px]"
          />
          <Button
            size="sm"
            className="h-7 px-2 label-micro"
            disabled={!presetName}
            onClick={async () => {
              try {
                await api.armPresetRecord(armId, presetName);
                toast.success(`saved ${presetName}`);
              } catch (e) {
                toast.error((e as Error).message);
              }
            }}
          >
            save
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="h-7 px-2 label-micro"
            disabled={!presetName || disabled}
            onClick={async () => {
              try {
                await api.armPreset(armId, presetName);
              } catch (e) {
                toast.error((e as Error).message);
              }
            }}
          >
            go to
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
