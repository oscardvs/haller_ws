// hmi/frontend/components/ArmPanel.tsx
"use client";
import { useMemo, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";

import { useTelemetry } from "@/lib/telemetry";
import { api } from "@/lib/api";
import { JointSlider } from "./JointSlider";
import { ModeToggle } from "./ModeToggle";
import { CameraTile } from "./CameraTile";

export function ArmPanel({ armId }: { armId: string }) {
  const arm = useTelemetry((s) => s.lastFrame?.arms[armId]);
  const [presetName, setPresetName] = useState("");
  const joints = useMemo(() => Object.entries(arm?.joints ?? {}), [arm]);

  if (!arm) return <div className="text-xs text-muted-foreground">no telemetry yet</div>;
  const disabled = arm.mode !== "manual";

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-2">
        <CardTitle>Arm: {armId}</CardTitle>
        <ModeToggle armId={armId} mode={arm.mode} />
      </CardHeader>
      <CardContent className="space-y-4">
        <CameraTile id={`${armId}_wrist`} role="wrist" />
        <div className="space-y-2">
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
        <div className="flex items-center gap-2">
          <Input
            placeholder="preset name"
            value={presetName}
            onChange={(e) => setPresetName(e.target.value)}
            className="max-w-[160px] h-8"
          />
          <Button
            size="sm"
            disabled={!presetName}
            onClick={async () => {
              try { await api.armPresetRecord(armId, presetName); toast.success(`saved ${presetName}`); }
              catch (e) { toast.error((e as Error).message); }
            }}
          >Save pose</Button>
          <Button
            size="sm"
            variant="outline"
            disabled={!presetName || disabled}
            onClick={async () => {
              try { await api.armPreset(armId, presetName); }
              catch (e) { toast.error((e as Error).message); }
            }}
          >Go to pose</Button>
        </div>
      </CardContent>
    </Card>
  );
}
