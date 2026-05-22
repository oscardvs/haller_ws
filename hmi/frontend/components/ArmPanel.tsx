// hmi/frontend/components/ArmPanel.tsx
"use client";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";

import { useTelemetry } from "@/lib/telemetry";
import { api, cameraStreamUrl, type CameraInfo } from "@/lib/api";
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
  const teleop = useTelemetry((s) => s.lastFrame?.teleop);
  const teleopRole: "leader" | "follower" | null =
    teleop?.running && teleop.leader === armId
      ? "leader"
      : teleop?.running && teleop.follower === armId
        ? "follower"
        : null;
  const isCalibrating = Boolean(arm?.calibration);
  const [presetName, setPresetName] = useState("");
  const [presets, setPresets] = useState<string[] | null>(null);
  const [wristCam, setWristCam] = useState<CameraInfo | null>(null);
  const joints = useMemo(() => Object.entries(arm?.joints ?? {}), [arm]);

  // Find the wrist camera bound to this arm (config.yaml: arm_id + role: wrist).
  useEffect(() => {
    let cancelled = false;
    api
      .cameras()
      .then((r) => {
        if (cancelled) return;
        const match = r.cameras.find(
          (c) => c.role === "wrist" && c.arm_id === armId,
        );
        setWristCam(match ?? null);
      })
      .catch(() => setWristCam(null));
    return () => {
      cancelled = true;
    };
  }, [armId]);

  const refreshPresets = useCallback(async () => {
    try {
      const { names } = await api.armPresetsList(armId);
      setPresets(names.sort((a, b) => a.localeCompare(b)));
    } catch (e) {
      // Don't toast on initial-load failure (backend may not be up yet); telemetry will reconnect.
      console.error("refreshPresets failed", e);
    }
  }, [armId]);

  useEffect(() => {
    refreshPresets();
  }, [refreshPresets]);

  if (!arm) {
    return (
      <Card className="border-dashed">
        <CardContent className="p-3 font-mono text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
          arm · {armId} · awaiting telemetry…
        </CardContent>
      </Card>
    );
  }

  const disabled = arm.mode !== "manual" || teleopRole !== null || isCalibrating;
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
          {isCalibrating ? (
            <span className="ml-2 inline-flex items-center gap-1.5 rounded-sm bg-blue-700 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.12em] text-blue-50">
              calibrating
            </span>
          ) : teleopRole ? (
            <span className="ml-2 inline-flex items-center gap-1.5 rounded-sm bg-[var(--haller-live)]/20 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.12em] text-[var(--haller-live)]">
              teleop · {teleopRole}
            </span>
          ) : null}
        </div>
        <ModeToggle armId={armId} mode={arm.mode} />
      </div>

      <CardContent className="p-3 space-y-3">
        <CameraTile
          id={wristCam?.id ?? `${armId}_wrist`}
          role="wrist"
          streamUrl={wristCam?.active ? cameraStreamUrl(wristCam.id) : undefined}
          active={wristCam?.active}
          width={wristCam?.width}
          height={wristCam?.height}
          fps={wristCam?.fps}
        />

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

        {/* Action row — Home + Free-drive. */}
        <div className="flex items-center gap-2 pt-2 border-t border-border/70">
          <span className="label-tracked text-muted-foreground shrink-0">
            Actions
          </span>
          <Button
            size="sm"
            className="h-7 px-2 label-micro"
            disabled={disabled}
            onClick={async () => {
              try {
                await api.armHome(armId);
                toast.success("homing arm");
              } catch (e) {
                toast.error(`home failed: ${(e as Error).message}`);
              }
            }}
          >
            home
          </Button>
          <Button
            size="sm"
            variant={torqueOn ? "outline" : "default"}
            className="h-7 px-2 label-micro"
            onClick={async () => {
              try {
                await api.armTorque(armId, !torqueOn);
                toast.message(
                  torqueOn
                    ? "torque off — arm is free-drivable"
                    : "torque engaged",
                );
              } catch (e) {
                toast.error(`torque toggle failed: ${(e as Error).message}`);
              }
            }}
          >
            {torqueOn ? "free-drive" : "engage"}
          </Button>
        </div>

        {/* Preset row — save + named-go. */}
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
                setPresetName("");
                refreshPresets();
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

        {/* Saved-poses list */}
        {presets && presets.length > 0 ? (
          <div className="flex flex-wrap items-center gap-1.5 pt-2">
            <span className="label-tracked text-muted-foreground shrink-0 mr-1">
              Saved
            </span>
            {presets.map((name) => (
              <span
                key={name}
                className="inline-flex items-center gap-1 rounded-sm border border-border bg-muted/40 pl-2 pr-1 h-6 font-mono text-[11px]"
              >
                <button
                  type="button"
                  disabled={disabled}
                  className="text-foreground/90 hover:text-[var(--haller-live)] disabled:cursor-not-allowed disabled:text-muted-foreground"
                  title={disabled ? "switch to manual to replay" : `go to ${name}`}
                  onClick={async () => {
                    try {
                      await api.armPreset(armId, name);
                    } catch (e) {
                      toast.error((e as Error).message);
                    }
                  }}
                >
                  {name}
                </button>
                <button
                  type="button"
                  className="ml-0.5 inline-flex h-4 w-4 items-center justify-center rounded-sm text-muted-foreground hover:bg-destructive/20 hover:text-destructive"
                  title={`delete ${name}`}
                  onClick={async () => {
                    try {
                      await api.armPresetDelete(armId, name);
                      toast.message(`deleted ${name}`);
                      refreshPresets();
                    } catch (e) {
                      toast.error((e as Error).message);
                    }
                  }}
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        ) : presets && presets.length === 0 ? (
          <div className="pt-2 label-micro text-muted-foreground">
            no saved poses yet — record one above
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
