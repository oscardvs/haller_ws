"use client";

/**
 * RecordingPanel — HMI-integrated bimanual dataset recorder (v0).
 *
 * Drives the in-process recorder (POST /record/start|stop, GET /record/status),
 * which logs both arms + cameras + base into a LeRobotDataset while you
 * demonstrate via the human-pose teleop. This replaces the old CLI-command-copy
 * flow: `lerobot-record` assumes a leader→follower pair and can't capture
 * Haller's two-follower bimanual rig, so recording lives inside the HMI.
 *
 * Workflow: start human-pose teleop first, then Start recording and demonstrate
 * with the dead-man held — the recorder logs the teleop's commanded joint
 * targets as `action` and the measured joints as `observation.state`.
 */
import { useEffect, useRef, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";

import { api, type RecordStatus } from "@/lib/api";

function slugify(s: string): string {
  return (
    s.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 60) || "task"
  );
}

export function RecordingPanel() {
  const [task, setTask] = useState("Pick the red cube and place it in the box");
  const [hfUser, setHfUser] = useState("");
  const [status, setStatus] = useState<RecordStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const repoId = `${hfUser || "local"}/haller_${slugify(task)}`;
  const recording = status?.recording ?? false;

  // Poll status while mounted so the live frame count updates during a take.
  useEffect(() => {
    const tick = async () => {
      try {
        setStatus(await api.recordStatus());
      } catch {
        /* backend not ready / recorder unavailable — keep last known status */
      }
    };
    tick();
    pollRef.current = setInterval(tick, 1000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  async function start() {
    setBusy(true);
    try {
      setStatus(await api.recordStart(repoId, task));
      toast.success(`recording → ${repoId}`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function stop(save: boolean) {
    setBusy(true);
    try {
      const s = await api.recordStop(save);
      setStatus(s);
      if (save) toast.success(`saved ${s.episode_frames} frames`);
      else toast("take discarded");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card className="overflow-hidden p-0">
      <div className="flex items-center justify-between gap-2 px-3 h-9 border-b border-border bg-card">
        <div className="flex items-center gap-2">
          <span className="label-micro text-muted-foreground">Recording</span>
          <Badge variant={recording ? "default" : "secondary"}>
            {recording ? "● rec" : "HMI · v0"}
          </Badge>
        </div>
        <span className="font-mono text-[10px] text-muted-foreground">
          {recording && status ? (
            <>
              frames: <span className="text-foreground">{status.episode_frames}</span>
            </>
          ) : (
            <>
              dataset: <span className="text-foreground">{repoId}</span>
            </>
          )}
        </span>
      </div>

      <CardContent className="p-3 space-y-3">
        <div className="text-[11px] text-muted-foreground">
          Records both arms + cameras + base into a LeRobotDataset from inside the
          HMI. Start human-pose teleop first, then Start recording and demonstrate
          with the dead-man held.
        </div>

        <div className="grid grid-cols-12 gap-2">
          <label className="col-span-8 flex flex-col gap-1">
            <span className="label-tracked text-muted-foreground">Task</span>
            <Input
              value={task}
              onChange={(e) => setTask(e.target.value)}
              disabled={recording}
              placeholder="What should the policy learn?"
              className="font-mono text-[12px] h-7"
            />
          </label>
          <label className="col-span-4 flex flex-col gap-1">
            <span className="label-tracked text-muted-foreground">HF user</span>
            <Input
              value={hfUser}
              onChange={(e) => setHfUser(e.target.value)}
              disabled={recording}
              placeholder="osrdvs"
              className="font-mono text-[12px] h-7"
            />
          </label>
        </div>

        <div className="rounded-sm border border-border bg-muted/30 p-2 font-mono text-[11px] break-all">
          {repoId}
        </div>

        {status?.last_error && (
          <div className="text-[11px] text-red-500 break-all">
            recorder error: {status.last_error}
          </div>
        )}

        <div className="flex items-center gap-2">
          {!recording ? (
            <Button
              size="sm"
              className="h-7 px-3 label-micro"
              disabled={busy}
              onClick={start}
            >
              start recording
            </Button>
          ) : (
            <>
              <Button
                size="sm"
                className="h-7 px-3 label-micro"
                disabled={busy}
                onClick={() => stop(true)}
              >
                stop &amp; save
              </Button>
              <Button
                size="sm"
                variant="secondary"
                className="h-7 px-3 label-micro"
                disabled={busy}
                onClick={() => stop(false)}
              >
                discard
              </Button>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
