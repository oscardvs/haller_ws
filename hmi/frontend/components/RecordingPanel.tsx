"use client";

/**
 * RecordingPanel — UI for the CLI dataset-collection flow (Phase 1).
 *
 * Recording itself runs out-of-process via scripts/record_dataset.sh because
 * lerobot-record wants exclusive control of the serial ports + cameras (the
 * HMI must be stopped while it runs). Instead of pretending the HMI can
 * launch it directly, this panel just *builds the shell command* from the
 * task + episode-count inputs and offers a Copy button.
 *
 * Phase 2 (HMI-integrated recorder) will replace this with start/save/discard
 * buttons that drive recording from inside the HMI process; see
 * docs/setup/dataset-collection.md.
 */
import { useMemo, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";

function slugify(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 60) || "task";
}

function shellQuote(s: string): string {
  // POSIX single-quote escape: 'foo'\''bar' for foo'bar.
  return `'${s.replace(/'/g, `'\\''`)}'`;
}

export function RecordingPanel() {
  const [task, setTask] = useState("Pick the red cube and place it in the box");
  const [episodes, setEpisodes] = useState(20);

  const slug = useMemo(() => slugify(task), [task]);
  const command = useMemo(
    () => `scripts/record_dataset.sh ${shellQuote(task)} ${episodes}`,
    [task, episodes],
  );

  return (
    <Card className="overflow-hidden p-0">
      <div className="flex items-center justify-between gap-2 px-3 h-9 border-b border-border bg-card">
        <div className="flex items-center gap-2">
          <span className="label-micro text-muted-foreground">Recording</span>
          <Badge variant="secondary">CLI · phase 1</Badge>
        </div>
        <span className="font-mono text-[10px] text-muted-foreground">
          dataset: <span className="text-foreground">…/so101_{slug}</span>
        </span>
      </div>

      <CardContent className="p-3 space-y-3">
        <div className="text-[11px] text-muted-foreground">
          Recording owns the serial ports + cameras, so the HMI must be stopped
          while it runs. Fill in the task + episode count, copy the command,
          stop the HMI in your other terminal, then run it.
        </div>

        <div className="grid grid-cols-12 gap-2">
          <label className="col-span-9 flex flex-col gap-1">
            <span className="label-tracked text-muted-foreground">Task</span>
            <Input
              value={task}
              onChange={(e) => setTask(e.target.value)}
              placeholder="What should the policy learn?"
              className="font-mono text-[12px] h-7"
            />
          </label>
          <label className="col-span-3 flex flex-col gap-1">
            <span className="label-tracked text-muted-foreground">Episodes</span>
            <Input
              type="number"
              min={1}
              max={500}
              value={episodes}
              onChange={(e) => {
                const n = Number(e.target.value);
                if (Number.isFinite(n)) setEpisodes(Math.max(1, Math.min(500, Math.round(n))));
              }}
              className="font-mono text-[12px] h-7"
            />
          </label>
        </div>

        <div className="rounded-sm border border-border bg-muted/30 p-2 font-mono text-[11px] break-all">
          {command}
        </div>

        <div className="flex items-center gap-2">
          <Button
            size="sm"
            className="h-7 px-3 label-micro"
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(command);
                toast.success("command copied to clipboard");
              } catch {
                toast.error("clipboard access denied — select and copy manually");
              }
            }}
          >
            copy command
          </Button>
          <a
            href="https://github.com/oscardvs/haller_ws/blob/main/docs/setup/dataset-collection.md"
            target="_blank"
            rel="noreferrer"
            className="label-micro text-muted-foreground hover:text-foreground"
          >
            docs/setup/dataset-collection.md
          </a>
        </div>
      </CardContent>
    </Card>
  );
}
