"use client";

/**
 * TeleopLauncher — pick leader + follower, start the bidirectional teleop loop.
 *
 * When idle: dropdowns + swap + start.
 * When running: live status (leader -> follower @ Hz, tick count) + stop.
 * Subscribes to telemetry for the live teleop status so it stays in sync with
 * the backend (other operators starting/stopping, E-STOP killing the session).
 */
import { useEffect, useMemo, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";

import { api } from "@/lib/api";
import { useTelemetry } from "@/lib/telemetry";

function NativeSelect({
  value,
  onChange,
  options,
  disabled,
  ariaLabel,
}: {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
  disabled?: boolean;
  ariaLabel?: string;
}) {
  return (
    <select
      aria-label={ariaLabel}
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
      className="h-7 rounded-sm border border-border bg-background px-2 font-mono text-[12px] focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-50"
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

export function TeleopLauncher({ armIds }: { armIds: string[] }) {
  const teleop = useTelemetry((s) => s.lastFrame?.teleop);
  const running = teleop?.running ?? false;

  const defaultLeader = armIds[0] ?? "";
  const defaultFollower = armIds.find((id) => id !== defaultLeader) ?? armIds[0] ?? "";
  const [leader, setLeader] = useState<string>(defaultLeader);
  const [follower, setFollower] = useState<string>(defaultFollower);

  // Stay in sync with what the backend says is running.
  useEffect(() => {
    if (running && teleop?.leader && teleop.follower) {
      setLeader(teleop.leader);
      setFollower(teleop.follower);
    }
  }, [running, teleop?.leader, teleop?.follower]);

  // Make sure follower auto-swaps off the chosen leader.
  useEffect(() => {
    if (leader === follower) {
      const other = armIds.find((id) => id !== leader);
      if (other) setFollower(other);
    }
  }, [leader, follower, armIds]);

  const armOptions = useMemo(
    () => armIds.map((id) => ({ value: id, label: id })),
    [armIds],
  );

  const canStart = armIds.length >= 2 && leader !== follower && !running;

  return (
    <Card className="p-0 overflow-hidden">
      <div className="flex items-center justify-between gap-2 px-3 h-9 border-b border-border bg-card">
        <div className="flex items-center gap-2">
          <span className="label-micro text-muted-foreground">Teleop</span>
          <Badge
            variant={running ? "default" : "secondary"}
            className={running ? "bg-[var(--haller-live)] text-background" : ""}
          >
            {running ? "running" : "idle"}
          </Badge>
          {running && teleop ? (
            <span className="font-mono text-[11px] text-muted-foreground">
              {teleop.leader} → {teleop.follower}
              {typeof teleop.hz === "number" ? ` · ${teleop.hz.toFixed(0)} Hz` : ""}
              {typeof teleop.tick_count === "number" ? ` · ${teleop.tick_count} ticks` : ""}
            </span>
          ) : null}
        </div>
        {running ? (
          <Button
            size="sm"
            variant="destructive"
            className="h-7 px-3 label-micro"
            onClick={async () => {
              try {
                await api.teleopStop();
                toast.message("teleop stopped");
              } catch (e) {
                toast.error(`teleop stop failed: ${(e as Error).message}`);
              }
            }}
          >
            stop
          </Button>
        ) : null}
      </div>

      {!running ? (
        <CardContent className="p-3">
          {armIds.length < 2 ? (
            <div className="label-micro text-muted-foreground">
              teleop needs ≥2 enabled arms in <code>hmi/backend/config.yaml</code>
            </div>
          ) : (
            <div className="flex flex-wrap items-center gap-2">
              <span className="label-tracked text-muted-foreground">Leader</span>
              <NativeSelect
                value={leader}
                onChange={setLeader}
                options={armOptions}
                ariaLabel="leader arm"
              />
              <Button
                size="sm"
                variant="outline"
                className="h-7 px-2 label-micro"
                onClick={() => {
                  const oldLeader = leader;
                  setLeader(follower);
                  setFollower(oldLeader);
                }}
                title="swap leader and follower"
              >
                ⇄
              </Button>
              <span className="label-tracked text-muted-foreground">Follower</span>
              <NativeSelect
                value={follower}
                onChange={setFollower}
                options={armOptions.filter((o) => o.value !== leader)}
                ariaLabel="follower arm"
              />
              <Button
                size="sm"
                className="h-7 px-3 label-micro"
                disabled={!canStart}
                onClick={async () => {
                  try {
                    await api.teleopStart(leader, follower);
                    toast.success(`teleop started: ${leader} → ${follower}`);
                  } catch (e) {
                    toast.error(`teleop start failed: ${(e as Error).message}`);
                  }
                }}
              >
                start
              </Button>
            </div>
          )}
        </CardContent>
      ) : (
        <CardContent className="p-3 text-[12px] font-mono text-muted-foreground">
          back-drive the leader (<span className="text-foreground">{teleop?.leader}</span>) by hand — the follower (<span className="text-foreground">{teleop?.follower}</span>) mirrors it. manual UI on participating arms is disabled.
          {teleop?.last_error ? (
            <div className="mt-2 text-destructive">last error: {teleop.last_error}</div>
          ) : null}
        </CardContent>
      )}
    </Card>
  );
}
