// hmi/frontend/components/TelemetryBar.tsx
"use client";
import { useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { useTelemetry } from "@/lib/telemetry";

export function TelemetryBar() {
  const { connected, lastFrame, start } = useTelemetry();
  useEffect(() => { start(); }, [start]);
  const t = lastFrame?.t;
  return (
    <div className="flex items-center gap-3 text-xs font-mono text-muted-foreground">
      <Badge variant={connected ? "default" : "destructive"}>
        {connected ? "live" : "disconnected"}
      </Badge>
      {t ? <span>t={new Date(t * 1000).toLocaleTimeString()}</span> : null}
      {lastFrame ? (
        <>
          <span>v={lastFrame.base.linear.toFixed(2)} m/s</span>
          <span>ω={lastFrame.base.angular.toFixed(2)} rad/s</span>
          <span>x={lastFrame.base.odom.x.toFixed(2)}</span>
          <span>y={lastFrame.base.odom.y.toFixed(2)}</span>
        </>
      ) : null}
    </div>
  );
}
