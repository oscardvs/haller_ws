"use client";

import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import { HumanTeleopPanel } from "@/components/HumanTeleopPanel";
import { SimViewTile } from "@/components/SimViewTile";

export default function HumanTeleopPage() {
  const [armIds, setArmIds] = useState<string[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.config()
      .then((cfg) => setArmIds(cfg.arms.map((a) => a.id)))
      .catch((e: Error) => setErr(e.message));
  }, []);

  if (err) {
    return (
      <main className="p-4 font-mono text-sm text-destructive">
        config load failed: {err}
      </main>
    );
  }

  return (
    <main className="p-4 space-y-3">
      <header className="flex items-baseline justify-between">
        <div>
          <h1 className="text-lg font-mono">Human Teleop</h1>
          <p className="text-[12px] text-muted-foreground">
            bimanual · monocular RGB · hold <kbd>SPACE</kbd> to drive
          </p>
        </div>
      </header>
      {armIds.length >= 2 ? (
        <HumanTeleopPanel armIds={armIds} />
      ) : (
        <div className="text-[12px] font-mono text-muted-foreground">
          human teleop needs ≥2 enabled arms in <code>hmi/backend/config.yaml</code>
        </div>
      )}
      {/* Pinned overlay, so you can watch the robot while driving it. Renders
          nothing unless the running config has a sim camera. */}
      <SimViewTile />
    </main>
  );
}
