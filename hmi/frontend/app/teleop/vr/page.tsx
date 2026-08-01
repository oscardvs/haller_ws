"use client";

import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import { DeepLinkChrome } from "@/components/DeepLinkChrome";
import { VRTeleopPanel } from "@/components/VRTeleopPanel";

export default function VRTeleopPage() {
  const [armIds, setArmIds] = useState<string[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.config()
      .then((cfg) => setArmIds(cfg.arms.map((a) => a.id)))
      .catch((e: Error) => setErr(e.message));
  }, []);

  if (err) {
    return (
      <>
        <DeepLinkChrome label="VR teleop" />
        <main className="p-4 font-mono text-sm text-destructive">
          config load failed: {err}
        </main>
      </>
    );
  }

  return (
    <>
      <DeepLinkChrome label="VR teleop" />
      <main className="p-4 space-y-3">
        <header>
          <h1 className="text-lg font-mono">VR Teleop</h1>
          <p className="text-[12px] text-muted-foreground">
            Meta Quest · passthrough AR in the headset browser · per-hand grip to drive · B/Y = E-STOP
          </p>
        </header>
        {armIds.length >= 2 ? (
          <VRTeleopPanel armIds={armIds} />
        ) : (
          <div className="space-y-1 text-[12px] font-mono text-muted-foreground">
            <div>
              VR teleop needs ≥2 enabled arms in{" "}
              <code>hmi/backend/config.yaml</code> — currently{" "}
              {armIds.length === 0 ? "none" : `only "${armIds[0]}"`}.
            </div>
            <div>
              The teleop session drives both sides every tick, so it needs two
              distinct arm handles. Single-arm teleop would be a real change to
              the commit loop, not a config flag.
            </div>
          </div>
        )}
      </main>
    </>
  );
}
