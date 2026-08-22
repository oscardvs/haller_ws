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
        {armIds.length ? (
          // One arm is a session, not a degenerate case: the absent side never
          // acquires and is never written to, and the panel pairs the arm it
          // does have to the hand the stance puts it under.
          <VRTeleopPanel armIds={armIds} />
        ) : (
          <div className="text-[12px] font-mono text-muted-foreground">
            No arms are enabled in <code>hmi/backend/config.yaml</code>, so
            there is nothing for a session to drive.
          </div>
        )}
      </main>
    </>
  );
}
