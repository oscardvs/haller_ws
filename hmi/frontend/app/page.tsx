// hmi/frontend/app/page.tsx
"use client";
import { useEffect, useState } from "react";
import { ArmPanel } from "@/components/ArmPanel";
import { BasePanel } from "@/components/BasePanel";
import { CamerasPanel } from "@/components/CamerasPanel";
import { RecordingPanel } from "@/components/RecordingPanel";
import { TeleopLauncher } from "@/components/TeleopLauncher";
import { api } from "@/lib/api";

export default function Dashboard() {
  const [cfg, setCfg] = useState<Awaited<ReturnType<typeof api.config>> | null>(null);
  useEffect(() => {
    api.config().then(setCfg).catch(console.error);
  }, []);

  if (!cfg) {
    return (
      <main className="p-3">
        <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          loading config…
        </div>
      </main>
    );
  }

  return (
    <main className="p-3 space-y-3">
      {/* Section register: tiny breadcrumb so dashboard reads as one screen of many. */}
      <div className="flex items-center gap-3 px-1">
        <span className="label-tracked text-muted-foreground">Overview</span>
        <span className="h-px flex-1 bg-border" />
        <span className="label-micro text-muted-foreground">
          {cfg.arms.length} arm{cfg.arms.length === 1 ? "" : "s"} · {cfg.cameras.length} cam
          {cfg.cameras.length === 1 ? "" : "s"}
        </span>
      </div>

      <TeleopLauncher armIds={cfg.arms.map((a) => a.id)} />

      <div className="grid grid-cols-12 gap-3">
        <div className="col-span-12 lg:col-span-7">
          <BasePanel />
        </div>
        <div className="col-span-12 lg:col-span-5 grid grid-cols-1 gap-3">
          {cfg.arms.map((arm) => (
            <ArmPanel key={arm.id} armId={arm.id} />
          ))}
        </div>
      </div>

      {/* Section: dataset collection. Cameras strip + CLI-command builder. */}
      <div className="flex items-center gap-3 px-1 pt-2">
        <span className="label-tracked text-muted-foreground">Dataset collection</span>
        <span className="h-px flex-1 bg-border" />
      </div>

      <div className="grid grid-cols-12 gap-3">
        <div className="col-span-12 lg:col-span-7">
          <CamerasPanel />
        </div>
        <div className="col-span-12 lg:col-span-5">
          <RecordingPanel />
        </div>
      </div>
    </main>
  );
}
