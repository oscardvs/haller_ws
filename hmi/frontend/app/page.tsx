// hmi/frontend/app/page.tsx
"use client";
import { useEffect, useState } from "react";
import { ArmPanel } from "@/components/ArmPanel";
import { BasePanel } from "@/components/BasePanel";
import { CamerasPanel } from "@/components/CamerasPanel";
import { RecordingPanel } from "@/components/RecordingPanel";
import { TeleopLauncher } from "@/components/TeleopLauncher";
import { api } from "@/lib/api";
import { fetchCalibrationStatus, type CalibrationStatusResponse } from "@/lib/calibration";
import { BACKEND_URL } from "@/lib/config";

export default function Dashboard() {
  const [cfg, setCfg] = useState<Awaited<ReturnType<typeof api.config>> | null>(null);
  const [calStatus, setCalStatus] = useState<CalibrationStatusResponse | null>(null);
  const [wizardArm, setWizardArm] = useState<string | null>(null);

  useEffect(() => {
    api.config().then(setCfg).catch(console.error);
  }, []);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const s = await fetchCalibrationStatus(BACKEND_URL);
        if (!cancelled) setCalStatus(s);
      } catch { /* offline */ }
    })();
    return () => { cancelled = true; };
  }, []);

  const uncalibrated = (calStatus?.arms ?? []).filter(a => !a.has_file);

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

      {uncalibrated.length > 0 && (
        <div className="space-y-2">
          {uncalibrated.map(a => (
            <div key={a.id}
                 className="bg-amber-900/30 border border-amber-700 rounded-md px-4 py-2 flex items-center justify-between">
              <span className="text-sm">Arm <strong>{a.id}</strong> has no calibration file.</span>
              <button
                className="underline text-amber-300 text-sm"
                onClick={() => setWizardArm(a.id)}
              >Calibrate {a.id}</button>
            </div>
          ))}
        </div>
      )}

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

      {wizardArm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card border rounded p-6 max-w-sm">
            <p className="text-sm mb-3">Wizard for {wizardArm} — T10 not yet implemented.</p>
            <button
              className="px-3 py-1.5 border rounded text-sm"
              onClick={async () => {
                setWizardArm(null);
                try {
                  setCalStatus(await fetchCalibrationStatus(BACKEND_URL));
                } catch { /* offline */ }
              }}
            >Close</button>
          </div>
        </div>
      )}
    </main>
  );
}
