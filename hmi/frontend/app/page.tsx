// hmi/frontend/app/page.tsx
"use client";
import { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { TelemetryBar } from "@/components/TelemetryBar";
import { ArmPanel } from "@/components/ArmPanel";
import { api } from "@/lib/api";

export default function Dashboard() {
  const [cfg, setCfg] = useState<Awaited<ReturnType<typeof api.config>> | null>(null);
  useEffect(() => { api.config().then(setCfg).catch(console.error); }, []);
  if (!cfg) return <div className="p-3 text-sm">Loading config…</div>;
  return (
    <main className="p-3 space-y-3">
      <div className="flex items-baseline justify-between">
        <h1 className="text-lg font-semibold">Haller HMI</h1>
        <TelemetryBar />
      </div>
      <div className="grid grid-cols-12 gap-3">
        <Card className="col-span-7">
          <CardHeader><CardTitle>Base</CardTitle></CardHeader>
          <CardContent>Base panel coming in Task 14.</CardContent>
        </Card>
        {cfg.arms.map((arm) => (
          <div key={arm.id} className="col-span-5">
            <ArmPanel armId={arm.id} />
          </div>
        ))}
      </div>
    </main>
  );
}
