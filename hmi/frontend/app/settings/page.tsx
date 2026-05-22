// hmi/frontend/app/settings/page.tsx
"use client";
import { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";

type ConfigBody = Awaited<ReturnType<typeof api.config>>;

export default function SettingsPage() {
  const [cfg, setCfg] = useState<ConfigBody | null>(null);
  const [health, setHealth] = useState<{ status: string } | null>(null);
  useEffect(() => {
    api.config().then(setCfg).catch(console.error);
    api.health().then(setHealth).catch(console.error);
  }, []);
  return (
    <main className="p-3 space-y-3">
      <h1 className="text-lg font-semibold">Settings</h1>
      <Card>
        <CardHeader><CardTitle>Health</CardTitle></CardHeader>
        <CardContent className="text-sm font-mono">
          status: <Badge>{health?.status ?? "…"}</Badge>
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle>Arms</CardTitle></CardHeader>
        <CardContent>
          {cfg?.arms.map((a) => (
            <div key={a.id} className="text-sm font-mono py-1 flex gap-3">
              <span className="w-16">{a.id}</span>
              <span>{a.model}</span>
              <span className="text-muted-foreground">{a.port}</span>
              <Badge variant="outline">{a.mode}</Badge>
            </div>
          ))}
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle>Cameras</CardTitle></CardHeader>
        <CardContent>
          {cfg?.cameras.map((c) => (
            <div key={c.id} className="text-sm font-mono py-1 flex gap-3">
              <span className="w-32">{c.id}</span>
              <span>{c.role}</span>
              <Badge variant="outline">{c.source}</Badge>
            </div>
          ))}
        </CardContent>
      </Card>
    </main>
  );
}
