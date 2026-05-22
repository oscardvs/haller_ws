// hmi/frontend/app/settings/page.tsx
"use client";
import { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
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
      <div className="flex items-center gap-3 px-1">
        <span className="label-tracked text-muted-foreground">Settings</span>
        <span className="h-px flex-1 bg-border" />
        <span className="label-micro text-muted-foreground">
          {cfg?.version ? `cfg ${cfg.version}` : "—"}
        </span>
      </div>

      <Card className="overflow-hidden p-0">
        <SectionHeader title="Health" right={health?.status ?? "…"} ok={health?.status === "ok"} />
        <CardContent className="p-3">
          <div className="font-mono text-[12px] text-muted-foreground">
            backend reachable · live websocket
          </div>
        </CardContent>
      </Card>

      <Card className="overflow-hidden p-0">
        <SectionHeader title="Arms" right={`${cfg?.arms.length ?? 0} configured`} />
        <CardContent className="p-0">
          <table className="w-full font-mono text-[12px]">
            <thead>
              <tr className="text-left label-micro text-muted-foreground border-b border-border">
                <th className="px-3 py-1.5 font-semibold">id</th>
                <th className="px-3 py-1.5 font-semibold">model</th>
                <th className="px-3 py-1.5 font-semibold">port</th>
                <th className="px-3 py-1.5 font-semibold">mode</th>
              </tr>
            </thead>
            <tbody>
              {cfg?.arms.map((a) => (
                <tr key={a.id} className="border-b border-border/60 last:border-0">
                  <td className="px-3 py-1.5 text-foreground">{a.id}</td>
                  <td className="px-3 py-1.5 text-muted-foreground">{a.model}</td>
                  <td className="px-3 py-1.5 text-muted-foreground">{a.port}</td>
                  <td className="px-3 py-1.5">
                    <span className="inline-flex items-center gap-1.5 px-1.5 py-0.5 rounded-[2px] border border-border label-micro">
                      <span className="inline-block h-1 w-1 rounded-full bg-muted-foreground" />
                      {a.mode}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      <Card className="overflow-hidden p-0">
        <SectionHeader title="Cameras" right={`${cfg?.cameras.length ?? 0} configured`} />
        <CardContent className="p-0">
          <table className="w-full font-mono text-[12px]">
            <thead>
              <tr className="text-left label-micro text-muted-foreground border-b border-border">
                <th className="px-3 py-1.5 font-semibold">id</th>
                <th className="px-3 py-1.5 font-semibold">role</th>
                <th className="px-3 py-1.5 font-semibold">source</th>
              </tr>
            </thead>
            <tbody>
              {cfg?.cameras.map((c) => (
                <tr key={c.id} className="border-b border-border/60 last:border-0">
                  <td className="px-3 py-1.5 text-foreground">{c.id}</td>
                  <td className="px-3 py-1.5 text-muted-foreground">{c.role}</td>
                  <td className="px-3 py-1.5 text-muted-foreground">{c.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </main>
  );
}

function SectionHeader({
  title,
  right,
  ok,
}: {
  title: string;
  right?: string;
  ok?: boolean;
}) {
  return (
    <div className="flex items-center justify-between gap-2 px-3 h-9 border-b border-border bg-card">
      <span className="font-mono text-[12px] font-semibold tracking-[0.12em] uppercase text-foreground">
        {title}
      </span>
      {right ? (
        <span
          className={`label-micro flex items-center gap-1.5 ${
            ok ? "text-[var(--haller-live)]" : "text-muted-foreground"
          }`}
        >
          {ok !== undefined && (
            <span
              className={`inline-block h-1.5 w-1.5 rounded-full ${
                ok ? "bg-[var(--haller-live)]" : "bg-muted-foreground"
              }`}
            />
          )}
          {right}
        </span>
      ) : null}
    </div>
  );
}
