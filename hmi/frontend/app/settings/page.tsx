// hmi/frontend/app/settings/page.tsx
"use client";
import { useCallback, useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { api } from "@/lib/api";
import { CalibrationStatusCard } from "@/components/CalibrationStatusCard";
import {
  fetchCalibrationStatus,
  type CalibrationStatusResponse,
} from "@/lib/calibration";
import { BACKEND_URL } from "@/lib/config";
import { useTelemetry, type ArmState } from "@/lib/telemetry";

const EMPTY_ARMS: Record<string, ArmState> = {};

type ConfigBody = Awaited<ReturnType<typeof api.config>>;

export default function SettingsPage() {
  const [cfg, setCfg] = useState<ConfigBody | null>(null);
  const [health, setHealth] = useState<{ status: string } | null>(null);
  useEffect(() => {
    api.config().then(setCfg).catch(console.error);
    api.health().then(setHealth).catch(console.error);
  }, []);

  const armsTelemetry = useTelemetry((s) => s.lastFrame?.arms ?? EMPTY_ARMS);
  const [calStatus, setCalStatus] = useState<CalibrationStatusResponse | null>(null);
  const [wizardArm, setWizardArm] = useState<string | null>(null);

  const refreshCal = useCallback(async () => {
    try {
      setCalStatus(await fetchCalibrationStatus(BACKEND_URL));
    } catch {
      /* offline */
    }
  }, []);
  useEffect(() => {
    void refreshCal();
  }, [refreshCal]);

  const allManual = Object.values(armsTelemetry).every((a) => a.mode === "manual");
  const someSession = calStatus?.current_session != null;
  const blockedReason = !allManual
    ? "Put every arm in manual first"
    : someSession
    ? "Another calibration is in progress"
    : undefined;
  const canStart = allManual && !someSession;

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

      <Card className="overflow-hidden p-0">
        <SectionHeader title="Calibration" right={`${calStatus?.arms.length ?? 0} arms`} />
        <CardContent className="p-3">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {(calStatus?.arms ?? []).map((s) => (
              <CalibrationStatusCard
                key={s.id}
                status={s}
                canStart={canStart}
                blockedReason={blockedReason}
                onCalibrate={() => setWizardArm(s.id)}
              />
            ))}
          </div>
        </CardContent>
      </Card>

      {wizardArm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card border rounded p-6 max-w-sm">
            <p className="text-sm mb-3">
              Wizard for {wizardArm} — T10 not yet implemented.
            </p>
            <button
              className="px-3 py-1.5 border rounded text-sm"
              onClick={async () => {
                setWizardArm(null);
                await refreshCal();
              }}
            >
              Close
            </button>
          </div>
        </div>
      )}
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
