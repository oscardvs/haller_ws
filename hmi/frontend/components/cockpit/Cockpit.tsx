"use client";

/**
 * The Haller cockpit — one fixed-viewport supervisory surface.
 *
 * Four rows, and the page itself never scrolls:
 *
 *   44px  header   identity · tabs · E-STOP
 *   26px  rail     link state · base telemetry · alerts
 *   1fr   content  the active tab, which owns its own min-h-0 and overflow
 *   34px  command  teleop · recorder · hint · census · clock
 *
 * Replaces the scrolling multi-page dashboard. `/base`, `/arm/[id]`,
 * `/settings` and `/teleop/human` still exist as unlinked deep links.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { api, type CameraInfo } from "@/lib/api";
import { useTelemetry } from "@/lib/telemetry";
import { useRecorder } from "@/lib/recorder";

import { CockpitHeader } from "./CockpitHeader";
import { TelemetryRail } from "./TelemetryRail";
import { CommandBar } from "./CommandBar";
import { AlertsPopover } from "./AlertsPopover";
import { TeleopPopover } from "./TeleopPopover";
import { RecordPopover } from "./RecordPopover";
import { OperateTab } from "./OperateTab";
import { CamerasTab } from "./CamerasTab";
import { SettingsTab } from "./SettingsTab";
import { useViewport, type PopId, type TabId } from "./lib";

type ConfigBody = Awaited<ReturnType<typeof api.config>>;

export function Cockpit() {
  const [tab, setTab] = useState<TabId>("operate");
  const [pop, setPop] = useState<PopId>(null);
  const [cfg, setCfg] = useState<ConfigBody | null>(null);
  const [cameras, setCameras] = useState<CameraInfo[] | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);
  const viewport = useViewport();

  const startTelemetry = useTelemetry((s) => s.start);
  const startPolling = useRecorder((s) => s.startPolling);

  const teleopRef = useRef<HTMLButtonElement>(null);
  const recRef = useRef<HTMLButtonElement>(null);
  const alertsRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    startTelemetry();
  }, [startTelemetry]);

  useEffect(() => startPolling(), [startPolling]);

  // Boot fetch. /config is required — without it there is no arm list and no
  // camera list, and an empty cockpit would look like a robot with nothing
  // attached rather than a UI that has not loaded. /cameras is allowed to fail
  // softly: the arm cards degrade to "no feed" tiles, which is honest.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const c = await api.config();
        if (cancelled) return;
        setCfg(c);
      } catch (e) {
        if (!cancelled) setBootError((e as Error).message);
        return;
      }
      try {
        const r = await api.cameras();
        if (!cancelled) setCameras(r.cameras);
      } catch {
        if (!cancelled) setCameras([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Escape closes whatever is open, from anywhere. Popover binds this too, but
  // only while one is mounted — this is the shell-level guarantee.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPop(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const closePop = useCallback(() => setPop(null), []);
  const goTab = useCallback((t: TabId) => {
    setTab(t);
    // A popover anchored to the command bar has no meaning once the surface
    // behind it changed.
    setPop(null);
  }, []);

  const armIds = cfg?.arms.map((a) => a.id) ?? [];
  const cams = cameras ?? [];
  const booting = cfg === null && bootError === null;

  return (
    <div
      className="grid h-screen w-full grid-rows-[44px_26px_minmax(0,1fr)_34px] overflow-hidden"
      style={{ position: "relative" }}
    >
      <CockpitHeader tab={tab} onTab={goTab} />

      <TelemetryRail
        alertsOpen={pop === "alerts"}
        onToggleAlerts={() => setPop(pop === "alerts" ? null : "alerts")}
        alertsRef={alertsRef}
      />

      {booting ? (
        <BootState />
      ) : bootError ? (
        <BootError message={bootError} />
      ) : cfg ? (
        <>
          {tab === "operate" && (
            <OperateTab
              armIds={armIds}
              cameras={cams}
              viewport={viewport}
              pop={pop}
              setPop={setPop}
            />
          )}
          {tab === "cameras" && <CamerasTab cameras={cams} />}
          {tab === "settings" && <SettingsTab cfg={cfg} cameras={cams} />}
        </>
      ) : null}

      <CommandBar
        tab={tab}
        pop={pop}
        onToggleTeleop={() => setPop(pop === "teleop" ? null : "teleop")}
        onToggleRec={() => setPop(pop === "rec" ? null : "rec")}
        teleopRef={teleopRef}
        recRef={recRef}
        census={census(armIds.length, cams)}
        version={cfg?.version ?? "—"}
      />

      {pop === "alerts" && (
        <AlertsPopover onClose={closePop} triggerRef={alertsRef} />
      )}
      {pop === "teleop" && (
        <TeleopPopover
          armIds={armIds}
          onClose={closePop}
          triggerRef={teleopRef}
          onTab={goTab}
        />
      )}
      {pop === "rec" && (
        <RecordPopover onClose={closePop} triggerRef={recRef} onTab={goTab} />
      )}
    </div>
  );
}

/** Names the calls in flight rather than showing an empty cockpit — an empty
 *  cockpit is indistinguishable from a robot with nothing configured. */
function BootState() {
  return (
    <div className="flex min-h-0 flex-col items-center justify-center gap-2.5 p-2">
      <span className="relative inline-flex h-2 w-2">
        <span className="absolute inline-flex h-full w-full animate-haller-ping rounded-full bg-[var(--haller-live)] opacity-70" />
        <span className="relative inline-flex h-2 w-2 rounded-full bg-[var(--haller-live)]" />
      </span>
      <span className="font-mono text-[11px] tracking-[0.18em] uppercase text-muted-foreground">
        loading config…
      </span>
      <span className="font-mono text-[11px] text-muted-foreground">
        GET /config · GET /cameras
      </span>
    </div>
  );
}

function BootError({ message }: { message: string }) {
  return (
    <div className="flex min-h-0 flex-col items-center justify-center gap-2.5 p-4">
      <span className="inline-flex items-center gap-2 rounded-md border border-[var(--haller-fault)] px-2.5 py-1.5 font-mono text-[11px] text-[var(--haller-fault)]">
        <span
          aria-hidden
          className="h-1.5 w-1.5 rounded-[1px] bg-[var(--haller-fault)]"
        />
        GET /config failed
      </span>
      <span className="max-w-[440px] text-center font-mono text-[11px] break-all text-muted-foreground">
        {message}
      </span>
      <span className="max-w-[440px] text-center text-[11px] text-pretty text-muted-foreground">
        The cockpit cannot draw an arm or camera it has never been told about.
        Check the backend is running and that NEXT_PUBLIC_BACKEND_URL points at
        it, then reload.
      </span>
    </div>
  );
}

function census(arms: number, cameras: CameraInfo[]): string {
  const live = cameras.filter((c) => c.active).length;
  return `${arms} arm${arms === 1 ? "" : "s"} · ${live}/${cameras.length} cams`;
}
