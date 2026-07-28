"use client";

/**
 * Header + telemetry rail for the routes that are NOT the cockpit.
 *
 * `/base`, `/arm/[id]`, `/settings` and `/teleop/human` survive the v3
 * redesign as unlinked deep links — nothing in the cockpit navigates to them,
 * but bookmarks, a second monitor and "open just the left arm" all still work.
 * They used to inherit this chrome from the root layout; the cockpit needs a
 * bare document, so they carry it themselves now.
 *
 * The E-STOP is here for the same reason it is in the cockpit header: there is
 * no route in this app from which the operator cannot stop the robot.
 */
import Link from "next/link";

import { EStopButton } from "./EStopButton";
import { TelemetryBar } from "./TelemetryBar";

export function DeepLinkChrome({ label }: { label: string }) {
  return (
    <header className="sticky top-0 z-40 border-b border-border bg-background/85 backdrop-blur supports-[backdrop-filter]:bg-background/70">
      <div className="flex h-11 items-center justify-between px-3">
        <div className="flex min-w-0 items-center gap-4">
          <Link
            href="/"
            className="group inline-flex shrink-0 items-center gap-2 select-none"
            aria-label="Haller HMI — back to cockpit"
          >
            <span className="relative inline-flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-haller-ping rounded-full bg-[var(--haller-live)] opacity-70" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-[var(--haller-live)]" />
            </span>
            <span className="font-mono text-[13px] font-semibold tracking-[0.3em] uppercase text-foreground">
              Haller
            </span>
          </Link>
          <span className="label-micro shrink-0 border-l border-border pl-3 text-muted-foreground">
            {label}
          </span>
          <Link
            href="/"
            className="label-micro truncate text-muted-foreground transition-colors hover:text-[var(--haller-live)]"
          >
            ← cockpit
          </Link>
        </div>
        <EStopButton />
      </div>
      <TelemetryBar />
    </header>
  );
}
