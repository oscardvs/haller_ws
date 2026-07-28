"use client";

/** Everything the backend has raised on the current frame, hanging off the
 *  rail's alert counter. Alerts are frame state, not a log — they are whatever
 *  telemetry says is true right now. */
import type { RefObject } from "react";

import { useTelemetry } from "@/lib/telemetry";
import { Popover, PopoverHeader } from "./Popover";

export function AlertsPopover({
  onClose,
  triggerRef,
}: {
  onClose: () => void;
  triggerRef: RefObject<HTMLButtonElement | null>;
}) {
  const alerts = useTelemetry((s) => s.lastFrame?.alerts ?? EMPTY);

  return (
    <Popover
      onClose={onClose}
      triggerRef={triggerRef}
      label="Alerts"
      className="top-[70px] right-2 flex max-h-[calc(100%-120px)] w-[min(380px,calc(100%-16px))] flex-col overflow-hidden"
    >
      <PopoverHeader
        title={`Alerts · ${alerts.length === 0 ? "none" : alerts.length}`}
        onClose={onClose}
      />
      <div className="flex min-h-0 flex-col gap-1.5 overflow-y-auto">
        {alerts.length === 0 ? (
          <span className="label-micro text-muted-foreground">
            nothing raised on the current frame
          </span>
        ) : (
          alerts.map((a, i) => {
            const colour =
              a.level === "error" ? "var(--haller-fault)" : "var(--haller-warn)";
            return (
              <div
                key={`${a.code}-${i}`}
                className="flex flex-col gap-1 rounded-md border p-2"
                style={{ borderColor: colour }}
              >
                <div className="flex items-center gap-2">
                  <span
                    aria-hidden
                    className="h-1.5 w-1.5 rounded-[1px]"
                    style={{ backgroundColor: colour }}
                  />
                  <span className="label-micro" style={{ color: colour }}>
                    {a.level}
                  </span>
                  <span className="font-mono text-[10px]">{a.code}</span>
                  <span className="ml-auto font-mono text-[9px] text-muted-foreground">
                    {a.source}
                  </span>
                </div>
                <span className="text-[11px] text-pretty text-muted-foreground">
                  {a.message}
                </span>
              </div>
            );
          })
        )}
      </div>
    </Popover>
  );
}

/** Stable identity so the selector doesn't hand zustand a fresh array every
 *  frame and re-render this on every tick of telemetry. */
const EMPTY: NonNullable<
  ReturnType<typeof useTelemetry.getState>["lastFrame"]
>["alerts"] = [];
