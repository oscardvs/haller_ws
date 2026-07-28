"use client";

/**
 * 44px header: identity, the six tabs, and the E-STOP.
 *
 * The E-STOP sits in the header on every tab for the same reason it used to
 * sit in the app rail — there is no state of this UI from which the operator
 * has to navigate to stop the robot. EStopButton itself is unchanged: always
 * red, always full opacity, pulsing while alerts are raised.
 */
import { EStopButton } from "@/components/EStopButton";
import { TABS, type TabId } from "./lib";

export function CockpitHeader({
  tab,
  onTab,
}: {
  tab: TabId;
  onTab: (t: TabId) => void;
}) {
  return (
    <header className="flex items-center justify-between gap-4 border-b border-border bg-[var(--haller-chrome)] px-2.5">
      <div className="flex min-w-0 items-center gap-4">
        <div className="inline-flex shrink-0 items-center gap-2 select-none">
          <span className="relative inline-flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-haller-ping rounded-full bg-[var(--haller-live)] opacity-70" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-[var(--haller-live)]" />
          </span>
          <span className="font-mono text-[13px] font-semibold tracking-[0.3em] uppercase">
            Haller
          </span>
        </div>

        <nav
          aria-label="Cockpit sections"
          className="flex shrink-0 items-center gap-0.5 rounded-md bg-muted p-0.5"
        >
          {TABS.map((t) => {
            const active = t.id === tab;
            return (
              <button
                key={t.id}
                type="button"
                onClick={() => onTab(t.id)}
                aria-current={active ? "page" : undefined}
                className={
                  "inline-flex h-[26px] items-center gap-1.5 rounded-sm px-2.5 font-mono text-[10px] font-semibold tracking-[0.12em] whitespace-nowrap uppercase transition-colors " +
                  (active
                    ? "bg-[var(--haller-live-soft)] text-[var(--haller-live)]"
                    : "text-muted-foreground hover:text-foreground")
                }
              >
                <span
                  aria-hidden
                  className="h-1 w-1 rounded-[1px]"
                  style={{
                    backgroundColor: active
                      ? "var(--haller-live)"
                      : "var(--haller-rail)",
                  }}
                />
                {t.label}
              </button>
            );
          })}
        </nav>
      </div>

      <EStopButton className="h-[30px]" />
    </header>
  );
}
