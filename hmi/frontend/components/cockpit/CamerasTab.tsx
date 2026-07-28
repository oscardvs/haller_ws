"use client";

/**
 * Every configured camera at once. Base-role cameras get a double-width cell —
 * they are the ones you actually read a scene from; wrist views are checks.
 *
 * Reserved and placeholder slots are shown, not hidden. "wrist_right exists in
 * config.yaml but is not wired" is a fact about the robot, and a grid that
 * quietly omits it makes the operator wonder whether they misremembered.
 */
import { cameraStreamUrl, type CameraInfo } from "@/lib/api";

export function CamerasTab({ cameras }: { cameras: CameraInfo[] }) {
  if (cameras.length === 0) {
    return (
      <div className="flex min-h-0 items-center justify-center p-2">
        <span className="label-micro text-muted-foreground">
          no cameras configured in hmi/backend/config.yaml
        </span>
      </div>
    );
  }

  return (
    <div className="grid min-h-0 auto-rows-[minmax(0,1fr)] grid-cols-3 gap-2 overflow-y-auto p-2">
      {cameras.map((c) => (
        <div
          key={c.id}
          className={
            "flex min-h-0 flex-col overflow-hidden rounded-lg bg-[var(--haller-inset)] shadow-[0_0_0_1px_var(--border)] " +
            (c.role === "base" ? "col-span-2" : "col-span-1")
          }
          data-camera-id={c.id}
        >
          <div className="flex h-7.5 shrink-0 items-center gap-2 border-b border-border bg-[var(--haller-chrome)] px-2.5">
            <span className="inline-flex shrink-0 items-center gap-1.5 label-micro">
              <span
                aria-hidden
                className="h-1.5 w-1.5 rounded-full"
                style={{
                  backgroundColor: c.active
                    ? "var(--haller-live)"
                    : "var(--haller-warn)",
                }}
              />
              {c.role}
            </span>
            <span className="shrink-0 font-mono text-[10px]">{c.id}</span>
            <span className="min-w-0 flex-1 truncate text-right font-mono text-[10px] text-muted-foreground">
              {c.source} ·{" "}
              {c.width && c.height && c.fps
                ? `${c.width}×${c.height} · ${c.fps} fps`
                : "not wired"}
            </span>
          </div>
          <div className="relative flex min-h-0 flex-1 items-center justify-center">
            {c.active ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={cameraStreamUrl(c.id)}
                alt={`${c.id} live feed`}
                className="absolute inset-0 h-full w-full object-contain"
              />
            ) : (
              <>
                <span className="scanlines absolute inset-0" aria-hidden />
                <span className="relative font-mono text-[10px] tracking-[0.16em] uppercase text-muted-foreground opacity-70">
                  {c.source === "placeholder" ? "reserved slot" : "no feed"}
                </span>
              </>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
