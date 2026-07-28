"use client";

/**
 * The Operate tab's main viewport: a chip per configured camera, and the
 * selected one filling the bay.
 *
 * Every camera the backend reports gets a chip, including reserved
 * placeholders and sim views — which slots exist but are not wired is
 * operational information, not clutter to hide.
 */
import { cameraStreamUrl, type CameraInfo } from "@/lib/api";
import { useSticky } from "./lib";

export function PrimaryCameraPanel({ cameras }: { cameras: CameraInfo[] }) {
  const [selected, setSelected] = useSticky<string | null>(
    "operate.primaryCam",
    null,
  );
  // Default to the first base-role camera: the one you can judge a drive from.
  const active =
    cameras.find((c) => c.id === selected) ??
    cameras.find((c) => c.role === "base") ??
    cameras[0] ??
    null;
  const live = active ? active.active : false;

  return (
    <div className="flex min-h-0 flex-col overflow-hidden rounded-lg bg-[var(--haller-inset)] shadow-[0_0_0_1px_var(--border)]">
      <div className="flex h-8 shrink-0 items-center gap-2 border-b border-border bg-[var(--haller-chrome)] px-2">
        <div className="flex shrink-0 items-center gap-1 overflow-x-auto">
          {cameras.map((c) => {
            const on = active?.id === c.id;
            return (
              <button
                key={c.id}
                type="button"
                title={c.id}
                onClick={() => setSelected(c.id)}
                aria-pressed={on}
                className={
                  "inline-flex h-6 shrink-0 items-center gap-1.5 rounded-sm px-2.5 font-mono text-[10px] tracking-[0.06em] whitespace-nowrap " +
                  (on ? "bg-muted text-foreground" : "text-muted-foreground hover:bg-muted/50")
                }
              >
                <span
                  aria-hidden
                  className="h-1.5 w-1.5 rounded-full"
                  style={{
                    backgroundColor: c.active
                      ? "var(--haller-live)"
                      : "var(--haller-warn)",
                  }}
                />
                {shortLabel(c)}
              </button>
            );
          })}
        </div>
        <span className="min-w-0 flex-1 truncate text-right font-mono text-[10px] tracking-[0.04em] text-muted-foreground">
          {active ? meta(active) : "no cameras configured"}
        </span>
      </div>

      <div className="corner-frame relative flex min-h-0 flex-1 items-center justify-center">
        <span
          aria-hidden
          className="corner-tr absolute top-2 right-2 h-3 w-3 border-t border-r"
          style={{ borderColor: "var(--haller-rail)" }}
        />
        <span
          aria-hidden
          className="corner-bl absolute bottom-2 left-2 h-3 w-3 border-b border-l"
          style={{ borderColor: "var(--haller-rail)" }}
        />
        {active && live ? (
          // MJPEG multipart renders natively in <img>; no decoder needed.
          // eslint-disable-next-line @next/next/no-img-element
          <img
            key={active.id}
            src={cameraStreamUrl(active.id)}
            alt={`${active.id} live feed`}
            className="absolute inset-0 h-full w-full object-contain"
          />
        ) : (
          <span className="scanlines absolute inset-0" aria-hidden />
        )}
        {!(active && live) && (
          <span className="relative font-mono text-[10px] tracking-[0.2em] uppercase text-muted-foreground opacity-70">
            {active ? `${active.id} · no feed` : "no camera"}
          </span>
        )}
      </div>
    </div>
  );
}

function shortLabel(c: CameraInfo): string {
  if (c.source === "sim_camera") return c.id.replace(/_sim$/, " sim");
  if (c.role === "wrist") return c.arm_id ? `wrist ${c.arm_id}` : "wrist";
  return c.id;
}

function meta(c: CameraInfo): string {
  const shape =
    c.width && c.height && c.fps ? `${c.width}×${c.height} · ${c.fps} fps` : "— · —";
  return `${c.source} · ${shape} · ${c.active ? "streaming" : "no feed"}`;
}
