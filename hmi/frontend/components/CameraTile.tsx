// hmi/frontend/components/CameraTile.tsx
"use client";

/**
 * CameraTile — 16:9 slot that shows a live MJPEG stream or a placeholder.
 *
 * Two visual states:
 *  - Live  (streamUrl + active):  <img src={MJPEG endpoint}> fills the tile.
 *  - Idle  (placeholder source, inactive, or no streamUrl): dashed border,
 *          scanline pattern, "no feed" badge. Communicates "slot reserved
 *          but no camera attached" so the page reads identically once
 *          hardware lands.
 *
 * Browsers can render multipart/x-mixed-replace JPEG streams in <img>
 * directly — no JS decoder needed. The backend caps stream FPS to ~15 Hz
 * (see CameraManager.STREAM_FPS) which is fine for supervisory view.
 */
export function CameraTile({
  id,
  role,
  streamUrl,
  active,
  width,
  height,
  fps,
  collapsed,
}: {
  id: string;
  role: "wrist" | "base";
  streamUrl?: string;
  active?: boolean;
  width?: number;
  height?: number;
  fps?: number;
  /** Label strip only — no 16:9 body. Used when the viewport is too short to
   *  afford a picture, where the joint stack is the thing worth the pixels.
   *  The strip stays so the slot is still visibly *there*. */
  collapsed?: boolean;
}) {
  const live = Boolean(streamUrl) && active !== false;
  const resolution =
    width && height && fps ? `${width}×${height} · ${fps} fps` : "— × — · — fps";

  if (collapsed) {
    return (
      <div
        className="flex shrink-0 items-center gap-2 overflow-hidden rounded-sm border border-border bg-card/60 px-2 py-1"
        data-camera-id={id}
        data-collapsed
      >
        <span className="inline-flex shrink-0 items-center gap-1.5 label-micro text-foreground/90">
          <span
            className="inline-block h-1 w-1 rounded-full"
            style={{
              backgroundColor: live ? "var(--haller-live)" : "var(--haller-warn)",
            }}
          />
          {role}
        </span>
        <span className="truncate font-mono text-[10px] tracking-[0.08em] text-muted-foreground">
          {id}
        </span>
        <span className="ml-auto shrink-0 label-micro text-muted-foreground">
          {live ? "live" : "no feed"}
        </span>
      </div>
    );
  }

  return (
    <div
      className={
        "corner-frame relative aspect-video w-full overflow-hidden rounded-sm bg-card/60" +
        (live ? " border border-border" : " border border-dashed border-border/80 scanlines")
      }
      data-camera-id={id}
    >
      {/* Decorative corner-frame extras */}
      <span
        aria-hidden
        className="corner-tr absolute top-0 right-0 h-2.5 w-2.5 border-t border-r"
        style={{ borderColor: "var(--haller-rail)" }}
      />
      <span
        aria-hidden
        className="corner-bl absolute bottom-0 left-0 h-2.5 w-2.5 border-b border-l"
        style={{ borderColor: "var(--haller-rail)" }}
      />

      {/* Top-left meta row */}
      <div className="absolute top-1.5 left-2 z-10 flex items-center gap-1.5">
        <span className="inline-flex items-center gap-1.5 px-1.5 py-0.5 rounded-[2px] border border-border bg-background/70 label-micro text-foreground/90">
          <span
            className="inline-block h-1 w-1 rounded-full"
            style={{
              backgroundColor: live
                ? "var(--haller-live)"
                : "var(--haller-warn)",
            }}
          />
          {role}
        </span>
        <span className="font-mono text-[10px] tracking-[0.08em] text-muted-foreground">
          {id}
        </span>
      </div>

      {live ? (
        // The MJPEG stream. Disable lazy decoding so the first frame appears immediately.
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={streamUrl}
          alt={`${id} live feed`}
          className="absolute inset-0 h-full w-full object-cover"
          // Browsers cache multipart streams aggressively across navigations;
          // including a `key` on this element is the caller's job if needed.
        />
      ) : (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="flex items-center gap-2 px-2 py-1 rounded-sm border border-border bg-background/70">
            <span className="inline-block h-1.5 w-1.5 rounded-[1px] bg-muted-foreground" />
            <span className="label-micro text-muted-foreground">no feed</span>
          </div>
        </div>
      )}

      {/* Bottom-right resolution */}
      <div className="absolute bottom-1.5 right-2 z-10 font-mono text-[10px] tracking-[0.08em] text-muted-foreground bg-background/40 px-1 rounded-[2px]">
        {resolution}
      </div>
    </div>
  );
}
