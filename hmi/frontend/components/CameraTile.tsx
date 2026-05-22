// hmi/frontend/components/CameraTile.tsx
"use client";

/**
 * CameraTile — reserved 16:9 slot for a future video feed.
 *
 *  - Dashed border + scanline pattern communicates "placeholder".
 *  - Role pill (wrist/base) + monospace camera id.
 *  - Corner registration marks so the tile reads as a viewport.
 *  - Layout is dimensionally identical whether or not a feed is wired,
 *    so adding a real stream later doesn't reshuffle the page.
 */
export function CameraTile({
  id,
  role,
}: {
  id: string;
  role: "wrist" | "base";
}) {
  return (
    <div
      className="corner-frame relative aspect-video w-full overflow-hidden rounded-sm border border-dashed border-border/80 bg-card/60 scanlines"
      data-camera-id={id}
    >
      {/* Decorative corner-frame extras (top-right / bottom-left) */}
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
      <div className="absolute top-1.5 left-2 flex items-center gap-1.5">
        <span className="inline-flex items-center gap-1.5 px-1.5 py-0.5 rounded-[2px] border border-border bg-background/70 label-micro text-foreground/90">
          <span className="inline-block h-1 w-1 rounded-full bg-[var(--haller-warn)]" />
          {role}
        </span>
        <span className="font-mono text-[10px] tracking-[0.08em] text-muted-foreground">
          {id}
        </span>
      </div>

      {/* Center status badge */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="flex items-center gap-2 px-2 py-1 rounded-sm border border-border bg-background/70">
          <span className="inline-block h-1.5 w-1.5 rounded-[1px] bg-muted-foreground" />
          <span className="label-micro text-muted-foreground">no feed</span>
        </div>
      </div>

      {/* Bottom-right resolution placeholder */}
      <div className="absolute bottom-1.5 right-2 font-mono text-[10px] text-muted-foreground tracking-[0.08em]">
        — × — · — fps
      </div>
    </div>
  );
}
