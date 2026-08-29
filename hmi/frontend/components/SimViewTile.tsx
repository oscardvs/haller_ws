"use client";

/**
 * SimViewTile — picture-in-picture of the MuJoCo scene, pinned to the viewport.
 *
 * Against a sim backend there is no physical arm to look at, so without this
 * there is no way to watch the robot while driving it: you open the raw MJPEG
 * endpoint in a second window, or you fly blind.
 *
 * Deliberately self-contained: it reads nothing from the teleop session and
 * owns no shared state, so it can sit alongside any panel without touching it.
 *
 * Renders nothing at all when no sim camera is configured, which is the case
 * for every real-hardware config.
 */
import { useEffect, useState } from "react";

import { api, cameraStreamUrl, type CameraInfo } from "@/lib/api";

/** Prefer the three-quarter view — the overhead camera flattens away exactly
 *  the joints (shoulder_lift, elbow_flex) you want to watch while teleoping.
 *  Exported because the cockpit renders its own in-flow tile from the same
 *  choice: there must be one answer to "which sim camera do we show". */
export function pickSimCamera(cameras: CameraInfo[]): CameraInfo | null {
  const sim = cameras.filter((c) => c.source === "sim_camera");
  if (sim.length === 0) return null;
  return sim.find((c) => c.id.includes("threequarter")) ?? sim[0];
}

export function SimViewTile({
  placement = "pinned",
}: {
  /** `pinned` floats over the deep-link teleop page, which has no room
   *  reserved for it — hence the clearance note below. `inline` is the
   *  cockpit, whose Teleop tab gives the tile a column of its own, and an
   *  in-flow tile cannot cover anything. */
  placement?: "pinned" | "inline";
} = {}) {
  const [cam, setCam] = useState<CameraInfo | null>(null);
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api.cameras()
      .then((r) => { if (!cancelled) setCam(pickSimCamera(r.cameras)); })
      .catch(() => { if (!cancelled) setCam(null); });
    return () => { cancelled = true; };
  }, []);

  if (!cam) return null;

  const live = cam.active !== false;

  return (
    // Pinned: bottom-left, lifted clear of the dead-man state badge that sits
    // at the very bottom of the teleop panel — that badge must never be
    // covered.
    <div
      className={
        placement === "pinned"
          ? "fixed bottom-16 left-4 z-40 w-[260px] rounded-sm border border-border bg-background/95 shadow-lg backdrop-blur-sm"
          : "flex w-full flex-col overflow-hidden rounded-lg border border-border bg-card"
      }
      data-sim-view={cam.id}
    >
      <div className="flex items-center justify-between gap-2 border-b border-border px-2 py-1">
        <span className="inline-flex items-center gap-1.5 font-mono text-[10px] tracking-[0.08em] text-muted-foreground">
          <span
            className="inline-block h-1 w-1 rounded-full"
            style={{ backgroundColor: live ? "var(--haller-live)" : "var(--haller-warn)" }}
          />
          SIM · {cam.id}
        </span>
        <button
          type="button"
          onClick={() => setCollapsed((c) => !c)}
          className="font-mono text-[10px] text-muted-foreground hover:text-foreground"
          aria-expanded={!collapsed}
        >
          {collapsed ? "show" : "hide"}
        </button>
      </div>

      {!collapsed && (
        <div className="relative aspect-video w-full overflow-hidden bg-card/60">
          {live ? (
            // MJPEG multipart renders natively in <img>; no decoder needed.
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={cameraStreamUrl(cam.id)}
              alt={`${cam.id} live sim feed`}
              className="absolute inset-0 h-full w-full object-cover"
            />
          ) : (
            <div className="absolute inset-0 grid place-items-center font-mono text-[10px] text-muted-foreground">
              no feed
            </div>
          )}
        </div>
      )}
    </div>
  );
}
