"use client";

/**
 * <video> + <canvas> overlay. Pure render — owns no state besides refs.
 *
 *   - The parent attaches a MediaStream to `video.current.srcObject`.
 *   - On every animation frame, the parent passes the latest landmark result
 *     and calls drawOverlay() through the imperative handle.
 */
import { forwardRef, useImperativeHandle, useRef } from "react";
import type { GhostSides, OverlaySides } from "@/lib/mediapipe";
export type { OverlaySides } from "@/lib/mediapipe";

export type CameraOverlayHandle = {
  video: HTMLVideoElement | null;
  /** `aspect` is what buildGhostSides needs to un-skew a direction expressed in
   *  landmarks normalized to [0,1] on both axes; only the canvas knows it. */
  aspect: () => number;
  draw: (sides: OverlaySides, ghost?: GhostSides) => void;
};

const INSTRUMENT_LINE = "oklch(80% 0.18 142)";
const AMBER = "oklch(75% 0.16 70)";
/** The ghost is a target, not a reading. Dimmed and dashed so it never reads as
 *  live tracking, and it goes UNDER the skeleton so matching is legible. */
const GHOST_IDLE = "oklch(70% 0.02 250 / 0.55)";
const GHOST_MATCHED = "oklch(80% 0.18 142 / 0.55)";

export const CameraOverlay = forwardRef<
  CameraOverlayHandle,
  {
    aspectRatio?: string;
    /** Fill the parent instead of holding an aspect ratio. The cockpit gives
     *  this a grid row of a fixed-height page, where a 16:9 box would either
     *  overflow it or leave a band of dead space. */
    fill?: boolean;
  }
>(
  function CameraOverlay({ aspectRatio = "16/9", fill = false }, ref) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useImperativeHandle(ref, () => ({
      get video() { return videoRef.current; },
      aspect() {
        const cv = canvasRef.current;
        const vd = videoRef.current;
        const w = vd?.videoWidth || cv?.clientWidth || 0;
        const h = vd?.videoHeight || cv?.clientHeight || 0;
        return h > 0 ? w / h : 1;
      },
      draw(sides: OverlaySides, ghost?: GhostSides) {
        const cv = canvasRef.current;
        const vd = videoRef.current;
        if (!cv || !vd) return;
        const w = (cv.width = vd.videoWidth || cv.clientWidth);
        const h = (cv.height = vd.videoHeight || cv.clientHeight);
        const ctx = cv.getContext("2d");
        if (!ctx) return;
        ctx.clearRect(0, 0, w, h);

        const drawGhost = (side: GhostSides["left"]) => {
          if (!side || side.pose.length < 2) return;
          ctx.save();
          ctx.strokeStyle = side.matched ? GHOST_MATCHED : GHOST_IDLE;
          ctx.lineWidth = 6;
          ctx.lineCap = "round";
          ctx.lineJoin = "round";
          ctx.setLineDash([10, 7]);
          ctx.beginPath();
          side.pose.forEach(([x, y], i) => {
            const px = x * w, py = y * h;
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
          });
          ctx.stroke();
          // A ring at the wrist: the end of the limb is where the eye goes when
          // lining two arms up, and the dashed line alone is hard to land on.
          const [wx, wy] = side.pose[side.pose.length - 1];
          ctx.setLineDash([]);
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.arc(wx * w, wy * h, 12, 0, Math.PI * 2);
          ctx.stroke();
          ctx.restore();
        };
        drawGhost(ghost?.left ?? null);
        drawGhost(ghost?.right ?? null);

        const drawSide = (side: OverlaySides["left"]) => {
          if (!side) return;
          const colour = side.lost ? AMBER : INSTRUMENT_LINE;
          ctx.strokeStyle = colour;
          ctx.fillStyle = colour;
          ctx.lineWidth = 2;

          // Body skeleton: shoulder → elbow → wrist.
          ctx.beginPath();
          for (let i = 0; i < side.pose.length; i++) {
            const [x, y] = side.pose[i];
            const px = x * w, py = y * h;
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
          }
          ctx.stroke();

          // Hand landmarks as 4px ticks.
          for (const [x, y] of side.hand) {
            ctx.fillRect(x * w - 2, y * h - 2, 4, 4);
          }

          // Pinch line: thumb-tip to index-tip (assumed first two hand entries).
          if (side.hand.length >= 2) {
            const [tx, ty] = side.hand[0];
            const [ix, iy] = side.hand[1];
            ctx.beginPath();
            ctx.setLineDash(side.pinch01 < 0.3 ? [4, 3] : []);
            ctx.lineWidth = 1.5;
            ctx.moveTo(tx * w, ty * h);
            ctx.lineTo(ix * w, iy * h);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.lineWidth = 2;
          }
        };

        drawSide(sides.left);
        drawSide(sides.right);
      },
    }));

    return (
      <div
        className={fill ? "relative h-full w-full" : "relative w-full"}
        style={fill ? undefined : { aspectRatio }}
      >
        <video
          ref={videoRef}
          autoPlay muted playsInline
          className="absolute inset-0 w-full h-full object-cover"
          style={{ transform: "scaleX(-1)" }}
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ transform: "scaleX(-1)" }}
        />
      </div>
    );
  },
);
