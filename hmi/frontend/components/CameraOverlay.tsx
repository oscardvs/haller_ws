"use client";

/**
 * <video> + <canvas> overlay. Pure render — owns no state besides refs.
 *
 *   - The parent attaches a MediaStream to `video.current.srcObject`.
 *   - On every animation frame, the parent passes the latest landmark result
 *     and calls drawOverlay() through the imperative handle.
 */
import { forwardRef, useImperativeHandle, useRef } from "react";

export type OverlaySides = {
  left:  { lost: boolean; pose: [number, number][]; hand: [number, number][]; pinch01: number } | null;
  right: { lost: boolean; pose: [number, number][]; hand: [number, number][]; pinch01: number } | null;
};

export type CameraOverlayHandle = {
  video: HTMLVideoElement | null;
  draw: (sides: OverlaySides) => void;
};

const INSTRUMENT_LINE = "oklch(80% 0.18 142)";
const AMBER = "oklch(75% 0.16 70)";

export const CameraOverlay = forwardRef<CameraOverlayHandle, { aspectRatio?: string }>(
  function CameraOverlay({ aspectRatio = "16/9" }, ref) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useImperativeHandle(ref, () => ({
      get video() { return videoRef.current; },
      draw(sides: OverlaySides) {
        const cv = canvasRef.current;
        const vd = videoRef.current;
        if (!cv || !vd) return;
        const w = (cv.width = vd.videoWidth || cv.clientWidth);
        const h = (cv.height = vd.videoHeight || cv.clientHeight);
        const ctx = cv.getContext("2d");
        if (!ctx) return;
        ctx.clearRect(0, 0, w, h);

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
      <div className="relative w-full" style={{ aspectRatio }}>
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
