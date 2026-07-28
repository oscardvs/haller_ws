"use client";

/**
 * Base drive console: crosshair pad, speed governor, halt.
 *
 * The odometry readouts that used to sit beside the pad now live in the
 * telemetry rail, where they are visible from every tab.
 *
 * Two things here are load-bearing:
 *
 *  1. The 10 Hz cmd_vel interval is owned by this component, so it stops when
 *     the Operate tab unmounts. A cockpit that keeps posting velocities while
 *     the operator is reading Settings is a cockpit that can be surprised.
 *
 *  2. Drive keys ignore events from text inputs. BasePanel bound WASD to
 *     `window` unconditionally while RecordingPanel put a task field on the
 *     same page, so typing "place the red cube" drove the base. That is fixed
 *     here and the keys are additionally scoped to this tab by mounting.
 */
import { useEffect, useRef, useState } from "react";

import { api } from "@/lib/api";
import { isEditableTarget, useSticky } from "./lib";

const SEND_HZ = 10;

export function DriveConsole() {
  const [speed, setSpeed] = useSticky("operate.speed", 0.4);
  const cmd = useRef({ linear: 0, angular: 0 });
  const [knob, setKnob] = useState({ x: 0, y: 0 });
  const [driving, setDriving] = useState(false);
  const pad = useRef<HTMLDivElement>(null);
  const speedRef = useRef(speed);
  useEffect(() => {
    speedRef.current = speed;
  }, [speed]);

  // Fixed-rate send. Reads the governor through a ref so changing speed does
  // not tear down and restart the interval mid-drive.
  useEffect(() => {
    const t = setInterval(() => {
      api
        .cmdVel(cmd.current.linear * speedRef.current, cmd.current.angular * speedRef.current)
        .catch(() => {});
    }, 1000 / SEND_HZ);
    return () => {
      clearInterval(t);
      // One last zero on the way out, so leaving the tab mid-drive stops the
      // base instead of leaving the last non-zero command standing.
      api.cmdVel(0, 0).catch(() => {});
    };
  }, []);

  // Keyboard drive.
  useEffect(() => {
    const pressed = new Set<string>();
    const update = () => {
      let l = 0;
      let a = 0;
      if (pressed.has("w") || pressed.has("arrowup")) l += 1;
      if (pressed.has("s") || pressed.has("arrowdown")) l -= 1;
      if (pressed.has("a") || pressed.has("arrowleft")) a += 1;
      if (pressed.has("d") || pressed.has("arrowright")) a -= 1;
      cmd.current = { linear: l, angular: a };
      setKnob({ x: -a, y: -l });
      setDriving(l !== 0 || a !== 0);
    };
    const down = (e: KeyboardEvent) => {
      if (isEditableTarget(e.target)) return;
      pressed.add(e.key.toLowerCase());
      update();
    };
    // Key-up is NOT gated on the target. If the operator presses "w" on the
    // page and releases it after focus has moved into a field, the key must
    // still lift — a stuck drive key is the failure that matters here.
    const up = (e: KeyboardEvent) => {
      pressed.delete(e.key.toLowerCase());
      update();
    };
    // Focus loss (alt-tab) never delivers key-up either.
    const blur = () => {
      pressed.clear();
      update();
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", blur);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", blur);
    };
  }, []);

  // Pointer drive.
  useEffect(() => {
    const el = pad.current;
    if (!el) return;
    let dragging = false;
    const move = (clientX: number, clientY: number) => {
      const rect = el.getBoundingClientRect();
      const dx = (clientX - (rect.left + rect.width / 2)) / (rect.width / 2);
      const dy = (clientY - (rect.top + rect.height / 2)) / (rect.height / 2);
      const clip = (v: number) => Math.max(-1, Math.min(1, v));
      const x = clip(dx);
      const y = clip(dy);
      setKnob({ x, y });
      cmd.current = { linear: -y, angular: -x };
      setDriving(true);
    };
    const onDown = (e: PointerEvent) => {
      dragging = true;
      move(e.clientX, e.clientY);
    };
    const onMove = (e: PointerEvent) => {
      if (dragging) move(e.clientX, e.clientY);
    };
    const onUp = () => {
      if (!dragging) return;
      dragging = false;
      cmd.current = { linear: 0, angular: 0 };
      setKnob({ x: 0, y: 0 });
      setDriving(false);
    };
    el.addEventListener("pointerdown", onDown);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    return () => {
      el.removeEventListener("pointerdown", onDown);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
    };
  }, []);

  const speedPct = ((speed - 0.1) / 0.9) * 100;

  return (
    <div className="flex shrink-0 flex-nowrap items-center gap-3 rounded-lg bg-card p-2.5 shadow-[0_0_0_1px_var(--border)]">
      <div className="flex shrink-0 flex-col items-center gap-1.5">
        <div
          ref={pad}
          role="application"
          aria-label="base drive pad"
          className="relative h-[132px] w-[132px] cursor-grab touch-none rounded-md border border-border bg-background select-none active:cursor-grabbing"
          style={{
            backgroundImage:
              "radial-gradient(circle at center, var(--haller-grid) 1px, transparent 1px)",
            backgroundSize: "15px 15px",
          }}
        >
          <span
            aria-hidden
            className="absolute top-0 bottom-0 left-1/2 w-px -translate-x-1/2"
            style={{ background: "var(--haller-grid)" }}
          />
          <span
            aria-hidden
            className="absolute top-1/2 right-0 left-0 h-px -translate-y-1/2"
            style={{ background: "var(--haller-grid)" }}
          />
          <span
            aria-hidden
            className="absolute top-1/2 left-1/2 h-[62%] w-[62%] -translate-x-1/2 -translate-y-1/2 rounded-full border border-border"
          />
          <span className="absolute top-0.5 left-1/2 -translate-x-1/2 label-micro text-muted-foreground">
            fwd
          </span>
          <span className="absolute bottom-0.5 left-1/2 -translate-x-1/2 label-micro text-muted-foreground">
            rev
          </span>
          <span className="absolute top-1/2 left-0.5 -translate-y-1/2 label-micro text-muted-foreground">
            ccw
          </span>
          <span className="absolute top-1/2 right-0.5 -translate-y-1/2 label-micro text-muted-foreground">
            cw
          </span>
          <div
            className="absolute -mt-[15px] -ml-[15px] h-[30px] w-[30px] rounded-md"
            style={{
              left: `${50 + knob.x * 34}%`,
              top: `${50 + knob.y * 34}%`,
              background: driving ? "var(--haller-live)" : "var(--haller-rail)",
              boxShadow: driving
                ? "0 0 0 2px var(--haller-live-soft), 0 0 18px var(--haller-live-soft)"
                : "none",
            }}
          >
            <span
              aria-hidden
              className="absolute inset-1.5 rounded-sm border border-background"
            />
          </div>
        </div>
        <span className="font-mono text-[9px] tracking-[0.14em] uppercase text-muted-foreground">
          wasd · arrows · drag
        </span>
      </div>

      <div className="flex min-w-0 flex-1 flex-col gap-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <span className="label-tracked text-muted-foreground">Mobile base</span>
          <span className="font-mono text-[11px] font-semibold tracking-[0.12em] uppercase">
            haller-base
          </span>
          <span aria-hidden className="h-px flex-1 bg-border" />
          <span className="label-micro text-muted-foreground">
            send · {SEND_HZ} hz
          </span>
        </div>

        <div>
          <div className="mb-1.5 flex items-baseline justify-between">
            <label
              htmlFor="speed-governor"
              className="label-tracked text-muted-foreground"
            >
              Speed governor
            </label>
            <span data-num className="font-mono text-[12px] tabular-nums">
              {speed.toFixed(2)}×
            </span>
          </div>
          <input
            id="speed-governor"
            type="range"
            min={0.1}
            max={1}
            step={0.05}
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="haller-range h-4 w-full cursor-pointer"
            style={
              {
                "--pct": `${speedPct}%`,
              } as React.CSSProperties
            }
          />
          <div className="mt-1 flex justify-between font-mono text-[9px] text-muted-foreground">
            <span>0.10</span>
            <span>0.50</span>
            <span>1.00</span>
          </div>
        </div>

        <button
          type="button"
          onClick={() => {
            cmd.current = { linear: 0, angular: 0 };
            setKnob({ x: 0, y: 0 });
            setDriving(false);
            api.cmdVel(0, 0).catch(() => {});
          }}
          className="h-7.5 rounded-md border border-[var(--haller-fault)] label-micro tracking-[0.2em] text-[var(--haller-fault)] hover:bg-[oklch(0.62_0.245_27/0.12)]"
        >
          halt drive
        </button>
      </div>
    </div>
  );
}
