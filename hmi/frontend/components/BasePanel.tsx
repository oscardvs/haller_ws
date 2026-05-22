// hmi/frontend/components/BasePanel.tsx
"use client";
import { useEffect, useRef, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { useTelemetry } from "@/lib/telemetry";
import { CameraTile } from "./CameraTile";

const SEND_HZ = 10;

/**
 * BasePanel — supervisory drive console for the 4-wheeled base.
 *
 *  Left:  virtual joystick on a crosshair pad (mouse + WASD).
 *  Right: speed governor, kill button, live odometry mini-readout.
 *  Top:   forward-facing base camera tile.
 *
 * The joystick visualizes WHAT IS BEING SENT (cmd_vel), not where the
 * stick was last released — so operators can see the full chain.
 */
export function BasePanel() {
  const base = useTelemetry((s) => s.lastFrame?.base);
  const [speed, setSpeed] = useState(0.4);
  const cmd = useRef({ linear: 0, angular: 0 });
  const pad = useRef<HTMLDivElement>(null);
  const [knob, setKnob] = useState({ x: 0, y: 0 });
  const [active, setActive] = useState(false);

  // Send at fixed rate while non-zero, then one final zero
  useEffect(() => {
    const t = setInterval(() => {
      api
        .cmdVel(cmd.current.linear * speed, cmd.current.angular * speed)
        .catch(() => {});
    }, 1000 / SEND_HZ);
    return () => clearInterval(t);
  }, [speed]);

  // keyboard
  useEffect(() => {
    const pressed = new Set<string>();
    const update = () => {
      let l = 0,
        a = 0;
      if (pressed.has("w") || pressed.has("arrowup")) l += 1;
      if (pressed.has("s") || pressed.has("arrowdown")) l -= 1;
      if (pressed.has("a") || pressed.has("arrowleft")) a += 1;
      if (pressed.has("d") || pressed.has("arrowright")) a -= 1;
      cmd.current = { linear: l, angular: a };
      setKnob({ x: -a, y: -l });
      setActive(l !== 0 || a !== 0);
    };
    const down = (e: KeyboardEvent) => {
      pressed.add(e.key.toLowerCase());
      update();
    };
    const up = (e: KeyboardEvent) => {
      pressed.delete(e.key.toLowerCase());
      update();
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  // joystick (mouse + touch)
  useEffect(() => {
    const el = pad.current;
    if (!el) return;
    let dragging = false;
    const r = () => el.getBoundingClientRect();
    const move = (clientX: number, clientY: number) => {
      const rect = r();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const dx = (clientX - cx) / (rect.width / 2);
      const dy = (clientY - cy) / (rect.height / 2);
      const clipped = (v: number) => Math.max(-1, Math.min(1, v));
      const x = clipped(dx);
      const y = clipped(dy);
      setKnob({ x, y });
      cmd.current = { linear: -y, angular: -x };
      setActive(true);
    };
    const onDown = (e: MouseEvent) => {
      dragging = true;
      move(e.clientX, e.clientY);
    };
    const onMove = (e: MouseEvent) => {
      if (dragging) move(e.clientX, e.clientY);
    };
    const onUp = () => {
      dragging = false;
      cmd.current = { linear: 0, angular: 0 };
      setKnob({ x: 0, y: 0 });
      setActive(false);
    };
    el.addEventListener("mousedown", onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      el.removeEventListener("mousedown", onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  const v = base?.linear;
  const w = base?.angular;
  const x = base?.odom.x;
  const y = base?.odom.y;
  const yaw = base?.odom.yaw;
  const scan = base?.scan_min_range;

  const fmt = (n: number | undefined | null, d = 2) =>
    typeof n === "number" ? n.toFixed(d) : "—";

  return (
    <Card className="overflow-hidden p-0">
      {/* Header strip */}
      <div className="flex items-center justify-between gap-2 px-3 h-9 border-b border-border bg-card">
        <div className="flex items-center gap-2">
          <span className="label-micro text-muted-foreground">Mobile base</span>
          <span className="font-mono text-[12px] font-semibold tracking-[0.12em] uppercase text-foreground">
            haller-base
          </span>
        </div>
        <span className="label-micro text-muted-foreground">
          send · {SEND_HZ} Hz
        </span>
      </div>

      <CardContent className="p-3 space-y-3">
        <CameraTile id="base_front" role="base" />

        <div className="grid grid-cols-12 gap-3 items-stretch">
          {/* JOYSTICK PAD */}
          <div className="col-span-7">
            <div className="label-tracked text-muted-foreground mb-1.5">
              Drive
            </div>
            <div
              ref={pad}
              className="relative aspect-square w-full max-w-[260px] mx-auto rounded-sm border border-border bg-background/40 touch-none cursor-grab active:cursor-grabbing select-none"
              tabIndex={0}
              style={{
                backgroundImage:
                  "radial-gradient(circle at center, var(--haller-grid) 1px, transparent 1px)",
                backgroundSize: "16px 16px",
              }}
            >
              {/* Crosshair */}
              <span
                aria-hidden
                className="absolute left-1/2 top-0 bottom-0 w-px -translate-x-1/2"
                style={{ background: "var(--haller-grid)" }}
              />
              <span
                aria-hidden
                className="absolute top-1/2 left-0 right-0 h-px -translate-y-1/2"
                style={{ background: "var(--haller-grid)" }}
              />
              {/* Deadzone ring */}
              <span
                aria-hidden
                className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border border-border/60"
                style={{ width: "62%", height: "62%" }}
              />
              {/* Axis labels */}
              <span className="absolute top-1 left-1/2 -translate-x-1/2 label-micro text-muted-foreground/70">
                FWD
              </span>
              <span className="absolute bottom-1 left-1/2 -translate-x-1/2 label-micro text-muted-foreground/70">
                REV
              </span>
              <span className="absolute left-1 top-1/2 -translate-y-1/2 label-micro text-muted-foreground/70 [writing-mode:vertical-rl] rotate-180">
                CCW
              </span>
              <span className="absolute right-1 top-1/2 -translate-y-1/2 label-micro text-muted-foreground/70 [writing-mode:vertical-rl]">
                CW
              </span>
              {/* Knob */}
              <div
                className={`absolute h-8 w-8 -mt-4 -ml-4 rounded-[3px] transition-shadow ${
                  active
                    ? "bg-[var(--haller-live)] shadow-[0_0_0_2px_var(--haller-live-soft),0_0_16px_var(--haller-live-soft)]"
                    : "bg-muted-foreground/60"
                }`}
                style={{
                  left: `${50 + knob.x * 40}%`,
                  top: `${50 + knob.y * 40}%`,
                }}
              >
                <span
                  aria-hidden
                  className="absolute inset-1.5 rounded-[2px] border border-background/60"
                />
              </div>
            </div>
            <div className="mt-1.5 text-center font-mono text-[10px] tracking-[0.16em] uppercase text-muted-foreground">
              wasd · arrows · drag
            </div>
          </div>

          {/* GOVERNOR + READOUT */}
          <div className="col-span-5 flex flex-col gap-3">
            <div>
              <div className="flex items-baseline justify-between mb-1">
                <span className="label-tracked text-muted-foreground">
                  Speed
                </span>
                <span data-num className="font-mono text-[12px] text-foreground">
                  {speed.toFixed(2)}×
                </span>
              </div>
              <Slider
                min={0.1}
                max={1.0}
                step={0.05}
                value={[speed]}
                onValueChange={(next) => {
                  const v = Array.isArray(next) ? next[0] : (next as number);
                  setSpeed(v);
                }}
              />
              <div className="mt-1 flex justify-between font-mono text-[10px] text-muted-foreground tabular-nums">
                <span>0.10</span>
                <span>0.50</span>
                <span>1.00</span>
              </div>
            </div>

            <Button
              variant="destructive"
              className="h-8 label-micro tracking-[0.22em]"
              onClick={() => {
                cmd.current = { linear: 0, angular: 0 };
                setKnob({ x: 0, y: 0 });
                setActive(false);
              }}
            >
              halt drive
            </Button>

            {/* Live readout — six tight rows */}
            <div className="grid grid-cols-2 gap-x-3 gap-y-1 border-t border-border pt-2">
              <Readout label="v"   value={`${fmt(v)} m/s`}   />
              <Readout label="ω"   value={`${fmt(w)} rad/s`} />
              <Readout label="x"   value={fmt(x)} />
              <Readout label="y"   value={fmt(y)} />
              <Readout label="ψ"   value={`${fmt(yaw)} rad`} />
              <Readout label="lidar" value={fmt(scan)} />
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function Readout({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="label-micro text-muted-foreground w-10">{label}</span>
      <span data-num className="font-mono text-[12px] text-foreground tabular-nums">
        {value}
      </span>
    </div>
  );
}
