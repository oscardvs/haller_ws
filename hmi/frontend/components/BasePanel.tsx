// hmi/frontend/components/BasePanel.tsx
"use client";
import { useEffect, useRef, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { useTelemetry } from "@/lib/telemetry";
import { CameraTile } from "./CameraTile";

const SEND_HZ = 10;

export function BasePanel() {
  const base = useTelemetry((s) => s.lastFrame?.base);
  const [speed, setSpeed] = useState(0.4);
  const cmd = useRef({ linear: 0, angular: 0 });
  const pad = useRef<HTMLDivElement>(null);
  const [knob, setKnob] = useState({ x: 0, y: 0 });

  // Send at fixed rate while non-zero, then one final zero
  useEffect(() => {
    const t = setInterval(() => {
      api.cmdVel(cmd.current.linear * speed, cmd.current.angular * speed).catch(() => {});
    }, 1000 / SEND_HZ);
    return () => clearInterval(t);
  }, [speed]);

  // keyboard
  useEffect(() => {
    const pressed = new Set<string>();
    const update = () => {
      let l = 0, a = 0;
      if (pressed.has("w") || pressed.has("arrowup")) l += 1;
      if (pressed.has("s") || pressed.has("arrowdown")) l -= 1;
      if (pressed.has("a") || pressed.has("arrowleft")) a += 1;
      if (pressed.has("d") || pressed.has("arrowright")) a -= 1;
      cmd.current = { linear: l, angular: a };
    };
    const down = (e: KeyboardEvent) => { pressed.add(e.key.toLowerCase()); update(); };
    const up = (e: KeyboardEvent) => { pressed.delete(e.key.toLowerCase()); update(); };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
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
    };
    const onDown = (e: MouseEvent) => { dragging = true; move(e.clientX, e.clientY); };
    const onMove = (e: MouseEvent) => { if (dragging) move(e.clientX, e.clientY); };
    const onUp = () => { dragging = false; cmd.current = { linear: 0, angular: 0 }; setKnob({ x: 0, y: 0 }); };
    el.addEventListener("mousedown", onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => { el.removeEventListener("mousedown", onDown); window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, []);

  return (
    <Card>
      <CardHeader><CardTitle>Base</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <CameraTile id="base_front" role="base" />
        <div className="grid grid-cols-2 gap-3 items-center">
          <div
            ref={pad}
            className="relative w-full aspect-square bg-muted/30 rounded-full border touch-none"
            tabIndex={0}
          >
            <div
              className="absolute w-8 h-8 -mt-4 -ml-4 rounded-full bg-emerald-500"
              style={{
                left: `${50 + knob.x * 40}%`,
                top: `${50 + knob.y * 40}%`,
              }}
            />
          </div>
          <div className="space-y-2">
            <div className="text-xs font-mono">speed {speed.toFixed(2)}×</div>
            <Slider min={0.1} max={1.0} step={0.05} value={[speed]} onValueChange={(next) => {
              const v = Array.isArray(next) ? next[0] : (next as number);
              setSpeed(v);
            }} />
            <Button variant="destructive" onClick={() => { cmd.current = { linear: 0, angular: 0 }; setKnob({ x: 0, y: 0 }); }}>
              STOP
            </Button>
            <div className="text-xs font-mono text-muted-foreground">
              v={base?.linear.toFixed(2) ?? "—"} m/s  ω={base?.angular.toFixed(2) ?? "—"} rad/s
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
