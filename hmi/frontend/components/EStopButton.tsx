// hmi/frontend/components/EStopButton.tsx
"use client";
import { Button } from "@/components/ui/button";

export function EStopButton({ className }: { className?: string }) {
  return (
    <Button
      variant="destructive"
      size="lg"
      className={`rounded-full h-16 w-16 text-xs font-bold ${className ?? ""}`}
      aria-label="Emergency stop"
    >
      E-STOP
    </Button>
  );
}
