// hmi/frontend/components/EStopButton.tsx
"use client";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { toast } from "sonner";

export function EStopButton({ className }: { className?: string }) {
  const handleClick = async () => {
    try {
      await api.estop();
      toast.error("E-STOP triggered — torque off, base zeroed");
    } catch (e) {
      toast.error(`E-STOP failed: ${(e as Error).message}`);
    }
  };
  return (
    <Button
      variant="destructive"
      size="lg"
      onClick={handleClick}
      className={`rounded-full h-16 w-16 text-xs font-bold ${className ?? ""}`}
      aria-label="Emergency stop"
    >
      E-STOP
    </Button>
  );
}
