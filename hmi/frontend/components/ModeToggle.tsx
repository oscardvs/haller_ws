// hmi/frontend/components/ModeToggle.tsx
"use client";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { toast } from "sonner";

const ORDER: Array<"auto" | "manual" | "stop"> = ["auto", "manual", "stop"];

export function ModeToggle({ armId, mode }: { armId: string; mode: "auto" | "manual" | "stop" }) {
  return (
    <div className="flex items-center gap-2">
      <Badge variant={mode === "stop" ? "destructive" : mode === "auto" ? "default" : "secondary"}>
        {mode}
      </Badge>
      <div className="flex gap-1">
        {ORDER.map((m) => (
          <Button
            key={m}
            size="sm"
            variant={m === mode ? "default" : "outline"}
            onClick={async () => {
              try {
                await api.armMode(armId, m);
              } catch (e) {
                toast.error(`mode change failed: ${(e as Error).message}`);
              }
            }}
          >
            {m}
          </Button>
        ))}
      </div>
    </div>
  );
}
