// hmi/frontend/components/CameraTile.tsx
"use client";
import { Badge } from "@/components/ui/badge";

export function CameraTile({ id, role }: { id: string; role: "wrist" | "base" }) {
  return (
    <div className="aspect-video bg-muted/30 rounded-md flex items-center justify-center border border-dashed">
      <div className="flex flex-col items-center gap-1 text-muted-foreground">
        <span className="text-xs uppercase">{role}</span>
        <span className="font-mono text-xs">{id}</span>
        <Badge variant="outline" className="text-xs">no feed</Badge>
      </div>
    </div>
  );
}
