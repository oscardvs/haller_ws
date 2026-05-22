"use client";

/**
 * CamerasPanel — strip of every configured camera, with live thumbnails for
 * the ones whose source is `opencv` and connected successfully at startup.
 *
 * Source of truth is GET /cameras (runtime state). Placeholder cameras still
 * appear so it's obvious which slots are reserved but not yet wired.
 */
import { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

import { api, cameraStreamUrl, type CameraInfo } from "@/lib/api";
import { CameraTile } from "./CameraTile";

export function CamerasPanel() {
  const [cams, setCams] = useState<CameraInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .cameras()
      .then((r) => setCams(r.cameras))
      .catch((e) => setError((e as Error).message));
  }, []);

  return (
    <Card className="overflow-hidden p-0">
      <div className="flex items-center justify-between gap-2 px-3 h-9 border-b border-border bg-card">
        <div className="flex items-center gap-2">
          <span className="label-micro text-muted-foreground">Cameras</span>
          <Badge variant="secondary">
            {cams ? `${cams.filter((c) => c.active).length}/${cams.length} live` : "…"}
          </Badge>
        </div>
      </div>

      <CardContent className="p-3">
        {error ? (
          <div className="label-micro text-destructive">cameras unavailable: {error}</div>
        ) : !cams ? (
          <div className="label-micro text-muted-foreground">loading…</div>
        ) : cams.length === 0 ? (
          <div className="label-micro text-muted-foreground">
            no cameras configured in <code>hmi/backend/config.yaml</code>
          </div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
            {cams.map((c) => (
              <CameraTile
                key={c.id}
                id={c.id}
                role={c.role}
                streamUrl={c.active ? cameraStreamUrl(c.id) : undefined}
                active={c.active}
                width={c.width}
                height={c.height}
                fps={c.fps}
              />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
