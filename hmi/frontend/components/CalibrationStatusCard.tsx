"use client";

import * as React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import type { CalibrationArmStatus } from "@/lib/calibration";

interface Props {
  status: CalibrationArmStatus;
  canStart: boolean;
  blockedReason?: string;
  onCalibrate: () => void;
}

export function CalibrationStatusCard({ status, canStart, blockedReason, onCalibrate }: Props) {
  const fileLabel = status.has_file
    ? `Calibrated · ${new Date((status.mtime ?? 0) * 1000).toLocaleString()}`
    : "No calibration file";
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle className="text-base font-medium">arm: {status.id}</CardTitle>
        <Badge variant={status.has_file ? "secondary" : "destructive"}>
          {status.has_file ? "OK" : "MISSING"}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-2">
        <p className="text-sm text-muted-foreground">{fileLabel}</p>
        <p className="text-xs text-muted-foreground break-all">{status.path}</p>
        <Button
          onClick={onCalibrate}
          disabled={!canStart || status.in_session}
          title={!canStart ? blockedReason : undefined}
          className="w-full"
        >
          {status.in_session ? "In session…" : "Calibrate"}
        </Button>
      </CardContent>
    </Card>
  );
}
