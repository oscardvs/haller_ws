// hmi/frontend/components/CalibrationWizard.tsx
"use client";

import * as React from "react";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useTelemetry } from "@/lib/telemetry";
import {
  startCalibration, captureNeutral, finishSweep, saveCalibration, abortCalibration,
  fetchCalibrationStatus,
  type CalibrationState, type JointCalibration,
} from "@/lib/calibration";
import { BACKEND_URL } from "@/lib/config";

interface Props { armId: string; onClose: () => void; }

type ProposedMap = Record<string, JointCalibration>;

export function CalibrationWizard({ armId, onClose }: Props) {
  const armFrame = useTelemetry(s => s.lastFrame?.arms?.[armId]);
  const calBlock = armFrame?.calibration;
  const stateFromTele = (calBlock?.state ?? null) as CalibrationState | null;
  const [phase, setPhase] = React.useState<CalibrationState>("homing");
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [proposed, setProposed] = React.useState<ProposedMap | null>(null);
  const [current, setCurrent] = React.useState<ProposedMap | null>(null);

  // Bootstrap: start the session if no session exists for this arm; re-attach if one already does.
  React.useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const status = await fetchCalibrationStatus(BACKEND_URL);
        if (cancelled) return;
        const active = status.current_session;
        if (active && active.arm_id === armId) {
          setPhase(active.state);
          if (active.state === "review") {
            setProposed(active.proposed ?? null);
            setCurrent(active.current ?? null);
          }
        } else if (!active) {
          await startCalibration(BACKEND_URL, armId);
          setPhase("homing");
        } else {
          setError(`Another session is active for arm '${active.arm_id}'.`);
        }
      } catch (e: any) { setError(e.message); }
    })();
    return () => { cancelled = true; };
  }, [armId]);

  // Mirror telemetry-driven state for non-review phases.
  React.useEffect(() => {
    if (stateFromTele && stateFromTele !== "review") setPhase(stateFromTele);
  }, [stateFromTele]);

  // Always abort on unmount unless the wizard reached "done" (Save handles its own cleanup).
  const isDoneRef = React.useRef(false);
  React.useEffect(() => () => {
    if (!isDoneRef.current) {
      abortCalibration(BACKEND_URL, armId).catch(() => {});
    }
  }, [armId]);

  const guarded = async (fn: () => Promise<void>) => {
    setBusy(true); setError(null);
    try { await fn(); } catch (e: any) { setError(e.message); }
    finally { setBusy(false); }
  };

  const ticks = (calBlock?.ticks ?? {}) as Record<string, number>;
  const mins  = (calBlock?.min   ?? {}) as Record<string, number>;
  const maxes = (calBlock?.max   ?? {}) as Record<string, number>;
  const joints = Object.keys(ticks).length ? Object.keys(ticks)
                 : Object.keys(proposed ?? current ?? {});

  return (
    <Sheet open onOpenChange={(o) => { if (!o) onClose(); }}>
      <SheetContent side="right" className="w-[480px] sm:max-w-md">
        <SheetHeader>
          <SheetTitle>Calibrate: {armId} arm</SheetTitle>
        </SheetHeader>

        {error && <p className="text-sm text-destructive py-2">{error}</p>}

        {phase === "homing" && (
          <section className="space-y-4 py-4">
            <h3 className="font-medium">Step 1 of 3 — Set neutral pose</h3>
            <p className="text-sm text-muted-foreground">
              Move the arm by hand into the pose you want to be "0°". Torque is off; the arm is back-drivable.
            </p>
            <Table>
              <TableHeader><TableRow><TableHead>Joint</TableHead><TableHead>Ticks</TableHead></TableRow></TableHeader>
              <TableBody>
                {joints.map(j => <TableRow key={j}><TableCell>{j}</TableCell><TableCell>{ticks[j] ?? "–"}</TableCell></TableRow>)}
              </TableBody>
            </Table>
            <div className="flex gap-2">
              <Button
                disabled={busy}
                onClick={() => guarded(async () => {
                  await captureNeutral(BACKEND_URL, armId);
                  setPhase("sweeping");
                })}
              >Capture neutral</Button>
              <Button variant="ghost" onClick={onClose}>Cancel</Button>
            </div>
          </section>
        )}

        {phase === "sweeping" && (
          <SweepingStep
            busy={busy} joints={joints}
            ticks={ticks} mins={mins} maxes={maxes}
            onDone={() => guarded(async () => {
              const res = await finishSweep(BACKEND_URL, armId);
              setProposed(res.proposed); setCurrent(res.current);
              setPhase("review");
            })}
            onCancel={onClose}
          />
        )}

        {phase === "review" && (
          <ReviewStep
            armId={armId} busy={busy} proposed={proposed} current={current}
            onSave={() => guarded(async () => {
              await saveCalibration(BACKEND_URL, armId);
              isDoneRef.current = true;
              onClose();
            })}
            onCancel={onClose}
          />
        )}
      </SheetContent>
    </Sheet>
  );
}

function SweepingStep({
  busy, joints, ticks, mins, maxes, onDone, onCancel,
}: {
  busy: boolean;
  joints: string[];
  ticks: Record<string, number>;
  mins: Record<string, number>;
  maxes: Record<string, number>;
  onDone: () => void;
  onCancel: () => void;
}) {
  return (
    <section className="space-y-4 py-4">
      <h3 className="font-medium">Step 2 of 3 — Range of motion</h3>
      <p className="text-sm text-muted-foreground">
        Wiggle every joint to its physical limits. The table records the
        extremes; click <strong>Done sweeping</strong> when every joint has both a min and a max.
      </p>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Joint</TableHead>
            <TableHead className="text-right">min</TableHead>
            <TableHead className="text-right">POS</TableHead>
            <TableHead className="text-right">max</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {joints.map(j => (
            <TableRow key={j}>
              <TableCell>{j}</TableCell>
              <TableCell className="text-right tabular-nums">{mins[j] ?? "–"}</TableCell>
              <TableCell className="text-right tabular-nums font-medium">{ticks[j] ?? "–"}</TableCell>
              <TableCell className="text-right tabular-nums">{maxes[j] ?? "–"}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <div className="flex gap-2">
        <Button disabled={busy} onClick={onDone}>Done sweeping</Button>
        <Button variant="ghost" onClick={onCancel}>Cancel</Button>
      </div>
    </section>
  );
}
function ReviewStep(props: any) {
  return <section className="py-4"><Button onClick={props.onSave}>Save</Button></section>;
}
