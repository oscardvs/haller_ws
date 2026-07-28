// hmi/frontend/components/CalibrationWizard.tsx
"use client";

/**
 * Modal calibration wizard, opened from the deep-linked /settings page.
 *
 * The state machine lives in lib/useCalibrationSession.ts, shared with the
 * cockpit's in-band Calibrate tab — re-attach-don't-restart, backend-ended
 * detection and confirm-before-discard are safety rules, and safety rules that
 * exist in two copies drift apart.
 *
 * This surface is modal, so it aborts the session on the way out: closing the
 * sheet IS ending the session. The cockpit's tab is not modal and does the
 * opposite.
 */
import * as React from "react";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  useCalibrationSession, confirmCopy, type ProposedMap,
} from "@/lib/useCalibrationSession";

interface Props { armId: string; onClose: () => void; }

export function CalibrationWizard({ armId, onClose }: Props) {
  const s = useCalibrationSession({
    armId,
    autoStart: true,
    abortOnUnmount: true,
    onClose,
  });
  const copy = confirmCopy(s.phase);

  return (
    <Sheet open onOpenChange={(o) => { if (!o) s.attemptClose(); }}>
      <SheetContent side="right" className="w-[480px] sm:max-w-md">
        <SheetHeader>
          <SheetTitle>Calibrate: {armId} arm</SheetTitle>
        </SheetHeader>

        {s.error && <p className="text-sm text-destructive py-2">{s.error}</p>}
        {s.blockError && (
          <p className="text-sm text-destructive py-2">Bus error: {s.blockError}</p>
        )}
        {s.sessionAborted && (
          <p className="text-sm text-destructive py-2">
            Calibration was aborted (bus error or arm disconnected). Close and retry.
          </p>
        )}

        {s.phase === "homing" && (
          <section className="space-y-4 py-4">
            <h3 className="font-medium">Step 1 of 3 — Set neutral pose</h3>
            <p className="text-sm text-muted-foreground">
              Move the arm by hand into the pose you want to be &quot;0°&quot;. Torque is off; the arm is back-drivable.
            </p>
            <Table>
              <TableHeader><TableRow><TableHead>Joint</TableHead><TableHead>Ticks</TableHead></TableRow></TableHeader>
              <TableBody>
                {s.joints.map(j => <TableRow key={j}><TableCell>{j}</TableCell><TableCell>{s.ticks[j] ?? "–"}</TableCell></TableRow>)}
              </TableBody>
            </Table>
            <div className="flex gap-2">
              <Button
                disabled={s.busy || s.sessionAborted}
                onClick={s.advance}
              >Capture neutral</Button>
              <Button variant="ghost" onClick={onClose}>Cancel</Button>
            </div>
          </section>
        )}

        {s.phase === "sweeping" && (
          <SweepingStep
            busy={s.busy} sessionAborted={s.sessionAborted} joints={s.joints}
            ticks={s.ticks} mins={s.mins} maxes={s.maxes}
            onDone={s.advance}
            onCancel={s.attemptClose}
          />
        )}

        {s.phase === "review" && (
          <ReviewStep
            busy={s.busy} proposed={s.proposed} current={s.current}
            onSave={s.advance}
            onCancel={s.attemptClose}
          />
        )}

        <AlertDialog open={s.confirmOpen} onOpenChange={(o) => { if (!o) s.keep(); }}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>{copy.title}</AlertDialogTitle>
              <AlertDialogDescription>{copy.body}</AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>{copy.keep}</AlertDialogCancel>
              <AlertDialogAction onClick={s.discard}>
                Discard
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </SheetContent>
    </Sheet>
  );
}

function SweepingStep({
  busy, sessionAborted, joints, ticks, mins, maxes, onDone, onCancel,
}: {
  busy: boolean;
  sessionAborted: boolean;
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
        <Button disabled={busy || sessionAborted} onClick={onDone}>Done sweeping</Button>
        <Button variant="ghost" onClick={onCancel}>Cancel</Button>
      </div>
    </section>
  );
}

function ReviewStep({
  busy, proposed, current, onSave, onCancel,
}: {
  busy: boolean;
  proposed: ProposedMap | null;
  current: ProposedMap | null;
  onSave: () => void;
  onCancel: () => void;
}) {
  const joints = Object.keys(proposed ?? {});
  return (
    <section className="space-y-4 py-4">
      <h3 className="font-medium">Step 3 of 3 — Review</h3>
      <p className="text-sm text-muted-foreground">
        Review the new calibration. The previous file is preserved as a <code>.bak-&lt;ts&gt;</code> sibling.
      </p>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Joint</TableHead>
            <TableHead className="text-right">old min → new min</TableHead>
            <TableHead className="text-right">old max → new max</TableHead>
            <TableHead className="text-right">old offset → new offset</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {joints.map(j => {
            const p = proposed?.[j]; const c = current?.[j];
            return (
              <TableRow key={j}>
                <TableCell>{j}</TableCell>
                <TableCell className="text-right tabular-nums">
                  <span>{c?.range_min ?? "—"}</span> → <strong>{p?.range_min}</strong>
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  <span>{c?.range_max ?? "—"}</span> → <strong>{p?.range_max}</strong>
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  <span>{c?.homing_offset ?? "—"}</span> → <strong>{p?.homing_offset}</strong>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
      <div className="flex gap-2">
        <Button disabled={busy} onClick={onSave}>Save</Button>
        <Button variant="ghost" onClick={onCancel}>Cancel</Button>
      </div>
    </section>
  );
}
