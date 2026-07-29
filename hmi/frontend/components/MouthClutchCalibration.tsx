"use client";

/**
 * Mouth-clutch calibration and speech-safety verification.
 *
 * Three windows, recorded as full traces rather than folded into a number:
 *   1. "talk"   — speak normally for a few seconds.
 *   2. "open"   — hold a relaxed open.
 *   3. "verify" — speak normally again, and check the derived thresholds
 *                 against it. `engaged: false` is the whole safety claim.
 *
 * WHY A TRACE AND NOT TWO NUMBERS
 * -------------------------------
 * The previous version recorded max(talk) and min(open) and derived the engage
 * threshold as a fraction of the gap between them. Measured on a real operator
 * that put the threshold at 0.60 against an observed jaw peak of 0.72 — 83% of
 * everything their jaw could do, to be sustained continuously while both arms
 * were moving. It could not be held, and a 22 s attempt never engaged once.
 *
 * The failure was the statistic, not the constant. Speech peaks are as high as
 * a deliberate open — a speech peak IS a wide-open jaw — so amplitude barely
 * separates them. What separates them is duration: speech drives the jaw back
 * toward closed on every consonant, several times a second, and a hold does
 * not. That statistic (the highest level SUSTAINED across a window) cannot be
 * recovered from a peak, so the browser records the samples and the backend
 * computes it. Deriving it here would put a second copy of the safety policy
 * in the client, where it could quietly disagree with the one that runs.
 *
 * The browser decides nothing. It records raw MediaPipe blendshape scores and
 * displays what it is told.
 */
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

import { api, type JawTrace, type MouthAnalysis, type MouthVerdict } from "@/lib/api";
import type { JawTraceRecorder } from "@/lib/mediapipe";

/** Must match safety.MOUTH_MIN_SEPARATION on the backend. */
export const MOUTH_MIN_SEPARATION = 0.25;

export type MouthCalib = {
  /** Highest level normal speech SUSTAINED for a hold window. */
  talk_hold: number | null;
  /** Level sustained through the whole deliberate-open capture. */
  open_hold: number | null;
  /** Loudest instantaneous speech sample. Shown, never used to decide. */
  talk_peak: number | null;
};

type Phase = "talk" | "open" | "verify";

export function mouthCalibReady(v: MouthCalib): boolean {
  return (
    v.talk_hold !== null &&
    v.open_hold !== null &&
    v.open_hold - v.talk_hold >= MOUTH_MIN_SEPARATION
  );
}

const PROMPT: Record<Phase, string> = {
  talk: "speak normally — a sentence or two, at working volume",
  open: "hold a RELAXED open, the way you could for a whole session — not your widest",
  verify: "speak normally again. the clutch must not engage once",
};

export function MouthClutchCalibration({
  liveJawOpen,
  value,
  onChange,
  recorder,
  analyze = api.mouthAnalyze,
}: {
  /** Current jawOpen score [0,1], or null when no face is tracked. */
  liveJawOpen: number | null;
  value: MouthCalib;
  onChange: (next: MouthCalib) => void;
  /** Fed by the render loop, so no sample is lost to a skipped re-render. */
  recorder: JawTraceRecorder;
  /** Injected so tests do not need a live backend. */
  analyze?: (body: {
    talk?: JawTrace; open?: JawTrace; verify?: JawTrace;
    calib?: { talk_hold: number; open_hold: number; talk_peak?: number | null };
  }) => Promise<MouthAnalysis>;
}) {
  const [capturing, setCapturing] = useState<Phase | null>(null);
  const [talkTrace, setTalkTrace] = useState<JawTrace | null>(null);
  const [openTrace, setOpenTrace] = useState<JawTrace | null>(null);
  const [report, setReport] = useState<MouthAnalysis | null>(null);
  const [verdict, setVerdict] = useState<MouthVerdict | null>(null);
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  // Only to redraw the live counter — the samples themselves live in the
  // recorder, which the render loop feeds directly.
  const [, setTick] = useState(0);

  useEffect(() => {
    if (capturing === null) return;
    const id = setInterval(() => setTick((t) => t + 1), 100);
    return () => clearInterval(id);
  }, [capturing]);

  const ready = mouthCalibReady(value);

  const stop = async (phase: Phase) => {
    const trace = recorder.stop();
    setCapturing(null);
    if (trace.length === 0) {
      // No face for the whole window: there is nothing to analyse, and a stale
      // previous calibration is better than one invented from nothing.
      setFailure("no face was tracked during that window");
      return;
    }
    setFailure(null);
    const nextTalk = phase === "talk" ? trace : talkTrace;
    const nextOpen = phase === "open" ? trace : openTrace;
    if (phase === "talk") setTalkTrace(trace);
    if (phase === "open") setOpenTrace(trace);

    setBusy(true);
    try {
      if (phase === "verify") {
        const out = await analyze({
          verify: trace,
          calib: {
            talk_hold: value.talk_hold!, open_hold: value.open_hold!,
            talk_peak: value.talk_peak,
          },
        });
        setVerdict(out.verify ?? null);
      } else {
        const out = await analyze({
          talk: nextTalk ?? undefined, open: nextOpen ?? undefined,
        });
        setReport(out);
        // A capture that failed to produce thresholds leaves NO calibration
        // rather than the previous one. Silently keeping the old numbers after
        // a recalibration the operator watched fail is how you end up driving
        // on thresholds nobody meant to be using.
        onChange(out.calib
          ? { talk_hold: out.calib.talk_hold, open_hold: out.calib.open_hold,
              talk_peak: out.calib.talk_peak ?? null }
          : { talk_hold: null, open_hold: null, talk_peak: null });
        // The thresholds moved; any earlier verdict was about the old ones.
        setVerdict(null);
      }
    } catch (e) {
      setFailure((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const toggle = (phase: Phase) => {
    if (capturing === phase) {
      void stop(phase);
      return;
    }
    if (capturing !== null) return;   // the other buttons are disabled anyway
    setCapturing(phase);
    recorder.start();
  };

  const label = (phase: Phase, verb: string) =>
    capturing === phase
      ? `${phase} · stop (${recorder.count})`
      : `${phase} · ${verb}`;

  const num = (v: number | null | undefined) =>
    v === null || v === undefined ? "—" : v.toFixed(2);

  return (
    <Card className="p-0">
      <CardContent className="p-3 flex flex-col gap-2 text-[12px] font-mono">
        <div className="flex justify-between">
          <span className="text-muted-foreground">clutch</span>
          <span>mouth</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">live jaw</span>
          <span className="tabular-nums">{num(liveJawOpen)}</span>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="outline" className="h-7 flex-1"
                  disabled={busy || (capturing !== null && capturing !== "talk")}
                  onClick={() => toggle("talk")}>
            {label("talk", "capture")}
          </Button>
          <Button size="sm" variant="outline" className="h-7 flex-1"
                  disabled={busy || (capturing !== null && capturing !== "open")}
                  onClick={() => toggle("open")}>
            {label("open", "capture")}
          </Button>
        </div>
        <div className="text-muted-foreground">
          {capturing === null
            ? "start already talking, or already open — not before. the window records from your first sample"
            : PROMPT[capturing]}
        </div>

        {/* Sustained vs peak, side by side, because the gap between them is the
            entire reason this works and the operator should see it. */}
        <div className="flex justify-between">
          <span className="text-muted-foreground">speech held..peak</span>
          <span className="tabular-nums">
            {num(value.talk_hold)} .. {num(value.talk_peak)}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">open held</span>
          <span className="tabular-nums">{num(value.open_hold)}</span>
        </div>
        {report?.thresholds ? (
          <div className="flex justify-between">
            <span className="text-muted-foreground">engage..release</span>
            <span className="tabular-nums">
              {num(report.thresholds.t_engage)} .. {num(report.thresholds.t_release)}
            </span>
          </div>
        ) : null}

        {report?.problems?.length ? (
          <div className="text-[var(--instrument-warn,oklch(75%_0.16_70))]">
            {report.problems.join("; ")}
          </div>
        ) : null}
        {failure ? (
          <div className="text-[var(--instrument-warn,oklch(75%_0.16_70))]">
            {failure}
          </div>
        ) : null}

        {/* The verification is the only evidence that speech cannot drive the
            robot. Until it has been run against this operator, say so. */}
        <Button size="sm" variant="outline" className="h-7"
                disabled={busy || !ready || (capturing !== null && capturing !== "verify")}
                onClick={() => toggle("verify")}>
          {label("verify", ready ? "speech test" : "needs calibration")}
        </Button>
        {verdict ? (
          <div
            className={verdict.engaged
              ? "text-[var(--instrument-warn,oklch(75%_0.16_70))]"
              : "text-[var(--instrument-line,oklch(80%_0.18_142))]"}
          >
            {verdict.engaged
              ? `FAIL — speech engaged the clutch at ${Math.round(verdict.first_engage_ms ?? 0)}ms. do not drive on this calibration`
              : `PASS — speech peaked at ${num(verdict.peak)}, sustained ${num(verdict.sustained)}, ${num(verdict.margin)} below engage`}
          </div>
        ) : ready ? (
          <div className="text-muted-foreground">
            not yet verified against your speech
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
