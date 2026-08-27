"use client";

/**
 * One episode, played.
 *
 * Three facts about a LeRobot v3.0 dataset shape this whole component:
 *
 * 1. Many episodes are packed into ONE mp4. Changing episode inside the same
 *    file is a seek; only changing file is a load. `videoSrcKey` is the
 *    identity of the loaded file and the `src` is set from it and nothing
 *    else — assigning `src` on every episode change costs a full re-buffer on
 *    every arrow key, which is the difference between triaging 46 takes and
 *    giving up.
 * 2. The file runs straight on into the next episode, BUTT-JOINED with no gap:
 *    episode N's `to_timestamp` IS episode N+1's `from_timestamp`, and the
 *    frames either side of that boundary decode to different images (measured
 *    on file-001 of local/so101_pick_cube at 60.5666). Nothing stops playback
 *    there, so "watch episode 3" quietly becomes "watch episodes 3, 4 and 5"
 *    unless the clamp below holds it.
 *
 *    `to_timestamp` is EXCLUSIVE — an episode owns the frames whose PTS lies
 *    in [from, to) — so the last real frame sits at exactly `to - 1/fps`.
 *    That, and not some small epsilon, is the clamp. Measured on the same
 *    file, whose last episode ends at its container duration of 82.567:
 *    seeking to 82.567, 82.557 and 82.534 all decode NO FRAME, and 82.533
 *    (= to - 1/30) decodes the last one. A 0.01 s epsilon is smaller than a
 *    frame period at 30 fps, so it lands past the end and shows a blank frame
 *    instead of the take — on the last episode of every file, which is 7 of
 *    the 46 here. `- 1/fps` is also the largest clamp that cannot leak into
 *    the next episode, so it is the correct value rather than a working one.
 * 3. `playbackRate` is a property of the media ELEMENT and resets to 1 on
 *    every source swap. It is re-applied on `loadedmetadata`, not once.
 */
import {
  forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState,
} from "react";
import { PauseIcon, PlayIcon } from "lucide-react";

import { labVideoUrl, sliceFor, videoSrcKey, type LabEpisode } from "@/lib/lab";
import { Empty, Panel, PanelHead } from "@/components/lab/ui";
import { CameraPicker } from "@/components/lab/CameraPicker";

export type EpisodePlayerHandle = {
  togglePlay(): void;
  /** Move `delta` steps along RATES. The keyboard path to speed. */
  stepRate(delta: number): void;
};

/** Review speeds. 0.5 for a grasp, 4 for the approach nobody needs to watch. */
const RATES = [0.5, 1, 1.5, 2, 4] as const;

/** Survives a reload because the operator's review speed is a habit, not a
 *  property of the episode. Private mode has no storage and 1x is a fine
 *  default, so every access is guarded. */
const RATE_KEY = "haller.lab.rate";

function readRate(): number | null {
  try {
    const raw = window.localStorage.getItem(RATE_KEY);
    const n = raw === null ? NaN : Number(raw);
    return (RATES as readonly number[]).includes(n) ? n : null;
  } catch {
    return null;
  }
}

function writeRate(v: number): void {
  try {
    window.localStorage.setItem(RATE_KEY, String(v));
  } catch {
    /* storage is off; the rate still applies to this session */
  }
}

export const EpisodePlayer = forwardRef<
  EpisodePlayerHandle,
  {
    repoId: string;
    episode: LabEpisode | null;
    videoKeys: string[];
    videoKey: string | null;
    onVideoKey: (k: string) => void;
    /** The dataset's own frame rate, from `DatasetDetail.fps`. The clamp is
     *  one frame period wide and cannot be derived from anything else. */
    fps: number;
    /** Episode-relative seconds, clamped to [0, duration] — the trace charts
     *  plot against episode time, not file time. */
    onTime?: (episodeRelativeSeconds: number) => void;
  }
>(function EpisodePlayer(
  { repoId, episode, videoKeys, videoKey, fps, onVideoKey, onTime },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [pos, setPos] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const [failed, setFailed] = useState(false);

  const ready = episode !== null && videoKey !== null;
  const slice = episode ? sliceFor(episode, videoKey) : null;
  const from = slice?.from_timestamp ?? 0;
  const to = slice?.to_timestamp ?? 0;
  const dur = Math.max(0, to - from);
  /** One frame period. The clamp, the park position and the "already at the
   *  end" test are all expressed in it — every one of them is really asking
   *  "is there another frame after this one". */
  const frame = fps > 0 ? 1 / fps : 1 / 30;
  /** The last frame that actually exists in this episode. */
  const lastFrameAt = Math.max(from, to - frame);
  const srcKey =
    episode !== null && videoKey !== null
      ? videoSrcKey(repoId, episode, videoKey)
      : null;

  /** Which file the element currently holds. Null when nothing is loaded. */
  const loadedKey = useRef<string | null>(null);
  /** Where to land once metadata arrives — a fresh file cannot be seeked yet. */
  const pendingSeek = useRef<number | null>(null);
  const onTimeRef = useRef(onTime);
  const rateRef = useRef(rate);

  useEffect(() => { onTimeRef.current = onTime; });
  useEffect(() => { rateRef.current = rate; }, [rate]);

  useEffect(() => {
    const stored = readRate();
    if (stored !== null) setRate(stored);
  }, []);

  useEffect(() => {
    const v = videoRef.current;
    if (v) v.playbackRate = rate;
  }, [rate]);

  // The element unmounts with the empty state, so the loaded identity has to
  // go with it or the next episode would be treated as a seek into a file
  // that is no longer there.
  useEffect(() => {
    if (!ready) {
      loadedKey.current = null;
      setPlaying(false);
      setPos(0);
    }
  }, [ready]);

  const epIndex = episode?.episode_index ?? null;

  useEffect(() => {
    const v = videoRef.current;
    if (!v || srcKey === null || epIndex === null || videoKey === null) return;
    setFailed(false);
    if (loadedKey.current !== srcKey) {
      loadedKey.current = srcKey;
      pendingSeek.current = from;
      v.src = labVideoUrl(repoId, videoKey, epIndex);
      v.load();
    } else {
      // Same packed file, different episode: a seek, never a load.
      try { v.currentTime = from; } catch { /* not seekable yet */ }
    }
    setPos(0);
    onTimeRef.current?.(0);
  }, [srcKey, from, repoId, videoKey, epIndex]);

  const emit = useCallback((absolute: number) => {
    const rel = Math.min(dur, Math.max(0, absolute - from));
    setPos(rel);
    onTimeRef.current?.(rel);
  }, [dur, from]);

  const handleTimeUpdate = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    // The clamp. Trigger and park are the SAME instant — the last frame that
    // exists — because there is nothing between it and the next episode to
    // park in.
    if (dur > 0 && v.currentTime > lastFrameAt) {
      if (!v.paused) v.pause();
      v.currentTime = lastFrameAt;
      emit(lastFrameAt);
      return;
    }
    emit(v.currentTime);
  }, [dur, emit, lastFrameAt]);

  const handleLoadedMetadata = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    v.playbackRate = rateRef.current;
    const want = pendingSeek.current;
    pendingSeek.current = null;
    if (want !== null) {
      try { v.currentTime = want; } catch { /* unseekable source */ }
    }
    emit(v.currentTime);
  }, [emit]);

  const seekRelative = useCallback((rel: number) => {
    const v = videoRef.current;
    if (!v || dur <= 0) return;
    const t = Math.min(dur, Math.max(0, rel));
    try { v.currentTime = from + t; } catch { /* unseekable source */ }
    emit(from + t);
  }, [dur, emit, from]);

  const applyRate = useCallback((next: number) => {
    setRate(next);
    writeRate(next);
    const v = videoRef.current;
    if (v) v.playbackRate = next;
  }, []);

  const togglePlay = useCallback(() => {
    const v = videoRef.current;
    if (!v || !ready) return;
    if (v.paused) {
      // Parked on the clamp: play means play the episode, not its last frame.
      if (dur > 0 && v.currentTime >= lastFrameAt) v.currentTime = from;
      void v.play().catch(() => { /* the element reports it via onError */ });
    } else {
      v.pause();
    }
  }, [dur, from, lastFrameAt, ready]);

  useImperativeHandle(ref, () => ({
    togglePlay,
    stepRate(delta: number) {
      const i = RATES.indexOf(rateRef.current as (typeof RATES)[number]);
      const at = i < 0 ? RATES.indexOf(1) : i;
      applyRate(RATES[Math.min(RATES.length - 1, Math.max(0, at + delta))]);
    },
  }), [applyRate, togglePlay]);

  const frac = dur > 0 ? Math.min(1, Math.max(0, pos / dur)) : 0;

  return (
    <Panel className="min-h-0 flex-1">
      <PanelHead
        title="player"
        right={
          episode ? (
            <>
              Ep {episode.label}{" "}
              <span className="opacity-60">idx {episode.episode_index}</span>
            </>
          ) : (
            "—"
          )
        }
      >
        <CameraPicker keys={videoKeys} value={videoKey} onChange={onVideoKey} />
      </PanelHead>

      <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-hidden p-2.5">
        {episode !== null && videoKey !== null ? (
          // Capped by HEIGHT, not width: a 4:3 frame stretched across a wide
          // review column pushes the traces off the fixed viewport.
          <div className="relative aspect-[4/3] w-[min(100%,calc(38vh*4/3))] shrink-0 self-center overflow-hidden rounded-md border border-border bg-[var(--haller-inset)]">
            <video
              ref={videoRef}
              preload="metadata"
              playsInline
              aria-label={`episode ${episode.label} video`}
              className="h-full w-full object-contain"
              onLoadedMetadata={handleLoadedMetadata}
              onTimeUpdate={handleTimeUpdate}
              onSeeked={handleTimeUpdate}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              onEnded={() => setPlaying(false)}
              onError={() => setFailed(true)}
            />
            {failed && (
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="scanlines absolute inset-0" aria-hidden />
                <span className="relative font-mono text-[10px] tracking-[0.14em] uppercase text-[var(--haller-warn)]">
                  no video for this key
                </span>
              </div>
            )}
          </div>
        ) : (
          <Empty>pick an episode</Empty>
        )}

        <div className="flex shrink-0 items-center gap-2.5">
          <button
            type="button"
            disabled={!ready}
            onClick={togglePlay}
            aria-label={playing ? "pause" : "play"}
            className="inline-flex h-7 w-9 shrink-0 items-center justify-center rounded-md border border-border bg-secondary text-foreground transition-colors hover:bg-muted disabled:opacity-50"
          >
            {playing
              ? <PauseIcon size={12} aria-hidden />
              : <PlayIcon size={12} aria-hidden />}
          </button>

          {/* Fixed width: a readout that grows from 9.9 to 10.0 shifts the
              whole transport bar under the pointer. */}
          <span
            data-num
            className="min-w-[86px] shrink-0 font-mono text-[10px] tabular-nums text-muted-foreground"
          >
            {pos.toFixed(1)} / {dur.toFixed(1)} s
          </span>

          <div
            role="slider"
            tabIndex={ready ? 0 : -1}
            aria-label="scrub episode"
            aria-valuemin={0}
            aria-valuemax={Number(dur.toFixed(1))}
            aria-valuenow={Number(pos.toFixed(1))}
            aria-valuetext={`${pos.toFixed(1)} of ${dur.toFixed(1)} seconds`}
            aria-disabled={!ready}
            onClick={(e) => {
              if (!ready) return;
              const r = e.currentTarget.getBoundingClientRect();
              const f = Math.min(1, Math.max(0, (e.clientX - r.left) / (r.width || 1)));
              seekRelative(f * dur);
            }}
            onKeyDown={(e) => {
              if (!ready) return;
              if (e.key === "ArrowLeft") { e.preventDefault(); seekRelative(pos - 1); }
              else if (e.key === "ArrowRight") { e.preventDefault(); seekRelative(pos + 1); }
            }}
            className={
              "flex h-6 min-w-0 flex-1 items-center rounded-sm " +
              (ready ? "cursor-pointer" : "opacity-50")
            }
          >
            <div className="relative h-1.5 w-full overflow-hidden rounded-[2px] bg-[var(--input)]">
              <div
                className="absolute inset-y-0 left-0 bg-[var(--haller-live)]"
                style={{ width: `${frac * 100}%` }}
              />
            </div>
          </div>

          <button
            type="button"
            disabled={!ready}
            onClick={() => applyRate(RATES[(RATES.indexOf(rate as (typeof RATES)[number]) + 1) % RATES.length])}
            aria-label={`playback rate ${rate}x — cycles`}
            title="playback rate"
            className="inline-flex h-7 w-[52px] shrink-0 items-center justify-center rounded-md border border-border bg-secondary label-micro tracking-[0.12em] text-foreground transition-colors hover:bg-muted disabled:opacity-50"
          >
            <span data-num className="tabular-nums">{rate}×</span>
          </button>
        </div>
      </div>
    </Panel>
  );
});
