"use client";

/**
 * The triage list: every episode the current filters returned, in the order the
 * SERVER returned them. Nothing here sorts or filters — this is a window onto a
 * paged result, and a window that re-orders what it is shown is lying about the
 * page it is on.
 *
 * The scroll container is the panel body and carries `data-episode-list`; each
 * row carries `data-episode-index` with the STORED index, so the pane above can
 * put the selected take back on screen after an arrow-key walk.
 */
import { useEffect, useRef } from "react";

import type { LabEpisode, Mark } from "@/lib/lab";
import { Button, Empty, HeadRow, Panel, PanelHead, Refusal } from "./ui";
import { EpisodeRow } from "./EpisodeRow";

/** Header and row line share one template so the columns actually line up;
 *  the leading 18px is the selection checkbox. */
const GRID = "18px 92px minmax(0,1fr) 88px";

const COLS = [
  { key: "sel", label: "" },
  { key: "ep", label: "ep" },
  { key: "take", label: "verdict · task" },
  { key: "len", label: "len · share", align: "right" as const },
];

export function EpisodeList({
  episodes,
  total,
  loading,
  error,
  selectedIndex,
  selection,
  evalSet,
  onSelect,
  onToggleSelect,
  onMark,
  onNote,
  onLoadMore,
}: {
  episodes: LabEpisode[];
  total: number;
  loading: boolean;
  error: string | null;
  selectedIndex: number | null;
  selection: Set<number>;
  evalSet: Set<number>;
  onSelect: (episodeIndex: number) => void;
  onToggleSelect: (episodeIndex: number, shiftKey: boolean) => void;
  onMark: (episodeIndex: number, mark: Mark) => void;
  onNote: (episodeIndex: number, note: string) => void;
  onLoadMore?: () => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const sentinelRef = useRef<HTMLDivElement>(null);

  const hasMore = episodes.length < total;
  const canLoadMore = Boolean(onLoadMore) && hasMore;

  // The observer fires from a callback that outlives the render it was made in,
  // so it reads the current answer rather than a captured one.
  const latest = useRef({ loading, hasMore, onLoadMore });
  useEffect(() => {
    latest.current = { loading, hasMore, onLoadMore };
  });

  useEffect(() => {
    const el = sentinelRef.current;
    // jsdom has no IntersectionObserver, and a browser that scrolls in jumps
    // can miss it — the explicit button below is the path that always exists,
    // so its absence costs nothing but a click.
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(
      (entries) => {
        const s = latest.current;
        if (!s.onLoadMore || s.loading || !s.hasMore) return;
        if (entries.some((e) => e.isIntersecting)) s.onLoadMore();
      },
      { root: scrollRef.current, rootMargin: "160px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [canLoadMore]);

  return (
    <Panel className="flex-1">
      <PanelHead title="episodes" right={`${episodes.length} of ${total}`} />

      <div
        ref={scrollRef}
        data-episode-list
        className="flex min-h-0 flex-1 flex-col overflow-y-auto"
      >
        {episodes.length > 0 && (
          <HeadRow cols={COLS} className="shrink-0" style={{ gridTemplateColumns: GRID }} />
        )}

        {error ? (
          <div className="p-2.5">
            <Refusal>{error}</Refusal>
          </div>
        ) : episodes.length === 0 ? (
          // A refetch after a mark keeps the rows it already has: blanking the
          // list the operator is triaging reads as "the marks deleted it".
          <Empty>{loading ? "reading…" : "no episodes match these filters"}</Empty>
        ) : (
          episodes.map((ep) => (
            <div key={ep.index} data-episode-index={ep.index} className="shrink-0">
              <EpisodeRow
                episode={ep}
                selected={selectedIndex === ep.index}
                checked={selection.has(ep.index)}
                inEval={evalSet.has(ep.index)}
                onSelect={() => onSelect(ep.index)}
                onToggleSelect={(shiftKey) => onToggleSelect(ep.index, shiftKey)}
                onMark={(m) => onMark(ep.index, m)}
                onNote={(n) => onNote(ep.index, n)}
              />
            </div>
          ))
        )}

        {canLoadMore && (
          <div className="shrink-0 px-2.5 py-2">
            <div ref={sentinelRef} aria-hidden className="h-px" />
            <Button className="w-full" disabled={loading} onClick={() => onLoadMore?.()}>
              {loading ? "reading…" : `load more · ${total - episodes.length} left`}
            </Button>
          </div>
        )}
      </div>
    </Panel>
  );
}
