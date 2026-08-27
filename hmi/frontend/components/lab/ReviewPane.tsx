"use client";

/**
 * Review — watch a take, decide whether a policy should learn from it.
 *
 * This is the surface that decides what gets trained on, so everything here is
 * arranged around one loop: watch, judge, next. The player, the gripper and the
 * joint traces are ONE instrument — they share a playhead, so the frame on
 * screen and the sample under the cursor are the same instant — and the list on
 * the right is the queue. A triage pass is 70 episodes at a keystroke each,
 * which is why the marks are optimistic and the keyboard reaches everything.
 *
 * Three rules this pane exists to hold:
 *
 * 1. NOTHING assumes a channel or a camera count. The two datasets on this disk
 *    are a 6-channel single-arm one with a single camera key and a 12-channel
 *    bimanual one with three, and every count comes from `detail.video_keys`
 *    and `trace.names`.
 * 2. The list is a WINDOW onto a server-sorted, server-filtered result. Sorting
 *    or filtering here would order one page and call it the answer.
 * 3. The eval split is never recomputed in the browser. `lab.split` is the only
 *    source for the `val` badges; a second implementation would drift and the
 *    badges would lie about which demonstrations the policy has already seen —
 *    the one error on this page that cannot be spotted by looking at it.
 */
import {
  Fragment, useCallback, useEffect, useMemo, useRef, useState,
} from "react";
import { toast } from "sonner";

import {
  gripperGuides,
  isBusy,
  isForbidden,
  isMissing,
  lab,
  reason,
  REMOTE_REFUSED,
  rigLabel,
  type DatasetDetail,
  type DatasetSummary,
  type LabEpisode,
  type Mark,
  type SplitMode,
  type Trace,
} from "@/lib/lab";
import { isEditableTarget, useSticky } from "@/components/cockpit/lib";
import {
  Button, Chip, Empty, MARK_COLOR, MarkBar, Note, Panel, PanelHead, Refusal,
} from "@/components/lab/ui";
import { DatasetShelf } from "@/components/lab/DatasetShelf";
import { EpisodePlayer, type EpisodePlayerHandle } from "@/components/lab/EpisodePlayer";
import { EpisodeList } from "@/components/lab/EpisodeList";
import {
  DEFAULT_FILTERS, EpisodeFilters, type EpisodeFilterState,
} from "@/components/lab/EpisodeFilters";
import { BulkBar } from "@/components/lab/BulkBar";
import { TraceChart } from "@/components/lab/charts/TraceChart";
import { GripperChart } from "@/components/lab/charts/GripperChart";
import { AutoclassifyDialog } from "@/components/lab/AutoclassifyDialog";
import { PruneDialog } from "@/components/lab/PruneDialog";
import { DeleteDatasetDialog } from "@/components/lab/DeleteDatasetDialog";

/** One page of the episode list. The window GROWS rather than being stitched
 *  from separate requests: a list assembled out of pages fetched under
 *  different marks shows two different answers in one column. */
const PAGE = 100;

/** Trailing coalesce on an `epoch`-driven re-read. A mark is a keystroke and
 *  the optimistic patch has already landed, so a burst of J/K/J/K becomes one
 *  reconcile instead of one request per key. The first read is not delayed. */
const REFRESH_MS = 180;

const STALE_NOTE =
  "at least one mark no longer describes the episode at its index — " +
  "an episode was pruned and the survivors renumbered. re-check before training.";

/* ─── keyboard triage ─────────────────────────────────────────────────────
   The bindings and the hint row are built from ONE table, so the hint cannot
   drift from the handler. The table is pure DATA: the handlers used to live on
   it as closures, and a closure over `playerRef` in a list the hint row also
   renders reaches a ref during render. The actions are looked up from the same
   table by `id` inside the keydown effect instead, so a binding added here
   without an action does not compile. This is the reason this beats a
   spreadsheet: the operator's hands never leave the keys between watching and
   judging. */

/** A failure that persists until the operator does something about it, so it
 *  gets a panel and never a toast. A 409 carries the backend's own sentence and
 *  is quoted rather than paraphrased; it is a state to clear, not a fault. */
type Failure = { text: string; tone: "warn" | "fault" };

const failureOf = (e: unknown): Failure => ({
  text: reason(e),
  tone: isBusy(e) ? "warn" : "fault",
});

type BindingId = "play" | "step" | "keep" | "reject" | "rate";

type Binding = {
  /** Names the action. Typed, so a new row here forces a new case below. */
  id: BindingId;
  /** Matched against the normalised key — lowercased, with " " as "space". */
  keys: string[];
  /** How the row reads. Same order as `keys`. */
  hint: string[];
  label: string;
};

const BINDINGS: readonly Binding[] = [
  { id: "play", keys: ["space"], hint: ["space"], label: "play / pause" },
  { id: "step", keys: ["j", "l"], hint: ["J", "L"], label: "prev / next episode" },
  { id: "keep", keys: ["k"], hint: ["K"], label: "toggle keep" },
  { id: "reject", keys: ["r"], hint: ["R"], label: "toggle reject" },
  { id: "rate", keys: ["[", "]"], hint: ["[", "]"], label: "slower / faster" },
];

/** No held-out plan means no `val` badges. One shared empty set rather than a
 *  fresh one per render, so the list below is not re-rendered by it. */
const NO_EVAL: Set<number> = new Set();

export function ReviewPane({
  repoId,
  onPickDataset,
}: {
  repoId: string | null;
  onPickDataset: (repoId: string | null) => void;
}): React.ReactElement {
  /* ---- the dataset ------------------------------------------------------ */
  const [detail, setDetail] = useState<DatasetDetail | null>(null);
  /** Only for the two numbers `detail` does not carry: bytes on disk, and
   *  whether the marks survived the last renumbering. */
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [detailError, setDetailError] = useState<Failure | null>(null);
  /** The build predates the Lab. Not an error and not empty — a third state. */
  const [noLab, setNoLab] = useState(false);

  /** Bumped after every mutation. In the dependency list of the reads below so
   *  a mark refreshes the counts, the tag vocabulary and the held-out plan. */
  const [epoch, setEpoch] = useState(0);
  const bump = useCallback(() => setEpoch((n) => n + 1), []);

  /* ---- the list --------------------------------------------------------- */
  const [filters, setFilters] = useState<EpisodeFilterState>(DEFAULT_FILTERS);
  const [limit, setLimit] = useState(PAGE);
  const [rows, setRows] = useState<LabEpisode[]>([]);
  const [total, setTotal] = useState(0);
  const [listLoading, setListLoading] = useState(false);
  const [listError, setListError] = useState<string | null>(null);

  /* ---- selection -------------------------------------------------------- */
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [selection, setSelection] = useState<Set<number>>(() => new Set());
  /** Where a shift-range starts. A plain click moves it; a shift-click never
   *  does, so the range can be re-dragged from the same anchor. */
  const anchor = useRef<number | null>(null);
  const [bulkBusy, setBulkBusy] = useState(false);

  /* ---- the instrument --------------------------------------------------- */
  /** The trace, TAGGED with the episode it was read for. Tagged rather than
   *  cleared the moment the selection moves: a clear is a setState in the
   *  effect body, and a trace that outlives its take puts episode N's samples
   *  under episode N+1's video — the one thing this instrument promises is
   *  that the frame and the sample are the same instant. */
  const [traceRead, setTraceRead] = useState<{
    index: number;
    trace: Trace | null;
    error: string | null;
  } | null>(null);
  const [playheadT, setPlayheadT] = useState<number | null>(null);
  const [videoKey, setVideoKey] = useSticky<string | null>("lab.review.camera", null);
  const [overlay, setOverlay] = useSticky<boolean>("lab.review.overlay", true);
  const playerRef = useRef<EpisodePlayerHandle>(null);

  const [dialog, setDialog] = useState<"autoclass" | "prune" | "delete" | null>(null);

  /* ---- the held-out plan -------------------------------------------------
     Sticky under `lab.split.*` so Review and Train read the same plan. Read
     only: the knobs live in the launcher, and this pane never recomputes the
     split — see the header note. */
  const [evalSplit] = useSticky<number>("lab.split.eval", 0.2);
  const [splitSeed] = useSticky<number>("lab.split.seed", 42);
  const [splitMode] = useSticky<SplitMode>("lab.split.mode", "random");
  const [evalPlan, setEvalPlan] = useState<Set<number>>(() => new Set());
  /** Derived rather than cleared when the fraction goes to zero: a `val` badge
   *  that outlives the plan it came from claims the policy never saw a take it
   *  is about to train on. */
  const evalSet = evalSplit > 0 ? evalPlan : NO_EVAL;

  /* ── the dataset, read whole ───────────────────────────────────────────
     `detail` carries the FULL episode list, which is what the dialogs, the tag
     vocabulary and the mark counts are built from — the paged list is a window
     and cannot answer "how many are still unmarked". The backend caches the
     grade against a file stamp and re-reads the marks every time, so this is
     cheap after the first call and never serves a stale decision. */
  useEffect(() => {
    // Nothing to clear on the way out: with no `repoId` this pane is the
    // dataset shelf, and neither `detail` nor `summary` is read past it.
    if (!repoId) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      void (async () => {
        try {
          const d = await lab.detail(repoId);
          if (cancelled) return;
          setDetail(d);
          setNoLab(false);
          setDetailError(null);
        } catch (e) {
          if (cancelled) return;
          if (isMissing(e)) {
            setNoLab(true);
            setDetail(null);
            setDetailError(null);
          } else {
            // The previous read stays on screen. Blanking the list an operator
            // is triaging reads as "the marks deleted it".
            setDetailError(failureOf(e));
          }
        }
        try {
          const r = await lab.datasets();
          if (!cancelled) {
            setSummary(r.datasets.find((d) => d.repo_id === repoId) ?? null);
          }
        } catch {
          // Costs the size readout and the stale badge, nothing else.
          if (!cancelled) setSummary(null);
        }
      })();
    }, epoch === 0 ? 0 : REFRESH_MS);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [repoId, epoch]);

  /* ── one page of the list ──────────────────────────────────────────────
     Sort and filter are the SERVER's, and the request always starts at offset 0
     with the whole held window as its limit. See `PAGE`. */
  useEffect(() => {
    if (!repoId) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      // Raised where the request actually starts rather than in the effect
      // body: during the coalesce window nothing is in flight yet.
      setListLoading(true);
      void (async () => {
        try {
          const page = await lab.episodes({
            repo_id: repoId,
            sort: filters.sort,
            order: filters.order,
            filter_mark: filters.mark,
            filter_verdict: filters.verdict,
            tag: filters.tag,
            q: filters.q || null,
            offset: 0,
            limit,
          });
          if (cancelled) return;
          setRows(page.episodes);
          setTotal(page.total);
          setListError(null);
        } catch (e) {
          if (cancelled) return;
          if (isMissing(e)) setNoLab(true);
          else setListError(reason(e));
        } finally {
          if (!cancelled) setListLoading(false);
        }
      })();
    }, epoch === 0 ? 0 : REFRESH_MS);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [repoId, filters, limit, epoch]);

  /* ── the held-out set ──────────────────────────────────────────────────
     `epoch` is in the deps because the split is planned over the KEPT list —
     marking one more take keep moves which episodes the trainer never sees. */
  useEffect(() => {
    if (!repoId || evalSplit <= 0) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      void (async () => {
        try {
          const plan = await lab.split(repoId, evalSplit, splitSeed, splitMode);
          if (!cancelled) setEvalPlan(new Set(plan.eval_episodes));
        } catch {
          // No plan means no badges. An invented one would claim the policy
          // has not seen episodes it trained on.
          if (!cancelled) setEvalPlan(NO_EVAL);
        }
      })();
    }, epoch === 0 ? 0 : REFRESH_MS);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [repoId, evalSplit, splitSeed, splitMode, epoch]);

  /* ── the selected episode's trace ──────────────────────────────────────
     Not keyed on `epoch`: a mark is an opinion about the numbers, it does not
     change them. */
  useEffect(() => {
    if (!repoId || selectedIndex === null) return;
    const index = selectedIndex;
    let cancelled = false;
    void (async () => {
      try {
        const t = await lab.trace(repoId, index);
        if (!cancelled) setTraceRead({ index, trace: t, error: null });
      } catch (e) {
        if (cancelled) return;
        setTraceRead({
          index,
          trace: null,
          error: isMissing(e) ? "this backend serves no traces" : reason(e),
        });
      }
    })();
    return () => { cancelled = true; };
  }, [repoId, selectedIndex]);

  /* ── the camera key ────────────────────────────────────────────────────
     Sticky across a tab glance, but validated against THIS dataset's keys:
     `left_wrist` is a real key on the bimanual corpus and does not exist on the
     solo one, and a key the dataset has never heard of plays nothing. */
  useEffect(() => {
    const keys = detail?.video_keys ?? [];
    if (keys.length === 0) {
      if (videoKey !== null) setVideoKey(null);
      return;
    }
    if (videoKey === null || !keys.includes(videoKey)) setVideoKey(keys[0]);
  }, [detail, videoKey, setVideoKey]);

  /* ── derived readings ──────────────────────────────────────────────────── */

  /** The read is only an answer about the take it was taken for. */
  const forSelected = traceRead?.index === selectedIndex ? traceRead : null;
  const trace = forSelected?.trace ?? null;
  const traceError = forSelected?.error ?? null;

  const episodes = useMemo(() => detail?.episodes ?? [], [detail]);

  /** One source for the counts: the full episode list, which the optimistic
   *  patch below writes to as well, so the header never disagrees with a row. */
  const counts = useMemo(() => {
    const c = { keep: 0, reject: 0, unset: 0 };
    for (const e of episodes) {
      if (e.mark === "keep") c.keep += 1;
      else if (e.mark === "reject") c.reject += 1;
      else c.unset += 1;
    }
    return c;
  }, [episodes]);

  const tagVocab = useMemo(() => {
    const seen = new Set<string>();
    for (const e of episodes) for (const t of e.tags ?? []) seen.add(t);
    return [...seen].sort();
  }, [episodes]);

  const rowByIndex = useMemo(
    () => new Map(rows.map((r) => [r.index, r] as const)),
    [rows],
  );

  /** The take under the player. Memoised because the keyboard bindings close
   *  over it, and the playhead re-renders this component ~60 times a second —
   *  a fresh binding list per frame would re-register the window listener at
   *  the same rate. Read from the list window first, then from the full
   *  episode list, so a take filtered out from under the player still plays. */
  const selected = useMemo<LabEpisode | null>(
    () =>
      selectedIndex === null
        ? null
        : rowByIndex.get(selectedIndex)
          ?? episodes.find((e) => e.index === selectedIndex)
          ?? null,
    [selectedIndex, rowByIndex, episodes],
  );

  /* ── selection ─────────────────────────────────────────────────────────── */

  const select = useCallback((index: number) => {
    setSelectedIndex(index);
    setPlayheadT(null);
  }, []);

  // Land the selected row back on screen after a J/L walk. The list owns its
  // own scroll box, so this reaches into it by the attributes it publishes.
  useEffect(() => {
    if (selectedIndex === null) return;
    const list = document.querySelector("[data-episode-list]");
    const row = list?.querySelector(`[data-episode-index="${selectedIndex}"]`);
    if (row instanceof HTMLElement) row.scrollIntoView({ block: "nearest" });
  }, [selectedIndex, rows]);

  // Something to watch as soon as there is something to watch. Adjusted during
  // render, not from an effect: an effect commits one frame of empty player
  // under a list that already has rows, and the fix for that is not a timer.
  if (selectedIndex === null && rows.length > 0) {
    setSelectedIndex(rows[0].index);
    setPlayheadT(null);
  }

  const onToggleSelect = useCallback(
    (index: number, shiftKey: boolean) => {
      const at = rows.findIndex((r) => r.index === index);
      const from =
        anchor.current === null
          ? -1
          : rows.findIndex((r) => r.index === anchor.current);
      setSelection((prev) => {
        const next = new Set(prev);
        // The range is taken from the CURRENT LIST ORDER, not from the numeric
        // indices: the list is sorted server-side, and what the operator drags
        // across is what they see.
        if (shiftKey && at >= 0 && from >= 0) {
          const [lo, hi] = from <= at ? [from, at] : [at, from];
          for (let i = lo; i <= hi; i += 1) next.add(rows[i].index);
          return next;
        }
        if (next.has(index)) next.delete(index);
        else next.add(index);
        return next;
      });
      if (!shiftKey) anchor.current = index;
    },
    [rows],
  );

  const clearSelection = useCallback(() => {
    setSelection(new Set());
    anchor.current = null;
  }, []);

  /** Everything the list has actually READ, which is not the same as everything
   *  the filter matched — the pages beyond are takes nobody has looked at. */
  const selectAll = useCallback(() => {
    setSelection(new Set(rows.map((r) => r.index)));
  }, [rows]);

  /* ── writes ────────────────────────────────────────────────────────────── */

  const failed = useCallback((e: unknown, what: string) => {
    toast.error(isForbidden(e) ? REMOTE_REFUSED : `${what}: ${reason(e)}`);
  }, []);

  /** Patch one episode in BOTH the list window and the full episode list, so a
   *  mark and the counts it changes move together. */
  const patch = useCallback((index: number, p: Partial<LabEpisode>) => {
    setRows((rs) => rs.map((r) => (r.index === index ? { ...r, ...p } : r)));
    setDetail((d) =>
      d === null
        ? d
        : { ...d, episodes: d.episodes.map((e) => (e.index === index ? { ...e, ...p } : e)) },
    );
  }, []);

  /**
   * Mark one episode, on screen first.
   *
   * OPTIMISTIC on purpose. A triage pass is 70 episodes at a keystroke each; a
   * round-trip per keypress makes the pass unusable, and worse, it makes the
   * operator wait to find out whether the key registered. The write is rolled
   * back to the exact previous mark if the backend refuses, and the `epoch`
   * bump reconciles with the server a beat later.
   */
  const markOne = useCallback(
    async (index: number, mark: Mark, note?: string) => {
      if (!repoId) return;
      const before = rowByIndex.get(index) ?? episodes.find((e) => e.index === index);
      patch(index, note === undefined ? { mark } : { mark, note });
      try {
        await lab.mark(repoId, index, mark, note);
        bump();
      } catch (e) {
        if (before) patch(index, { mark: before.mark, note: before.note });
        failed(e, `Ep ${before?.label ?? index + 1}`);
      }
    },
    [repoId, rowByIndex, episodes, patch, bump, failed],
  );

  const onMark = useCallback(
    (index: number, mark: Mark) => { void markOne(index, mark); },
    [markOne],
  );

  /** A note is written with the mark that is already on the take — `lab.mark`
   *  takes both, and sending a note without one would clear the decision. */
  const onNote = useCallback(
    (index: number, note: string) => {
      const cur = rowByIndex.get(index) ?? episodes.find((e) => e.index === index);
      void markOne(index, cur?.mark ?? "unset", note);
    },
    [rowByIndex, episodes, markOne],
  );

  /** Bulk is server-authoritative, unlike a single mark: it is one request for
   *  many rows, nobody is holding a key down, and a partial optimistic patch
   *  over a refusal would leave the list claiming decisions that were never
   *  stored. */
  const runBulk = useCallback(
    async (body: { status?: Mark; tags_add?: string[]; tags_remove?: string[] }) => {
      const eps = [...selection];
      if (!repoId || eps.length === 0) return;
      setBulkBusy(true);
      try {
        const r = await lab.bulk({ repo_id: repoId, episodes: eps, ...body });
        toast.success(`${r.updated} episode${r.updated === 1 ? "" : "s"} updated`);
        bump();
      } catch (e) {
        failed(e, "bulk edit");
      } finally {
        setBulkBusy(false);
      }
    },
    [repoId, selection, bump, failed],
  );

  /* ── the playhead ──────────────────────────────────────────────────────
     One animation frame, not one `timeupdate`: the element fires ~30 times a
     second and each one would re-render two charts. */
  const frame = useRef(0);
  const pending = useRef(0);
  const onTime = useCallback((t: number) => {
    pending.current = t;
    if (typeof requestAnimationFrame === "undefined") {
      setPlayheadT(t);
      return;
    }
    if (frame.current !== 0) return;
    frame.current = requestAnimationFrame(() => {
      frame.current = 0;
      setPlayheadT(pending.current);
    });
  }, []);
  useEffect(() => () => {
    if (frame.current !== 0) cancelAnimationFrame(frame.current);
  }, []);

  /* ── keyboard ──────────────────────────────────────────────────────────── */

  const step = useCallback(
    (delta: number) => {
      if (rows.length === 0) return;
      const at = selectedIndex === null ? -1 : rows.findIndex((r) => r.index === selectedIndex);
      const next =
        at < 0
          ? (delta > 0 ? 0 : rows.length - 1)
          : Math.min(rows.length - 1, Math.max(0, at + delta));
      select(rows[next].index);
    },
    [rows, selectedIndex, select],
  );

  const toggleMark = useCallback(
    (mark: Mark) => {
      if (selected === null) return;
      // Pressing the mark that is already on returns the take to `unset` —
      // "I have not judged this" is a state worth being able to get back to.
      void markOne(selected.index, selected.mark === mark ? "unset" : mark);
    },
    [selected, markOne],
  );

  useEffect(() => {
    if (repoId === null) return;
    // Keyed off the SAME table the hint row renders. `Record<BindingId, …>` is
    // what makes that a promise: a row added to `BINDINGS` with no case here
    // fails to compile instead of becoming a key that does nothing.
    const actions: Record<BindingId, (key: string) => void> = {
      play: () => playerRef.current?.togglePlay(),
      step: (k) => step(k === "j" ? -1 : 1),
      keep: () => toggleMark("keep"),
      reject: () => toggleMark("reject"),
      rate: (k) => playerRef.current?.stepRate(k === "[" ? -1 : 1),
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      // A dialog owns the keyboard while it is open, and the note box is a text
      // field — without this guard, typing a note marks episodes.
      if (dialog !== null || isEditableTarget(e.target)) return;
      const key = e.key === " " ? "space" : e.key.toLowerCase();
      const b = BINDINGS.find((x) => x.keys.includes(key));
      if (!b) return;
      // Space scrolls the page otherwise, and the page must never scroll.
      e.preventDefault();
      actions[b.id](key);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [dialog, repoId, step, toggleMark]);

  /* ── the dataset picker ────────────────────────────────────────────────── */

  if (repoId === null) {
    return (
      <div className="grid min-h-0 grid-rows-[auto_minmax(0,1fr)] gap-2 overflow-hidden p-2">
        <Note className="shrink-0 px-0.5">
          Pick a corpus to review. Marking a take keep or reject is what decides
          which demonstrations a policy learns from — nothing downstream reads
          anything else.
        </Note>
        <DatasetShelf layout="grid" onOpen={onPickDataset} refreshKey={epoch} />
      </div>
    );
  }

  if (noLab) {
    return (
      <div className="grid min-h-0 overflow-hidden p-2">
        <Panel>
          <PanelHead title="review" right={repoId} />
          <Empty>this backend has no lab — review needs the /lab routes</Empty>
        </Panel>
      </div>
    );
  }

  const task = detail?.episodes.find((e) => e.task)?.task ?? summary?.task ?? null;
  const stale = summary?.stale === true;
  const videoKeys = detail?.video_keys ?? [];

  return (
    <div className="grid min-h-0 grid-rows-[auto_auto_minmax(0,1fr)] overflow-hidden">
      {/* ---- header strip ------------------------------------------------ */}
      <div className="flex h-9 shrink-0 items-center gap-2 overflow-hidden border-b border-border bg-[var(--haller-chrome)] px-2.5">
        <span
          className="max-w-[16rem] shrink truncate font-mono text-[11px] font-semibold"
          title={repoId}
        >
          {repoId}
        </span>
        {detail && (
          <Chip tabIndex={-1} className="pointer-events-none">
            {rigLabel(detail.rig)}
          </Chip>
        )}
        <span
          className="min-w-0 flex-1 truncate text-[10px] text-muted-foreground"
          title={task ?? undefined}
        >
          {task ?? "no task recorded"}
        </span>

        {stale && (
          <Chip
            on
            colour="var(--haller-warn)"
            tabIndex={-1}
            className="pointer-events-none"
            title={STALE_NOTE}
          >
            stale marks
          </Chip>
        )}

        <MarkBar marks={counts} className="w-14 shrink-0" />
        <span
          data-num
          className="shrink-0 font-mono text-[10px] whitespace-nowrap tabular-nums text-muted-foreground"
        >
          <span style={{ color: MARK_COLOR.keep }}>{counts.keep}</span> kept ·{" "}
          <span style={{ color: MARK_COLOR.reject }}>{counts.reject}</span> rejected ·{" "}
          {counts.unset} unmarked
        </span>

        <Button onClick={() => onPickDataset(null)} title="back to the shelf">
          change dataset
        </Button>
        <Button
          disabled={!detail}
          onClick={() => setDialog("autoclass")}
          title="propose marks from the traces — nothing is written until you read the diff"
        >
          autoclassify
        </Button>
        <Button
          tone="danger"
          disabled={!detail || counts.reject === 0}
          onClick={() => setDialog("prune")}
          title="permanently remove the takes marked reject"
        >
          prune rejected
        </Button>
        <Button
          tone="danger"
          onClick={() => setDialog("delete")}
          title="destroy this dataset — there is no undo"
        >
          delete dataset
        </Button>
        <Button
          tone="primary"
          onClick={() =>
            toast.message(
              counts.keep === 0
                ? "nothing is marked keep yet — a run needs a kept set"
                : `${repoId} stays selected — open the train tab to launch on ${counts.keep} kept`,
            )
          }
          title="the train tab launches on the kept set of the selected dataset"
        >
          train on kept →
        </Button>
      </div>

      {detailError && (
        <div className="shrink-0 px-2 pt-2">
          {/* The backend's own sentence, quoted. A 409 names the operator's
              next move and is not a fault. */}
          <Refusal tone={detailError.tone}>{detailError.text}</Refusal>
        </div>
      )}

      {/* ---- the instrument, and the queue -------------------------------- */}
      <div className="grid min-h-0 grid-cols-[minmax(0,1fr)_27rem] gap-2 overflow-hidden p-2">
        {/* LEFT: one episode, three views of it, sharing a playhead. */}
        <div className="grid min-h-0 grid-rows-[minmax(0,1fr)_auto_auto] gap-2 overflow-hidden">
          <EpisodePlayer
            ref={playerRef}
            repoId={repoId}
            episode={selected}
            videoKeys={videoKeys}
            videoKey={videoKey}
            onVideoKey={setVideoKey}
            // The dataset's own rate. 0 until `detail` lands, which is the
            // player's "assume 30" — the clamp is one frame period wide and
            // there is nothing else here to derive it from.
            fps={detail?.fps ?? 0}
            onTime={onTime}
          />

          <div className="flex min-h-0 shrink-0 flex-col gap-2">
            {traceError && <Refusal>{traceError}</Refusal>}
            <GripperChart
              trace={trace}
              playheadT={playheadT}
              guides={gripperGuides(selected)}
            />
          </div>

          <div className="flex min-h-0 flex-col gap-1.5 overflow-hidden">
            <TraceChart
              trace={trace}
              playheadT={playheadT}
              overlay={overlay}
              onOverlay={setOverlay}
            />
            <div className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-1 px-0.5">
              {BINDINGS.map((b) => (
                <span key={b.label} className="inline-flex items-center gap-1">
                  {b.hint.map((h, i) => (
                    <Fragment key={h}>
                      {i > 0 && (
                        <span className="font-mono text-[9px] text-muted-foreground opacity-50">
                          /
                        </span>
                      )}
                      <kbd className="inline-flex h-4 min-w-4 items-center justify-center rounded-[3px] border border-border bg-secondary px-1 font-mono text-[9px] text-muted-foreground">
                        {h}
                      </kbd>
                    </Fragment>
                  ))}
                  <span className="label-micro pl-0.5 text-muted-foreground opacity-70">
                    {b.label}
                  </span>
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* RIGHT: what the server matched, in the order the server matched it. */}
        <div className="grid min-h-0 grid-rows-[auto_minmax(0,1fr)_auto] gap-2 overflow-hidden">
          <Panel className="shrink-0">
            <PanelHead
              title="filters"
              right={`${total} match${total === 1 ? "" : "es"}`}
            >
              <Button
                tone="ghost"
                className="ml-auto"
                onClick={() => {
                  setFilters(DEFAULT_FILTERS);
                  setLimit(PAGE);
                }}
                title="clear every filter"
              >
                clear
              </Button>
            </PanelHead>
            <EpisodeFilters
              value={filters}
              onChange={(v) => {
                // A filter change is a different result set, so the window
                // starts again at the first page.
                setFilters(v);
                setLimit(PAGE);
              }}
              tags={tagVocab}
              counts={counts}
            />
          </Panel>

          <EpisodeList
            episodes={rows}
            total={total}
            loading={listLoading}
            error={listError}
            selectedIndex={selectedIndex}
            selection={selection}
            evalSet={evalSet}
            onSelect={select}
            onToggleSelect={onToggleSelect}
            onMark={onMark}
            onNote={onNote}
            onLoadMore={() => setLimit((l) => l + PAGE)}
          />

          {/* The card only exists while the selection does. `BulkBar` already
              renders nothing at zero, but an empty Panel still draws its 1px
              ring — a hairline under the list that means nothing. */}
          {selection.size > 0 && (
            <Panel className="shrink-0">
              <BulkBar
                count={selection.size}
                busy={bulkBusy}
                knownTags={tagVocab}
                onMark={(m) => void runBulk({ status: m })}
                onTag={(t) => void runBulk({ tags_add: [t] })}
                onUntag={(t) => void runBulk({ tags_remove: [t] })}
                onClear={clearSelection}
                onSelectAll={selectAll}
              />
            </Panel>
          )}
        </div>
      </div>

      {/* ---- dialogs ----------------------------------------------------- */}
      {dialog === "autoclass" && (
        <AutoclassifyDialog
          repoId={repoId}
          onClose={() => setDialog(null)}
          onApplied={bump}
        />
      )}
      {dialog === "prune" && (
        <PruneDialog
          repoId={repoId}
          episodes={episodes}
          onClose={() => setDialog(null)}
          onPruned={() => {
            // The survivors are renumbered by a detached job, so every index
            // the selection holds is about to name a different take.
            clearSelection();
            setSelectedIndex(null);
            bump();
          }}
        />
      )}
      {dialog === "delete" && (
        <DeleteDatasetDialog
          repoId={repoId}
          episodes={episodes.length}
          durationS={episodes.reduce((s, e) => s + (e.duration_s || 0), 0)}
          // The one number `detail` does not carry. NaN reads as "—" rather
          // than as a confident 0 B.
          sizeBytes={summary?.size_bytes ?? NaN}
          onClose={() => setDialog(null)}
          onDeleted={() => {
            bump();
            onPickDataset(null);
          }}
        />
      )}
    </div>
  );
}
