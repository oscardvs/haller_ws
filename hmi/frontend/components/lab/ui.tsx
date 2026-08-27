"use client";

/**
 * The Lab's shared component vocabulary.
 *
 * Every surface under `components/lab/` is built from these, so the workspace
 * reads as part of the cockpit rather than as a page that was pasted into it.
 * Nothing here invents a colour, a radius or a type scale: the panel chrome is
 * the same `rounded-lg bg-card shadow-[0_0_0_1px_var(--border)]` the Dataset
 * tab already uses, the segmented control is the Teleop tab's stance picker,
 * and every label is `.label-micro` or `.label-tracked` from globals.css.
 *
 * The house register for copy is lowercase, terse and factual — "no episodes
 * yet", not "No Episodes Found!". Sentence case is for prose that explains a
 * consequence; labels never get it.
 */
import { useCallback, useEffect, useRef } from "react";

import type { Mark, Verdict } from "@/lib/lab";

/* ─── surfaces ────────────────────────────────────────────────────────── */

/** A card. `min-h-0` and `overflow-hidden` are on by default because the
 *  cockpit is a fixed-viewport grid and a panel that grows past its row is how
 *  the whole shell starts scrolling. */
export function Panel({
  children,
  className = "",
  ...rest
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={
        "flex min-h-0 flex-col overflow-hidden rounded-lg bg-card " +
        "shadow-[0_0_0_1px_var(--border)] " + className
      }
      {...rest}
    >
      {children}
    </div>
  );
}

/** The 34px strip every panel wears: name on the left, a reading on the right. */
export function PanelHead({
  title,
  right,
  children,
  className = "",
}: {
  title: string;
  /** A terse reading — counts, sizes, a status. Mono, muted, never wraps. */
  right?: React.ReactNode;
  /** Controls that belong in the head rather than the body. */
  children?: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={
        "flex h-8.5 shrink-0 items-center justify-between gap-2 border-b " +
        "border-border px-3 " + className
      }
    >
      <span className="label-tracked shrink-0 text-muted-foreground">{title}</span>
      {children}
      {right !== undefined && (
        <span className="min-w-0 truncate text-right font-mono text-[10px] whitespace-nowrap text-muted-foreground">
          {right}
        </span>
      )}
    </div>
  );
}

/** Dead space that says what its emptiness means. The scanlines are the same
 *  texture a camera bay with no feed wears, so "nothing here" looks the same
 *  everywhere in the cockpit. */
export function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="relative flex min-h-[80px] flex-1 items-center justify-center p-4">
      <span className="scanlines absolute inset-0" aria-hidden />
      <span className="relative max-w-[46ch] text-center font-mono text-[10px] tracking-[0.14em] uppercase text-muted-foreground opacity-70">
        {children}
      </span>
    </div>
  );
}

/** The backend refused, or cannot. Distinct from `Empty`: this is a sentence
 *  the robot said, and it is quoted rather than paraphrased. */
export function Refusal({
  children,
  tone = "warn",
}: {
  children: React.ReactNode;
  tone?: "warn" | "fault";
}) {
  const colour = tone === "fault" ? "var(--haller-fault)" : "var(--haller-warn)";
  return (
    <div
      className="rounded-md border px-2.5 py-2 font-mono text-[10px] text-pretty"
      style={{ borderColor: colour, color: colour }}
    >
      {children}
    </div>
  );
}

/* ─── navigation ──────────────────────────────────────────────────────── */

/** The Collect · Review · Train sub-nav. Same shape as the cockpit header's
 *  tab strip one level down, so the hierarchy is legible without a breadcrumb. */
export function SubNav<T extends string>({
  items,
  value,
  onChange,
  label,
  children,
}: {
  items: readonly { id: T; label: string; hint?: string }[];
  value: T;
  onChange: (id: T) => void;
  label: string;
  /** Trailing content — a dataset name, a count, a deep link. */
  children?: React.ReactNode;
}) {
  return (
    <div className="flex h-8 shrink-0 items-center gap-2 border-b border-border bg-[var(--haller-chrome)] px-2.5">
      <nav
        aria-label={label}
        className="flex shrink-0 items-center gap-0.5 rounded-md bg-muted p-0.5"
      >
        {items.map((it) => {
          const active = it.id === value;
          return (
            <button
              key={it.id}
              type="button"
              onClick={() => onChange(it.id)}
              aria-current={active ? "page" : undefined}
              title={it.hint}
              className={
                "inline-flex h-[22px] items-center gap-1.5 rounded-sm px-2.5 " +
                "font-mono text-[10px] font-semibold tracking-[0.12em] uppercase " +
                "whitespace-nowrap transition-colors " +
                (active
                  ? "bg-[var(--haller-live-soft)] text-[var(--haller-live)]"
                  : "text-muted-foreground hover:text-foreground")
              }
            >
              <span
                aria-hidden
                className="h-1 w-1 rounded-[1px]"
                style={{
                  backgroundColor: active ? "var(--haller-live)" : "var(--haller-rail)",
                }}
              />
              {it.label}
            </button>
          );
        })}
      </nav>
      {children}
    </div>
  );
}

/** A radiogroup that reads as one control. The Teleop tab's stance picker,
 *  generalised — same `bg-muted p-1` well, same `bg-card` active face. */
export function Segmented<T extends string | null>({
  options,
  value,
  onChange,
  label,
  disabled = false,
  className = "",
}: {
  options: readonly { value: T; label: string; hint?: string }[];
  value: T;
  onChange: (v: T) => void;
  label: string;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={"flex items-center gap-1 rounded-md bg-muted p-1 " + className}
    >
      {options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          disabled={disabled}
          title={o.hint}
          onClick={() => onChange(o.value)}
          className={
            "h-6 flex-1 rounded-sm px-2 label-micro whitespace-nowrap disabled:opacity-55 " +
            (o.value === value
              ? "bg-card text-foreground"
              : "text-muted-foreground hover:text-foreground")
          }
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/* ─── controls ────────────────────────────────────────────────────────── */

const INPUT =
  "h-7.5 w-full min-w-0 rounded-md border border-input bg-background px-2.5 " +
  "font-mono text-[11px] disabled:opacity-50";

/** Label above, control below, optional hint under it. The hint is where a
 *  unit or a consequence goes — never a restatement of the label. */
export function Field({
  label,
  hint,
  children,
  className = "",
}: {
  label: string;
  hint?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <label className={"flex min-w-0 flex-col gap-1.5 " + className}>
      <span className="label-tracked text-muted-foreground">{label}</span>
      {children}
      {hint !== undefined && (
        <span className="text-[10px] text-pretty text-muted-foreground">{hint}</span>
      )}
    </label>
  );
}

export function TextInput(props: React.InputHTMLAttributes<HTMLInputElement>) {
  const { className = "", ...rest } = props;
  return <input type="text" className={INPUT + " " + className} {...rest} />;
}

/** A number input that reports numbers.
 *
 *  `onChange` fires with the parsed value; an unparseable box reports the
 *  fallback rather than NaN, because NaN in a spec is a training run that
 *  fails a minute after launch instead of a field that refuses now. */
export function NumberInput({
  value,
  onChange,
  fallback = 0,
  className = "",
  ...rest
}: Omit<React.InputHTMLAttributes<HTMLInputElement>, "onChange" | "value"> & {
  value: number;
  onChange: (v: number) => void;
  fallback?: number;
}) {
  return (
    <input
      type="number"
      value={Number.isFinite(value) ? value : ""}
      onChange={(e) => {
        const n = Number(e.target.value);
        onChange(Number.isFinite(n) ? n : fallback);
      }}
      className={INPUT + " tabular-nums " + className}
      data-num
      {...rest}
    />
  );
}

export function Select({
  className = "",
  children,
  ...rest
}: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select className={INPUT + " " + className} {...rest}>
      {children}
    </select>
  );
}

export type ButtonTone = "default" | "primary" | "danger" | "ghost";

const TONE: Record<ButtonTone, string> = {
  default: "border border-border bg-secondary text-foreground hover:bg-muted",
  primary: "bg-primary text-primary-foreground hover:opacity-90",
  danger:
    "border border-[var(--haller-fault)] bg-[oklch(0.62_0.245_27/0.2)] text-[var(--haller-fault)]",
  ghost: "border border-transparent text-muted-foreground hover:text-foreground",
};

/** The cockpit's button: 28px, micro label, tracked. Sizes come from the row
 *  it sits in, not from a variant zoo. */
export function Button({
  tone = "default",
  className = "",
  children,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { tone?: ButtonTone }) {
  return (
    <button
      type="button"
      className={
        "inline-flex h-7 shrink-0 items-center justify-center gap-1.5 rounded-md px-2.5 " +
        "label-micro tracking-[0.12em] transition-colors disabled:opacity-50 " +
        TONE[tone] + " " + className
      }
      {...rest}
    >
      {children}
    </button>
  );
}

/** A filter chip. `on` is the selected state; chips are a set, not a radio. */
export function Chip({
  on = false,
  count,
  colour,
  className = "",
  children,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  on?: boolean;
  count?: number;
  /** Overrides the active colour — used to make the keep/reject chips carry
   *  their own semantics rather than all reading as "selected". */
  colour?: string;
}) {
  const active = colour ?? "var(--haller-live)";
  return (
    <button
      type="button"
      aria-pressed={on}
      className={
        "inline-flex h-5.5 shrink-0 items-center gap-1.5 rounded-full border px-2 " +
        "label-micro transition-colors disabled:opacity-50 " +
        (on ? "" : "border-border text-muted-foreground hover:text-foreground ") +
        className
      }
      style={on ? { borderColor: active, color: active } : undefined}
      {...rest}
    >
      {children}
      {count !== undefined && (
        // The space is load-bearing, not spacing: without it the accessible
        // name concatenates to "reject4" and a screen reader reads one token,
        // so the chip cannot be asked for by its word either.
        <span data-num className="tabular-nums opacity-70">{" "}{count}</span>
      )}
    </button>
  );
}

/* ─── dataset vocabulary ──────────────────────────────────────────────── */

export const MARK_COLOR: Record<Mark, string> = {
  keep: "var(--haller-live)",
  reject: "var(--haller-fault)",
  unset: "var(--haller-rail)",
};

export const VERDICT_COLOR: Record<Verdict, string> = {
  PASS: "var(--haller-live)",
  SUSPECT: "var(--haller-warn)",
  FAIL: "var(--haller-fault)",
};

/** The grader's opinion, as a pill. Deliberately quieter than a mark: it is
 *  a reading, and the operator's decision has to out-rank it visually. */
export function VerdictTag({ verdict }: { verdict: Verdict | null }) {
  if (!verdict) return null;
  const colour = VERDICT_COLOR[verdict];
  return (
    <span
      className="inline-flex h-4 shrink-0 items-center rounded-[3px] px-1 label-micro"
      style={{ color: colour, background: "color-mix(in oklch, " + colour + " 16%, transparent)" }}
    >
      {verdict}
    </span>
  );
}

/** A small square that carries the mark without words — for dense rows where
 *  a pill per episode would be all you could see. */
export function MarkDot({ mark }: { mark: Mark }) {
  return (
    <span
      aria-hidden
      className="h-1.5 w-1.5 shrink-0 rounded-[1px]"
      style={{ backgroundColor: MARK_COLOR[mark] }}
    />
  );
}

/** keep / reject / unset as one stacked bar. Reads as a progress meter for a
 *  campaign: how much of this dataset has been judged at all. */
export function MarkBar({
  marks,
  className = "",
}: {
  marks: { keep: number; reject: number; unset: number };
  className?: string;
}) {
  const total = marks.keep + marks.reject + marks.unset;
  return (
    <div
      className={"flex h-1 overflow-hidden rounded-[2px] bg-muted " + className}
      role="img"
      aria-label={`${marks.keep} keep, ${marks.reject} reject, ${marks.unset} unmarked`}
      title={`${marks.keep} keep · ${marks.reject} reject · ${marks.unset} unmarked`}
    >
      {(["keep", "unset", "reject"] as const).map((k) =>
        marks[k] > 0 ? (
          <span
            key={k}
            style={{ flex: marks[k], backgroundColor: MARK_COLOR[k] }}
          />
        ) : null,
      )}
      {total === 0 && <span className="flex-1" />}
    </div>
  );
}

/* ─── dialogs ─────────────────────────────────────────────────────────── */

/**
 * A modal over the whole viewport.
 *
 * `position: fixed` rather than absolute-in-shell: the cockpit grid is
 * `height:100vh; overflow:hidden`, and a destructive confirmation is the one
 * thing that should not be clipped by whichever panel happened to open it.
 *
 * Escape closes and the backdrop closes, but neither is the confirm path —
 * every destructive dialog here still requires its own explicit act.
 */
export function Dialog({
  title,
  onClose,
  children,
  footer,
  wide = false,
}: {
  title: string;
  onClose: () => void;
  children: React.ReactNode;
  footer?: React.ReactNode;
  wide?: boolean;
}) {
  const cardRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.stopPropagation(); onClose(); }
    };
    document.addEventListener("keydown", onKey, true);
    return () => document.removeEventListener("keydown", onKey, true);
  }, [onClose]);

  // Focus lands inside the card, so the keyboard triage shortcuts behind it
  // (space, J, L, K, R) cannot reach the list while a dialog is open.
  useEffect(() => {
    const first = cardRef.current?.querySelector<HTMLElement>(
      "input, select, textarea, button",
    );
    first?.focus();
  }, []);

  const onBackdrop = useCallback(
    (e: React.MouseEvent) => { if (e.target === e.currentTarget) onClose(); },
    [onClose],
  );

  return (
    <div
      className="fixed inset-0 z-[80] flex items-center justify-center bg-[oklch(0_0_0/0.6)] p-4"
      onMouseDown={onBackdrop}
    >
      <div
        ref={cardRef}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className={
          "flex max-h-[86vh] w-full flex-col overflow-hidden rounded-xl border " +
          "border-border bg-popover shadow-[0_24px_60px_oklch(0_0_0/0.5)] " +
          (wide ? "max-w-[52rem]" : "max-w-[34rem]")
        }
      >
        <div className="flex h-9 shrink-0 items-center justify-between gap-2 border-b border-border px-3.5">
          <span className="label-tracked text-muted-foreground">{title}</span>
          <button
            type="button"
            onClick={onClose}
            aria-label={`close ${title}`}
            className="rounded-sm px-1 text-[14px] leading-none text-muted-foreground hover:text-foreground"
          >
            ×
          </button>
        </div>
        <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-3.5">
          {children}
        </div>
        {footer !== undefined && (
          <div className="flex shrink-0 items-center justify-end gap-2 border-t border-border px-3.5 py-2.5">
            {footer}
          </div>
        )}
      </div>
    </div>
  );
}

/** The consequence box inside a destructive dialog. Loud on purpose: this is
 *  the sentence that has to be read, and it names what is lost. */
export function WarnBox({
  tone = "warn",
  children,
}: {
  tone?: "warn" | "fault";
  children: React.ReactNode;
}) {
  const colour = tone === "fault" ? "var(--haller-fault)" : "var(--haller-warn)";
  return (
    <div
      className="rounded-md border px-3 py-2.5 text-[11px] text-pretty"
      style={{
        borderColor: colour,
        color: colour,
        background: "color-mix(in oklch, " + colour + " 10%, transparent)",
      }}
    >
      {children}
    </div>
  );
}

/** Prose inside a panel or dialog: 11px, pretty-wrapped, muted. */
export function Note({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <p className={"text-[11px] text-pretty text-muted-foreground " + className}>
      {children}
    </p>
  );
}

/** A `label · value` pair for a stats row. Digits are tabular so a column of
 *  them lines up. */
export function Stat({
  label,
  value,
  colour,
}: {
  label: string;
  value: React.ReactNode;
  colour?: string;
}) {
  return (
    <span className="inline-flex items-baseline gap-1.5 whitespace-nowrap">
      <span data-num className="font-mono text-[11px] tabular-nums" style={colour ? { color: colour } : undefined}>
        {value}
      </span>
      <span className="label-micro text-muted-foreground">{label}</span>
    </span>
  );
}

/** Column headers for a dense table. One place so every list in the Lab has
 *  the same header weight and tracking. */
export function HeadRow({
  cols,
  className = "",
  style,
}: {
  cols: { key: string; label: string; align?: "left" | "right" }[];
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <div
      className={
        "sticky top-0 z-10 grid gap-2 border-b border-border bg-muted px-2.5 py-1 " +
        "label-micro text-muted-foreground " + className
      }
      style={style}
    >
      {cols.map((c) => (
        <span key={c.key} className={c.align === "right" ? "text-right" : undefined}>
          {c.label}
        </span>
      ))}
    </div>
  );
}
