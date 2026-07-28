"use client";

/**
 * The cockpit's one popover surface.
 *
 * Escape closes, click-outside closes, and focus returns to the trigger — the
 * last one matters because an operator who opened the Teleop popover with the
 * keyboard and pressed Escape must not be dumped back at document start with
 * the E-STOP six tab-stops away.
 *
 * Deliberately not a shadcn Popover: those portal into <body>, and the shell is
 * a `height:100vh; overflow:hidden` grid whose overlays are positioned against
 * it. Rendering in place keeps the design's `left:8px; bottom:42px` honest.
 */
import { useEffect, useRef, type RefObject } from "react";

export function Popover({
  onClose,
  triggerRef,
  label,
  className = "",
  children,
}: {
  onClose: () => void;
  /** Focus goes back here on close. */
  triggerRef: RefObject<HTMLButtonElement | null>;
  label: string;
  className?: string;
  children: React.ReactNode;
}) {
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
      }
    };
    // pointerdown, not click: a click that starts inside the panel and ends
    // outside (a drag on the rate field, a text selection) must not close it.
    const onPointerDown = (e: PointerEvent) => {
      const t = e.target as Node | null;
      if (!t) return;
      if (panelRef.current?.contains(t)) return;
      if (triggerRef.current?.contains(t)) return; // the trigger toggles itself
      onClose();
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onPointerDown);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onPointerDown);
    };
  }, [onClose, triggerRef]);

  // Return focus on unmount, but only if it is still inside the panel we are
  // tearing down. If the operator has already clicked elsewhere, yanking focus
  // back to the trigger would be the rude kind of "helpful".
  useEffect(() => {
    const panel = panelRef.current;
    const trigger = triggerRef.current;
    return () => {
      if (panel && panel.contains(document.activeElement)) trigger?.focus();
    };
  }, [triggerRef]);

  return (
    <div
      ref={panelRef}
      role="dialog"
      aria-label={label}
      className={
        "absolute z-60 rounded-lg border border-border bg-popover p-3 shadow-[0_18px_40px_oklch(0_0_0/0.45)] " +
        className
      }
    >
      {children}
    </div>
  );
}

/** Header row every popover shares: title on the left, close on the right. */
export function PopoverHeader({
  title,
  onClose,
}: {
  title: string;
  onClose: () => void;
}) {
  return (
    <div className="mb-3 flex items-center justify-between gap-2">
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
  );
}
