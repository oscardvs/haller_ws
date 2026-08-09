'use client';

import { useCallback, useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useTheme } from 'next-themes';
import { Expand, Minus, Plus, RotateCcw, X } from 'lucide-react';

let mermaidLoader: Promise<typeof import('mermaid').default> | null = null;

function loadMermaid() {
  if (!mermaidLoader) {
    mermaidLoader = import('mermaid').then((m) => m.default);
  }
  return mermaidLoader;
}

/* Mermaid 11 (via khroma) can't parse CSS Color-4 syntax like oklch() or lab(),
   which is what the browser resolves our --color-fd-* vars into. So we mirror
   the palette here as plain hex and switch by theme. Keep these in sync with
   `app/global.css` if the theme changes. */
const PALETTE = {
  light: {
    bg:        '#f5f1e8',
    fg:        '#1c1812',
    muted:     '#ede7d9',
    mutedFg:   '#5e544a',
    card:      '#f0eadc',
    border:    '#d0c8b8',
    primary:   '#1c6845', /* Bambu PLA dark green — brand colour */
    primaryFg: '#fbf7ed',
  },
  dark: {
    bg:        '#161311',
    fg:        '#e7e1d4',
    muted:     '#1f1c18',
    mutedFg:   '#a99e88',
    card:      '#1c1815',
    border:    '#3a3127',
    primary:   '#3fb472', /* Bambu green lifted for dark surface */
    primaryFg: '#101810',
  },
} as const;

const MIN_SCALE = 0.25;
const MAX_SCALE = 12;

function clamp(v: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, v));
}

/** Natural size of a rendered mermaid SVG, read off its viewBox. */
function naturalSize(svg: string): { w: number; h: number } {
  const m = /viewBox="([\d.\-+eE]+) ([\d.\-+eE]+) ([\d.\-+eE]+) ([\d.\-+eE]+)"/.exec(svg);
  const w = m ? Number(m[3]) : 0;
  const h = m ? Number(m[4]) : 0;
  return w > 0 && h > 0 ? { w, h } : { w: 800, h: 600 };
}

/* Full-screen viewer: wheel to zoom about the cursor, drag to pan, Esc to
   close. The diagram is laid out at its aspect-correct fit size and everything
   on top of that is a CSS transform, so text stays vector-sharp at any zoom. */
function MermaidLightbox({
  svg,
  title,
  onClose,
}: {
  svg: string;
  title: string;
  onClose: () => void;
}) {
  const nat = naturalSize(svg);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [view, setView] = useState({ s: 1, x: 0, y: 0 });
  const [fit, setFit] = useState<{ w: number; h: number } | null>(null);
  const drag = useRef<{ px: number; py: number; x: number; y: number } | null>(null);

  const reset = useCallback(() => setView({ s: 1, x: 0, y: 0 }), []);

  /* Fit-to-viewport base size. `s = 1` means "as large as fits", so a diagram
     that the article column squeezed down is already bigger the moment it
     opens. */
  useEffect(() => {
    const measure = () => {
      const availW = window.innerWidth - 64;
      const availH = window.innerHeight - 140;
      const k = Math.min(availW / nat.w, availH / nat.h);
      setFit({ w: nat.w * k, h: nat.h * k });
    };
    measure();
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, [nat.w, nat.h]);

  /* Lock the page behind the overlay, and take Esc. */
  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      if (e.key === '0') reset();
      if (e.key === '+' || e.key === '=') setView((v) => ({ ...v, s: clamp(v.s * 1.3, MIN_SCALE, MAX_SCALE) }));
      if (e.key === '-') setView((v) => ({ ...v, s: clamp(v.s / 1.3, MIN_SCALE, MAX_SCALE) }));
    };
    window.addEventListener('keydown', onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener('keydown', onKey);
    };
  }, [onClose, reset]);

  /* Wheel zoom has to be a non-passive listener to keep the page from
     scrolling underneath, which rules out React's onWheel. */
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const px = e.clientX - (r.left + r.width / 2);
      const py = e.clientY - (r.top + r.height / 2);
      setView((v) => {
        const s = clamp(v.s * Math.exp(-e.deltaY * 0.0018), MIN_SCALE, MAX_SCALE);
        /* keep the point under the cursor pinned */
        const k = s / v.s;
        return { s, x: px - (px - v.x) * k, y: py - (py - v.y) * k };
      });
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { px: e.clientX, py: e.clientY, x: view.x, y: view.y };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    setView((v) => ({ ...v, x: d.x + (e.clientX - d.px), y: d.y + (e.clientY - d.py) }));
  };
  const onPointerUp = () => {
    drag.current = null;
  };

  const zoom = (k: number) =>
    setView((v) => ({ ...v, s: clamp(v.s * k, MIN_SCALE, MAX_SCALE) }));

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col bg-fd-background/97 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-label={title}
    >
      {/* toolbar */}
      <div className="flex items-center justify-between gap-4 border-b border-fd-border px-4 py-2">
        <span className="h-label truncate">{title}</span>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => zoom(1 / 1.3)}
            aria-label="Zoom out"
            className="rounded-md p-1.5 text-fd-muted-foreground hover:bg-fd-accent hover:text-fd-accent-foreground"
          >
            <Minus className="size-4" />
          </button>
          <span className="h-mono w-14 text-center text-xs tabular-nums text-fd-muted-foreground">
            {Math.round(view.s * 100)}%
          </span>
          <button
            type="button"
            onClick={() => zoom(1.3)}
            aria-label="Zoom in"
            className="rounded-md p-1.5 text-fd-muted-foreground hover:bg-fd-accent hover:text-fd-accent-foreground"
          >
            <Plus className="size-4" />
          </button>
          <button
            type="button"
            onClick={reset}
            aria-label="Reset zoom"
            className="rounded-md p-1.5 text-fd-muted-foreground hover:bg-fd-accent hover:text-fd-accent-foreground"
          >
            <RotateCcw className="size-4" />
          </button>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="ml-1 rounded-md p-1.5 text-fd-muted-foreground hover:bg-fd-accent hover:text-fd-accent-foreground"
          >
            <X className="size-4" />
          </button>
        </div>
      </div>

      {/* stage */}
      <div
        ref={wrapRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onDoubleClick={reset}
        className="relative flex flex-1 items-center justify-center overflow-hidden touch-none select-none cursor-grab active:cursor-grabbing"
      >
        {fit && (
          <div
            style={{
              width: fit.w,
              height: fit.h,
              transform: `translate(${view.x}px, ${view.y}px) scale(${view.s})`,
            }}
            className="[&_svg]:!max-w-none [&_svg]:size-full"
            dangerouslySetInnerHTML={{ __html: svg }}
          />
        )}
      </div>

      <p className="border-t border-fd-border px-4 py-1.5 text-center text-xs text-fd-muted-foreground">
        scroll to zoom · drag to pan · double-click to reset · Esc to close
      </p>
    </div>
  );
}

export function Mermaid({ chart, title }: { chart: string; title?: string }) {
  const id = useId().replace(/:/g, '_');
  const containerRef = useRef<HTMLDivElement>(null);
  const [svg, setSvg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [zoomed, setZoomed] = useState(false);

  useEffect(() => setMounted(true), []);

  useEffect(() => {
    if (!mounted) return;
    let cancelled = false;

    loadMermaid().then(async (mermaid) => {
      const t = PALETTE[resolvedTheme === 'dark' ? 'dark' : 'light'];

      try {
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: 'strict',
          theme: 'base',
          fontFamily:
            'var(--font-mono), "JetBrains Mono", ui-monospace, monospace',
          themeVariables: {
            fontFamily:
              'var(--font-mono), "JetBrains Mono", ui-monospace, monospace',
            fontSize: '13px',
            background: t.bg,

            primaryColor: t.card,
            primaryTextColor: t.fg,
            primaryBorderColor: t.mutedFg,

            secondaryColor: t.muted,
            secondaryTextColor: t.fg,
            secondaryBorderColor: t.border,

            tertiaryColor: t.bg,
            tertiaryTextColor: t.fg,
            tertiaryBorderColor: t.border,

            lineColor: t.primary,
            arrowheadColor: t.primary,
            textColor: t.fg,
            mainBkg: t.card,
            nodeBorder: t.mutedFg,
            clusterBkg: 'transparent',
            clusterBorder: t.border,
            edgeLabelBackground: t.bg,
            titleColor: t.fg,

            actorBkg: t.card,
            actorBorder: t.mutedFg,
            actorTextColor: t.fg,
            actorLineColor: t.border,
            signalColor: t.fg,
            signalTextColor: t.fg,
            labelBoxBkgColor: t.card,
            labelBoxBorderColor: t.border,
            labelTextColor: t.fg,
            activationBorderColor: t.primary,
            activationBkgColor: t.card,
            sequenceNumberColor: t.primaryFg,
            noteBkgColor: t.muted,
            noteBorderColor: t.border,
            noteTextColor: t.fg,

            /* Pie slices: fixed hexes shared by both themes, picked dark
               enough that white in-slice text reads on every one. Chrome
               (legend, title, gaps) still follows the theme. */
            pie1: '#1f6feb',
            pie2: '#2f9e4f',
            pie3: '#8957e5',
            pie4: '#c2410c',
            pie5: '#a16207',
            pie6: '#57606a',
            pie7: '#0f766e',
            pie8: '#9f1239',
            pieOpacity: '1',
            pieStrokeColor: t.bg,
            pieStrokeWidth: '2px',
            pieOuterStrokeColor: t.border,
            pieOuterStrokeWidth: '1px',
            pieSectionTextColor: '#ffffff',
            pieSectionTextSize: '12px',
            pieLegendTextColor: t.fg,
            pieLegendTextSize: '13px',
            pieTitleTextColor: t.fg,
            pieTitleTextSize: '15px',
          },
        });

        const { svg, bindFunctions } = await mermaid.render(`mermaid-${id}`, chart);
        if (cancelled) return;
        setSvg(svg);
        setError(null);
        queueMicrotask(() => {
          if (containerRef.current && bindFunctions) bindFunctions(containerRef.current);
        });
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    });

    return () => {
      cancelled = true;
    };
  }, [chart, id, mounted, resolvedTheme]);

  if (error) {
    return (
      <pre className="text-sm text-red-600 dark:text-red-400 whitespace-pre-wrap">
        Mermaid render error: {error}
      </pre>
    );
  }
  if (!svg) {
    return <div className="mermaid-frame my-6 h-24 animate-pulse" aria-hidden />;
  }

  /* The lightbox mounts a second copy of the same markup, and mermaid scopes
     its <style> and its marker refs by the diagram id — so rename them, or the
     copy inherits (and fights over) the original's ids. */
  const zoomSvg = svg.replaceAll(`mermaid-${id}`, `mermaid-${id}-zoom`);

  return (
    <figure className="mermaid-frame my-6 group">
      <button
        type="button"
        onClick={() => setZoomed(true)}
        aria-label="Enlarge diagram"
        title="Enlarge diagram"
        className="absolute top-2.5 right-2.5 z-1 rounded-md p-1.5 text-fd-muted-foreground opacity-0 transition-opacity hover:bg-fd-accent hover:text-fd-accent-foreground group-hover:opacity-100 focus-visible:opacity-100"
      >
        <Expand className="size-4" />
      </button>
      <div
        ref={containerRef}
        onClick={(e) => {
          if ((e.target as HTMLElement).closest('a')) return;
          setZoomed(true);
        }}
        className="flex justify-center cursor-zoom-in [&_svg]:max-w-full [&_svg]:h-auto"
        dangerouslySetInnerHTML={{ __html: svg }}
      />
      {zoomed &&
        createPortal(
          <MermaidLightbox
            svg={zoomSvg}
            title={title ?? 'Diagram'}
            onClose={() => setZoomed(false)}
          />,
          document.body,
        )}
    </figure>
  );
}
