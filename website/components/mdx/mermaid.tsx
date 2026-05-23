'use client';

import { useEffect, useId, useRef, useState } from 'react';
import { useTheme } from 'next-themes';

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
    primary:   '#d75721',
    primaryFg: '#fbf7ed',
  },
  dark: {
    bg:        '#1a1612',
    fg:        '#e7e1d4',
    muted:     '#221d17',
    mutedFg:   '#a99e88',
    card:      '#1f1a15',
    border:    '#3a3127',
    primary:   '#e57b3f',
    primaryFg: '#1a1612',
  },
} as const;

export function Mermaid({ chart }: { chart: string }) {
  const id = useId().replace(/:/g, '_');
  const containerRef = useRef<HTMLDivElement>(null);
  const [svg, setSvg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

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

  return (
    <figure className="mermaid-frame my-6">
      <div
        ref={containerRef}
        className="flex justify-center [&_svg]:max-w-full [&_svg]:h-auto"
        dangerouslySetInnerHTML={{ __html: svg }}
      />
    </figure>
  );
}
