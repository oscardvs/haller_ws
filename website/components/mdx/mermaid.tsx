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
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'strict',
        theme: resolvedTheme === 'dark' ? 'dark' : 'default',
        fontFamily: 'var(--font-sans)',
      });

      try {
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
    return <div className="my-4 h-16 animate-pulse rounded bg-fd-muted/40" aria-hidden />;
  }

  return (
    <div
      ref={containerRef}
      className="my-4 flex justify-center [&_svg]:max-w-full"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
