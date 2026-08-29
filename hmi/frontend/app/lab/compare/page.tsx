"use client";

/**
 * `/lab/compare?runs=a,b,c` — the Lab's one deep link.
 *
 * A comparison of training runs is the one thing in this app worth keeping
 * after the tab is closed: it is the evidence behind "we shipped the 100k-step
 * ACT run and not the 50k one". So the whole state of this page is the query
 * string, and the page is unlinked chrome the way `/settings` and `/teleop/vr`
 * are — a bookmark, a second monitor and a pasted link all open the same thing.
 *
 * `useSearchParams` bails the client tree out of prerendering, so the reader
 * lives under a <Suspense> boundary. Without one the production build fails
 * outright (missing-suspense-with-csr-bailout), which is why the params live
 * in their own component rather than in the default export.
 *
 * This route is also the one surface in the cockpit that may scroll: the chart
 * count is set by whatever the trainer logged, and clipping a metric because
 * it landed in row four would hide exactly the run that failed.
 */
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";

import { DeepLinkChrome } from "@/components/DeepLinkChrome";
import { ComparePane } from "@/components/lab/ComparePane";
import { Empty, Note, Panel, PanelHead } from "@/components/lab/ui";

export default function CompareRunsPage() {
  return (
    <>
      <DeepLinkChrome label="Compare runs" />
      <main className="p-3 space-y-3">
        <Suspense
          fallback={
            <Panel>
              <PanelHead title="Compare runs" />
              <Empty>reading the url…</Empty>
            </Panel>
          }
        >
          <CompareParams />
        </Suspense>
      </main>
    </>
  );
}

function CompareParams() {
  const params = useSearchParams();
  const ids = parseIds(params.get("runs"));

  if (ids.length === 0) {
    return (
      <Panel>
        <PanelHead title="Compare runs" />
        <div className="p-3">
          <Note>
            No runs in the url. Tick the compare box beside each run in the
            Train tab&apos;s run list — it builds{" "}
            <span className="font-mono">/lab/compare?runs=a,b,c</span>, and that
            url is the whole state of this page: bookmark it, paste it into a
            note, open it on the other machine.
          </Note>
        </div>
      </Panel>
    );
  }

  return <ComparePane runIds={ids} />;
}

/** `runs=a,b,a` → `["a", "b"]`. Deduped in first-seen order: a repeated id
 *  would otherwise get a second colour and a second legend row, and read as
 *  two runs that happen to agree exactly. */
function parseIds(raw: string | null): string[] {
  const out: string[] = [];
  for (const part of (raw ?? "").split(",")) {
    const id = part.trim();
    if (id && !out.includes(id)) out.push(id);
  }
  return out;
}
