"use client";

/**
 * What was asked for differently.
 *
 * Only the spec keys whose values DISAGREE get a row. A train spec is ~14
 * keys and two runs from the same launcher usually differ in one of them;
 * printing all fourteen is how that one gets missed — the reader scans a wall
 * of identical values and stops reading. The keys that agree are collapsed to
 * a single line that still names them, because "these two runs also shared a
 * device and a seed" is a fact worth having, just not worth fourteen rows.
 *
 * Values are rendered, never interpreted: `episodes` is 46 integers and the
 * cell shows as much of that list as a row can hold with the whole of it on
 * the title, so the exact kept set is one hover away and never a paragraph.
 */
import { useMemo, useState } from "react";

import { Button, Note, Panel, PanelHead } from "@/components/lab/ui";
import type { Run } from "@/lib/lab";

/** A cell wider than this wraps the table past the pane; the full string
 *  rides on `title`. */
const CLIP = 44;

/** The key was not in this run's spec at all. Distinct from a key whose value
 *  is empty: an older run that predates a knob never chose one. */
const ABSENT = "—";

export function HparamDiff({ runs }: { runs: Run[] }) {
  const [open, setOpen] = useState(false);

  const { keys, valuesFor, differing, identical } = useMemo(() => {
    const seen: string[] = [];
    for (const r of runs) {
      for (const k of Object.keys(specOf(r))) if (!seen.includes(k)) seen.push(k);
    }
    seen.sort();
    const table = new Map<string, string[]>();
    for (const k of seen) {
      table.set(k, runs.map((r) => renderValue(specOf(r)[k])));
    }
    const differs = (k: string) => {
      const vals = table.get(k) ?? [];
      return vals.some((v) => v !== vals[0]);
    };
    return {
      keys: seen,
      valuesFor: table,
      differing: seen.filter(differs),
      identical: seen.filter((k) => !differs(k)),
    };
  }, [runs]);

  if (runs.length === 0) return null;

  return (
    <Panel>
      <PanelHead
        title="Hyper-parameters"
        right={`${differing.length} differ · ${identical.length} identical`}
      />
      <div className="flex min-h-0 flex-col gap-2.5 p-3">
        {runs.length < 2 && (
          <Note>
            One run, so nothing disagrees with anything. Add a second run to the
            <span className="font-mono"> runs=</span> list to see a diff.
          </Note>
        )}

        {keys.length === 0 && (
          <Note>these runs carry no spec — nothing to diff.</Note>
        )}

        {runs.length >= 2 && keys.length > 0 && differing.length === 0 && (
          <Note>
            Every spec key agrees across these runs. Whatever separates the
            curves above, it is not what was asked for.
          </Note>
        )}

        {differing.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse font-mono text-[10px]">
              <thead>
                <tr className="border-b border-border label-micro text-muted-foreground">
                  <th className="px-2.5 py-1 text-left font-semibold">key</th>
                  {runs.map((r) => (
                    <th
                      key={r.id}
                      className="max-w-[16rem] truncate px-2.5 py-1 text-left font-semibold"
                      title={r.id}
                    >
                      {r.name ?? r.id}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {differing.map((k) => (
                  <tr key={k} className="border-b border-border/60 last:border-0">
                    <td className="px-2.5 py-1 whitespace-nowrap text-muted-foreground">
                      {k}
                    </td>
                    {(valuesFor.get(k) ?? []).map((v, i) => (
                      <td key={runs[i].id} className="px-2.5 py-1" title={v}>
                        <span className="tabular-nums" data-num>{clip(v)}</span>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {identical.length > 0 && (
          <div className="flex flex-col gap-2 border-t border-border pt-2.5">
            <div className="flex items-start justify-between gap-2">
              <p className="min-w-0 flex-1 text-[11px] text-pretty text-muted-foreground">
                {identical.length} hyper-parameter
                {identical.length === 1 ? "" : "s"} identical across these runs:{" "}
                <span className="font-mono">{preview(identical)}</span>
              </p>
              <Button
                onClick={() => setOpen((v) => !v)}
                aria-expanded={open}
                aria-label={open ? "hide identical hyper-parameters" : "show identical hyper-parameters"}
              >
                {open ? "hide" : "show all"}
              </Button>
            </div>
            {open && (
              <div className="overflow-x-auto">
                <table className="w-full border-collapse font-mono text-[10px]">
                  <tbody>
                    {identical.map((k) => {
                      const v = (valuesFor.get(k) ?? [ABSENT])[0];
                      return (
                        <tr key={k} className="border-b border-border/60 last:border-0">
                          <td className="w-[14rem] px-2.5 py-1 whitespace-nowrap text-muted-foreground">
                            {k}
                          </td>
                          <td className="px-2.5 py-1" title={v}>
                            <span className="tabular-nums" data-num>{clip(v)}</span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </Panel>
  );
}

/** The spec as a bag of keys. `Run.spec` is `Partial<TrainSpec> &
 *  Record<string, unknown>`, and a run written by an older launcher may carry
 *  keys this build has never heard of — which is exactly what a diff should
 *  show rather than drop. */
function specOf(r: Run): Record<string, unknown> {
  return (r.spec ?? {}) as Record<string, unknown>;
}

/** One spec value on one line. Arrays are joined rather than JSON-encoded:
 *  the kept-episode list is 46 integers, and brackets and commas-with-quotes
 *  buy nothing a reader wants. */
function renderValue(v: unknown): string {
  if (v === undefined) return ABSENT;
  if (v === null) return "null";
  if (Array.isArray(v)) return v.length === 0 ? "[]" : v.map(String).join(", ");
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function clip(s: string): string {
  return s.length > CLIP ? s.slice(0, CLIP - 1) + "…" : s;
}

function preview(keys: string[]): string {
  const head = keys.slice(0, 6).join(", ");
  return keys.length > 6 ? `${head}, …` : head;
}
