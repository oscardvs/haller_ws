"""Diff a live payload against the TS type that claims to describe it.

The discipline this encodes: compare the two REAL components rather than either
against a fixture. A field typed from a contract LINE rather than from a
payload is the shape that produced every Track C defect on 2026-08-27, and all
of them type-checked.

THE ONE RULE THIS TOOL MUST OBEY: never report confident nonsense. Every case
it cannot actually judge is printed as `?` rather than passing quietly, because
a silent pass from a checker is worth less than no checker — it is the
check-that-cannot-fire, one level up.

Known-good on the wire side: the payload is parsed JSON, so a nested object
(`drops: {cameras: {}, arms: {}}`) contributes ONE top-level key. A regex over
Python source hoists `"arms":` into the top level and reports a field the route
has never sent; this tool cannot, because it never reads the producer's source.
That failure belongs to hand-rolled four-liners — haller-ws-95 hit it on
2026-08-27 and it is why this note exists.

Usage:
    python typediff.py                       # against the default backend
    python typediff.py --base http://127.0.0.1:8031 --run <run_id>
    from typediff import ts_fields, diff     # importable; nothing runs on import

`main()` walks the /lab run surface. For any other route, import it: pass
`ts_fields(read(f"{FRONTEND}/lib/api.ts"), "RecordStatus")` and a payload you
fetched yourself — that is how the 2026-08-27 `RecordStatus` reconcile and the
arm/roll/stop walk were run.

Its own tests are `test_typediff.py`, beside it: `npm run test:tools`, or
`python3 tools/test_typediff.py`. They pin the three confident-nonsense modes
this tool has actually produced. **They are NOT part of `npm test`**, which is
a JS runner — see the note at the top of that file.
"""
import argparse
import json
import os
import re
import urllib.request

#: This file lives at `hmi/frontend/tools/`, so the frontend is two levels up.
#: Derived from `__file__` rather than hardcoded, and that is not tidiness: the
#: original carried an absolute path into the SHARED tree, so running it from a
#: detached worktree read the shared tree's types while appearing to measure the
#: worktree — the same fake-isolation trap the editable install produces for
#: pytest, and invisible for the same reason, because both trees usually agree.
FRONTEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(path):
    with open(path) as f:
        return f.read()


def _take(fragment, out, nested):
    """One `name?: type` fragment into `out`, or record it as nested."""
    frag = fragment.strip()
    if not frag:
        return
    m = re.match(r"(\w+)(\??)\s*:\s*(.+)", frag, re.S)
    if not m:
        return
    declared = " ".join(m.group(3).split())
    if declared.startswith("{"):
        nested.append(m.group(1))       # inline object: named, never silently dropped
        return
    out[m.group(1)] = (declared, m.group(2) == "?")


def ts_fields(src, name):
    """`{field: (declared, optional)}` for `export type <name> = { … }`.

    RAISES rather than guessing. The original version required `\\n};`, so a
    ONE-LINE declaration (`export type LogPage = { offset: number };`) did not
    match and the regex ran on to the NEXT type in the file — then reported the
    payload against a type nobody asked about, with no error. That is the
    second confident-nonsense mode and it cost a wrong `LogPage` report.
    """
    m = re.search(rf"export type {re.escape(name)}\s*=\s*(?:\w+\s*&\s*)?\{{", src)
    if not m:
        raise LookupError(f"type `{name}` not found — check the spelling or the file")

    # Walk braces from the opening one so single-line and multi-line declarations
    # are the same code path, and so the block cannot run past its own type.
    i = m.end() - 1
    depth, end = 0, None
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                end = j
                break
    if end is None:
        raise LookupError(f"type `{name}` has no closing brace — refusing to guess")

    body = src[i + 1:end]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    body = re.sub(r"//.*", "", body)

    # Split on TOP-LEVEL `;` rather than by line. A line-anchored regex reads
    # only the first field of a one-line type — `{ offset: number; text: string }`
    # yielded `offset` alone — which is the same bug as the fall-through wearing
    # a different shape: a field silently absent, reported as agreement.
    out, nested, depth, buf = {}, [], 0, []
    for ch in body:
        if ch in "{[(<":
            depth += 1
        elif ch in "}])>":
            depth -= 1
        if ch == ";" and depth == 0:
            _take("".join(buf), out, nested)
            buf = []
        else:
            buf.append(ch)
    _take("".join(buf), out, nested)      # a trailing field with no `;`
    return out, nested


def jtype(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    return "object"


PRIMITIVES = ("string", "number", "boolean", "null")


def _union_parts(d):
    """Split a declaration on TOP-LEVEL `|` only.

    `Record<string, a | b>` is one part, not two — splitting naively on `|`
    invents union members that the declaration does not have, which is the
    invent-a-thing-the-source-never-said family again.
    """
    parts, depth, buf = [], 0, []
    for ch in d:
        if ch in "{[(<":
            depth += 1
        elif ch in "}])>":
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf).strip())
    return [p for p in parts if p]


def _kind_of(part):
    """What ONE union member is, as far as this file can tell."""
    p = part.strip()
    low = p.lower()
    if p.endswith("[]") or low.startswith("array<") or low.startswith("readonly"):
        return "array"
    if p.startswith("{") or low.startswith("record<") or low.startswith("partial<"):
        return "object"
    if low in PRIMITIVES or p.startswith('"') or p.startswith("'") or p.isdigit():
        return "prim"
    return "alias"          # a name this file cannot resolve


def compat(declared, actual):
    """True / False / None, where **None means CANNOT JUDGE**.

    The original returned a bool for everything, so an alias passed whenever its
    NAME happened to contain a matching substring — `RecordDrops` satisfied the
    object check because it contains "record". Agreement for the wrong reason is
    the thing this whole tool exists to catch, so an unresolvable alias is now
    reported rather than waved through.
    """
    d = declared.strip()
    parts = _union_parts(d)                       # top-level `|` only
    live = [p for p in parts if p.lower() not in ("null", "undefined")]
    anything = any(p.lower() in ("unknown", "any") for p in parts)

    if actual == "null":
        # By PART, not by substring: `"null" in "AnnullableFoo"` is the same
        # accidental-substring bug this function was rewritten to remove.
        return anything or any(p.lower() in ("null", "undefined") for p in parts)

    if anything:
        return True

    kinds = {_kind_of(p) for p in live}            # {"array"|"object"|"prim"|"alias"}

    # An ALIAS anywhere in the union means the declaration cannot be resolved
    # from this file alone — and that is true whichever way it would fall.
    # Returning False here was the second instance of the accidental-verdict
    # bug: `tags?: Tags` where `type Tags = string[]` is a correct declaration
    # and was reported as a confident MISMATCH against a wire array.
    if "alias" in kinds:
        return None
    if not kinds:
        return None

    if actual == "array":
        return "array" in kinds
    if actual == "object":
        # Every live part is a primitive or an array, so an object on the wire
        # is a REAL mismatch — this is the arm that used to return None and
        # made the discriminator unable to discriminate.
        return "object" in kinds
    # A primitive on the wire.
    return any(p.lower() == actual or p.startswith('"') and actual == "string"
               or p.isdigit() and actual == "number" for p in live)


def diff(label, parsed, payload):
    """Print one comparison. `parsed` is `(fields, nested)` from `ts_fields`."""
    print(f"\n{'=' * 66}\n{label}\n{'=' * 66}")
    fields, nested = parsed
    lines = 0
    tk, pk = set(fields), set(payload)

    for k in sorted(pk - tk):
        if k in nested:
            continue
        print(f"  ON WIRE, NOT IN TYPE : {k} = {jtype(payload[k])}")
        lines += 1
    for k in sorted(tk - pk):
        _, opt = fields[k]
        print(f"  IN TYPE, NOT ON WIRE : {k}{' (optional)' if opt else '  <-- REQUIRED'}")
        lines += 1
    for k in sorted(tk & pk):
        declared, _ = fields[k]
        a = jtype(payload[k])
        verdict = compat(declared, a)
        if verdict is False:
            print(f"  MISMATCH             : {k}: declared `{declared}` | wire `{a}`")
            lines += 1
        elif verdict is None:
            print(f"  ? CANNOT JUDGE       : {k}: declared `{declared}` | wire `{a}`"
                  f"  (alias — resolve by hand)")
            lines += 1
    for k in nested:
        print(f"  ? NOT DESCENDED      : {k} is an inline nested object; this tool"
              f" compares one level only")
        lines += 1

    if lines == 0:
        print("  agrees, and every field was judged")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8022",
                    help="backend base URL (the default port is whatever YOUR rig uses; "
                         "a dead port is a connection error, not a contract finding)")
    ap.add_argument("--run", default=None, help="a run id; omitted = the newest one")
    args = ap.parse_args()

    lab = read(f"{FRONTEND}/lib/lab.ts")

    def get(path):
        with urllib.request.urlopen(args.base + path, timeout=15) as r:
            return json.load(r)

    runs = get("/lab/runs")["runs"]
    if not runs:
        print("no runs in the store — point --base at a backend with a run store")
        return
    run_id = args.run or runs[0]["id"]

    diff("RunSummary  vs  GET /lab/runs [0]", ts_fields(lab, "RunSummary"), runs[0])

    summ, sn = ts_fields(lab, "RunSummary")
    extra, en = ts_fields(lab, "Run")
    diff(f"Run (= RunSummary & …)  vs  GET /lab/runs/{run_id}",
         ({**summ, **extra}, sn + en), get(f"/lab/runs/{run_id}"))

    cks = get(f"/lab/runs/{run_id}/checkpoints")["checkpoints"]
    if cks:
        diff("Checkpoint  vs  /checkpoints [0]", ts_fields(lab, "Checkpoint"), cks[0])
        diff("Checkpoint  vs  /checkpoints [last]", ts_fields(lab, "Checkpoint"), cks[-1])
    diff("MetricsPage vs  /metrics", ts_fields(lab, "MetricsPage"),
         get(f"/lab/runs/{run_id}/metrics"))
    diff("LogPage     vs  /log", ts_fields(lab, "LogPage"), get(f"/lab/runs/{run_id}/log"))
    diff("LabSystem   vs  /lab/system", ts_fields(lab, "LabSystem"), get("/lab/system"))


if __name__ == "__main__":
    main()
