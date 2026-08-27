#!/usr/bin/env python3
"""Tests for `typediff.py` — the three modes it has actually lied in.

Run: `npm run test:tools`, or `python3 tools/test_typediff.py`.

**These are deliberately NOT part of `npm test`.** That script is
`tsc --noEmit && vitest run`, a JavaScript toolchain, and coupling the frontend
suite to whichever `python3` happens to be on PATH would make the JS suite fail
for a reason that has nothing to do with the frontend. The cost is stated
rather than hidden: this file only runs when someone runs it. It is one command
and it is named in `package.json`, which is the most this territory can do
without reaching into the backend's pytest — that boundary is the integrator's
to move, not Track C's.

Every case below is a REGRESSION, not a hypothetical. Each one is a shape this
tool reported confidently and wrongly against a real payload on 2026-08-27, and
the whole reason it is in the repo at all is the port's own rule that the
guardrail belongs in the artifact and not in the habit.

No backend, no network, no fixtures on disk — pure functions over strings.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typediff import ts_fields, compat, _union_parts, jtype  # noqa: E402

FAILED = []


def check(label, got, want):
    if got != want:
        FAILED.append(f"{label}\n      got  {got!r}\n      want {want!r}")


# ── Mode 1: the one-line type fall-through ────────────────────────────────
#
# `ts_fields` required `\n};`, so a ONE-LINE declaration did not match and the
# search ran on to the NEXT type in the file. The log payload was then diffed
# against `Checkpoint` and reported three required fields missing and two
# unexpected — a full, confident, entirely fictional report. It did not error,
# which is what made it expensive.

SRC = """
export type LogPage = { offset: number; text: string };

export type Checkpoint = {
  step: number | null;
  path: string;
  has_model: boolean;
};

export type Run = RunSummary & {
  spec: Partial<TrainSpec> & Record<string, unknown>;
  argv?: string[];
  nested: { a: string; b: number };
};
"""


def test_one_line_type_does_not_fall_through():
    fields, _ = ts_fields(SRC, "LogPage")
    # The tell: `step`/`path`/`has_model` here would mean it read `Checkpoint`.
    check("one-line type reads its OWN fields",
          sorted(fields), ["offset", "text"])


# ── Mode 2: the line-anchored field regex ─────────────────────────────────
#
# The fix for mode 1 did not cure this, and testing is the only thing that
# found it: the field regex was line-anchored, so a one-line type yielded only
# its FIRST field. `LogPage` came back as `['offset']` and `text` vanished.
#
# That is mode 1 wearing different clothes — a field SILENTLY ABSENT, reported
# as agreement — and it is the more dangerous shape of the two, because a
# missing field produces a CLEAN report rather than a wrong one. Nobody
# investigates a clean report.

def test_one_line_type_yields_every_field_not_just_the_first():
    fields, _ = ts_fields(SRC, "LogPage")
    check("one-line type keeps its second field", "text" in fields, True)
    check("`text` is typed, not merely present",
          fields.get("text"), ("string", False))


def test_intersection_type_is_found_and_nested_named():
    fields, nested = ts_fields(SRC, "Run")
    check("`Run = RunSummary & {…}` is matched", "spec" in fields, True)
    # An inline object is NAMED as not-descended, never silently dropped —
    # dropping it would let a whole sub-object disagree in silence.
    check("inline nested object is named", nested, ["nested"])


def test_missing_type_raises_rather_than_guessing():
    try:
        ts_fields(SRC, "NoSuchType")
    except LookupError:
        return
    FAILED.append("missing type must RAISE, not return a guess")


# ── Mode 3: verdicts reached for the wrong reason ─────────────────────────
#
# `compat` returned a bool for everything, so an alias passed whenever its NAME
# happened to contain a matching substring: `RecordDrops` satisfied the object
# check because it contains "record". Three further instances were found while
# fixing it, all the same family — a verdict reached for a reason unrelated to
# the question.
#
# The dead arm is the one worth naming: `return None if <regex> else None`.
# Both arms `None`, so the regex was dead code and a REAL mismatch — `string`
# declared against a wire `object` — printed `? CANNOT JUDGE` instead of
# `MISMATCH`. It failed safe under the tool's own "never report confident
# nonsense" rule and was still a discriminator that could not discriminate,
# inside the very function rewritten to stop aliases passing by accident.
# **A `?` is what a reader skims: quieter is not the same as caught.**

def test_primitive_against_object_is_a_real_mismatch():
    check("`string` vs object is FALSE, not None", compat("string", "object"), False)
    check("`string | null` vs object is FALSE, not None",
          compat("string | null", "object"), False)


def test_unresolvable_alias_cannot_be_judged_either_way():
    # Was True by substring accident ("RecordDrops" contains "record").
    check("alias vs object is None", compat("RecordDrops", "object"), None)
    # Was False by the opposite accident: `type Tags = string[]` is a CORRECT
    # declaration against a wire array, reported as a confident MISMATCH.
    check("alias vs array is None", compat("Tags", "array"), None)


def test_null_is_matched_by_part_not_by_substring():
    # `"null" in "AnnullableFoo"` is True — the literal bug the rewrite claimed
    # to have removed, still sitting three lines above where it was removed.
    check("unrelated alias vs null is FALSE",
          compat("AnnullableFoo", "null"), False)
    check("a real nullable vs null is True",
          compat("string | null", "null"), True)


def test_generic_commas_do_not_invent_union_members():
    # Splitting naively on `|` breaks `Record<string, a | b>` into union parts
    # the declaration does not have — inventing a thing the source never said.
    check("generic is ONE part", _union_parts("Record<string, a | b>"),
          ["Record<string, a | b>"])
    check("a real union still splits", _union_parts("string | null"),
          ["string", "null"])
    check("`Record<…>` vs object is True",
          compat("Record<string, a | b>", "object"), True)


def test_jtype_reads_json_shapes_including_bool_before_number():
    # bool must be tested before int: `isinstance(True, int)` is True in Python,
    # so a boolean would otherwise report as `number` and every boolean field
    # would silently agree with a `number` declaration.
    check("bool is boolean, not number", jtype(True), "boolean")
    check("null", jtype(None), "null")
    check("array", jtype([1]), "array")
    check("object", jtype({"a": 1}), "object")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    if FAILED:
        print(f"FAILED {len(FAILED)}/{len(tests)}")
        for f in FAILED:
            print("  ✗ " + f)
        return 1
    print(f"typediff: {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
