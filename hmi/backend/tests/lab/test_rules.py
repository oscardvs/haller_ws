# hmi/backend/tests/lab/test_rules.py
"""The rules DSL: the grammar, the offsets, the worst arm, and the shell that
must not be there.

Three things are defended here, and the order is deliberate.

**That no rule can execute anything.** `autoclass/preview` is ungated so Oscar
can triage from the headset, so this string arrives over the LAN and is handled
on the machine that owns the servo bus. The battery below is the shape of an
actual attempt — `__import__`, a subscript into `().__class__`, a `lambda`, a
conditional expression, a statement separator — and every one of them must come
back as a `RuleError` with an offset. `test_the_module_never_calls_eval_exec_or_compile`
reads `rules.py` back with `ast` rather than trusting a comment, because the
whole file exists to not be a one-line `eval` and the way that regresses is
somebody "simplifying" it.

**That the offset is right.** The route turns a `RuleError` into a 400 that
reaches Oscar as a toast inside a headset, where "unexpected '(' at 14" is
actionable and "invalid syntax" is not. An offset that is merely present is not
the same as one that points at the character to blame, so the offsets are
asserted, not the messages alone.

**That a bare name is the WORST arm.** `tracking > 5` must mean "either arm
failed to track". The test that matters is the one where reading only the left
arm would have said the episode is fine: that is the same failure the per-arm
rewrite of `grade.py` exists to kill, arriving through a different door.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from haller_hmi.lab import catalog, rules
from haller_hmi.lab.schema import RigSpec

from . import _dataset

# ---- namespaces -----------------------------------------------------------


def _arm(side: str, **over) -> dict:
    """One `arms` entry from a graded episode, clean unless overridden."""
    arm = {
        "side": side, "verdict": "PASS", "why": "single clean grasp and release",
        "closes": 1, "reopened": True, "grip_min": 10.0, "grip_max": 100.0,
        "tracking": 0.1, "sweep_total": 40.0, "sweep": [40.0, 0.0],
        "closed_below": 40.0, "open_above": 70.0,
    }
    arm.update(over)
    return arm


def _episode(arms, **over) -> dict:
    """One `catalog.dataset_detail` episode, with the field names it really
    uses — `seconds` and `status`, which the namespace renames to `duration_s`
    and `mark`. A hand-written namespace would never catch that rename going
    wrong; this shape is why the fixture is built as an EPISODE."""
    ep = {
        "episode_index": 3, "label": 4, "frames": 100, "seconds": 4.0,
        "share": 0.25, "verdict": "PASS", "reasons": [], "arms": arms,
        "tasks": ["Test task"], "videos": {}, "status": "unset", "note": "",
        "tags": [],
    }
    ep.update(over)
    return ep


@pytest.fixture()
def solo() -> dict:
    """A single unprefixed arm — the kit's rig. No `left.` / `right.` names
    exist at all here, because there is nothing to tell apart."""
    return rules.build_namespace(_episode([_arm("")]), "solo")


@pytest.fixture()
def worst() -> dict:
    """Left arm spotless, right arm broken in every measurable way.

    This is the namespace the whole worst-arm rule is about: every rule below
    that reads a bare name would answer "fine" if the roll-up took the first
    arm, and columns 0..5 — everything the kit's constants could see — are the
    left arm.
    """
    return rules.build_namespace(
        _episode(
            [
                _arm("left"),
                _arm("right", verdict="FAIL", tracking=6.0, sweep_total=1.0,
                     closes=3, grip_min=0.0, grip_max=42.0, reopened=False),
            ],
            verdict="FAIL",
        ),
        "bimanual",
    )


# ---- every operator -------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("frames < 200", True), ("frames < 100", False),
    ("frames <= 100", True), ("frames <= 99", False),
    ("frames > 99", True), ("frames > 100", False),
    ("frames >= 100", True), ("frames >= 101", False),
    ("frames == 100", True), ("frames == 99", False),
    ("frames != 99", True), ("frames != 100", False),
])
def test_every_operator_compares_numbers(solo, text, expected):
    assert rules.match(text, solo) is expected


@pytest.mark.parametrize("text,expected", [
    ("verdict == 'PASS'", True), ("verdict == 'FAIL'", False),
    ("verdict != 'FAIL'", True), ("verdict != 'PASS'", False),
    ("mark == 'unset'", True), ("mark == 'reject'", False),
])
def test_the_string_names_compare_with_equality(solo, text, expected):
    assert rules.match(text, solo) is expected


@pytest.mark.parametrize("text,expected", [
    ("share == 0.25", True),
    ("share == .25", True),
    ("grip_min > -5", True),
    ("grip_min < -5", False),
    ("frames == 1e2", True),
])
def test_number_literals_cover_the_forms_an_operator_types(solo, text, expected):
    """`-5` in particular: Haller's gripper is calibrated to [-9.97, 100.27], so
    a negative literal is a rule somebody writes, not a curiosity."""
    assert rules.match(text, solo) is expected


def test_a_rule_evaluates_to_a_real_bool(solo):
    """`is True`, not truthiness: the diff this feeds carries the answer into
    JSON, where a numpy scalar or an int would serialise as something the UI
    renders differently."""
    result = rules.evaluate(rules.parse("frames > 1"), solo)

    assert result is True
    assert type(result) is bool


# ---- precedence -----------------------------------------------------------

def test_and_binds_tighter_than_or(solo):
    """`a or b and c` is `a or (b and c)`. Read the other way this is False, so
    the assertion distinguishes the two groupings rather than agreeing with
    both."""
    assert rules.match("frames == 100 or frames == 1 and frames == 2", solo) is True
    assert rules.match("frames == 1 and frames == 2 or frames == 100", solo) is True


def test_parentheses_override_and_over_or(solo):
    assert rules.match("(frames == 100 or frames == 1) and frames == 2", solo) is False
    assert rules.match("frames == 1 and (frames == 2 or frames == 100)", solo) is False


def test_not_binds_tighter_than_and(solo):
    """Both operands are FALSE, which is the only choice of inputs that tells
    the two readings apart: `(not a) and b` is False, `not (a and b)` is True."""
    text = "not frames == 1 and frames == 2"

    assert rules.match(text, solo) is False
    assert rules.match("not (frames == 1 and frames == 2)", solo) is True


def test_not_applies_to_a_whole_comparison_not_to_its_left_operand(solo):
    assert rules.match("not frames > 1000", solo) is True
    assert rules.match("not frames > 1", solo) is False


def test_not_stacks(solo):
    assert rules.match("not not frames > 1", solo) is True
    assert rules.match("not not not frames > 1", solo) is False


def test_a_group_can_hold_a_whole_expression(solo):
    assert rules.match(
        "(frames > 1 and (verdict == 'PASS' or mark == 'reject'))", solo) is True


# ---- the offset -----------------------------------------------------------

@pytest.mark.parametrize("text,pos,fragment", [
    pytest.param("", 0, "empty", id="empty"),
    pytest.param("frames > ", 9, "expected a number", id="missing-right-operand"),
    pytest.param("frames > 5 and", 14, "expected a number", id="dangling-and"),
    pytest.param("frames > 5 or", 13, "expected a number", id="dangling-or"),
    pytest.param("not", 3, "expected a number", id="dangling-not"),
    pytest.param("(frames > 5", 11, "expected ')'", id="unclosed-group"),
    pytest.param("()", 1, "expected a number", id="empty-group"),
    pytest.param("frames >> 5", 8, "expected a number", id="doubled-operator"),
    pytest.param("frames > 5 5", 11, "already complete", id="trailing-token"),
    pytest.param("frames = 5", 7, "'=='", id="single-equals"),
    pytest.param("frames ! 5", 7, "'!='", id="lone-bang"),
    pytest.param("share > 0.3 ; frames > 5", 12, "unexpected character", id="semicolon"),
    pytest.param("verdict == 'oops", 11, "unterminated", id="unterminated-string"),
    pytest.param("frames > 5 and (share > 0.3", 27, "expected ')'", id="unclosed-second"),
])
def test_a_malformed_rule_carries_the_offset_of_the_character_to_blame(
        text, pos, fragment):
    """"unexpected '(' at 14" is a 400 the operator can act on from inside a
    headset. "invalid syntax" is a debugging session they cannot run in there."""
    with pytest.raises(rules.RuleError) as excinfo:
        rules.parse(text)

    assert excinfo.value.pos == pos
    assert fragment in excinfo.value.message
    # The offset has to survive into the string the route puts in `detail`.
    assert str(pos) in str(excinfo.value)


def test_the_unclosed_group_error_names_the_paren_that_was_opened():
    """Two open parens and one close: the offset is where the rule ran out, and
    the message says which '(' is still waiting."""
    with pytest.raises(rules.RuleError) as excinfo:
        rules.parse("((frames > 5)")

    assert "opened at character 0" in excinfo.value.message


def test_a_rule_error_is_a_value_error_so_the_route_answers_400():
    """`api/errors.as_http` maps `ValueError` to 400 already; inheriting means a
    bad rule needs no rung of its own, and cannot come back as a 500."""
    assert issubclass(rules.RuleError, ValueError)

    with pytest.raises(rules.RuleError) as excinfo:
        rules.parse("frames >")

    assert isinstance(excinfo.value, ValueError)
    assert excinfo.value.message and isinstance(excinfo.value.pos, int)


def test_uppercase_keywords_are_told_they_are_uppercase(solo):
    """`AND` tokenises as a name, so the bare parser error is "the rule was
    already complete" — true, and not the thing the operator got wrong."""
    with pytest.raises(rules.RuleError) as excinfo:
        rules.parse("frames > 5 AND frames < 9")

    assert excinfo.value.pos == 11
    assert "lower case" in excinfo.value.message


# ---- unknown names --------------------------------------------------------

def test_an_unknown_name_names_itself_and_lists_the_valid_ones(solo):
    """Not a silent False. A rule that matches nothing because of a typo reads
    exactly like a dataset with no episodes worth fixing."""
    with pytest.raises(rules.RuleError) as excinfo:
        rules.match("trackng > 5", solo)

    message = excinfo.value.message
    assert excinfo.value.pos == 0
    assert "'trackng'" in message
    for name in ("tracking", "frames", "duration_s", "share", "verdict", "mark"):
        assert name in message


def test_an_unknown_side_names_the_rig_rather_than_answering_false():
    """`left.tracking` on a right-only rig. Silent False would make a typo look
    like "no episodes matched", which is the answer an operator acts on."""
    names = [n for n in _dataset.state_names("bimanual") if n.startswith("right_")]
    rig = RigSpec.from_info({"features": {"observation.state": {
        "dtype": "float32", "shape": [len(names)], "names": names}}})
    assert rig.rig == "right"
    ns = rules.build_namespace(_episode([_arm("right")]), rig)

    with pytest.raises(rules.RuleError) as excinfo:
        rules.match("left.tracking > 5", ns)

    assert "'right'" in excinfo.value.message      # the rig, named
    assert "left" in excinfo.value.message


def test_a_solo_rig_says_there_are_no_sides_at_all(solo):
    with pytest.raises(rules.RuleError) as excinfo:
        rules.match("right.tracking > 5", solo)

    assert "'solo'" in excinfo.value.message
    assert "unprefixed" in excinfo.value.message


def test_an_unknown_field_on_a_known_side_lists_that_arms_fields(worst):
    with pytest.raises(rules.RuleError) as excinfo:
        rules.match("left.frames > 5", worst)

    assert "'left.frames'" in excinfo.value.message
    assert "tracking" in excinfo.value.message


def test_an_unknown_name_is_raised_even_when_the_other_branch_already_decided(solo):
    """No short-circuit. If `or` stopped at the first true comparison, a typo in
    the second branch would raise on some episodes and not others — the hardest
    kind of 400 to reproduce from a headset."""
    with pytest.raises(rules.RuleError):
        rules.match("frames > 1 or nope > 1", solo)
    with pytest.raises(rules.RuleError):
        rules.match("frames > 1000 and nope > 1", solo)


# ---- comparisons that cannot mean anything --------------------------------

def test_comparing_a_string_to_a_number_is_a_rule_error_not_a_type_error(solo):
    """A bare `TypeError` escapes `as_http` as a 500. This is bad input, and the
    operator has to be told which side of their comparison is text."""
    with pytest.raises(rules.RuleError) as excinfo:
        rules.match("verdict > 3", solo)

    assert excinfo.value.pos == 8            # the operator, not the operands
    assert "verdict" in excinfo.value.message
    assert "'PASS'" in excinfo.value.message


@pytest.mark.parametrize("text", [
    "verdict == 3", "3 != mark", "frames == 'PASS'", "mark < 'x'",
])
def test_text_and_numbers_never_compare(solo, text):
    with pytest.raises(rules.RuleError):
        rules.match(text, solo)


def test_ordering_two_strings_is_refused_rather_than_answered_alphabetically(solo):
    """`verdict > 'FAIL'` looks like a question about severity and would be
    answered as one about the alphabet."""
    with pytest.raises(rules.RuleError) as excinfo:
        rules.match("verdict > 'FAIL'", solo)

    assert "== and !=" in excinfo.value.message


def test_an_unmeasurable_value_is_refused_loudly():
    """`grade.py` writes None for UNMEASURABLE, never for zero. A gripperless
    arm answering False to `closes == 0` would pass every episode of that
    dataset through a grasp rule without a word."""
    ns = rules.build_namespace(
        _episode([_arm("", closes=None, reopened=None, grip_min=None, grip_max=None)]),
        "solo")

    assert ns["closes"] is None
    with pytest.raises(rules.RuleError) as excinfo:
        rules.match("closes == 0", ns)

    assert "not measured" in excinfo.value.message


# ---- the worst arm --------------------------------------------------------

def test_a_bare_name_reads_the_worst_arm_not_the_first(worst):
    """The one that matters. Reading only the left arm — which is every column
    the kit's `GRIPPER_IDX = 5` can see — says this episode tracked fine. It did
    not: the right arm was 6° off, and a reject rule written as `tracking > 5`
    has to catch it."""
    assert rules.match("left.tracking > 5", worst) is False
    assert rules.match("right.tracking > 5", worst) is True
    assert rules.match("tracking > 5", worst) is True


@pytest.mark.parametrize("text", [
    pytest.param("tracking > 5", id="tracking-max"),
    pytest.param("closes > 1", id="closes-max"),
    pytest.param("sweep_total < 5", id="sweep-min"),
    pytest.param("grip_min < 5", id="grip-min-min"),
    pytest.param("grip_max < 50", id="grip-max-min"),
])
def test_each_roll_up_takes_the_worse_direction(worst, text):
    """max for tracking and closes (more is worse), min for the sweep and the
    grip range (less is worse). Every one of these fires on the bare name and
    reads False off the left arm alone."""
    assert rules.match(text, worst) is True
    assert rules.match(f"left.{text}", worst) is False
    assert rules.match(f"right.{text}", worst) is True


def test_reopened_is_true_only_when_every_arm_reopened():
    """`all`, not `any`: "the gripper let go" is a statement about the episode,
    and an episode where one jaw is still holding the cube is not one where it
    let go."""
    both = rules.build_namespace(
        _episode([_arm("left"), _arm("right")]), "bimanual")
    one = rules.build_namespace(
        _episode([_arm("left"), _arm("right", reopened=False)]), "bimanual")

    assert rules.match("reopened == true", both) is True
    assert rules.match("reopened == true", one) is False
    assert rules.match("left.reopened == true", one) is True
    assert rules.match("right.reopened == false", one) is True


def test_a_solo_rigs_bare_names_are_simply_its_one_arms_values():
    ns = rules.build_namespace(
        _episode([_arm("", tracking=6.0, sweep_total=1.0)]), "solo")

    assert ns["tracking"] == 6.0
    assert ns["sweep_total"] == 1.0
    assert not [k for k in ns if "." in k]


def test_an_arm_that_cannot_supply_a_value_is_left_out_of_the_roll_up():
    """A gripperless arm has None for the grasp numbers. Counting that as zero
    would make `grip_min < 5` fire on the arm that has no jaws at all."""
    ns = rules.build_namespace(
        _episode([_arm("left", grip_min=None), _arm("right", grip_min=7.0)]),
        "bimanual")

    assert ns["grip_min"] == 7.0
    assert rules.match("grip_min < 5", ns) is False


def test_the_namespace_uses_the_episodes_own_field_names(worst):
    """`duration_s` is the episode's `seconds` and `mark` is its `status` — the
    two renames between the catalog's spelling and the frozen HTTP contract's.
    A rule naming either is the thing that breaks if the rename drifts."""
    assert worst["duration_s"] == 4.0
    assert worst["mark"] == "unset"
    assert rules.match("duration_s > 3 and mark == 'unset'", worst) is True


# ---- tags -----------------------------------------------------------------

def test_a_tag_comparison_is_membership_over_every_tag():
    ns = rules.build_namespace(
        _episode([_arm("")], tags=["blurry", "dropped"]), "solo")

    assert rules.match("tag == 'blurry'", ns) is True
    assert rules.match("tag == 'dropped'", ns) is True
    assert rules.match("tag == 'dark'", ns) is False


def test_tags_is_the_same_name_as_tag():
    """The contract's table names the row `tags` and its example writes
    `tag == 'blurry'`. Both spellings work rather than one of them silently
    being an unknown name."""
    ns = rules.build_namespace(_episode([_arm("")], tags=["blurry"]), "solo")

    assert rules.match("tags == 'blurry'", ns) is True
    assert rules.match("tag == 'blurry'", ns) is True


def test_tag_not_equal_means_no_tag_equals_it():
    """THE ASYMMETRY. `tag != 'x'` is the negation of `tag == 'x'`, not
    `any(t != 'x')` — which would be true for a ['blurry', 'dark'] episode asked
    about 'blurry', and would make "the ones I have not called blurry" select
    the blurry ones."""
    tagged = rules.build_namespace(
        _episode([_arm("")], tags=["blurry", "dark"]), "solo")
    untagged = rules.build_namespace(_episode([_arm("")]), "solo")

    assert rules.match("tag != 'blurry'", tagged) is False
    assert rules.match("tag != 'green'", tagged) is True
    assert rules.match("tag != 'blurry'", untagged) is True
    assert rules.match("tag == 'blurry'", untagged) is False


def test_a_tag_only_compares_with_a_quoted_string():
    ns = rules.build_namespace(_episode([_arm("")], tags=["blurry"]), "solo")

    with pytest.raises(rules.RuleError) as excinfo:
        rules.match("tag == 5", ns)
    assert "text" in excinfo.value.message

    with pytest.raises(rules.RuleError) as excinfo:
        rules.match("tag < 'blurry'", ns)
    assert "membership" in excinfo.value.message


def test_a_plain_list_in_a_hand_built_namespace_still_means_membership():
    """`build_namespace` wraps tags, but a caller assembling a namespace by hand
    would reasonably pass the list straight through, and a list compared to a
    string is False forever with nothing said."""
    assert rules.match("tag == 'blurry'", {"tag": ["blurry", "dark"]}) is True
    assert rules.match("tag == 'green'", {"tag": ["blurry", "dark"]}) is False


# ---- booleans -------------------------------------------------------------

def test_true_and_false_are_names_because_the_grammar_has_no_boolean_literal(solo):
    """The frozen grammar's operands are NUMBER, STRING and IDENT. Binding
    `true`/`false` in the namespace keeps `reopened == true` inside that grammar
    instead of forcing `reopened == 1` on the operator."""
    assert rules.match("reopened == true", solo) is True
    assert rules.match("reopened == false", solo) is False
    assert rules.match("reopened == 1", solo) is True


# ---- hostile input --------------------------------------------------------

HOSTILE = [
    "__import__('os').system('x')",
    "().__class__",
    "().__class__.__base__.__subclasses__()",
    "1 if x else 2",
    "open('/etc/passwd')",
    "frames.__class__",
    "frames.__class__ == 'int'",
    "lambda: 1",
    "frames; import os",
    "import os",
    "frames > 5; print(1)",
    "eval('1')",
    "exec('x=1')",
    "compile('1', '', 'eval') == 1",
    "globals()",
    "[x for x in ()] == 1",
    "{}['a'] == 1",
    "frames[0] == 1",
    "frames + 1 > 2",
    "frames > 5 == True",
    "`frames`",
    "frames := 5",
]


@pytest.mark.parametrize("text", HOSTILE)
def test_hostile_input_is_a_rule_error_with_an_offset(solo, text):
    """None of these is a Python expression this module can run, because this
    module runs nothing: there is no call syntax, no subscript, no attribute
    access and no arithmetic in the grammar to abuse."""
    with pytest.raises(rules.RuleError) as excinfo:
        rules.match(text, solo)

    assert 0 <= excinfo.value.pos <= len(text)


def test_a_hostile_rule_touches_no_file(tmp_path, solo):
    """The one that would matter. If `open(...)` were ever reachable this file
    would exist, and the box it would be written on has NO BACKUP OF ANY KIND."""
    victim = tmp_path / "pwned"

    with pytest.raises(rules.RuleError):
        rules.match(f"open('{victim}', 'w') == 1", solo)
    with pytest.raises(rules.RuleError):
        rules.match(f"__import__('pathlib').Path('{victim}').touch() == 1", solo)

    assert not victim.exists()


def test_deeply_nested_parens_raise_rather_than_blowing_the_recursion_limit():
    """10 000 characters of nesting. Recursive descent recurses once per level,
    so without the cap this is a `RecursionError` — which is not a `RuleError`,
    carries no offset, and reaches the operator as a 500."""
    depth = 5_000
    text = "(" * depth + "frames > 1" + ")" * depth
    assert len(text) >= 10_000

    with pytest.raises(rules.RuleError) as excinfo:
        rules.parse(text)

    assert "nests" in excinfo.value.message
    assert str(rules.MAX_DEPTH) in excinfo.value.message
    assert excinfo.value.pos == rules.MAX_DEPTH


def test_a_deep_stack_of_nots_is_capped_too():
    """`not` recurses through the same descent, so it needs the same cap."""
    with pytest.raises(rules.RuleError) as excinfo:
        rules.parse("not " * 5_000 + "frames > 1")

    assert "nests" in excinfo.value.message


def test_a_long_flat_rule_is_fine(solo):
    """The depth cap is on NESTING, not on length: `and` is n-ary, so a thousand
    joined comparisons neither recurse nor get refused."""
    text = " and ".join(["frames > 1"] * 1_000)

    assert rules.match(text, solo) is True


def test_the_module_never_calls_eval_exec_or_compile():
    """Read back with `ast`, not with a grep: the module docstring says the
    words `eval`, `exec` and `compile` on purpose, and a textual check would
    either trip on that or be weakened until it stopped catching anything.

    This is the assertion that survives a future "simplification" of the parser
    into the one-line `eval` it exists instead of.

    `getattr`/`setattr` are in the banned set alongside the three obvious ones
    because a namespace lookup that reached for an attribute would put the DSL
    back in touch with the object graph it is kept away from — and because a
    convenience `getattr(rig, "rig", rig)` is exactly the harmless-looking line
    that erodes the rule. This module resolves names in a dict and nowhere else.
    """
    banned = {"eval", "exec", "compile", "__import__", "getattr", "setattr"}
    tree = ast.parse(Path(rules.__file__).read_text())

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = ast.unparse(node.func)
        # Both spellings: the bare builtin, and the same name reached through a
        # module (`builtins.eval`). `re.compile` is a regex, not a Python one,
        # and is the single allowed use of any of these words as a call.
        assert target not in banned, target
        assert target == "re.compile" or target.rpartition(".")[2] not in banned, target


def test_the_module_imports_nothing_that_could_run_anything():
    """stdlib and one sibling. No `os`, no `subprocess`, no `importlib` — and
    no `lerobot`/`torch`, which the serving process bans outright."""
    allowed = {"__future__", "operator", "re", "dataclasses", "typing", "schema"}
    tree = ast.parse(Path(rules.__file__).read_text())

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])

    assert imported <= allowed, imported


# ---- against a real graded episode ----------------------------------------

def test_the_namespace_lines_up_with_a_catalog_episode(tmp_path, monkeypatch):
    """End to end through `catalog.dataset_detail`, on the bimanual fixture
    whose RIGHT arm never moved.

    Every hand-built namespace above agrees with itself by construction. This is
    the one that fails if a key the catalog actually emits gets renamed: the
    episode goes in exactly as the detail view produced it, and `rig` goes in as
    the plain string the detail view carries.
    """
    home = tmp_path / "lerobot"
    root = home / "local" / "rules_bimanual"
    _dataset.make_dataset(root, n_episodes=1, rig="bimanual",
                          arm_content={"left": _dataset.CLEAN,
                                       "right": _dataset.STILL})
    monkeypatch.setenv("HF_LEROBOT_HOME", str(home))
    catalog._detail_cache.clear()
    catalog._frames_cache.clear()

    detail = catalog.dataset_detail("local/rules_bimanual")
    episode = detail["episodes"][0]
    ns = rules.build_namespace(episode, detail["rig"])

    assert ns["duration_s"] == episode["seconds"]
    assert ns["mark"] == "unset"
    assert ns["verdict"] == "FAIL"
    # The left arm swept 40°, the right one did not move at all.
    assert rules.match("left.sweep_total > 5", ns) is True
    assert rules.match("right.sweep_total > 5", ns) is False
    assert rules.match("sweep_total > 5", ns) is False
    assert rules.match("verdict == 'FAIL' and duration_s > 1", ns) is True


def test_a_rig_spec_may_be_passed_instead_of_its_name(tmp_path):
    """`build_namespace(episode, rig)` takes the `RigSpec` the catalog built or
    the `.rig` string it reports; the spec is read for one thing only, which is
    naming the rig when a rule asks for a side that does not exist."""
    root = tmp_path / "solo"
    _dataset.make_dataset(root, n_episodes=1)
    rig = RigSpec.from_info(json.loads((root / "meta" / "info.json").read_text()))
    assert rig.rig == "solo"

    ns = rules.build_namespace(_episode([_arm("")]), rig)

    with pytest.raises(rules.RuleError) as excinfo:
        rules.match("left.tracking > 1", ns)
    assert "'solo'" in excinfo.value.message
