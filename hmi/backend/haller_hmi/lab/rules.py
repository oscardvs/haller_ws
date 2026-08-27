# hmi/backend/haller_hmi/lab/rules.py
"""The operator-authored comparison DSL, parsed by hand because `eval` here is
remote code execution.

`POST /lab/datasets/autoclass/preview` is UNGATED, deliberately: it writes
nothing, and Oscar triages episodes from inside the headset over the LAN,
against a server started `--host 0.0.0.0` because that is how the Quest reaches
the HMI at all. So `params["reject_if"]` is an arbitrary string typed by anyone
who can reach the port, arriving on the machine that owns the servo bus.
`eval(text)` on it is not a shortcut, it is a shell on the robot.

`eval(text, {"__builtins__": {}}, ns)` is not the fix either, and this file says
so out loud because a great many people believe it is. The restricted-builtins
sandbox is escapable from a bare literal — `().__class__.__base__.__subclasses__()`
walks from an empty tuple to every class the interpreter has imported, and from
there to `os`. `compile()` and `exec()` are the same hole with more steps. There
is no safe subset; there is only not calling it. **`eval`, `exec` and `compile`
do not appear in this module, and `tests/lab/test_rules.py` reads this source
back to keep it that way.**

What replaces it is a tokeniser and a recursive-descent parser over the grammar
frozen in the contract, which has no call syntax, no subscript, no attribute
access and no arithmetic to abuse in the first place:

    expr       := or_expr
    or_expr    := and_expr ("or" and_expr)*
    and_expr   := not_expr ("and" not_expr)*
    not_expr   := "not" not_expr | primary
    primary    := "(" expr ")" | comparison
    comparison := operand OP operand
    OP         := "<" | "<=" | ">" | ">=" | "==" | "!="
    operand    := NUMBER | 'STRING' | IDENT ("." IDENT)*

Everything else is a parse error CARRYING THE CHARACTER OFFSET. The route turns
that into a 400 the operator reads as a toast in a headset, where "unexpected
'(' (at character 14)" tells them where to look and "invalid syntax" starts a
debugging session they cannot run from in there.

The namespace is flat and per episode (`build_namespace`), and its one load-
bearing rule is that a BARE name on a bimanual rig is the WORST arm's value, not
the first arm's — see `build_namespace`, where the reason is written down.
"""
from __future__ import annotations

import operator
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # only `.rig` is read off it, and only for error messages.
    from .schema import RigSpec

__all__ = [
    "ARM_NAMES",
    "MAX_DEPTH",
    "And",
    "Compare",
    "Expr",
    "Literal",
    "Name",
    "Not",
    "Or",
    "RuleError",
    "build_namespace",
    "evaluate",
    "match",
    "parse",
]

#: How deep `(` and `not` may nest. Recursive descent recurses once per level,
#: so an unbounded rule reaches Python's own recursion limit and raises
#: `RecursionError` — which is not a `RuleError`, carries no offset, and reaches
#: the operator as a 500 instead of a 400. 32 is far more nesting than a
#: readable rule has and far less than the interpreter's limit.
MAX_DEPTH = 32

#: The comparison operators, in the order error messages list them.
OPERATORS = ("<", "<=", ">", ">=", "==", "!=")

#: Matched before the single-character ones, or `<=` tokenises as `<` then `=`.
_TWO_CHAR_OPS = ("<=", ">=", "==", "!=")

#: Ordering operators. They need numbers on both sides: `verdict > 'FAIL'` would
#: otherwise compare two strings lexicographically and answer a question about
#: the alphabet that the operator meant to ask about severity.
_ORDERING = frozenset({"<", "<=", ">", ">="})

_KEYWORDS = frozenset({"and", "or", "not"})

_APPLY = {
    "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
    "==": operator.eq, "!=": operator.ne,
}

#: The per-arm values reachable as `left.<name>` / `right.<name>`. `sweep` is
#: excluded because it is a list and this DSL compares scalars; `closed_below` /
#: `open_above` are excluded because they are the thresholds a verdict was
#: reached WITH, not a measurement of the episode.
ARM_NAMES = (
    "tracking", "sweep_total", "closes", "grip_min", "grip_max",
    "reopened", "verdict",
)

#: Bare-name roll-ups across the arms, and the direction each one is worst in.
#: `all` for `reopened`: "every arm let go" is the only reading under which a
#: bare `reopened` is a statement about the episode rather than about whichever
#: arm the column order happened to put first.
_WORST_ARM = {
    "tracking": max,        # more error is worse
    "closes": max,          # more grasp attempts is worse
    "sweep_total": min,     # less motion is worse
    "grip_min": min,
    "grip_max": min,
    "reopened": all,
}

#: Rig name, stashed for the "no left arm on this rig" message. `$` is not in
#: the identifier charset, so no rule can ever name this key.
_RIG_KEY = "$rig"


class RuleError(ValueError):
    """A malformed or unusable rule, with the offset of the character to blame.

    A `ValueError` on purpose: `api/errors.py` already maps that to a 400, so a
    bad rule reaches the operator as bad input without the routes layer needing
    a rung of its own.
    """

    def __init__(self, message: str, pos: int) -> None:
        self.message = message
        self.pos = int(pos)
        super().__init__(f"{message} (at character {self.pos})")


# ---- tokens ---------------------------------------------------------------

_NUMBER, _STRING, _NAME, _KEYWORD, _OP, _LPAREN, _RPAREN, _EOF = (
    "number", "string", "name", "keyword", "op", "(", ")", "end")

# A dotted name is ONE token. `left.tracking` is a namespace key, not an
# attribute access on a `left` object — this module never touches a Python
# attribute, which is why `frames.__class__` is merely an unknown name here.
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")

# A leading `-` is part of the LITERAL, never an operation: Haller's gripper is
# calibrated to [-9.97, 100.27], so `grip_min < -5` is a rule an operator
# writes. `5 - 3` therefore tokenises as two numbers and fails to parse, which
# is the honest answer for a grammar with no arithmetic in it.
_NUMBER_RE = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


@dataclass(frozen=True)
class Token:
    kind: str
    value: Any
    pos: int
    text: str        # the source slice, so errors quote what was typed


def _tokenise(source: str) -> list[Token]:
    tokens: list[Token] = []
    i, n = 0, len(source)
    while i < n:
        ch = source[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "()":
            kind = _LPAREN if ch == "(" else _RPAREN
            tokens.append(Token(kind, ch, i, ch))
            i += 1
            continue
        pair = source[i:i + 2]
        if pair in _TWO_CHAR_OPS:
            tokens.append(Token(_OP, pair, i, pair))
            i += 2
            continue
        if ch in "<>":
            tokens.append(Token(_OP, ch, i, ch))
            i += 1
            continue
        if ch == "=":
            raise RuleError("'=' does not compare anything — write '==' to test equality", i)
        if ch == "!":
            raise RuleError("'!' on its own is not an operator — write '!=' for 'is not'", i)
        if ch in "'\"":
            # No escape processing, deliberately: a tag or a verdict never
            # contains a quote, and an escape grammar is one more place for a
            # clever string to mean something other than it looks like.
            end = source.find(ch, i + 1)
            if end < 0:
                raise RuleError(f"unterminated string — no closing {ch}", i)
            tokens.append(Token(_STRING, source[i + 1:end], i, source[i:end + 1]))
            i = end + 1
            continue
        if ch.isdigit() or ch in "-.":
            m = _NUMBER_RE.match(source, i)
            if m:
                tokens.append(Token(_NUMBER, float(m.group()), i, m.group()))
                i = m.end()
                continue
        m = _NAME_RE.match(source, i)
        if m:
            word = m.group()
            kind = _KEYWORD if word in _KEYWORDS else _NAME
            tokens.append(Token(kind, word, i, word))
            i = m.end()
            continue
        raise RuleError(f"unexpected character {ch!r}", i)
    tokens.append(Token(_EOF, None, n, ""))
    return tokens


# ---- syntax tree ----------------------------------------------------------
# Frozen dataclasses, and `And`/`Or` are N-ARY rather than a chain of binary
# nodes. That is not tidiness: a binary chain of 5 000 `and`s parses in a loop
# but EVALUATES by recursing 5 000 deep, so the depth cap on parentheses would
# not have protected the evaluator at all.

@dataclass(frozen=True)
class Literal:
    value: float | str
    text: str
    pos: int


@dataclass(frozen=True)
class Name:
    name: str
    pos: int


@dataclass(frozen=True)
class Compare:
    op: str
    left: Literal | Name
    right: Literal | Name
    pos: int         # the operator's offset — where a type error is reported


@dataclass(frozen=True)
class Not:
    operand: Any
    pos: int


@dataclass(frozen=True)
class And:
    parts: tuple[Any, ...]
    pos: int


@dataclass(frozen=True)
class Or:
    parts: tuple[Any, ...]
    pos: int


Expr = Compare | Not | And | Or
Operand = Literal | Name


# ---- parser ---------------------------------------------------------------

def _what(tok: Token) -> str:
    return "the end of the rule" if tok.kind == _EOF else repr(tok.text)


def _case_hint(tok: Token) -> str:
    """`AND` is the mistake this catches. It tokenises as a name, so without the
    hint the operator is told their comparison is missing an operator — true,
    but not the thing they got wrong."""
    if tok.kind == _NAME and tok.value.lower() in _KEYWORDS:
        return " — write 'and', 'or' and 'not' in lower case"
    return ""


def _deeper(depth: int, pos: int) -> int:
    if depth >= MAX_DEPTH:
        raise RuleError(f"the rule nests more than {MAX_DEPTH} levels deep", pos)
    return depth + 1


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._i = 0

    def _peek(self) -> Token:
        return self._tokens[self._i]

    def _take(self) -> Token:
        tok = self._tokens[self._i]
        if tok.kind != _EOF:      # EOF is never consumed, so _peek always works
            self._i += 1
        return tok

    def parse(self) -> Expr:
        node = self._expr(0)
        tok = self._peek()
        if tok.kind != _EOF:
            raise RuleError(
                f"unexpected {_what(tok)} — the rule was already complete before "
                f"it; join comparisons with 'and' / 'or'{_case_hint(tok)}", tok.pos)
        return node

    def _expr(self, depth: int) -> Expr:
        return self._or(depth)

    def _or(self, depth: int) -> Expr:
        parts = [self._and(depth)]
        while self._peek().kind == _KEYWORD and self._peek().value == "or":
            self._take()
            parts.append(self._and(depth))
        return parts[0] if len(parts) == 1 else Or(tuple(parts), parts[0].pos)

    def _and(self, depth: int) -> Expr:
        parts = [self._not(depth)]
        while self._peek().kind == _KEYWORD and self._peek().value == "and":
            self._take()
            parts.append(self._not(depth))
        return parts[0] if len(parts) == 1 else And(tuple(parts), parts[0].pos)

    def _not(self, depth: int) -> Expr:
        tok = self._peek()
        if tok.kind == _KEYWORD and tok.value == "not":
            self._take()
            return Not(self._not(_deeper(depth, tok.pos)), tok.pos)
        return self._primary(depth)

    def _primary(self, depth: int) -> Expr:
        tok = self._peek()
        if tok.kind == _LPAREN:
            self._take()
            inner = self._expr(_deeper(depth, tok.pos))
            close = self._peek()
            if close.kind != _RPAREN:
                raise RuleError(
                    f"expected ')' to close the '(' opened at character {tok.pos}, "
                    f"got {_what(close)}{_case_hint(close)}", close.pos)
            self._take()
            return inner
        return self._comparison()

    def _comparison(self) -> Compare:
        left = self._operand()
        tok = self._peek()
        if tok.kind != _OP:
            # This is where a function call dies: `open('/etc/passwd')` parses
            # `open` as a name and then finds `(` where an operator belongs.
            raise RuleError(
                f"expected a comparison operator ({', '.join(OPERATORS)}), got "
                f"{_what(tok)}{_case_hint(tok)}", tok.pos)
        self._take()
        right = self._operand()
        return Compare(str(tok.value), left, right, tok.pos)

    def _operand(self) -> Operand:
        tok = self._peek()
        if tok.kind in (_NUMBER, _STRING):
            self._take()
            return Literal(tok.value, tok.text, tok.pos)
        if tok.kind == _NAME:
            self._take()
            return Name(str(tok.value), tok.pos)
        raise RuleError(
            f"expected a number, a quoted string or a name, got {_what(tok)}"
            f"{_case_hint(tok)}", tok.pos)


def parse(text: str) -> Expr:
    """Parse one rule. Raises `RuleError(message, pos)` on anything malformed."""
    source = text if isinstance(text, str) else str(text or "")
    if not source.strip():
        raise RuleError("the rule is empty — write a comparison like frames < 60", 0)
    return _Parser(_tokenise(source)).parse()


# ---- values ---------------------------------------------------------------

class _Tags:
    """A set-valued operand: `tag == 'blurry'` asks whether ANY tag matches.

    Membership is the only thing tags support, so the identifier is bound to
    this rather than to a list — a list would compare equal to nothing an
    operator can type.
    """

    __slots__ = ("values",)

    def __init__(self, values) -> None:
        self.values = tuple(str(v) for v in values)

    def __repr__(self) -> str:
        return f"tags{self.values!r}"


def _source(node: Operand) -> str:
    return node.name if isinstance(node, Name) else node.text


def _describe(node: Operand, value: Any) -> str:
    """`verdict ('PASS')` for a name, `3` for a literal — an error about a
    comparison has to show both what was typed and what it turned out to be."""
    return f"{_source(node)} ({value!r})" if isinstance(node, Name) else node.text


def _unknown_name(node: Name, ns: dict) -> RuleError:
    """Never a silent False. A rule that quietly matches nothing because of a
    typo reads exactly like a dataset with no episodes to fix."""
    name = node.name
    visible = sorted(k for k in ns if not k.startswith("$"))
    if "." in name:
        side = name.partition(".")[0]
        sides = sorted({k.partition(".")[0] for k in visible if "." in k})
        if side not in sides:
            rig = ns.get(_RIG_KEY) or ""
            whose = f"this dataset's rig is {rig!r}" if rig else "this dataset"
            has = ", ".join(sides) if sides else (
                "none — this rig has a single unprefixed arm, so use the bare names")
            return RuleError(
                f"there is no {side!r} arm to read {name!r} from: {whose} "
                f"(arms: {has})", node.pos)
        fields = sorted(k.partition(".")[2] for k in visible if k.startswith(side + "."))
        return RuleError(
            f"unknown name {name!r} — {side} has: {', '.join(fields)}", node.pos)
    return RuleError(
        f"unknown name {name!r} — valid names are: {', '.join(visible)}", node.pos)


def _resolve(node: Operand, ns: dict) -> Any:
    if isinstance(node, Literal):
        return node.value
    if node.name not in ns:
        raise _unknown_name(node, ns)
    value = ns[node.name]
    # A hand-built namespace may hand `tags` through as a plain list; treat any
    # collection as membership rather than comparing a list to a string and
    # answering False forever.
    if isinstance(value, (list, tuple, set, frozenset)):
        return _Tags(value)
    return value


def _kind(value: Any) -> str | None:
    # `bool` is an `int` and is left that way on purpose: `reopened == 1` and
    # `reopened == true` then both work, and `reopened == 'yes'` is still the
    # text-versus-number error it should be.
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "text"
    return None


def _compare_tags(node: Compare, left: Any, right: Any) -> bool:
    if isinstance(left, _Tags) and isinstance(right, _Tags):
        raise RuleError(
            "compare tag with a quoted tag name, e.g. tag == 'blurry'", node.pos)
    if isinstance(left, _Tags):
        tags, other, other_node = left, right, node.right
    else:
        tags, other, other_node = right, left, node.left
    if node.op not in ("==", "!="):
        raise RuleError(
            f"tags are a membership test, so {node.op!r} is not defined on them "
            "— use tag == 'blurry' or tag != 'blurry'", node.pos)
    if not isinstance(other, str):
        raise RuleError(
            f"a tag is text; it cannot be tested against "
            f"{_describe(other_node, other)}", node.pos)
    hit = any(t == other for t in tags.values)
    # THE ASYMMETRY, and it is deliberate: `tag != 'x'` is the NEGATION of
    # `tag == 'x'` — "no tag equals x" — and NOT `any(t != 'x')`, which would be
    # true for a `['blurry', 'dark']` episode asked about 'blurry'. An operator
    # writing `tag != 'blurry'` means "the ones I have not called blurry".
    return hit if node.op == "==" else not hit


def _compare(node: Compare, ns: dict) -> bool:
    left = _resolve(node.left, ns)
    right = _resolve(node.right, ns)
    if isinstance(left, _Tags) or isinstance(right, _Tags):
        return _compare_tags(node, left, right)

    for operand, value in ((node.left, left), (node.right, right)):
        if value is None:
            # `grade.py` uses None for UNMEASURABLE, not for zero. Answering
            # False here would silently pass every episode of a gripperless rig
            # through a grasp rule; the loud version tells the operator their
            # rule does not apply to this dataset.
            raise RuleError(
                f"{_source(operand)} is not measured on this dataset, so there "
                "is nothing to compare — an arm with no gripper column has no "
                "grasp numbers", node.pos)
        if _kind(value) is None:
            raise RuleError(
                f"{_source(operand)} is a {type(value).__name__}, which cannot be "
                "compared", node.pos)

    if _kind(left) != _kind(right):
        raise RuleError(
            f"cannot compare {_describe(node.left, left)} with "
            f"{_describe(node.right, right)} — text and numbers are never "
            "comparable", node.pos)
    if _kind(left) == "text" and node.op in _ORDERING:
        raise RuleError(
            f"{node.op!r} needs numbers, and {_describe(node.left, left)} is text "
            "— verdict, mark and tag compare with == and != only", node.pos)
    return bool(_APPLY[node.op](left, right))


def _eval(node: Expr, ns: dict) -> bool:
    if isinstance(node, Compare):
        return _compare(node, ns)
    if isinstance(node, Not):
        return not _eval(node.operand, ns)
    if isinstance(node, (And, Or)):
        # Every branch is evaluated, no short-circuit. There is nothing to skip
        # past — no calls, no side effects, no cost — and short-circuiting would
        # make a bad comparison raise on some episodes and not others, which is
        # the hardest kind of 400 to reproduce from a headset.
        results = [_eval(part, ns) for part in node.parts]
        return all(results) if isinstance(node, And) else any(results)
    raise TypeError(f"not a rule node: {node!r}")


def evaluate(expr: Expr, namespace: dict) -> bool:
    """Evaluate a parsed rule against one episode's namespace.

    Raises `RuleError` for an unknown name or an impossible comparison — those
    are mistakes in the RULE, which is why they are the same error class the
    parser raises and reach the operator as the same 400.
    """
    return _eval(expr, namespace)


def match(text: str, namespace: dict) -> bool:
    """`parse` + `evaluate`, for a one-shot check.

    Autoclassifying a dataset parses ONCE and evaluates per episode; this is for
    the single-episode caller, and for tests.
    """
    return _eval(parse(text), namespace)


# ---- namespace ------------------------------------------------------------

def build_namespace(episode: dict, rig: RigSpec | str) -> dict:
    """The flat per-episode namespace, from one `catalog.dataset_detail` episode.

    **A bare name is the WORST arm's value, not the first arm's.** `tracking > 5`
    has to mean "either arm failed to track". If it meant "whichever arm the
    column order put first failed to track", then on Haller's 12-dim state —
    where columns 0..5 are the LEFT arm — an operator's reject rule would pass a
    dataset whose right arm was dead, silently, for exactly the reason the
    per-arm rewrite of `grade.py` exists: the kit's `GRIPPER_IDX = 5` graded the
    left arm by coincidence and never looked at the right one at all. So
    `tracking` and `closes` roll up with `max` (more is worse), `sweep_total`,
    `grip_min` and `grip_max` with `min` (less is worse), and `reopened` with
    `all` (true only when EVERY arm let go). Reach one named arm with
    `left.tracking` / `right.tracking`.

    An arm that cannot supply a value (no gripper column) is left OUT of the
    roll-up rather than counted as zero; if no arm can supply it the bare name
    is None, and comparing it raises rather than quietly answering False.

    `rig` may be a `RigSpec` or its `.rig` string. It is used for one thing:
    naming the rig when a rule asks for a side this dataset does not have.
    """
    arms = [a for a in (episode.get("arms") or ()) if isinstance(a, dict)]
    # Both spellings, bound to the same value: the contract's table calls the
    # row `tags` and its example writes `tag == 'blurry'`, and an operator
    # should not have to know which of the two the parser wanted.
    tags = _Tags(episode.get("tags") or ())

    # Spelled out rather than reached for with `getattr`: this module touches no
    # Python attribute anywhere, and `tests/lab/test_rules.py` holds it to that.
    if isinstance(rig, str):
        rig_name = rig
    elif rig is None:
        rig_name = ""
    else:
        rig_name = str(rig.rig or "")

    ns: dict[str, Any] = {
        _RIG_KEY: rig_name,
        "frames": episode.get("frames"),
        "duration_s": episode.get("seconds"),   # `seconds` on a catalog episode
        "share": episode.get("share"),
        "verdict": episode.get("verdict"),
        "mark": episode.get("status"),          # `status` on a catalog episode
        "tag": tags,
        "tags": tags,
        # The frozen grammar has no boolean literal, so without these the only
        # way to write "this arm never let go" is `reopened == 0`, which is a
        # comparison the operator has to stop and think about.
        "true": True,
        "false": False,
    }

    for name, worst in _WORST_ARM.items():
        values = [a.get(name) for a in arms]
        values = [v for v in values if v is not None]
        ns[name] = worst(values) if values else None

    for arm in arms:
        side = str(arm.get("side") or "")
        if not side:
            # The unprefixed solo arm. `.tracking` is not an identifier, and its
            # values are already the bare names above.
            continue
        for name in ARM_NAMES:
            ns[f"{side}.{name}"] = arm.get(name)
    return ns
