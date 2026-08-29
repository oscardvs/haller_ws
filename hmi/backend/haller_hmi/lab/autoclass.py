# hmi/backend/haller_hmi/lab/autoclass.py
"""Four ways to PROPOSE marks, and one preview → apply → revert contract over
all of them.

Nothing here is a port; the kit has no autoclassifier. What it has is a button
that graded 46 episodes and wrote 46 marks, and the whole shape of this module
is a reaction to that: every mode produces a DIFF the operator reads first, and
`apply` is a second, separately gated call. An autoclassifier that silently
rewrites 46 marks is worse than none, because the marks it overwrites are the
only record of decisions a human actually made.

Three things follow from that, and each of them is a constraint rather than a
preference.

**SUSPECT is left alone by `grade`.** FAIL → reject and PASS → keep, and the
third rung is deliberately missing. SUSPECT means "worth looking at"; resolving
it for the operator converts a request to look into a decision they did not
make. That line is the design, not an omission — see `_plan_grade`.

**`policy-loss` never marks anything.** It returns `diff: []` unconditionally
and a ranking, and `apply` on it raises. A high loss is as often a
rare-but-correct demonstration as a bad one, and deleting the tail of the loss
distribution is how a policy loses the only examples of the case it fails.

**The token is the whole safety story of apply.** The operator confirmed a diff
computed against one dataset state; applying it to a different state applies
decisions they never saw. So `preview` binds `(repo_id, mode, params, every
episode's frame count, every current mark)` into a sha256, `apply` recomputes
it against the dataset as it is NOW, and a mismatch refuses.

## Why the token carries its own plan, and why nothing is stored

`apply(repo_id, token)` is handed no mode and no params — so the token has to
carry them. It is `<base64url(plan)>.<sha256>`: the plan half says what to run,
the digest half says which dataset state it was confirmed against. `apply`
re-runs the plan, re-derives the digest from the live dataset and compares.

NOTHING is stored between preview and apply. There is no pending-diff table and
no session, which means a backend restart mid-triage does not invalidate a
token — there was nothing to lose. It also means the diff is RECOMPUTED at
apply rather than replayed: with the digest matching, the inputs are the same,
so the recomputed diff is the one that was shown, and the token stays a few
hundred bytes instead of growing with the dataset.

**The digest is not a signature and must never be treated as one.** It is
unkeyed, so anyone who can reach the port can mint a token — which is fine,
because `apply` is `require_local`-gated and a forged token can only ask for
something the operator could have asked for directly. Adding an HMAC here would
suggest this is an authorisation boundary. It is a staleness guard.

## The apply/revert pair

The batch is `uuid4().hex[:12]` and is recorded with EVERY prior entry,
including "there was no entry" stored as `null`, so revert restores ABSENCE
rather than manufacturing an `unset` mark nobody chose. `review.record_batch`
runs BEFORE the marks are written: a crash between the two leaves an undo
record for a change that never happened (a revert that restores what is already
there), while the other order would leave changes with no undo at all.

## numpy only

sklearn is banned outright and torch is banned package-wide — this module is
imported by the serving process, which is the teleop latency path. kNN over 46
episodes and ~29 features is a z-score, a dot product and a sort; a dependency
that pulls in scipy to do that would cost more import time than the whole Lab.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import uuid
from pathlib import Path

import numpy as np

from . import catalog, rules
from . import review as review_mod
from .schema import RigSpec

__all__ = [
    "EPISODE_LOSS_FILENAME",
    "KNN_K",
    "KNN_MIN_CONFIDENCE",
    "MODES",
    "PROPAGATE",
    "StaleTokenError",
    "apply",
    "preview",
    "revert",
]

MODES = ("grade", "rules", "knn", "policy-loss")

#: kNN defaults, from the contract. `k` is small because a review in progress
#: has few marks to learn from, not because 5 is tuned.
KNN_K = 5
KNN_MIN_CONFIDENCE = 0.6

#: What kNN may propagate. `mark` writes a keep/reject; `tags` writes tags and
#: leaves the mark exactly where it was.
PROPAGATE = ("mark", "tags")

#: Per-episode loss, written by a training/eval runner into its own run
#: directory. LeRobot does not log this — the mode is WIRED and DATA-GATED, and
#: reports `available: false` rather than inventing a proxy metric.
EPISODE_LOSS_FILENAME = "episode_loss.jsonl"

#: Percentile of |action - state| kept alongside the mean. The mean says how
#: well the arm tracked on average; the p95 says whether it lost the target for
#: a moment, which is the difference between a good demo and one with a stall
#: in it that averaging hides.
TRACKING_PERCENTILE = 95


class StaleTokenError(RuntimeError):
    """The dataset moved between preview and apply. The route answers 409.

    A `RuntimeError` and deliberately NOT a `ValueError`: `api/errors.py` maps
    `ValueError` to 400, and 400 tells the operator their request was
    malformed. It was not — the token was well-formed and correct when it was
    issued, and the world changed under it. That is a conflict, and the honest
    answer is "re-run the preview and look at the new diff".

    `api/errors.as_http()` has no rung for this class, so the route that
    exposes `apply` must catch it and raise `HTTPException(409, ...)` itself.
    """


# ---- token ----------------------------------------------------------------

def _canonical(obj) -> str:
    """The one JSON spelling everything hashes through.

    `sort_keys` so two dicts that differ only in key order hash the same, and
    `allow_nan=False` so a NaN that reached `params` raises here rather than
    travelling on as the literal `NaN`, which is not JSON and which reaches the
    browser as a parse error on a page that was working a second ago.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _state(detail: dict) -> tuple[dict, dict]:
    """The dataset state the token is computed against: every episode's length
    and every episode's current mark.

    Unmarked episodes are included as `unset` on purpose — an episode marked
    between preview and apply must change the digest, and leaving the unmarked
    ones out would make "nothing was marked" and "one episode was marked keep
    and another unmarked" collide in the same shape.
    """
    frames = {str(e["episode_index"]): int(e["frames"]) for e in detail["episodes"]}
    marks = {str(e["episode_index"]): str(e.get("status") or review_mod.UNSET)
             for e in detail["episodes"]}
    return frames, marks


def _digest(repo_id: str, mode: str, params: dict, frames: dict, marks: dict) -> str:
    return hashlib.sha256(
        _canonical([str(repo_id), mode, params, frames, marks]).encode("utf-8")
    ).hexdigest()


def _encode_token(repo_id: str, mode: str, params: dict,
                  frames: dict, marks: dict) -> str:
    plan = _canonical({"mode": mode, "params": params}).encode("utf-8")
    # Unpadded urlsafe base64: the token travels in a JSON body and in nothing
    # else, but '=' and '+' in a value that looks like an id invite exactly one
    # person to paste it into a query string.
    head = base64.urlsafe_b64encode(plan).decode("ascii").rstrip("=")
    return f"{head}.{_digest(repo_id, mode, params, frames, marks)}"


_BAD_TOKEN = (
    "that is not an autoclassify token — re-run the preview and apply the "
    "token it returns"
)


def _decode_token(token: str) -> tuple[str, dict, str]:
    """`(mode, params, digest)`, or a ValueError the route answers 400 with.

    A malformed token is BAD INPUT (400); a well-formed one whose digest no
    longer matches is a CONFLICT (409, `StaleTokenError`). Keeping the two
    apart is what lets the page tell "you sent something that is not a token"
    from "someone marked an episode while you were reading the diff".
    """
    head, sep, digest = str(token or "").partition(".")
    if not sep or not head or len(digest) != 64:
        raise ValueError(_BAD_TOKEN)
    if any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(_BAD_TOKEN)
    try:
        raw = base64.urlsafe_b64decode(head + "=" * (-len(head) % 4))
        plan = json.loads(raw.decode("utf-8"))
    except Exception as e:
        # base64, utf-8 and json all mean one thing here: not our token.
        raise ValueError(_BAD_TOKEN) from e
    if not isinstance(plan, dict):
        # ValueError, not TypeError: this arrives from an HTTP body, so it is
        # bad INPUT and the ladder in api/errors.py must answer 400, not 500.
        raise ValueError(_BAD_TOKEN)  # noqa: TRY004
    return _check_mode(plan.get("mode")), _check_params(plan.get("params")), digest


# ---- parameter checking ---------------------------------------------------

def _check_mode(mode) -> str:
    text = str(mode or "").strip()
    if text not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, got {mode!r}")
    return text


def _check_params(params) -> dict:
    if params is None:
        return {}
    if not isinstance(params, dict):
        # ValueError for the same reason `_decode_token` gives: 400, not 500.
        raise ValueError(  # noqa: TRY004
            f"params must be an object, got {type(params).__name__}")
    return dict(params)


def _knn_params(params: dict) -> tuple[int, float, str]:
    try:
        k = int(params.get("k", KNN_K))
    except (TypeError, ValueError):
        raise ValueError(f"k must be a whole number, got {params.get('k')!r}") from None
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    try:
        min_conf = float(params.get("min_confidence", KNN_MIN_CONFIDENCE))
    except (TypeError, ValueError):
        raise ValueError(
            f"min_confidence must be a number, got {params.get('min_confidence')!r}"
        ) from None
    if not math.isfinite(min_conf) or not 0.0 <= min_conf <= 1.0:
        raise ValueError(f"min_confidence must be between 0 and 1, got {min_conf}")
    propagate = str(params.get("propagate") or PROPAGATE[0]).strip()
    if propagate not in PROPAGATE:
        raise ValueError(
            f"propagate must be one of {', '.join(PROPAGATE)}, got {propagate!r}")
    return k, min_conf, propagate


# ---- public API -----------------------------------------------------------

def preview(repo_id: str, mode: str, params: dict | None = None) -> dict:
    """Compute a diff without writing anything.

    Ungated by the frozen contract, because it writes nothing and Oscar triages
    episodes from inside the headset. That is also why `rules` goes through
    `lab/rules.py`'s hand-written parser and never `eval`: this call is reachable
    from the LAN, on the machine that owns the servo bus.
    """
    mode = _check_mode(mode)
    params = _check_params(params)
    detail = catalog.dataset_detail(repo_id)
    frames, marks = _state(detail)

    out: dict = {
        "mode": mode,
        "token": _encode_token(repo_id, mode, params, frames, marks),
        "diff": [],
    }
    if mode == "policy-loss":
        out.update(_plan_policy_loss(detail, params))
    else:
        out["diff"] = _plan(mode, detail, params)
    return out


def apply(repo_id: str, token: str) -> dict:
    """Write the marks a previewed diff proposed, recording an undo batch first.

    Gated (`require_local`). Raises `StaleTokenError` when the dataset moved
    since the preview, and `ValueError` for `policy-loss`, which never marks.
    """
    mode, params, digest = _decode_token(token)
    if mode == "policy-loss":
        # Refused before the digest is even checked: re-running the preview
        # would not make this appliable, so "your token is stale" would send
        # the operator round a loop that has no exit.
        raise ValueError(
            "policy-loss is a sort order, never a mark — there is nothing to "
            "apply. A high loss is as often a rare-but-correct demonstration "
            "as a bad one; use the ranking to choose what to WATCH."
        )

    detail = catalog.dataset_detail(repo_id)
    frames, marks = _state(detail)
    if _digest(repo_id, mode, params, frames, marks) != digest:
        raise StaleTokenError(
            f"{repo_id} changed since that preview was computed — an episode "
            "was marked, added or pruned. Re-run the preview and read the new "
            "diff before applying it."
        )

    plan = _plan(mode, detail, params)
    if not plan:
        # No batch is recorded for an empty apply. `review.MAX_BATCHES` is 20,
        # so an undo entry that restores nothing would push a real one off the
        # end of the window the operator can actually reach.
        return {"applied": 0, "batch": ""}

    root = Path(detail["root"])
    stored = (review_mod.load(root).get("episodes") or {})
    before = {
        int(entry["episode"]): (stored.get(str(entry["episode"])) or None)
        for entry in plan
    }
    batch_id = uuid.uuid4().hex[:12]
    # Undo FIRST. A crash between these two lines leaves a batch that restores
    # what is already there; the other order leaves changes nothing can undo.
    review_mod.record_batch(root, batch_id, mode, before)

    episode_frames = {int(e["episode_index"]): int(e["frames"])
                      for e in detail["episodes"]}
    if mode == "knn" and _knn_params(params)[2] == "tags":
        applied = _apply_tags(root, plan, episode_frames)
    else:
        # Per-mark frame counts go in with the marks: a mark that does not
        # record the length of the episode it was made about cannot be told
        # from a mark that survived a prune (see `review.stale_marks`).
        review_mod.set_many(
            root,
            {int(e["episode"]): e["to"] for e in plan},
            fingerprint=detail.get("fingerprint") or None,
            episode_frames=episode_frames,
        )
        applied = len(plan)
    return {"applied": applied, "batch": batch_id}


def revert(repo_id: str, batch: str) -> dict:
    """Restore every mark a batch overwrote, absence included.

    Resolves the dataset by PATH rather than through `dataset_detail`: undoing a
    mark must keep working while the parquet is mid-write, and a detail read
    would refuse with `DatasetBusyError` for a reason that has nothing to do
    with the sidecar this touches.
    """
    root = catalog.dataset_root(repo_id)
    if not root.exists():
        raise FileNotFoundError(f"no dataset at {root}")
    return {"reverted": review_mod.revert_batch(root, batch)}


# ---- planning -------------------------------------------------------------

def _plan(mode: str, detail: dict, params: dict) -> list[dict]:
    if mode == "grade":
        return _plan_grade(detail)
    if mode == "rules":
        return _plan_rules(detail, params)
    if mode == "knn":
        return _plan_knn(detail, params)
    raise ValueError(f"mode {mode!r} produces no diff")  # pragma: no cover


def _entry(episode: dict, to: str, why: str, confidence: float) -> dict:
    """One diff row. `episode` is the STORED index; the UI renders it as
    `Ep {index+1} (idx {index})`, because Oscar counts episodes 1-based in
    conversation and that off-by-one is how the wrong demonstration gets
    deleted."""
    return {
        "episode": int(episode["episode_index"]),
        "from": str(episode.get("status") or review_mod.UNSET),
        "to": to,
        "why": why,
        "confidence": _finite(confidence),
    }


#: `grade`'s verdict → mark map. SUSPECT is ABSENT, and that absence is the
#: design: it means "worth looking at", and a classifier that resolves it has
#: converted a request to look into a decision the operator did not make. Do not
#: add a third key here without changing the contract first.
_VERDICT_MARK = {"FAIL": review_mod.REJECT, "PASS": review_mod.KEEP}


def _plan_grade(detail: dict) -> list[dict]:
    """The rule ladder already shown in the detail view, applied.

    `confidence` is 1.0 by definition — the ladder is deterministic. The field
    is here so one UI component renders all four modes.
    """
    out = []
    for ep in detail["episodes"]:
        to = _VERDICT_MARK.get(str(ep.get("verdict")))
        if to is None or to == ep.get("status"):
            continue
        reasons = "; ".join(str(r) for r in ep.get("reasons") or ())
        out.append(_entry(ep, to, f"{ep['verdict']}: {reasons}" if reasons
                          else str(ep.get("verdict")), 1.0))
    return out


def _plan_rules(detail: dict, params: dict) -> list[dict]:
    """An operator-authored rule per outcome, parsed ONCE and evaluated per
    episode.

    `reject_if` is evaluated first and an episode it matches is never considered
    for `keep_if` — the operator wrote the reject rule to carve exceptions out
    of the keep rule, and evaluating them the other way round would silently
    invert that.

    Unlike kNN this is allowed to overwrite an existing mark: a rule is an
    explicit statement about the whole dataset, and an operator who writes
    `reject_if: tracking > 5` after marking by hand means it.
    """
    reject_src = str(params.get("reject_if") or "").strip()
    keep_src = str(params.get("keep_if") or "").strip()
    if not reject_src and not keep_src:
        raise ValueError(
            "nothing to match: pass reject_if, keep_if or both, e.g. "
            "{\"reject_if\": \"verdict == 'FAIL'\"}")

    # Parsed here, so a syntax error is one RuleError with one offset rather
    # than the same error raised again for every episode in the dataset.
    reject = rules.parse(reject_src) if reject_src else None
    keep = rules.parse(keep_src) if keep_src else None
    rig = str(detail.get("rig") or "")

    out = []
    for ep in detail["episodes"]:
        ns = rules.build_namespace(ep, rig)
        if reject is not None and rules.evaluate(reject, ns):
            to, why = review_mod.REJECT, f"reject_if: {reject_src}"
        elif keep is not None and rules.evaluate(keep, ns):
            to, why = review_mod.KEEP, f"keep_if: {keep_src}"
        else:
            continue
        if to == ep.get("status"):
            continue
        out.append(_entry(ep, to, why, 1.0))
    return out


# ---- kNN ------------------------------------------------------------------

def _finite(value) -> float:
    """Every float that leaves this module goes through here.

    NaN is not JSON. A single NaN in a response body is not a wrong number on
    the page, it is a parse error that blanks the whole review — so an
    unrepresentable value becomes 0.0 at the boundary rather than downstream.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def _tracking_percentiles(root: Path, rig: RigSpec) -> dict[int, dict[str, float]]:
    """p95 of |action - state| per arm, per episode.

    The catalog's per-arm block already carries the MEAN; the percentile needs
    the raw frames, so this reads `catalog._frames` — the SAME cached
    action/state table `dataset_detail` just graded from. On a cache hit it is a
    numpy pass and no disk read at all; on a miss it costs the one parquet read
    the detail view would have paid anyway. The alternative, re-deriving it from
    `episode_trace`, would answer with a stride-downsampled p95, which is a
    different number wearing the same name.

    Measured on `local/so101_pick_cube` (46 episodes, 29 500 frames): the whole
    feature matrix takes 27 ms with the frames table warm and 42 ms cold, which
    is why a preview does not need a cache of its own.

    Same columns as `grade.tracking`: an arm's non-gripper joints only, so mean
    and p95 describe one block and can be compared with each other.
    """
    df = catalog._frames(root)
    out: dict[int, dict[str, float]] = {}
    for ep, sub in df.groupby("episode_index"):
        state = np.stack(sub["observation.state"].to_numpy())
        action = np.stack(sub["action"].to_numpy())
        err = np.abs(action - state)
        per_arm: dict[str, float] = {}
        for arm in rig.arms:
            idx = list(arm.joint_idx)
            per_arm[arm.side] = (
                float(np.percentile(err[:, idx], TRACKING_PERCENTILE)) if idx else 0.0
            )
        out[int(ep)] = per_arm
    return out


def _feature_matrix(detail: dict) -> np.ndarray:
    """One row per episode: 3 episode-level columns, then a block per arm.

    Per arm: tracking_mean, tracking_p95, sweep_total, closes, reopened,
    grip_min, grip_max, grip_range, then that arm's per-joint sweeps. The width
    is FIXED within one dataset because the rig is, so nothing is padded.

    A value the rig cannot supply — `grip_min` on an arm with no gripper column
    — becomes 0.0 rather than NaN. It is the same 0.0 for every episode, so the
    column has zero variance and contributes nothing to any distance, which is
    the correct weight for a measurement that does not exist here.

    Measured widths: **16 on a solo rig, 29 on a bimanual one** — 3 + 13 per
    arm, an SO-101 arm having 5 non-gripper joints. The contract's addendum
    labels the per-arm block 14 and then lists 13 named features (and so says
    17 / 31); the eight scalars plus five sweeps below are the named ones, and
    no fourteenth is invented to make the count come out.
    """
    # `from_info` reads `features["observation.state"]` and nothing else, and
    # the joint columns are all this needs — the gripper CALIBRATION only moves
    # the grade thresholds, which are not a tracking statistic.
    rig = RigSpec.from_info({"features": detail.get("features") or {}})
    p95 = _tracking_percentiles(Path(detail["root"]), rig)

    rows: list[list[float]] = []
    for ep in detail["episodes"]:
        row = [_finite(ep.get("frames")), _finite(ep.get("seconds")),
               _finite(ep.get("share"))]
        per_arm = p95.get(int(ep["episode_index"]), {})
        for arm in ep.get("arms") or ():
            grip_min, grip_max = arm.get("grip_min"), arm.get("grip_max")
            grip_range = (None if grip_min is None or grip_max is None
                          else grip_max - grip_min)
            row += [
                _finite(arm.get("tracking")),
                _finite(per_arm.get(str(arm.get("side") or ""))),
                _finite(arm.get("sweep_total")),
                _finite(arm.get("closes")),
                _finite(arm.get("reopened")),
                _finite(grip_min), _finite(grip_max), _finite(grip_range),
            ]
            row += [_finite(v) for v in arm.get("sweep") or ()]
        rows.append(row)

    widths = {len(r) for r in rows}
    if len(widths) > 1:
        raise ValueError(
            f"episodes of {detail.get('repo_id')} do not share a feature width "
            f"({sorted(widths)}) — the arms differ between episodes of one rig")
    return np.asarray(rows, dtype=float) if rows else np.zeros((0, 0))


#: A column whose spread is this small RELATIVE to its own magnitude counts as
#: constant, not merely as one that varies a little.
#:
#: The contract says a ZERO-VARIANCE column stays 0, and this is that rule
#: applied at the precision the data actually has. LeRobot records float32, so a
#: measurement that is conceptually identical across episodes still wobbles in
#: its last bits — a synthetic dataset whose every episode tracks at exactly
#: 0.1° gives `tracking` a standard deviation of ~1e-8, purely from where each
#: episode's joint values land in float32. Dividing by that promotes the last
#: bits of a float to a full unit-variance dimension, and the nearest-neighbour
#: ranking is then decided by rounding error instead of by the episode. float32
#: carries ~7 significant digits, so 1e-6 is one digit clear of the noise and
#: several orders below any spread that means something.
_CONSTANT_REL = 1e-6


def _zscore(x: np.ndarray) -> np.ndarray:
    """Per column, across this dataset's episodes.

    A constant column stays 0 instead of dividing by zero. Most columns of a
    single-task dataset are constant — every episode has the same gripper
    calibration and the same joint count — so `x / 0` here is the normal case,
    not the edge case, and one NaN would poison every distance in the matrix.
    """
    if x.size == 0:
        return x
    mean = x.mean(axis=0)
    sd = x.std(axis=0)
    out = np.zeros_like(x)
    live = sd > _CONSTANT_REL * np.maximum(np.abs(mean), 1.0)
    out[:, live] = (x[:, live] - mean[live]) / sd[live]
    return out


#: Below this norm a z-scored vector is treated as having no direction at all.
#: An episode sitting at the dataset mean in every live column lands here, and
#: its remaining direction is the rounding error of the z-score rather than
#: anything about the episode — normalising that would hand it a confident
#: nearest neighbour chosen by the last bit of a float. Z-scored columns have
#: unit variance, so a real vector's norm is order sqrt(live columns); 1e-9 is
#: far below any of them and far above the noise.
_NO_DIRECTION = 1e-9


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= _NO_DIRECTION or nb <= _NO_DIRECTION:
        # No direction, so no neighbours — 0.0, never a division that produces
        # NaN and never a similarity invented out of rounding error.
        return 0.0
    value = float(np.dot(a, b)) / (na * nb)
    if not math.isfinite(value):
        return 0.0
    return max(-1.0, min(1.0, value))


def _nearest(vec: np.ndarray, z: np.ndarray, pool: list[int], k: int) -> list[tuple]:
    """The k most similar members of `pool`, most similar first.

    Ties break on the episode index, and that is load-bearing rather than tidy:
    `apply` RECOMPUTES the diff, so a neighbour order that depended on dict or
    float ordering would let apply write a different diff than the one the
    operator confirmed, with the token matching.
    """
    scored = [(_cosine(vec, z[j]), j) for j in pool]
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return scored[:k]


def _vote(neighbours: list[tuple], label_of) -> tuple[str, float] | None:
    """Similarity-weighted vote. `(winner, confidence)`, or None for no evidence.

    Negative similarities are clamped to zero rather than shifted into a
    positive range: an anti-correlated neighbour is not weak evidence FOR its
    own label, it is evidence about a different kind of episode, and letting it
    vote at all would make "nothing like this has been reviewed" read as a
    confident answer.
    """
    weights: dict[str, float] = {}
    for sim, j in neighbours:
        weight = max(sim, 0.0)
        label = label_of(j)
        weights[label] = weights.get(label, 0.0) + weight
    total = sum(weights.values())
    if total <= 0.0:
        return None
    winner = min(weights.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return winner, weights[winner] / total


def _plan_knn(detail: dict, params: dict) -> list[dict]:
    k, min_conf, propagate = _knn_params(params)
    episodes = detail["episodes"]
    if not episodes:
        return []
    z = _zscore(_feature_matrix(detail))
    if propagate == "tags":
        return _knn_tags(episodes, z, k, min_conf)
    return _knn_marks(episodes, z, k, min_conf)


def _knn_marks(episodes: list[dict], z: np.ndarray, k: int,
               min_conf: float) -> list[dict]:
    """Propagate keep/reject from the marked episodes to the unmarked ones.

    **Only episodes whose current mark is `unset` are ever proposed.** kNN
    exists to extend a review you started, not to overrule one you finished: an
    episode you marked is one you looked at, and a neighbourhood vote is not
    evidence against your own eyes.

    With fewer than `k` marked episodes the vote runs over the ones that exist
    rather than refusing — but note that it is then unanimous by construction,
    which is why `min_confidence` is not a substitute for having reviewed a
    handful of episodes first.
    """
    marks = [str(e.get("status") or review_mod.UNSET) for e in episodes]
    pool = [i for i, m in enumerate(marks) if m != review_mod.UNSET]
    if not pool:
        return []

    out = []
    for i, ep in enumerate(episodes):
        if marks[i] != review_mod.UNSET:
            continue
        neighbours = _nearest(z[i], z, pool, k)
        voted = _vote(neighbours, lambda j: marks[j])
        if voted is None:
            continue
        winner, confidence = voted
        # Below the threshold the episode is left OUT of the diff entirely
        # rather than proposed weakly: a row an operator has to second-guess
        # costs more attention than the episode it was about.
        if confidence < min_conf:
            continue
        nearest = episodes[neighbours[0][1]]["episode_index"]
        agree = sum(1 for _s, j in neighbours if marks[j] == winner)
        out.append(_entry(
            ep, winner,
            f"{agree} of {len(neighbours)} nearest marked episodes are "
            f"{winner} — nearest is Ep {nearest + 1} (idx {nearest})",
            confidence,
        ))
    return out


def _knn_tags(episodes: list[dict], z: np.ndarray, k: int,
              min_conf: float) -> list[dict]:
    """Propagate TAGS from the tagged episodes to the untagged ones.

    The mark is untouched — `from` and `to` both carry the episode's current
    mark, and the proposed tags ride in an additive `tags` key. **That key is
    not in the frozen HTTP diff shape**; it is reported to the integrator rather
    than assumed, because a UI that does not know about it renders these rows as
    a no-op.

    The candidate rule is the tag-shaped reading of the mark rule above: an
    episode that already carries tags is one somebody looked at, so this only
    ever ADDS tags to episodes that have none.
    """
    tags = [[str(t) for t in (e.get("tags") or ())] for e in episodes]
    pool = [i for i, t in enumerate(tags) if t]
    if not pool:
        return []

    out = []
    for i, ep in enumerate(episodes):
        if tags[i]:
            continue
        neighbours = _nearest(z[i], z, pool, k)
        total = sum(max(sim, 0.0) for sim, _j in neighbours)
        if total <= 0.0:
            continue
        weights: dict[str, float] = {}
        for sim, j in neighbours:
            for tag in tags[j]:
                weights[tag] = weights.get(tag, 0.0) + max(sim, 0.0)
        chosen = sorted(
            (tag for tag, w in weights.items() if w / total >= min_conf),
            key=lambda tag: (-weights[tag], tag),
        )
        if not chosen:
            continue
        # The WEAKEST accepted tag's confidence: every tag on this row is at
        # least this well supported, which is the only number a single
        # confidence column on a multi-tag row can honestly mean.
        confidence = min(weights[tag] / total for tag in chosen)
        mark = str(ep.get("status") or review_mod.UNSET)
        entry = _entry(ep, mark, "tagged like its "
                       f"{len(neighbours)} nearest reviewed episodes: "
                       f"{', '.join(chosen)}", confidence)
        entry["tags"] = chosen
        out.append(entry)
    return out


def _apply_tags(root: Path, plan: list[dict], episode_frames: dict[int, int]) -> int:
    """Write a tags-mode plan, grouped so identical tag sets share one write.

    `review.bulk_update` rewrites the whole sidecar per call; one call per
    episode would be 46 atomic writes of a file the review page is polling.
    """
    by_tags: dict[tuple, list[int]] = {}
    for entry in plan:
        by_tags.setdefault(tuple(entry.get("tags") or ()), []).append(
            int(entry["episode"]))
    applied = 0
    for tag_set, eps in by_tags.items():
        if not tag_set:
            continue
        applied += review_mod.bulk_update(
            root, eps, tags_add=list(tag_set), episode_frames=episode_frames)
    return applied


# ---- policy loss ----------------------------------------------------------

def _plan_policy_loss(detail: dict, params: dict) -> dict:
    """A ranked sort order, hardest to fit first. Never a mark.

    Per-episode loss is not something LeRobot logs, so this reads
    `<run_dir>/episode_loss.jsonl` when a runner wrote one and otherwise says
    `available: false` with a reason. It is WIRED and DATA-GATED — a proxy
    metric computed here and labelled "policy loss" would be a number nobody
    could check against a training curve.
    """
    from . import runs  # local: the run store is not on the preview path for
                        # any other mode, and this keeps the module's import
                        # graph to what every caller actually needs.

    run_id = str(params.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("policy-loss needs params {\"run_id\": \"<train run>\"}")

    rdir = runs.run_dir(run_id)          # RUN_ID_RE + containment, or ValueError
    if not rdir.exists():
        raise FileNotFoundError(f"no run {run_id}")

    path = rdir / EPISODE_LOSS_FILENAME
    if not path.exists():
        return {
            "diff": [], "ranking": [], "available": False,
            "reason": (
                f"{run_id} wrote no {EPISODE_LOSS_FILENAME}. Per-episode loss is "
                "not something LeRobot logs; a run has to be launched with "
                "per-episode evaluation for this ranking to exist."
            ),
        }

    known = {int(e["episode_index"]) for e in detail["episodes"]}
    losses, dropped = _read_episode_loss(path, known)
    if not losses:
        return {
            "diff": [], "ranking": [], "available": False,
            "reason": (
                f"{EPISODE_LOSS_FILENAME} in {run_id} holds no usable rows for "
                f"this dataset's episodes ({dropped} row(s) named something else)."
            ),
        }

    # Hardest first, ties on the episode index so two identical losses do not
    # swap places between two polls of the same page.
    order = sorted(losses.items(), key=lambda kv: (-kv[1], kv[0]))
    ranking = [{"episode": ep, "score": _finite(score), "rank": n}
               for n, (ep, score) in enumerate(order, start=1)]
    reason = ""
    if dropped:
        reason = (f"{dropped} row(s) named episodes this dataset does not have "
                  "— the run may predate a prune.")
    # `diff` is ALWAYS empty here, and `apply` refuses this mode outright.
    return {"diff": [], "ranking": ranking, "available": True, "reason": reason}


#: Key spellings accepted in `episode_loss.jsonl`. Nothing writes this file yet,
#: so the reader takes both the LeRobot-ish and the plain spelling rather than
#: forcing a runner author to guess which one this side wanted.
_LOSS_EPISODE_KEYS = ("episode_index", "episode")
_LOSS_VALUE_KEYS = ("loss", "score")


def _read_episode_loss(path: Path, known: set[int]) -> tuple[dict[int, float], int]:
    """`{episode: mean loss}` plus the number of rows that named an episode this
    dataset does not have.

    Repeated rows for one episode AVERAGE. A runner writing one row per episode
    is unaffected; one that appends a row per evaluation pass gets the mean
    rather than whichever pass happened to be written last.

    A corrupt line costs one episode's rank, never the whole ranking — same rule
    `runs.read_metrics` applies to a metrics row caught mid-write.
    """
    totals: dict[int, list[float]] = {}
    dropped = 0
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        episode = _first(row, _LOSS_EPISODE_KEYS)
        value = _first(row, _LOSS_VALUE_KEYS)
        if episode is None or value is None:
            continue
        try:
            episode = int(episode)
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        if episode not in known:
            dropped += 1
            continue
        totals.setdefault(episode, []).append(value)
    return {ep: sum(v) / len(v) for ep, v in totals.items()}, dropped


def _first(row: dict, keys: tuple[str, ...]):
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return None
