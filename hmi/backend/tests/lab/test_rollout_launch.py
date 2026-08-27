# hmi/backend/tests/lab/test_rollout_launch.py
"""`POST /lab/runs/rollout` and the trained-rate gate it exists to enforce.

**Check (a): the DECLARED control rate against the rate the policy was TRAINED
at**, refused at launch. Its sibling check (b), measured-vs-declared, lives in
the child and is tested in `test_rollout_runner.py`; the two are different
checks on different events and neither substitutes for the other.

This file is organised around the CLAIM rather than around the modules,
because the claim spans three of them and no single-module test can state it:

    <checkpoint>/train_config.json -> dataset.repo_id     runs.trained_dataset
                                   -> meta/info.json      catalog.dataset_fps
                                   -> fps vs control_hz   routes_runs
And that chain is the ONLY route to the number. Verified 2026-08-27 by walking
every key of a real ACT checkpoint's `train_config.json` AND its policy
`config.json`: neither records an fps or a control rate anywhere, so a broken
link has no fallback that would not be an invention. `test_the_real_kit_
checkpoint_*` below pin that against the real bytes rather than a fixture.

**The launch harness cannot see a broken child.** `$HALLER_LAB_PYTHON` is
`true`, which ignores its arguments and exits 0 — so a launch "succeeds" and
its run directory appears whether or not the module named in `RUNNERS`
resolves. That is how four non-existent launch targets survived every test
until `f02dd81`. Nothing here asserts a rollout RAN; `test_runs.py`'s
`test_every_runner_target_is_importable` is what pins the target, in a
subprocess. What these tests own is the refusal: what never reaches `launch`
at all.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from haller_hmi.api import gate
from haller_hmi.api.deps import LabDeps
from haller_hmi.lab import catalog, runs
from haller_hmi.lab.routes_runs import build_runs_router

from ._dataset import make_dataset

#: The dataset the fake checkpoints below claim to have been trained on.
REPO = "local/smoke"

LOOPBACK_HOST = "127.0.0.1"

#: The kit's real trained ACT checkpoint. Read-only, like everything under
#: `~/vr-teleop-kit`, and the only place a `train_config.json` this port did
#: not itself write can be compared against.
KIT_CHECKPOINT = Path(
    "/home/odesha/vr-teleop-kit/outputs/runs/"
    "train-20260826-213350-act_so101_pick_cube/train/checkpoints/060000/"
    "pretrained_model"
)
needs_kit = pytest.mark.skipif(
    not (KIT_CHECKPOINT / "train_config.json").is_file(),
    reason=f"no real checkpoint at {KIT_CHECKPOINT}",
)


# ---- fixtures --------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_ambient_override(monkeypatch):
    monkeypatch.delenv(gate.REMOTE_CONTROL_ENV, raising=False)


@pytest.fixture(autouse=True)
def cold_caches():
    catalog._detail_cache.clear()
    catalog._frames_cache.clear()
    yield
    catalog._detail_cache.clear()
    catalog._frames_cache.clear()


@pytest.fixture
def store(tmp_path, monkeypatch) -> Path:
    """An empty run store and an interpreter that cannot roll out.

    `true` exits 0 and writes no `result.json`, so a launched run resolves to
    `died`. Every assertion below is about whether a run directory appears AT
    ALL, which is the one thing this stand-in reports honestly.
    """
    monkeypatch.setenv(runs.RUNS_DIR_ENV, str(tmp_path / "runs"))
    monkeypatch.setenv(runs.LAB_PYTHON_ENV, shutil.which("true") or "/bin/true")
    return tmp_path / "runs"


@pytest.fixture
def home(tmp_path, monkeypatch, store) -> Path:
    home = tmp_path / "lerobot"
    home.mkdir(parents=True)
    monkeypatch.setenv("HF_LEROBOT_HOME", str(home))
    return home


@pytest.fixture
def client(home) -> TestClient:
    deps = LabDeps(
        get_cameras=lambda: None,
        get_recorder=lambda: None,
        lerobot_home=lambda: home,
        allow_remote_control=lambda: False,
    )
    app = FastAPI()
    app.include_router(build_runs_router(deps))
    return TestClient(app, client=(LOOPBACK_HOST, 51000))


# ---- helpers ---------------------------------------------------------------

def make_checkpoint(
    path: Path,
    *,
    repo_id: str | None = REPO,
    episodes: list[int] | None = None,
    dataset_block: bool = True,
    body: str | None = None,
) -> Path:
    """A checkpoint directory shaped like LeRobot's, with `train_config.json`.

    Only the keys this gate reads are written; a real one carries forty more.
    `body` writes raw bytes instead, for the cannot-parse case.
    """
    path.mkdir(parents=True, exist_ok=True)
    config_path = path / "train_config.json"
    if body is not None:
        config_path.write_text(body)
        return path
    config: dict = {"policy": {"type": "act"}, "steps": 100}
    if dataset_block:
        dataset: dict = {"root": None, "repo_type": "dataset"}
        if repo_id is not None:
            dataset["repo_id"] = repo_id
        if episodes is not None:
            dataset["episodes"] = episodes
        config["dataset"] = dataset
    config_path.write_text(json.dumps(config, indent=2))
    return path


def launched(store: Path) -> list[Path]:
    """Run directories that exist, newest name last."""
    return sorted(p for p in store.iterdir() if p.is_dir()) if store.exists() else []


def spec_of(store: Path, run_id: str) -> dict:
    return json.loads((store / run_id / "spec.json").read_text())


# ============================================================================
# runs.trained_dataset — what a checkpoint says it was trained on
# ============================================================================

def test_the_training_dataset_is_read_from_the_checkpoints_own_config(tmp_path):
    ckpt = make_checkpoint(tmp_path / "pretrained_model", repo_id="local/thing")
    found = runs.trained_dataset(ckpt)
    assert found["repo_id"] == "local/thing"
    assert found["reason"] == ""
    assert found["config_path"] == str(ckpt / "train_config.json")


def test_the_training_episode_list_keeps_lerobots_order(tmp_path):
    """The eval split is the TAIL of this list, so sorting it here would
    describe a holdout that never happened. The real kit checkpoint's list is
    unsorted for exactly that reason."""
    order = [18, 22, 13, 30, 0, 44]
    ckpt = make_checkpoint(tmp_path / "pretrained_model", episodes=order)
    assert runs.trained_dataset(ckpt)["episodes"] == order


def test_a_checkpoint_that_is_not_there_reports_the_path_and_guesses_nothing(
        tmp_path):
    found = runs.trained_dataset(tmp_path / "nowhere")
    assert found["repo_id"] is None
    assert str(tmp_path / "nowhere") in found["reason"]


def test_a_checkpoint_with_no_train_config_says_so_and_names_the_file(tmp_path):
    ckpt = tmp_path / "pretrained_model"
    ckpt.mkdir()
    found = runs.trained_dataset(ckpt)
    assert found["repo_id"] is None
    assert "train_config.json" in found["reason"]


def test_a_train_config_that_will_not_parse_is_reported_not_raised(tmp_path):
    ckpt = make_checkpoint(tmp_path / "pretrained_model", body="{not json")
    found = runs.trained_dataset(ckpt)
    assert found["repo_id"] is None
    assert "could not be read" in found["reason"]


@pytest.mark.parametrize("kwargs, wanted", [
    ({"dataset_block": False}, "no 'dataset' block"),
    ({"repo_id": None}, "names no dataset.repo_id"),
    ({"repo_id": "   "}, "names no dataset.repo_id"),
])
def test_a_config_that_names_no_dataset_reports_which_part_is_missing(
        tmp_path, kwargs, wanted):
    ckpt = make_checkpoint(tmp_path / "pretrained_model", **kwargs)
    found = runs.trained_dataset(ckpt)
    assert found["repo_id"] is None
    assert wanted in found["reason"]


@needs_kit
def test_the_real_kit_checkpoint_carries_the_dataset_it_was_trained_on():
    """The contract test: two real components, not a fixture.

    `make_checkpoint` above writes what this port BELIEVES the shape is, so on
    its own it would prove only that the port agrees with itself.
    """
    found = runs.trained_dataset(KIT_CHECKPOINT)
    assert found["repo_id"] == "local/so101_pick_cube"
    assert found["reason"] == ""
    assert len(found["episodes"]) == 35


@needs_kit
def test_the_real_kit_checkpoint_records_no_rate_of_its_own():
    """Why the dataset chain is the ONLY route rather than the preferred one.

    If a checkpoint ever starts recording its own fps this test fails, and it
    should: the gate could then read one number instead of three files, and
    `trained_dataset`'s whole reason for reporting a broken link would be
    weaker. Failing here is the notification.
    """
    for name in ("train_config.json", "config.json"):
        blob = json.loads((KIT_CHECKPOINT / name).read_text())
        found: list[str] = []

        def walk(node, path="", found=found):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key.lower() in ("fps", "control_hz", "hz"):
                        found.append(f"{path}/{key}")
                    walk(value, f"{path}/{key}")
            elif isinstance(node, list):
                for item in node:
                    walk(item, path)

        walk(blob)
        assert found == [], f"{name} now records a rate at {found}"


# ============================================================================
# catalog.dataset_fps — and why it must not default when its neighbours do
# ============================================================================

def test_the_datasets_declared_fps_is_read_from_info_json(home):
    make_dataset(home / REPO, n_episodes=1, fps=25)
    assert catalog.dataset_fps(REPO) == 25


def test_a_dataset_that_is_not_there_has_no_fps_rather_than_a_default(home):
    assert catalog.dataset_fps("local/gone") is None


@pytest.mark.parametrize("value", [None, 0, -1, "thirty"])
def test_an_unusable_fps_is_none_and_never_thirty(home, value):
    make_dataset(home / REPO, n_episodes=1)
    info_path = home / REPO / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    if value is None:
        info.pop("fps")
    else:
        info["fps"] = value
    info_path.write_text(json.dumps(info))
    assert catalog.dataset_fps(REPO) is None


def test_a_dataset_whose_metadata_cannot_be_read_has_no_fps(home):
    """The directory is THERE and `info.json` is not readable — a truncated
    parquet-era state, and what a dataset looks like mid-recording. Distinct
    from an absent directory, and it was the one `_info` branch no test reached:
    a mutation defaulting THIS path to 30 survived the first matrix.
    """
    make_dataset(home / REPO, n_episodes=1)
    (home / REPO / "meta" / "info.json").write_text("{truncated mid-writ")
    assert catalog.dataset_fps(REPO) is None


# ============================================================================
# catalog.dataset_rate_provenance — whether the trained fps means anything
# ============================================================================
#
# Check (a) compares an int against an int. Nothing in `info.json`'s `fps`
# records where that int came from, so a PASS against a pre-invariant-10
# dataset is a declaration agreeing with a declaration. These pin the
# discriminator, and — the load-bearing one — that it never refuses.

def _write_rate_block(home, block, repo_id=REPO):
    """Give a dataset the recorder's provenance block."""
    info_path = home / repo_id / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    if block is None:
        info.pop(catalog.RATE_INFO_KEY, None)
    else:
        info[catalog.RATE_INFO_KEY] = block
    info_path.write_text(json.dumps(info))


def test_the_provenance_key_matches_the_recorders(home):
    """`catalog` spells `haller_rate` literally; `recorder.py` owns it.

    It cannot be imported — `recorder.py` imports lerobot and `lab/` is banned
    from it in the serving process — so the two spellings are pinned equal HERE,
    at the source, without importing anything. Crude on purpose, like
    `test_the_forbidden_path_appears_nowhere_in_this_module`.

    The drift is silent and in the worst direction: rename the recorder's key
    and every Haller-recorded dataset starts reading as "not measured", which
    is the exact claim this field exists to make truthfully. Nothing else
    fails, because absence is a legitimate value.
    """
    recorder_src = (
        Path(__file__).resolve().parents[2] / "haller_hmi" / "recorder.py"
    ).read_text()
    assert f'RATE_INFO_KEY = "{catalog.RATE_INFO_KEY}"' in recorder_src, (
        f"recorder.py no longer spells its rate block "
        f"{catalog.RATE_INFO_KEY!r}; catalog.dataset_rate_provenance would "
        f"report every measured dataset as unmeasured"
    )


def test_a_dataset_with_the_recorders_block_is_attested_as_measured(home):
    make_dataset(home / REPO, n_episodes=1, fps=30)
    _write_rate_block(home, {"fps_written": 30, "measured_hz": 29.87,
                             "samples": 300, "target_hz": 30})
    assert catalog.dataset_rate_provenance(REPO) == {
        "measured": True, "measured_hz": 29.87}


def test_a_dataset_without_the_block_is_not_attested_and_does_not_refuse(home):
    """Both real datasets on this box are this case, including the one the only
    real trained checkpoint here used. `False` means "nothing attests it", NOT
    "it was declared" — a third tool that measured honestly and wrote no block
    lands here too, and the field must not claim more than it knows."""
    make_dataset(home / REPO, n_episodes=1, fps=30)
    assert catalog.dataset_rate_provenance(REPO) == {
        "measured": False, "measured_hz": None}


def test_a_dataset_that_is_not_there_cannot_be_asked_about_provenance(home):
    assert catalog.dataset_rate_provenance("local/gone") == {
        "measured": None, "measured_hz": None}


def test_unreadable_metadata_is_no_answer_rather_than_not_measured(home):
    """`None` and `False` are different claims and the difference is the point:
    `False` says the dataset carries no attestation, `None` says there was no
    dataset to ask. Collapsing them would report a missing dataset as a
    measured-rate failure."""
    make_dataset(home / REPO, n_episodes=1)
    (home / REPO / "meta" / "info.json").write_text("{truncated mid-writ")
    assert catalog.dataset_rate_provenance(REPO) == {
        "measured": None, "measured_hz": None}


@pytest.mark.parametrize("block", [
    {"fps_written": 30},                       # no measured_hz at all
    {"measured_hz": "twenty-nine"},            # unparseable
    {"measured_hz": None},
])
def test_a_damaged_figure_costs_the_audit_value_not_the_discriminator(
        home, block):
    """The block being PRESENT is what says the recorder wrote this dataset, so
    `fps` is measured however mangled the figure beside it is. Two claims, and
    only one is damaged — folding them together would downgrade a genuinely
    measured dataset to unattested over a bad float."""
    make_dataset(home / REPO, n_episodes=1, fps=30)
    _write_rate_block(home, block)
    assert catalog.dataset_rate_provenance(REPO) == {
        "measured": True, "measured_hz": None}


@pytest.mark.parametrize("block", ["not-a-dict", 30, [1, 2]])
def test_a_block_that_is_not_an_object_is_not_an_attestation(home, block):
    make_dataset(home / REPO, n_episodes=1, fps=30)
    _write_rate_block(home, block)
    assert catalog.dataset_rate_provenance(REPO) == {
        "measured": False, "measured_hz": None}


def test_the_listing_still_says_thirty_for_the_dataset_the_gate_refuses(home):
    """The two functions disagree ON PURPOSE, and this pins the disagreement.

    `list_datasets` falls back to 30 so one malformed directory cannot blank a
    page. `dataset_fps` must not, because a silent 30 in a gate compares
    against a number nobody measured and reports agreement — a check that
    cannot fire. Both behaviours asserted together, in one test, because a
    future reader will otherwise "fix" the inconsistency in whichever file they
    happen to be looking at.
    """
    make_dataset(home / REPO, n_episodes=1)
    info_path = home / REPO / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info.pop("fps")
    info_path.write_text(json.dumps(info))

    assert catalog.dataset_fps(REPO) is None
    row = next(r for r in catalog.list_datasets() if r["repo_id"] == REPO)
    assert row["fps"] == 30


# ============================================================================
# POST /lab/runs/rollout — the gate
# ============================================================================

def test_a_rate_that_matches_the_training_launches_and_answers_with_the_id(
        home, store, client):
    make_dataset(home / REPO, n_episodes=2, fps=30)
    ckpt = make_checkpoint(home / "ckpt")
    response = client.post("/lab/runs/rollout", json={
        "policy_path": str(ckpt), "control_hz": 30})
    assert response.status_code == 200, response.text
    assert set(response.json()) == {"id"}
    assert len(launched(store)) == 1


def test_both_rates_are_stamped_into_the_spec_that_launched(home, store, client):
    """The ruling is that both numbers are recorded either way, so a run's own
    spec answers "what was this run against?" after the dataset has moved on."""
    make_dataset(home / REPO, n_episodes=2, fps=30)
    ckpt = make_checkpoint(home / "ckpt")
    run_id = client.post("/lab/runs/rollout", json={
        "policy_path": str(ckpt), "control_hz": 30}).json()["id"]

    spec = spec_of(store, run_id)
    assert spec["control_hz"] == 30
    assert spec["control_hz_trained"] == 30
    assert spec["control_hz_trained_repo_id"] == REPO
    assert spec["control_hz_trained_source"] == str(ckpt / "train_config.json")
    assert spec["control_hz_mismatch_override"] is False


def test_the_spec_records_whether_the_trained_rate_was_ever_measured(
        home, store, client):
    """`control_hz_trained` is the number; this says whether it means anything.

    Without it every PASS reads identically, and a PASS against a
    pre-invariant-10 dataset is a declaration agreeing with a declaration. The
    contract already makes both numbers reconstructible after the arm has
    moved; this makes the WORTH of their agreement reconstructible too.
    """
    make_dataset(home / REPO, n_episodes=2, fps=30)
    _write_rate_block(home, {"fps_written": 30, "measured_hz": 30.04,
                             "samples": 300, "target_hz": 30})
    ckpt = make_checkpoint(home / "ckpt")
    run_id = client.post("/lab/runs/rollout", json={
        "policy_path": str(ckpt), "control_hz": 30}).json()["id"]

    spec = spec_of(store, run_id)
    assert spec["control_hz_trained"] == 30
    assert spec["control_hz_trained_measured"] is True
    assert spec["control_hz_trained_measured_hz"] == 30.04


def test_an_unattested_rate_launches_and_says_so_rather_than_refusing(
        home, store, client):
    """**The load-bearing one.** Ruled 2026-08-27: do NOT refuse on absent
    provenance.

    This package refuses a rollout below the rate floor on purpose, so "refuse
    when unsure" has a live precedent here that does not apply. Refusing would
    block the only real trained checkpoint on this box over a number that is
    probably fine, converting a caveat into a blockade. Refusing is the wrong
    response to "we cannot tell"; recording that we cannot tell is the right
    one. So a 200 AND the honest stamp, asserted together — a test for only the
    stamp would pass on a route that refused.
    """
    make_dataset(home / REPO, n_episodes=2, fps=30)   # no rate block: legacy
    ckpt = make_checkpoint(home / "ckpt")
    response = client.post("/lab/runs/rollout", json={
        "policy_path": str(ckpt), "control_hz": 30})

    assert response.status_code == 200, response.json()
    spec = spec_of(store, response.json()["id"])
    assert spec["control_hz_trained"] == 30
    assert spec["control_hz_trained_measured"] is False
    assert spec["control_hz_trained_measured_hz"] is None


def test_provenance_is_no_answer_when_the_link_to_the_dataset_broke(
        home, store, client):
    """`False` would be a claim about a dataset nobody found. The rate is None
    here, so its provenance has no subject and must not read as a verdict.

    `allow_rate_mismatch` and `action_names` are both required to REACH the
    stamp: an unresolvable dataset refuses first for having no declared rate to
    compare, and again for nothing naming the action vector. Without them this
    test asserts about a spec that was never written — it failed exactly that
    way when first run.
    """
    make_dataset(home / REPO, n_episodes=2, fps=30)
    ckpt = make_checkpoint(home / "ckpt", repo_id="local/gone")
    run_id = client.post("/lab/runs/rollout", json={
        "policy_path": str(ckpt), "control_hz": 30,
        "allow_rate_mismatch": True,
        "action_names": ["shoulder_pan.pos", "gripper.pos"],
    }).json()["id"]

    spec = spec_of(store, run_id)
    assert spec["control_hz_trained"] is None
    assert spec["control_hz_trained_measured"] is None
    assert spec["control_hz_trained_measured_hz"] is None


def test_provenance_never_changes_what_the_gate_decides(home, store, client):
    """Attested or not, check (a) is the same exact match on the same integers.

    Asserted in BOTH directions on ONE dataset state, because the risk is that
    provenance quietly becomes an input to the gate — a refusal that consults
    it would be a second question wearing the first one's answer.
    """
    make_dataset(home / REPO, n_episodes=2, fps=30)
    ckpt = make_checkpoint(home / "ckpt")

    # Unattested: agreement still passes, divergence still refuses.
    assert client.post("/lab/runs/rollout", json={
        "policy_path": str(ckpt), "control_hz": 30}).status_code == 200
    assert client.post("/lab/runs/rollout", json={
        "policy_path": str(ckpt), "control_hz": 25}).status_code == 400

    # Attested: identical verdicts.
    _write_rate_block(home, {"fps_written": 30, "measured_hz": 29.9})
    assert client.post("/lab/runs/rollout", json={
        "policy_path": str(ckpt), "control_hz": 30}).status_code == 200
    assert client.post("/lab/runs/rollout", json={
        "policy_path": str(ckpt), "control_hz": 25}).status_code == 400


def test_a_measured_rate_that_rounded_is_stamped_unrounded(home, store, client):
    """The audit value's whole reason to exist: `fps` 30 against a measured
    29.62 passes check (a) exactly and sits 1.3% off — inside gate (b)'s 10%
    floor, so nothing downstream ever mentions it. The spec is the only place
    that distance survives."""
    make_dataset(home / REPO, n_episodes=2, fps=30)
    _write_rate_block(home, {"fps_written": 30, "measured_hz": 29.62})
    ckpt = make_checkpoint(home / "ckpt")
    run_id = client.post("/lab/runs/rollout", json={
        "policy_path": str(ckpt), "control_hz": 30}).json()["id"]

    spec = spec_of(store, run_id)
    assert spec["control_hz_trained"] == 30
    assert spec["control_hz_trained_measured_hz"] == 29.62


def test_an_undeclared_rate_defaults_to_the_one_the_policy_was_trained_at(
        home, store, client):
    make_dataset(home / REPO, n_episodes=2, fps=25)
    ckpt = make_checkpoint(home / "ckpt")
    run_id = client.post("/lab/runs/rollout", json={
        "policy_path": str(ckpt)}).json()["id"]

    spec = spec_of(store, run_id)
    assert spec["control_hz"] == 25
    assert spec["control_hz_declared_by"] == "trained_fps"


def test_a_declared_rate_is_recorded_as_declared_even_when_it_agrees(
        home, store, client):
    """Otherwise every run reads as a deliberate agreement between two numbers
    and no later reader can tell a choice from a default."""
    make_dataset(home / REPO, n_episodes=2, fps=30)
    ckpt = make_checkpoint(home / "ckpt")
    run_id = client.post("/lab/runs/rollout", json={
        "policy_path": str(ckpt), "control_hz": 30}).json()["id"]
    assert spec_of(store, run_id)["control_hz_declared_by"] == "request"


@pytest.mark.parametrize("declared", [15, 4.8, 29, 31, 60])
def test_a_rate_the_policy_was_not_trained_at_is_refused_in_both_directions(
        home, store, client, declared):
    """**Two-sided on purpose.** Slower is the kit's own 4.8-against-30 failure;
    faster is the same error mirrored, applying deltas sized for 33 ms over
    17 ms. A one-sided gate passes 60 and would have to be argued back.

    29 and 31 are here because the gate is EXACT, not a tolerance: `fps` is an
    int in lerobot's schema and `control_hz` is a declared value, so nothing
    between them produces a near miss by accident — only a typo or a decision,
    and a decision belongs in the override where it leaves evidence.
    """
    make_dataset(home / REPO, n_episodes=2, fps=30)
    ckpt = make_checkpoint(home / "ckpt")
    response = client.post("/lab/runs/rollout", json={
        "policy_path": str(ckpt), "control_hz": declared})

    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert "30" in detail and str(declared) in detail
    assert launched(store) == [], "a refused rollout must not leave a run behind"


def test_the_refusal_names_both_rates_and_the_dataset_it_read_them_from(
        home, client):
    make_dataset(home / REPO, n_episodes=2, fps=30)
    ckpt = make_checkpoint(home / "ckpt")
    detail = client.post("/lab/runs/rollout", json={
        "policy_path": str(ckpt), "control_hz": 15}).json()["detail"]
    assert "15" in detail and "30" in detail and REPO in detail
    assert "allow_rate_mismatch" in detail


def test_the_override_launches_and_the_run_says_so_forever(home, store, client):
    """The kit's real failure was not that a 4.8 Hz run happened. It was that
    "success" was reported with that number attached to nothing."""
    make_dataset(home / REPO, n_episodes=2, fps=30)
    ckpt = make_checkpoint(home / "ckpt")
    run_id = client.post("/lab/runs/rollout", json={
        "policy_path": str(ckpt), "control_hz": 15,
        "allow_rate_mismatch": True}).json()["id"]

    spec = spec_of(store, run_id)
    assert spec["control_hz"] == 15
    assert spec["control_hz_trained"] == 30
    assert spec["control_hz_mismatch_override"] is True
    summary = json.loads((store / run_id / "run.json").read_text())["spec_summary"]
    assert "rate override" in summary and "30" in summary


def test_a_checkpoint_with_no_recorded_dataset_is_refused_not_guessed_around(
        home, store, client):
    """The chain has no fallback. Inferring the dataset from the run directory
    or from whatever is selected would compare against the WRONG dataset's fps
    and report agreement."""
    make_dataset(home / REPO, n_episodes=2, fps=30)
    ckpt = tmp = home / "ckpt"
    tmp.mkdir()
    response = client.post("/lab/runs/rollout", json={
        "policy_path": str(ckpt), "control_hz": 30})
    assert response.status_code == 400, response.text
    assert "train_config.json" in response.json()["detail"]
    assert launched(store) == []


def test_a_policy_whose_training_dataset_is_gone_is_refused_by_name(
        home, store, client):
    """A prune renames the survivors' home and a delete removes it. The rate
    the policy was trained at goes with it, and that is a refusal rather than
    an assumption."""
    ckpt = make_checkpoint(home / "ckpt", repo_id="local/pruned_away")
    response = client.post("/lab/runs/rollout", json={
        "policy_path": str(ckpt), "control_hz": 30})
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert "local/pruned_away" in detail
    assert launched(store) == []


def test_an_unreadable_chain_still_needs_a_rate_even_under_the_override(
        home, store, client):
    """Overriding says "run it anyway", not "pick a number for me"."""
    ckpt = make_checkpoint(home / "ckpt", repo_id="local/gone")
    response = client.post("/lab/runs/rollout", json={
        "policy_path": str(ckpt), "allow_rate_mismatch": True})
    assert response.status_code == 400, response.text
    assert "control_hz" in response.json()["detail"]
    assert launched(store) == []


def test_an_overridden_unreadable_chain_launches_with_the_rate_spelled_out(
        home, store, client):
    ckpt = make_checkpoint(home / "ckpt", repo_id="local/gone")
    run_id = client.post("/lab/runs/rollout", json={
        "policy_path": str(ckpt), "control_hz": 30,
        "allow_rate_mismatch": True,
        "action_names": ["shoulder_pan.pos", "gripper.pos"],
    }).json()["id"]

    spec = spec_of(store, run_id)
    assert spec["control_hz"] == 30
    assert spec["control_hz_trained"] is None
    assert "local/gone" in spec["control_hz_trained_reason"]
    assert "repo_id" not in spec


def test_the_rig_comes_from_the_trained_dataset_and_the_request_cannot_name_one(
        home, store, client):
    """A rollout whose observation space differs from the recording is a policy
    being shown a world it has never seen, so the joint layout is derived from
    the dataset the policy was TRAINED on rather than from anything the caller
    sends."""
    make_dataset(home / REPO, n_episodes=2, fps=30)
    ckpt = make_checkpoint(home / "ckpt", repo_id=REPO)
    run_id = client.post("/lab/runs/rollout", json={
        "policy_path": str(ckpt), "control_hz": 30,
        "repo_id": "local/somewhere_else",
    }).json()["id"]
    assert spec_of(store, run_id)["repo_id"] == REPO


def test_action_names_alongside_a_known_dataset_is_refused_as_two_sources(
        home, store, client):
    """The child prefers `action_names` over `repo_id`, so accepting both would
    let the rig come from one dataset and the rate from another with nothing
    comparing them."""
    make_dataset(home / REPO, n_episodes=2, fps=30)
    ckpt = make_checkpoint(home / "ckpt")
    response = client.post("/lab/runs/rollout", json={
        "policy_path": str(ckpt), "control_hz": 30,
        "action_names": ["shoulder_pan.pos"]})
    assert response.status_code == 400, response.text
    assert "action_names" in response.json()["detail"]
    assert launched(store) == []


def test_a_missing_policy_path_is_a_400_that_names_the_field(home, store, client):
    response = client.post("/lab/runs/rollout", json={"control_hz": 30})
    assert response.status_code == 400, response.text
    assert "policy_path" in response.json()["detail"]
    assert launched(store) == []


def test_a_duration_past_the_ceiling_is_refused_at_the_door(home, store, client):
    """Refused BEFORE launch, against the child's own constant rather than a
    copy of it — a browser button that starts a doomed run and reports it dead
    two minutes later is worse than a refusal."""
    make_dataset(home / REPO, n_episodes=2, fps=30)
    ckpt = make_checkpoint(home / "ckpt")
    response = client.post("/lab/runs/rollout", json={
        "policy_path": str(ckpt), "control_hz": 30,
        "duration_s": runs.MAX_ROLLOUT_DURATION_S + 1})
    assert response.status_code == 400, response.text
    assert str(int(runs.MAX_ROLLOUT_DURATION_S)) in response.json()["detail"]
    assert launched(store) == []


def test_the_route_and_the_child_refuse_the_same_duration(home, client):
    """One constant, two enforcement points. Pinning them against each other is
    what stops the route's copy drifting — there is no copy, and this is the
    test that says so if one is ever reintroduced."""
    from haller_hmi.runners import rollout_runner

    assert rollout_runner.MAX_ROLLOUT_DURATION_S is runs.MAX_ROLLOUT_DURATION_S


@pytest.mark.parametrize("bad, wanted", [
    ('"fast"', "must be a number"),
    ("0", "must be greater than 0"),
    ("-5", "must be greater than 0"),
    ("NaN", "must be greater than 0"),
    ("Infinity", "must be greater than 0"),
])
def test_an_unusable_control_hz_is_refused_rather_than_coerced(
        home, store, client, bad, wanted):
    """Sent as RAW BODY TEXT, not through the json= encoder.

    `NaN` and `Infinity` are not legal JSON and the encoder refuses to write
    them — but Python's `json.loads`, which is what Starlette parses request
    bodies with, ACCEPTS both by default. So they are reachable over the wire
    and unreachable through a test that builds its body with `json=`: the
    harness would have declared the case impossible while the route stayed open
    to it. A nan `control_hz` becomes a period of nan seconds.
    """
    make_dataset(home / REPO, n_episodes=2, fps=30)
    ckpt = make_checkpoint(home / "ckpt")
    response = client.post(
        "/lab/runs/rollout",
        content=(
            f'{{"policy_path": {json.dumps(str(ckpt))}, "control_hz": {bad}, '
            f'"allow_rate_mismatch": true}}'
        ),
        headers={"Content-Type": "application/json"})
    assert response.status_code == 400, response.text
    # The OVERRIDE is set, so the rate gate is not what refuses here — this
    # pins the number validator itself. Without that, `NaN` was refused by the
    # rate gate (nan != 30) and a mutation removing the `isfinite` check
    # SURVIVED: the assertion read the right status for the wrong reason, and
    # under this very override a nan would have reached the child as a period
    # of nan seconds.
    assert wanted in response.json()["detail"]
    assert "control_hz" in response.json()["detail"]
    assert launched(store) == []


def test_the_spec_may_arrive_wrapped_the_way_the_train_route_accepts_it(
        home, store, client):
    make_dataset(home / REPO, n_episodes=2, fps=30)
    ckpt = make_checkpoint(home / "ckpt")
    response = client.post("/lab/runs/rollout", json={
        "spec": {"policy_path": str(ckpt), "control_hz": 30}})
    assert response.status_code == 200, response.text
    assert len(launched(store)) == 1


def test_the_launched_kind_is_rollout_and_not_train(home, store, client):
    make_dataset(home / REPO, n_episodes=2, fps=30)
    ckpt = make_checkpoint(home / "ckpt")
    run_id = client.post("/lab/runs/rollout", json={
        "policy_path": str(ckpt), "control_hz": 30}).json()["id"]
    record = json.loads((store / run_id / "run.json").read_text())
    assert record["kind"] == "rollout"
    assert record["argv"][2] == runs.RUNNERS["rollout"]
