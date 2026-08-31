"""The headless episode loop, against the real bimanual sim rig.

Real `MjModel`, real renders, real contacts - no mocks anywhere the physics is
the thing under test, for `test_scene_reset.py`'s reason: what these are about
is the addressing, the clock and the reproducibility, and a mock has none of
them.

The ONE stub in this file is a fake monitor, used only to drive the loop's
TERMINATION paths. The predicate itself has its own suite
(`tests/sim/test_task_success.py`, `test_insertion.py`) and nothing here
re-tests it; what is tested here is that the loop ends when the authority says
success and not on any opinion of its own.

The rig is `config.bimanual-sim.yaml` itself and not a fixture YAML. That file
is what an evaluation actually runs against, and a test rig that drifted from it
would pass while the real one stopped rendering a camera.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import pytest

from haller_hmi.config import load_config
from haller_hmi.sim.episode import (
    ARM_JOINT_ORDER,
    REASON_DRIVER_STOP,
    REASON_SUCCESS,
    REASON_TIMEOUT,
    STATE_KEY,
    EpisodeRunner,
    EpisodeSpec,
    frames_are_rgb,
    image_key,
    state_names,
)
from haller_hmi.sim.task import SuccessSpec

BACKEND = Path(__file__).resolve().parents[2]
RIG = BACKEND / "config.bimanual-sim.yaml"

#: Short episodes, so a whole test file is seconds rather than minutes. 30 Hz
#: is the rig's own `telemetry.hz` and is left alone: it is the rate its
#: datasets are recorded at, and every sim-second threshold in `sim/task.py` is
#: defined against it.
SHORT = EpisodeSpec(control_hz=30.0, max_episode_s=1.0)


# ---- drivers -------------------------------------------------------------

class Hold:
    """Holds whatever pose the first observation of the episode showed.

    The trivial driver, and deliberately so: every test below is about the loop,
    and a driver that does anything interesting would put its own behaviour into
    the assertions.
    """

    def __init__(self) -> None:
        self.pose: list[float] | None = None
        self.seeds: list[int] = []
        self.calls = 0

    def reset(self, seed: int) -> None:
        self.pose = None
        self.seeds.append(seed)

    def act(self, obs: dict) -> list[float]:
        self.calls += 1
        if self.pose is None:
            self.pose = list(obs[STATE_KEY])
        return self.pose


class StopAfter:
    """Answers None once it has acted `n` times."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.calls = 0

    def reset(self, seed: int) -> None:
        del seed
        self.calls = 0

    def act(self, obs: dict) -> list[float] | None:
        if self.calls >= self.n:
            return None
        self.calls += 1
        return list(obs[STATE_KEY])


class Constant:
    """Commands one fixed vector, whatever it sees."""

    def __init__(self, values: list[float]) -> None:
        self.values = values

    def reset(self, seed: int) -> None:
        del seed

    def act(self, obs: dict) -> list[float]:
        del obs
        return list(self.values)


class _MonitorSucceedingOnPoll:
    """Stands in for `TaskMonitor` to exercise the loop's success path only.

    Actually solving pick-and-place from a test would need a scripted expert and
    minutes of sim per episode. What is being tested here is that the loop stops
    when the AUTHORITY says success, so the authority is the thing replaced.
    """

    def __init__(self, poll_number: int) -> None:
        self.poll_number = poll_number
        self.spec = SuccessSpec()
        self.target = None
        self.polls = 0
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1
        self.polls = 0

    def poll(self) -> dict:
        self.polls += 1
        return {"success": self.polls >= self.poll_number, "held_s": 0.5}

    def provenance(self) -> dict:
        return {"task": "stub", "predicate": "test", "target": None}


# ---- fixtures ------------------------------------------------------------

@pytest.fixture(scope="module")
def runner():
    """One world for the whole module.

    Building it is cheap (0.06 s) but the EGL renderers are not (0.34 s), and
    every test resets the world completely through `reset_episode`, so sharing
    costs nothing in isolation and saves most of the file's runtime.
    """
    with EpisodeRunner.from_config_path(RIG, SHORT) as r:
        yield r


def cube_poses(runner: EpisodeRunner) -> list[tuple[float, ...]]:
    """Every loose body's pose, rounded, as a comparable tuple."""
    with runner.world.view() as (_model, data):
        return [tuple(round(float(v), 9) for v in data.qpos[c.qadr:c.qadr + 7])
                for c in runner.scene.cubes]


# ---- the layout ----------------------------------------------------------

def test_the_state_layout_is_the_recorder_s():
    """The vector this loop builds must be the one the dataset was written in.

    `episode.ARM_JOINT_ORDER` is `builder.SO101_JOINTS` (MJCF CamelCase) and the
    names reported are LeRobot snake_case, so the correspondence between the two
    orderings is an assumption living in two files that cannot import each other
    (`sim/arm.py` reaches lerobot; `sim/episode.py` must not). This is where it
    is pinned. Get it wrong and the wrist is driven by the shoulder's number,
    which does not fail: it scores zero and reads as a bad policy.
    """
    from haller_hmi.recorder import SO101_JOINT_ORDER
    from haller_hmi.sim.arm import LEROBOT_TO_MJCF

    assert tuple(LEROBOT_TO_MJCF.values()) == ARM_JOINT_ORDER
    assert tuple(LEROBOT_TO_MJCF.keys()) == SO101_JOINT_ORDER
    assert state_names(("left", "right")) == [
        f"{side}_{joint}"
        for side in ("left", "right")
        for joint in SO101_JOINT_ORDER
    ]


def test_the_runner_reads_the_rig_from_the_config(runner):
    cfg = load_config(RIG)

    assert runner.arm_ids == ["left", "right"]
    assert [c.key for c in runner.cameras] == ["top", "left_wrist", "right_wrist"]
    assert len(runner.state_names) == 12
    assert runner.state_names[5] == "left_gripper"
    assert runner.state_names[11] == "right_gripper"
    # Five cameras render on this rig and three are recorded. The two that are
    # not are the operator's, and a policy must not be handed them.
    assert len([c for c in cfg.cameras if c.record]) == 3
    assert len(cfg.cameras) == 5


def test_a_config_with_no_sim_arm_is_refused(tmp_path):
    """Loudly, at construction. A `source: real` rig has no bench to score on,
    and the alternative failure is `build_scene` raising "scene needs at least
    one arm" three frames deeper."""
    rig = tmp_path / "real.yaml"
    rig.write_text(
        "arms:\n"
        "  - id: right\n"
        "    model: so101_follower\n"
        "    port: /dev/ttyACM0\n"
        "    calibration_id: haller_right\n"
        "    source: real\n"
        "    enabled: true\n"
        "cameras: []\n"
    )

    with pytest.raises(ValueError, match="no enabled `source: sim` arm"):
        EpisodeRunner.from_config_path(rig, SHORT)


# ---- the observation -----------------------------------------------------

def test_the_observation_is_state_plus_one_frame_per_recorded_camera(runner):
    runner.reset_episode(0)

    obs = runner.observe()

    assert set(obs) == {
        STATE_KEY,
        image_key("top"),
        image_key("left_wrist"),
        image_key("right_wrist"),
    }
    assert len(obs[STATE_KEY]) == 12
    assert all(isinstance(v, float) for v in obs[STATE_KEY])
    assert frames_are_rgb(obs)
    assert obs[image_key("top")].shape == (720, 960, 3)
    assert obs[image_key("left_wrist")].shape == (480, 640, 3)
    assert obs[image_key("right_wrist")].shape == (480, 640, 3)


def test_every_recorded_camera_renders_something(runner):
    """Not a black frame. A `mujoco.Renderer` that never had `update_scene`
    called, or one aimed at a camera name that does not exist, hands back a
    uniform buffer - which is a perfectly valid uint8 array and would satisfy
    every shape assertion above while the policy saw nothing."""
    runner.reset_episode(0)

    obs = runner.observe()

    for key in ("top", "left_wrist", "right_wrist"):
        frame = obs[image_key(key)]
        assert frame.std() > 1.0, f"{key} rendered a flat frame"
        assert frame.max() > frame.min()


def test_render_cameras_off_removes_the_image_keys_rather_than_staling_them():
    """A driver that reads no image can skip 90% of this loop's wall time
    (measured 2026-08-31: 8.27 ms per tick with the three recorded cameras,
    0.81 without).

    The keys are REMOVED and not filled with an old frame: a driver that then
    reads one gets a KeyError instead of silently acting on the previous
    episode's view.
    """
    spec = EpisodeSpec(control_hz=30.0, max_episode_s=0.2, render_cameras=False)
    with EpisodeRunner.from_config_path(RIG, spec) as r:
        r.reset_episode(0)

        obs = r.observe()

        assert set(obs) == {STATE_KEY}


# ---- seeds ---------------------------------------------------------------

def test_the_same_seed_reproduces_the_bench_and_a_different_one_does_not(runner):
    """Seeds ARE the experiment. A success rate whose layouts cannot be
    re-created is a number nobody can argue with or against."""
    runner.reset_episode(7)
    first = cube_poses(runner)
    runner.reset_episode(8)
    other = cube_poses(runner)
    runner.reset_episode(7)
    again = cube_poses(runner)

    assert first == again
    assert first != other


def test_a_seed_list_replays_identically_including_a_repeat(runner):
    """The property the whole scoring story rests on: run the list twice, get
    the same benches, and a seed repeated WITHIN a list lays out the same bench
    both times rather than continuing a stream of randomness."""
    seeds = [3, 4, 3]

    def layouts() -> list[list[tuple[float, ...]]]:
        out = []
        for seed in seeds:
            runner.reset_episode(seed)
            out.append(cube_poses(runner))
        return out

    first, second = layouts(), layouts()

    assert first == second
    assert first[0] == first[2], "the repeated seed did not repeat its bench"
    assert first[0] != first[1]


def test_every_record_carries_the_seed_it_was_played_on(runner):
    driver = Hold()

    records = list(runner.run([11, 12, 13], driver))

    assert [r.seed for r in records] == [11, 12, 13]
    assert [r.episode for r in records] == [0, 1, 2]
    assert driver.seeds == [11, 12, 13], "the driver was not reset per seed"


def test_a_whole_episode_is_reproducible_from_its_seed(runner):
    """Not just the reset: the physics too. `mj_resetData` clears the solver's
    warm-start accelerations and `SceneController` zeroes them again, which is
    what makes two runs of one seed take the same first step rather than nearly
    the same one."""
    first = runner.run_episode(0, 21, Hold())
    end_first = runner.state()
    cubes_first = cube_poses(runner)

    second = runner.run_episode(0, 21, Hold())

    assert second.row() == first.row()
    assert runner.state() == pytest.approx(end_first, abs=0.0)
    assert cube_poses(runner) == cubes_first


def test_the_arms_start_every_episode_from_the_same_pose(runner):
    """`SceneController` deliberately does not touch the arms (`scene.py` says
    why), so if this loop did not reset them either, episode N+1 would start
    wherever N happened to leave them and the seed would describe half a bench.
    """
    runner.reset_episode(0)
    home = runner.state()

    runner.run_episode(0, 1, Constant([40.0] * 12))
    moved = runner.state()
    runner.reset_episode(999)

    assert moved != pytest.approx(home, abs=1e-6), "the driver moved nothing"
    assert runner.state() == pytest.approx(home, abs=1e-9)


# ---- termination ---------------------------------------------------------

def test_a_timeout_is_a_failure(runner):
    record = runner.run_episode(0, 5, Hold())

    assert record.success is False
    assert record.reason == REASON_TIMEOUT
    assert record.steps == SHORT.max_steps == 30


def test_the_loop_stops_when_the_monitor_says_success(runner, monkeypatch):
    """And on that poll, not a tick later. `TaskMonitor` has already required
    the placement to hold for `settle_s` SIM seconds before it answers True, so
    there is nothing left for extra steps to confirm."""
    stub = _MonitorSucceedingOnPoll(poll_number=6)
    monkeypatch.setattr(runner, "monitor", stub)

    record = runner.run_episode(0, 5, Hold())

    # Poll 1 is the pre-loop initialiser, so poll 6 lands after 5 steps.
    assert record.steps == 5
    assert record.success is True
    assert record.reason == REASON_SUCCESS
    assert stub.resets == 1, "the monitor's held-time streak was not reset"


def test_a_driver_that_stops_ends_the_episode_and_is_still_scored(runner):
    """Returning None is not a verdict. The predicate has already been polled on
    the state the last action produced, so a script that finishes on a winning
    placement gets the success it earned."""
    record = runner.run_episode(0, 5, StopAfter(4))

    assert record.steps == 4
    assert record.reason == REASON_DRIVER_STOP
    assert record.success is False


def test_a_driver_that_stops_on_a_winning_frame_still_wins(runner, monkeypatch):
    monkeypatch.setattr(runner, "monitor", _MonitorSucceedingOnPoll(poll_number=4))

    record = runner.run_episode(0, 5, StopAfter(3))

    assert record.steps == 3
    assert record.success is True
    assert record.reason == REASON_SUCCESS


# ---- the action --------------------------------------------------------

def test_the_action_is_degrees_on_every_joint_including_the_gripper(runner):
    """The rule `rollout_runner`'s docstring states at length. The sim jaw's
    range is the calibrated one, [-9.97, 100.27] deg, so a `[0, 1]` gripper
    arriving here would clamp to "almost shut" on every command."""
    runner.reset_episode(0)
    lo, hi = runner.world.joint_range_deg("left", "left_Jaw")
    assert (lo, hi) == pytest.approx((-9.969465635276324, 100.26761414789407))

    # No rate cap, so the goal reaches the actuator in one call.
    runner.spec = EpisodeSpec(control_hz=30.0, max_episode_s=1.0,
                              max_speed_deg_s=None)
    try:
        runner.reset_episode(0)
        target = [0.0] * 12
        target[5] = 88.1        # the docstring's own example, in degrees
        runner.act(target)

        assert _ctrl_deg(runner, "left_Jaw") == pytest.approx(88.1)
    finally:
        runner.spec = SHORT


def test_a_goal_past_a_joint_limit_is_clamped_to_the_MJCF_s_range(runner):
    runner.spec = EpisodeSpec(control_hz=30.0, max_episode_s=1.0,
                              max_speed_deg_s=None)
    try:
        runner.reset_episode(0)
        _lo, hi = runner.world.joint_range_deg("left", "left_Rotation")

        runner.act([1000.0] * 12)

        assert _ctrl_deg(runner, "left_Rotation") == pytest.approx(hi)
    finally:
        runner.spec = SHORT


def test_the_step_cap_is_measured_in_SIM_seconds(runner):
    """`max_speed_deg_s / control_hz` and never against the wall clock. This
    loop runs at about four times real time on this box, so a wall-clock cap
    would let the arms cover four times the ground per tick that the bench
    allows."""
    runner.reset_episode(0)
    per_tick = SHORT.max_speed_deg_s / SHORT.control_hz
    assert per_tick == pytest.approx(2.0)

    runner.act([1000.0] * 12)
    after_one = _ctrl_deg(runner, "left_Rotation")
    runner.act([1000.0] * 12)
    after_two = _ctrl_deg(runner, "left_Rotation")

    assert after_one == pytest.approx(per_tick)
    assert after_two == pytest.approx(2 * per_tick)


@pytest.mark.parametrize("action, match", [
    ([0.0] * 11, "expected 12"),
    ([0.0] * 13, "expected 12"),
    ([float("nan")] + [0.0] * 11, "non-finite"),
    ([float("inf")] + [0.0] * 11, "non-finite"),
    (["nope"] + [0.0] * 11, "non-numeric"),
])
def test_a_malformed_action_is_refused(runner, action, match):
    """A NaN in `data.ctrl` does not raise. It propagates into qpos on the next
    step, every downstream reading turns to NaN, the predicate silently stops
    firing, and the run reports 0% about a policy that may be fine."""
    runner.reset_episode(0)

    with pytest.raises(ValueError, match=match):
        runner.act(action)


# ---- the clock -----------------------------------------------------------

def test_the_episode_clock_tracks_the_control_rate(runner):
    """30 ticks at 30 Hz is 1.000 sim seconds, not 1.020.

    `1 / 30` is 16.67 of the model's 0.002 s timesteps. A target recomputed from
    the current time each tick rounds up to 17 every time and the clock runs 2%
    fast against `settle_s` and every other sim-second threshold in
    `sim/task.py`.
    """
    record = runner.run_episode(0, 0, Hold())

    assert record.steps == 30
    assert record.sim_s == pytest.approx(1.0, abs=1e-9)
    assert float(runner.world.data.time) == pytest.approx(1.0, abs=1e-9)


def test_the_stepper_thread_is_never_started(runner):
    """This loop owns the physics clock. A background stepper would make the
    number of physics steps per control tick a property of how loaded the box
    was, and two runs of one seed would diverge."""
    assert runner.world.is_running() is False
    runner.run_episode(0, 0, Hold())
    assert runner.world.is_running() is False


# ---- provenance ----------------------------------------------------------

def test_provenance_carries_the_predicate_AND_the_numbers_it_fired_against(runner):
    """`TaskMonitor.provenance()` names the clauses; it does not say what
    `settle_s` was set to. A rate recorded without its thresholds is
    reproducible only up to the numbers that decide it."""
    prov = runner.provenance()

    assert prov["predicate"]["predicate"] == "haller_hmi.sim.task.cube_placed"
    assert prov["thresholds"] == {
        "zone_inset_m": 0.01, "lin_vel_eps": 0.01, "ang_vel_eps": 0.1,
        "settle_s": 0.5, "require_release": True,
    }
    # The seed reproduces the layout only against the spec that interpreted it.
    assert prov["scene"]["xy_jitter_m"] == 0.04
    assert prov["scene"]["min_separation_m"] == 0.06
    assert prov["episode"]["control_hz"] == 30.0
    assert prov["episode"]["max_steps"] == 30
    assert prov["camera_keys"] == ["top", "left_wrist", "right_wrist"]
    assert prov["state_names"] == runner.state_names
    assert prov["state_unit"] == prov["action_unit"] == "deg"
    # It ends up inside `sim_eval_summary.json`, so it has to survive the trip.
    assert json.loads(json.dumps(prov)) == prov


def test_the_monitor_follows_the_scene_from_the_one_config_key(tmp_path):
    """`sim_task: insertion` selects the bore scene AND the bore predicate
    together. Scoring an insertion bench with the cube predicate would label
    every episode a failure and read as a policy problem."""
    from haller_hmi.sim.task import InsertionMonitor

    rig = RIG.read_text().replace("sim_cubes: 3", "sim_cubes: 0\nsim_task: insertion")
    path = tmp_path / "insertion.yaml"
    path.write_text(rig)

    with EpisodeRunner.from_config_path(path, SHORT) as r:
        assert isinstance(r.monitor, InsertionMonitor)
        assert r.provenance()["predicate"]["task"] == "bimanual_insertion"


def test_the_wrong_task_s_thresholds_are_refused_rather_than_ignored(tmp_path):
    """A caller handing the cube thresholds to an insertion bench has a belief
    about what is being measured. Substituting the defaults silently would leave
    that belief intact while the number came from somewhere else."""
    from haller_hmi.sim.task import InsertionSpec

    rig = RIG.read_text().replace("sim_cubes: 3", "sim_cubes: 0\nsim_task: insertion")
    path = tmp_path / "insertion.yaml"
    path.write_text(rig)

    with pytest.raises(ValueError, match="scored by InsertionSpec"):
        EpisodeRunner.from_config_path(path, SHORT,
                                       success_spec=SuccessSpec())
    with pytest.raises(ValueError, match="scored by SuccessSpec"):
        EpisodeRunner.from_config_path(RIG, SHORT, success_spec=InsertionSpec())


def test_the_real_monitor_is_polled_every_tick_without_error(runner):
    """The predicate runs over live contacts on a bench whose cubes have just
    been re-dealt, once per control tick for the whole episode. Nothing here
    asserts a VERDICT - `tests/sim/test_task_success.py` owns that - only that
    the authority answers, in its own shape, every time it is asked."""
    seen = []
    real_poll = runner.monitor.poll

    def watched():
        verdict = real_poll()
        seen.append(verdict)
        return verdict

    runner.monitor.poll = watched
    try:
        record = runner.run_episode(0, 2, Hold())
    finally:
        del runner.monitor.poll

    # One pre-loop initialiser plus one per step.
    assert len(seen) == record.steps + 1 == 31
    for verdict in seen:
        assert {"success", "held_s", "per_cube", "target", "settle_s",
                "sim_time_s"} <= set(verdict)
        assert isinstance(verdict["success"], bool)
        assert set(verdict["per_cube"]) == {"cube_0", "cube_1", "cube_2"}
    # Sim time advanced monotonically under it, which is the clock `settle_s`
    # is measured in.
    times = [v["sim_time_s"] for v in seen]
    assert times == sorted(times)
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(1.0, abs=1e-9)


# ---- lifecycle -----------------------------------------------------------

def test_close_is_idempotent_and_a_closed_runner_refuses_to_render():
    """Renderers left to `Renderer.__del__` raise `EGLError` at interpreter
    shutdown, which lands in `run.log` above the result line and reads exactly
    like a run that crashed on the way out."""
    r = EpisodeRunner.from_config_path(RIG, SHORT)
    r.reset_episode(0)
    r.observe()
    assert r._renderers, "nothing was rendered, so nothing is being tested"

    r.close()
    r.close()

    assert r._renderers == {}
    with pytest.raises(RuntimeError, match="closed"):
        r.observe()


def test_the_module_imports_without_lerobot_or_torch():
    """`sim/episode.py` runs in the serving process AND in the lab child, and a
    scripted driver must not pay for a CUDA context to move a cube. In a
    subprocess, because pytest has already imported half the world into this
    one."""
    import subprocess
    import sys

    probe = ("import sys; from haller_hmi.sim import episode as m; "
             "print('torch' in sys.modules, 'lerobot' in sys.modules, bool(m))")

    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, check=True, timeout=120, cwd=str(BACKEND))

    assert out.stdout.strip() == "False False True", out.stderr


# ---- helpers -------------------------------------------------------------

def _ctrl_deg(runner: EpisodeRunner, joint: str) -> float:
    """The actuator target for one MJCF joint, in degrees.

    Read off `data.ctrl` rather than off the runner's own bookkeeping: what the
    clamp and the cap are for is what actually reaches the actuator.
    """
    with runner.world.view() as (model, data):
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint)
        assert act_id >= 0, f"no actuator named {joint!r}"
        return math.degrees(float(data.ctrl[act_id]))


def test_numpy_state_is_still_plain_floats(runner):
    """`observation.state` is a list of Python floats and not a numpy array.

    The protocol says `list[float]`, and a numpy array satisfies most duck-typed
    uses right up to `json.dumps`, which is where a driver's own logging would
    find out.
    """
    runner.reset_episode(0)

    state = runner.observe()[STATE_KEY]

    assert isinstance(state, list)
    assert not isinstance(state, np.ndarray)
    assert json.dumps(state)
