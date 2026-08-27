# hmi/backend/tests/lab/test_rollout_runner.py
"""`runners/rollout_runner.py` — without hardware, without lerobot, without a bus.

Every test here runs under the SERVING venv (lerobot 0.5.1), which cannot run
the thing being tested: the real child runs under `~/venvs/haller-lab` at
lerobot 0.6.1 + torch 2.11.0+cu130, and `lerobot.scripts.lerobot_rollout` does
not exist in 0.5.1 at all. That is the point rather than a limitation — the
module keeps every heavy import inside `_rollout`'s call tree, so the plan it
builds, the handshake it declares, the messages it sends, its refusals and its
rate gate are all reachable from a process that has no policy stack.

**Nothing here moves an arm and nothing here loads a checkpoint.** A rollout
moves real hardware, so a runner whose only testable surface was "run it" would
have no tests worth having.

The load-bearing one is `test_the_forbidden_path_appears_nowhere_in_this_module`.
It greps the module's own source for the kit's giveaways — the kit's rollout
child opens the serial port, pre-syncs the goal registers, energises the servos
and drives the arm — and asserts none of them appear. It is a crude test and it
is the right kind of crude: the ruling it guards ("the child owns the policy,
never the bus", contract addendum 2026-08-27) is architectural, so it cannot be
checked by exercising behaviour, and a future edit that "just connects to the
follower to read the joint angles" would take `/estop` out of the loop with no
test failing anywhere else.

The socket tests use a REAL loopback listener rather than a mock, and never a
background thread: `listen(1)` accepts the connection into the backlog, so the
test can connect, accept and then write the exact bytes it wants the child to
read, all on one thread. It is a socket, not a stand-in ingest — it never
answers on its own, and every reply it sends is written by the test that wants
it.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from haller_hmi import safety
from haller_hmi.runners import rollout_runner as rr

BACKEND = Path(__file__).resolve().parents[2]

LEROBOT_HOME = Path("/home/odesha/robot-data/lerobot")
BIMANUAL = "local/haller_pick_the_red_cube_and_place_it_in_the_box"

#: Haller's recorder spelling: side-prefixed, no `.pos` suffix, left arm first.
BIMANUAL_NAMES = [
    "left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
    "left_wrist_flex", "left_wrist_roll", "left_gripper",
    "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
    "right_wrist_flex", "right_wrist_roll", "right_gripper",
]

#: The kit's dataset spelling: no side prefix, `.pos` suffix. `local/so101_pick_cube`
#: is recorded like this, and it is the case where the columns CANNOT say which
#: arm they came from.
UNPREFIXED_NAMES = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
]


# ---- helpers --------------------------------------------------------------

def make_spec(tmp_path: Path, **over) -> dict:
    """The minimum `lab/runs.launch` puts on disk, plus what a rollout needs.

    `run_id` and `run_dir` are stamped in by `launch`, so every real spec has
    them. `policy_path` must EXIST — `build_plan` checks it before anything
    else costs a CUDA context — so it is a real directory here, empty because
    nothing under test ever opens it.
    """
    run_dir = tmp_path / "rollout-20260827-143000"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    spec = {
        "run_id": "rollout-20260827-143000",
        "run_dir": str(run_dir),
        "policy_path": str(ckpt),
        "duration_s": 30.0,
        "control_hz": 30.0,
        "action_names": list(BIMANUAL_NAMES),
    }
    spec.update(over)
    return spec


def rig_of(names) -> rr.RigSpec:
    return rr.RigSpec.from_info({"features": {"observation.state": {"names": list(names)}}})


def metric_rows(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sidecar(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / rr.RATE_SIDECAR).read_text())


def free_port() -> int:
    """A loopback port with nothing on it. Bound, read back, released."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture()
def peer():
    """`(client, server_end)` over a real connected loopback TCP pair.

    No thread: `listen(1)` lets the connect complete into the backlog, so
    `connect()` and `accept()` both return on this one thread. The test writes
    every byte the child reads, so nothing here implements the ingest — which
    is the thing this chunk is explicitly not allowed to build.
    """
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    host, port = listener.getsockname()
    client = rr.IngestClient(host, port)
    client.connect()
    server, _ = listener.accept()
    server.settimeout(2.0)
    try:
        yield client, server
    finally:
        server.close()
        client.close()
        listener.close()


def read_line(sock: socket.socket) -> dict:
    """One newline-delimited JSON message off the wire."""
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(65536)
        assert chunk, "peer closed without sending a line"
        buf += chunk
    return json.loads(buf.split(b"\n", 1)[0])


def send_line(sock: socket.socket, msg: dict) -> None:
    sock.sendall((json.dumps(msg) + "\n").encode())


# ---- the two-interpreter rule --------------------------------------------

def test_importing_the_runner_pulls_in_neither_lerobot_nor_torch():
    """`lab/runs.py` names this module as a STRING and never imports it — but
    `_common` imports `lab.runs` back, and a module-scope `import torch` here
    would ride that edge into the serving process, which owns the bus and the
    teleop latency path."""
    probe = ("import sys; from haller_hmi.runners import rollout_runner as m; "
             "print('torch' in sys.modules, 'lerobot' in sys.modules, bool(m))")

    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, check=True, timeout=120, cwd=str(BACKEND))

    assert out.stdout.strip() == "False False True", out.stderr


def test_dry_run_prints_the_plan_and_the_handshake_and_opens_no_socket(tmp_path):
    """A subprocess, because pytest has already imported half the world into
    this one and `sys.modules` here would prove nothing.

    Both halves matter. No heavy import, so this is usable as a preflight from a
    box with the GPU busy; and no socket, so it is usable with the arms powered
    down and the HMI stopped — `socket.socket` and `socket.create_connection`
    are replaced with something that raises, so a dry run that reached for
    either fails the subprocess rather than quietly connecting."""
    spec = make_spec(tmp_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    probe = textwrap.dedent(f"""
        import socket, sys
        def _no(*a, **k):
            raise AssertionError("a dry run opened a socket")
        socket.socket = _no
        socket.create_connection = _no
        sys.argv = ["rollout_runner", {str(spec_path)!r}, "--dry-run"]
        from haller_hmi.runners import rollout_runner
        code = rollout_runner.main()
        print("EXIT", code)
        print("HEAVY", "torch" in sys.modules, "lerobot" in sys.modules)
    """)

    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, check=True, timeout=120, cwd=str(BACKEND))

    lines = out.stdout.strip().splitlines()
    plan = json.loads(lines[0].removeprefix("rollout plan: "))
    hello = json.loads(lines[1].removeprefix("handshake: "))
    assert plan["ingest_url"] == rr.DEFAULT_INGEST_URL
    assert plan["duration_s"] == 30.0
    assert hello == rr.hello_message(rr.build_plan(spec))
    assert lines[-2] == "EXIT 0"
    assert lines[-1] == "HEAVY False False"


def test_the_forbidden_path_appears_nowhere_in_this_module():
    """The architectural tripwire. See the module docstring.

    Handing the bus to a child means `/estop` cannot drop torque during a
    rollout, and a SIGSEGV or a SIGKILL in that child leaves the arm energised
    on its last goal until someone reaches the bench PSU. The ruling that
    closed that path cannot be checked by exercising behaviour, so it is
    checked by reading the source."""
    source = Path(rr.__file__).read_text()

    for needle in ("follower.connect", "enable_torque",
                   "presync_goal_positions", "SO101Follower"):
        assert needle not in source, (
            f"{needle!r} is in rollout_runner.py — this child streams targets "
            "to the server, which owns the bus. See the contract's rollout "
            "addendum before removing this assertion."
        )


# ---- the handshake --------------------------------------------------------

def test_the_handshake_declares_degrees_and_the_declared_rate(tmp_path):
    """The unit is a property of the SOURCE, declared once.

    A source declaring degrees gets no gripper special-case downstream. That is
    what stops a legitimate 88.1 deg gripper command being read as a `[0, 1]`
    fraction, clamped to 1.0 and opened fully."""
    hello = rr.hello_message(rr.build_plan(make_spec(tmp_path, control_hz=30.0)))

    assert hello["type"] == "policy_hello"
    assert hello["unit"] == "deg"
    assert hello["control_hz_declared"] == 30.0
    assert hello["run_id"] == "rollout-20260827-143000"
    assert hello["rig"] == "bimanual"


def test_the_handshake_is_exactly_the_frozen_keys(tmp_path):
    """Frozen with Track A. An extra key here is a message to the integrator,
    not an edit — the server validates the shape it was promised."""
    hello = rr.hello_message(rr.build_plan(make_spec(tmp_path)))

    assert set(hello) == {"type", "run_id", "unit", "control_hz_declared", "rig"}


def test_the_declared_rate_falls_back_to_the_datasets_fps(tmp_path):
    """`control_hz` is what the policy is asked to run at; a spec that only
    carries the dataset's `fps` means the same thing, because the rate a policy
    should run at IS the rate its data was recorded at."""
    plan = rr.build_plan(make_spec(tmp_path, control_hz=None, fps=25.0))

    assert plan["control_hz_declared"] == 25.0


# ---- the action message ---------------------------------------------------

def test_an_action_message_is_degrees_with_no_per_message_unit_field():
    """No per-message `unit`, on purpose: a per-message unit is a per-message
    chance to disagree with the declaration."""
    rig = rig_of(BIMANUAL_NAMES)
    values = [float(i) for i in range(12)]
    values[5] = 88.1        # the left gripper, in DEGREES

    msg = rr.action_message("rollout-1", 41, rr.action_from_vector(values, rig),
                            t_ms=1724759112345)

    assert set(msg) == {"type", "seq", "t_ms", "run_id", "action"}
    assert "unit" not in msg
    assert all("unit" not in side for side in msg["action"].values())
    assert msg["type"] == "policy_action"
    assert msg["seq"] == 41
    assert msg["t_ms"] == 1724759112345
    assert msg["run_id"] == "rollout-1"
    # 88.1 survives as 88.1. Through the `[0, 1]` path it would have clamped to
    # 1.0 and opened the jaw fully, silently, in the direction of dropping the
    # object.
    assert msg["action"]["left"]["gripper"] == 88.1


def test_an_action_carries_the_plain_joint_names_goal_deg_uses():
    """The SAME plain joint -> float per side that
    `human_teleop.status()["goal_deg"]` publishes, so it lands on the commit
    chain exactly as any other leader's target does. Not `left_shoulder_pan`,
    not `shoulder_pan.pos`."""
    rig = rig_of(BIMANUAL_NAMES)

    action = rr.action_from_vector([0.0] * 12, rig)

    assert sorted(action) == ["left", "right"]
    assert sorted(action["left"]) == [
        "elbow_flex", "gripper", "shoulder_lift", "shoulder_pan",
        "wrist_flex", "wrist_roll",
    ]
    assert sorted(action["right"]) == sorted(action["left"])


def test_a_solo_rig_has_no_key_for_the_absent_side():
    """ABSENT, not present-and-empty. `{"right": {}}` reads as "the right arm
    was commanded to nothing", which is a different claim and a wrong one."""
    rig = rig_of(UNPREFIXED_NAMES)

    action = rr.action_from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], rig, side="left")

    assert list(action) == ["left"]
    assert "right" not in action
    assert action["left"]["shoulder_pan"] == 1.0
    assert action["left"]["gripper"] == 6.0


def test_a_solo_dataset_recorded_on_the_right_arm_keys_the_right_side():
    rig = rig_of(UNPREFIXED_NAMES)

    action = rr.action_from_vector([0.0] * 6, rig, side="right")

    assert list(action) == ["right"]


def test_an_unprefixed_dataset_refuses_to_guess_which_arm(tmp_path):
    """The two arms are 40 cm apart. A rollout aimed at the wrong one is a
    collision, not a typo — and the refusal happens at PLAN time, on the spec,
    not on the first action message with the arm already live."""
    with pytest.raises(SystemExit) as excinfo:
        rr.build_plan(make_spec(tmp_path, action_names=UNPREFIXED_NAMES))

    assert "spec['side']" in str(excinfo.value)


def test_a_side_prefixed_dataset_needs_no_side_in_the_spec(tmp_path):
    """The columns already say which arm they came from, so nothing is guessed
    and nothing is asked for."""
    plan = rr.build_plan(make_spec(tmp_path, action_names=BIMANUAL_NAMES))

    assert plan["sides"] == ["left", "right"]
    assert plan["rig"] == "bimanual"


# ---- the bounded run ------------------------------------------------------

@pytest.mark.parametrize("duration", [0, -1, None])
def test_a_duration_that_is_not_positive_is_refused(tmp_path, duration):
    with pytest.raises(SystemExit) as excinfo:
        rr.build_plan(make_spec(tmp_path, duration_s=duration))

    assert "duration_s must be > 0" in str(excinfo.value)


def test_a_duration_past_the_ceiling_is_refused(tmp_path):
    """A policy loop started from a browser button must not be able to run
    until someone notices."""
    with pytest.raises(SystemExit) as excinfo:
        rr.build_plan(make_spec(tmp_path, duration_s=rr.MAX_ROLLOUT_DURATION_S + 1))

    assert str(int(rr.MAX_ROLLOUT_DURATION_S)) in str(excinfo.value)


def test_a_checkpoint_that_is_not_there_is_named(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        rr.build_plan(make_spec(tmp_path, policy_path=str(tmp_path / "nope")))

    assert str(tmp_path / "nope") in str(excinfo.value)


def test_no_checkpoint_at_all_names_the_spec_key(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        rr.build_plan(make_spec(tmp_path, policy_path=""))

    assert "policy_path" in str(excinfo.value)


# ---- the ingest endpoint --------------------------------------------------

def test_the_ingest_endpoint_is_the_spec_then_the_env_then_the_default(monkeypatch):
    monkeypatch.delenv(rr.INGEST_URL_ENV, raising=False)
    assert rr.ingest_url({}) == rr.DEFAULT_INGEST_URL

    monkeypatch.setenv(rr.INGEST_URL_ENV, "tcp://10.0.0.2:9000")
    assert rr.ingest_url({}) == "tcp://10.0.0.2:9000"

    # Spec wins, so one run can be pointed elsewhere without touching the
    # environment every other run on this server inherits.
    assert rr.ingest_url({"ingest_url": "tcp://127.0.0.1:1234"}) == "tcp://127.0.0.1:1234"


def test_an_empty_ingest_endpoint_names_the_spec_key_and_the_env_var():
    with pytest.raises(SystemExit) as excinfo:
        rr.parse_ingest("")

    detail = str(excinfo.value)
    assert "ingest_url" in detail
    assert rr.INGEST_URL_ENV in detail
    assert rr.DEFAULT_INGEST_URL in detail


def test_an_unsupported_ingest_scheme_is_named_not_coerced():
    """The coercion nobody notices is the one that connects to the wrong
    thing. Track A owns the ingest; if it is not TCP, this side changes."""
    with pytest.raises(SystemExit) as excinfo:
        rr.parse_ingest("ws://127.0.0.1:8000/ws/policy")

    assert "'ws'" in str(excinfo.value)


def test_a_missing_ingest_is_refused_with_a_message_naming_it():
    """Nothing listening is a REFUSAL, not a fallback.

    The whole architecture is that the server owns the bus and this child only
    streams targets at it, so when the server is not there this child has
    nothing it is allowed to do instead."""
    port = free_port()
    client = rr.IngestClient("127.0.0.1", port)

    with pytest.raises(SystemExit) as excinfo:
        client.connect(timeout=2.0)

    detail = str(excinfo.value)
    assert f"tcp://127.0.0.1:{port}" in detail
    assert rr.INGEST_URL_ENV in detail


# ---- the preflight is the server's answer ---------------------------------

def test_the_preflight_refuses_on_the_servers_own_sentence(peer):
    """This child cannot see the recorder or the teleop session, so it does not
    try: `lab/lease.bus_conflict` runs in the process that can, and its sentence
    comes back here as the refusal."""
    client, server = peer
    refusal = ("cannot start a rollout: an episode is being recorded into "
               "local/foo. Stop the recording first.")
    send_line(server, {"type": rr.REFUSED_TYPE, "detail": refusal})

    with pytest.raises(SystemExit) as excinfo:
        client.handshake({"type": rr.HELLO_TYPE}, timeout=2.0)

    assert str(excinfo.value) == refusal


def test_an_ack_that_says_not_ok_is_also_a_refusal(peer):
    client, server = peer
    send_line(server, {"type": rr.ACK_TYPE, "ok": False,
                       "detail": "cannot start a rollout: a teleop session is driving"})

    with pytest.raises(SystemExit) as excinfo:
        client.handshake({"type": rr.HELLO_TYPE}, timeout=2.0)

    assert "teleop session" in str(excinfo.value)


def test_an_unacknowledged_handshake_names_the_endpoint(peer):
    """A connection that is accepted and never answered is the shape a
    half-built ingest has, and it must not look like a hang."""
    client, _server = peer

    with pytest.raises(SystemExit) as excinfo:
        client.handshake({"type": rr.HELLO_TYPE}, timeout=0.2)

    detail = str(excinfo.value)
    assert client.url in detail
    assert rr.ACK_TYPE in detail


def test_the_handshake_goes_out_before_anything_else_and_carries_the_unit(peer, tmp_path):
    client, server = peer
    hello = rr.hello_message(rr.build_plan(make_spec(tmp_path)))
    send_line(server, {"type": rr.ACK_TYPE, "ok": True, "server_pid": 4242})

    ack = client.handshake(hello, timeout=2.0)

    assert read_line(server) == hello
    assert ack["ok"] is True
    # The server's pid, so the bus check can tell the NORMAL holder of the port
    # from a foreign one.
    assert client.server_pid == 4242


def test_an_action_survives_the_wire_as_degrees(peer, tmp_path):
    """End to end over a real socket: what the child sends is what the frozen
    shape says, newline-delimited, one message per line."""
    client, server = peer
    rig = rig_of(BIMANUAL_NAMES)
    values = [0.0] * 12
    values[5] = 88.1

    client.send(rr.action_message("rollout-1", 0, rr.action_from_vector(values, rig)))

    msg = read_line(server)
    assert msg["type"] == "policy_action"
    assert msg["action"]["left"]["gripper"] == 88.1
    assert "unit" not in msg


# ---- the bus check --------------------------------------------------------

def test_the_bus_check_never_reports_the_server_itself(monkeypatch):
    """**It must NOT refuse because the HMI holds the port.** Under this
    architecture the server holding the bus is the NORMAL, REQUIRED state — it
    is what lets `/estop` walk every motor in-process during a rollout. A check
    that treated the server's fd as a conflict would refuse every rollout
    forever and look like a hardware fault."""
    monkeypatch.setattr(rr, "port_holders", lambda device: [
        "pid 4242: uvicorn haller_hmi.server:app",
        "pid 9001: python -m some.other.thing",
    ])

    holders = rr.foreign_port_holders("/dev/haller_arm_leader", 4242)

    assert holders == ["pid 9001: python -m some.other.thing"]


def test_the_bus_check_reports_nothing_when_the_server_declared_no_pid(monkeypatch):
    """A holder list this child cannot ATTRIBUTE is not evidence of a conflict.
    The server's own answer at handshake is authoritative anyway, and reporting
    an unattributable list would refuse every rollout on the server's own fd."""
    monkeypatch.setattr(rr, "port_holders", lambda device: ["pid 4242: uvicorn"])

    assert rr.foreign_port_holders("/dev/haller_arm_leader", None) == []


def test_the_bus_check_asks_nothing_without_a_device(monkeypatch):
    called = []
    monkeypatch.setattr(rr, "port_holders", lambda device: called.append(device) or [])

    assert rr.foreign_port_holders("", 4242) == []
    assert called == []


# ---- the rate gate --------------------------------------------------------

def test_the_rate_floor_is_a_fraction_of_the_declared_rate():
    assert rr.rate_floor_hz(30.0) == pytest.approx(30.0 * safety.MIN_RATE_FRACTION)


def test_the_rate_fraction_IS_track_as_constant_and_not_a_local_copy():
    """The floor must come FROM `safety`, not merely be a number that agrees
    with it.

    This was a name-probe with a local 0.90 fallback until Track A published
    `MIN_RATE_FRACTION = 0.9`. Both said 0.9, so every reading agreed whether
    the probe was aimed correctly or not — "a test that passes on the fallback
    path cannot tell the two apart". The identity check below is the one that
    can: it moves the published value and requires this module to move with it,
    which no local copy can do.
    """
    assert rr.rate_floor_fraction() is safety.MIN_RATE_FRACTION


def test_moving_the_published_value_moves_the_floor(monkeypatch):
    """0.5, deliberately not 0.9: a value the old fallback could have produced
    would pass whether or not this module reads Track A at all.

    Patched on `rr`, not on `safety`, because `from ..safety import
    MIN_RATE_FRACTION` binds the value at IMPORT time — which is right for a
    constant and is exactly what makes the identity assertion above meaningful.
    The two tests do different halves: that one proves the floor IS Track A's
    object, this one proves `rate_floor_hz` derives from it rather than from a
    second literal.
    """
    monkeypatch.setattr(rr, "MIN_RATE_FRACTION", 0.5)

    assert rr.rate_floor_fraction() == 0.5
    assert rr.rate_floor_hz(30.0) == 15.0


def test_the_floor_is_imported_rather_than_probed_with_a_fallback():
    """The fallback is GONE, and its absence is the point.

    Before publication a missing constant was normal. After it, a lookup that
    cannot find it is a rename, a move or a typo — never normal — and swallowing
    that would hide the exact drift the probe existed to prevent. A warning was
    the softer option and is not enough: it still resolves to a number and still
    runs the arm. Absence is now an ImportError at module load, the loudest and
    earliest it can be.
    """
    source = Path(rr.__file__).read_text()

    assert "MIN_CONTROL_HZ_FRACTION" not in source, "the local fallback is back"
    assert "_PUBLISHED_FRACTION" not in source, "the name probe is back"
    assert "from ..safety import MIN_RATE_FRACTION" in source


def test_the_probe_names_a_module_that_stays_stdlib_only():
    """The probe's module is imported by this child and by these tests, so it
    must not reach lerobot.

    `haller_hmi.tick` will own the rate MEASUREMENT and therefore `arm.py` and
    therefore lerobot; `haller_hmi.safety` imports enum, math and dataclasses.
    That constraint is why the constant lives in safety.py, and this asserts the
    probe keeps pointing at a light module rather than following the constant if
    it ever moves somewhere heavy.

    In a SUBPROCESS, for the reason
    `test_importing_the_runner_pulls_in_neither_lerobot_nor_torch` above already
    gives: by the time the whole backend suite reaches this file, other modules
    have imported lerobot into THIS interpreter, so `sys.modules` here says
    nothing about what the probe pulled in. Asserting it in-process passes
    alone and fails in the suite — which is what the first version of this test
    did.
    """
    probe = ("import sys; import haller_hmi.safety; "
             "print('torch' in sys.modules, 'lerobot' in sys.modules)")

    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, check=True, timeout=120, cwd=str(BACKEND))

    assert out.stdout.strip() == "False False", out.stderr


def test_the_rate_gate_refuses_below_the_floor_and_records_both_numbers(tmp_path):
    """The kit's own rollout ran at 4.8 Hz against a 30 Hz target. That is not
    a slow rollout, it is a different dynamical system: action deltas sized for
    33 ms steps applied over 208 ms."""
    with pytest.raises(SystemExit) as excinfo:
        rr.gate_control_rate(tmp_path, 30.0, 4.8)

    detail = str(excinfo.value)
    assert "4.80 Hz" in detail
    assert "allow_slow" in detail

    record = sidecar(tmp_path)
    assert record["control_hz_declared"] == 30.0
    assert record["control_hz_measured"] == 4.8
    assert record["control_hz_override"] is False
    assert record["refused"]


def test_the_rate_gate_permits_with_the_override_and_records_both_numbers(tmp_path):
    """Override or not, the numbers are stamped. The kit's failure was not that
    a 4.8 Hz run happened — it was that "success" was reported with that number
    attached nowhere."""
    record = rr.gate_control_rate(tmp_path, 30.0, 4.8, override=True)

    assert record["control_hz_declared"] == 30.0
    assert record["control_hz_measured"] == 4.8
    assert record["control_hz_override"] is True
    assert record["refused"] == ""
    assert sidecar(tmp_path) == record


def test_a_healthy_rate_is_recorded_too(tmp_path):
    record = rr.gate_control_rate(tmp_path, 30.0, 29.4)

    assert record["refused"] == ""
    assert sidecar(tmp_path)["control_hz_measured"] == 29.4


def test_the_rate_numbers_reach_the_metrics_stream_the_runs_page_already_tails(tmp_path):
    """`lab/runs.write_result` carries exactly four keys and is shared with
    `load()`, so it is not the place to add fields. The sidecar is the run's
    final word; this row is what puts the rate on the timeline without a new
    route to read it."""
    with pytest.raises(SystemExit):
        rr.gate_control_rate(tmp_path, 30.0, 4.8)
    rr.gate_control_rate(tmp_path, 30.0, 28.0, source="run")

    rows = metric_rows(tmp_path)
    assert [r["kind"] for r in rows] == ["rate", "rate"]
    assert rows[0]["control_hz_measured"] == 4.8
    assert rows[0]["control_hz_source"] == "warmup"
    assert rows[1]["control_hz_measured"] == 28.0
    assert rows[1]["control_hz_source"] == "run"


def test_the_refusal_names_the_declared_rate_and_the_floor():
    detail = rr.control_rate_refusal(30.0, 4.8, fraction=0.9)

    assert "27.00 Hz" in detail
    assert "30.00 Hz" in detail
    assert "90%" in detail


def test_the_override_changes_only_whether_the_numbers_stop_the_run():
    assert rr.control_rate_refusal(30.0, 4.8, override=True, fraction=0.9) is None
    assert rr.control_rate_refusal(30.0, 4.8, override=False, fraction=0.9) is not None


# ---- the mid-run alarm ----------------------------------------------------

def test_a_healthy_run_never_alerts():
    watch = rr.RateWatch(27.0)

    for i in range(200):
        assert watch.tick(now=i / 30.0) is None


def test_one_slow_step_is_not_an_alert():
    """A single 0.5 s stall at 30 Hz is a 2 Hz instantaneous reading and not a
    fact about the run. Measuring over a trailing window is what stops the
    alarm firing on garbage collection."""
    watch = rr.RateWatch(27.0)
    t = 0.0
    for _ in range(60):
        watch.tick(now=t)
        t += 1 / 30.0
    t += 0.5

    assert watch.tick(now=t) is None


def test_a_sustained_slump_alerts_after_two_seconds_and_only_once():
    """Alert, not abort: the start gate is what refuses, and once the arm is
    moving the server's own staleness handling decides a source has stopped
    being usable. A second authority racing it tears down healthy runs."""
    watch = rr.RateWatch(27.0, under_s=2.0)
    alerts = []
    t = 0.0
    for _ in range(40):            # 40 ticks at 5 Hz = 8 s under the floor
        alert = watch.tick(now=t)
        if alert:
            alerts.append((t, alert))
        t += 0.2

    assert len(alerts) == 1
    first_t, message = alerts[0]
    assert 2.0 < first_t <= 2.8, f"alerted at {first_t:.1f}s"
    assert "27.00 Hz" in message


def test_recovering_rearms_the_alarm():
    """A run that slumps, recovers and slumps again has two things worth
    hearing about, not one."""
    watch = rr.RateWatch(27.0, under_s=2.0)
    alerts = 0
    t = 0.0
    for _ in range(2):
        for _ in range(20):        # 4 s at 5 Hz
            alerts += bool(watch.tick(now=t))
            t += 0.2
        for _ in range(120):       # 4 s at 30 Hz
            alerts += bool(watch.tick(now=t))
            t += 1 / 30.0

    assert alerts == 2


def test_the_rate_is_unknown_before_two_ticks():
    watch = rr.RateWatch(27.0)

    assert watch.rate() is None
    assert watch.tick(now=0.0) is None
    assert watch.rate() is None


# ---- the inbound observation ----------------------------------------------

def test_an_observation_is_decoded_into_what_inference_needs():
    obs = rr.decode_observation({
        "type": rr.OBSERVATION_TYPE, "seq": 7, "t_ms": 1724759112345,
        "state": [1, 2, 3], "images": {"top": "AAAA"},
    })

    assert obs == {"state": [1.0, 2.0, 3.0], "images": {"top": "AAAA"},
                   "seq": 7, "t_ms": 1724759112345}


@pytest.mark.parametrize("msg", [
    {"type": rr.ACK_TYPE, "ok": True},
    {"type": rr.OBSERVATION_TYPE},          # no state: nothing to infer from
    {"state": [1, 2, 3]},                   # no type
    "not a dict",
])
def test_anything_that_is_not_an_observation_decodes_to_none(msg):
    """An unrecognised inbound message is IGNORED, never guessed at. The
    inbound shape is Track A's and does not exist yet; a decoder that tried to
    salvage an unknown message would be inventing the contract."""
    assert rr.decode_observation(msg) is None


# ---- against the real recording -------------------------------------------

@pytest.mark.skipif(
    not (LEROBOT_HOME / BIMANUAL / "meta" / "info.json").is_file(),
    reason=f"{BIMANUAL} is not on this machine",
)
def test_the_real_bimanual_dataset_resolves_to_the_shape_the_server_expects(monkeypatch):
    """READ-ONLY against the real recording — one `meta/info.json` read, no
    write of any kind. This box has NO BACKUP.

    The anchor is that the action message built from THIS dataset's own columns
    is keyed the way `human_teleop.status()["goal_deg"]` is keyed, per side,
    with the gripper on both arms. The 12-dim layout is the reason the port
    works per arm at all: index 5 is the LEFT gripper here."""
    monkeypatch.setenv("HF_LEROBOT_HOME", str(LEROBOT_HOME))

    rig = rr.resolve_rig({"repo_id": BIMANUAL})
    action = rr.action_from_vector(list(range(rig.dim)), rig)

    assert rig.rig == "bimanual"
    assert rig.dim == 12
    assert sorted(action) == ["left", "right"]
    assert action["left"]["gripper"] == 5.0
    assert action["right"]["gripper"] == 11.0


def test_a_spec_with_no_columns_and_no_dataset_says_so():
    with pytest.raises(SystemExit) as excinfo:
        rr.resolve_rig({})

    detail = str(excinfo.value)
    assert "action_names" in detail
    assert "repo_id" in detail


# ---- the usual runner contract --------------------------------------------

def test_no_spec_path_is_a_usage_message_and_exit_two(monkeypatch, capsys):
    """Shared with every other runner via `_common.load_spec`: no run directory
    exists yet, so there is nowhere to write a `result.json` and none is
    written."""
    monkeypatch.setattr(sys, "argv", ["rollout_runner"])

    with pytest.raises(SystemExit) as excinfo:
        rr.main()

    assert excinfo.value.code == 2
    assert "usage:" in capsys.readouterr().out


def test_the_spec_can_ask_for_a_dry_run_without_the_flag(tmp_path, monkeypatch, capsys):
    """`lab/runs.launch` builds the child's argv itself and has no way to pass a
    flag, so `"dry_run": true` in the spec is the only route a dry run has from
    the UI."""
    spec = make_spec(tmp_path, dry_run=True)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    monkeypatch.setattr(sys, "argv", ["rollout_runner", str(spec_path)])

    assert rr.main() == 0

    out = capsys.readouterr().out
    assert "rollout plan:" in out
    assert "handshake:" in out
    assert "no socket was opened" in out
    # A dry run writes no result.json: nothing ran, so nothing finished.
    assert not (Path(spec["run_dir"]) / "result.json").exists()


def test_a_refused_rollout_ends_as_failed_with_the_refusal_on_one_line(tmp_path):
    """`_common.run_guarded` maps a string `SystemExit` — a preflight refusal —
    onto `failed` with the first line in `error`, and writes `result.json` in a
    `finally`. Without that file `lab/runs.load()` reports the run as `died`,
    which is a crash rather than a refusal."""
    from haller_hmi.runners import _common

    run_dir = tmp_path / "rollout-20260827-143000"
    run_dir.mkdir(parents=True, exist_ok=True)

    def refuse():
        raise SystemExit("cannot start a rollout: a teleop session is driving.\nStop it.")

    code = _common.run_guarded(run_dir, refuse)

    result = json.loads((run_dir / "result.json").read_text())
    assert code == 1
    assert result["status"] == "failed"
    assert result["error"] == "cannot start a rollout: a teleop session is driving."


def test_a_refused_spec_on_the_REAL_path_is_failed_not_died(tmp_path, monkeypatch):
    """`build_plan` runs INSIDE `run_guarded`, so its refusals reach
    `result.json`.

    Raised outside it there would be no `result.json` at all, and
    `lab/runs.load()` reports a dead pid with no result as `died` — a crash.
    A duration of zero is not a crash, and the runs table must not say it was.
    The socket module is booby-trapped for the length of this test: a refused
    spec must not have reached the ingest."""
    spec = make_spec(tmp_path, duration_s=0)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))

    def _no(*a, **k):
        raise AssertionError("a refused spec opened a socket")

    monkeypatch.setattr(socket, "create_connection", _no)
    monkeypatch.setattr(sys, "argv", ["rollout_runner", str(spec_path)])

    assert rr.main() == 1

    result = json.loads((Path(spec["run_dir"]) / "result.json").read_text())
    assert result["status"] == "failed"
    assert "duration_s must be > 0" in result["error"]


def test_stamping_the_rate_never_takes_the_run_down(tmp_path, monkeypatch):
    """An arm is moving. A stamp that cannot be written is a missing number,
    not a reason to stop telling the server where to put the arm."""
    missing = tmp_path / "gone"

    record = rr.gate_control_rate(missing, 30.0, 29.0)

    assert record["control_hz_measured"] == 29.0
    assert not missing.exists()


def test_the_module_answers_the_runners_map_by_name():
    """`lab/runs.RUNNERS` reaches every runner as a STRING through `-m`, never
    as an import. This asserts the module is reachable that way — the mapping
    itself belongs to `lab/runs.py` and is not this module's to assert."""
    out = subprocess.run(
        [sys.executable, "-m", "haller_hmi.runners.rollout_runner"],
        capture_output=True, text=True, timeout=120, cwd=str(BACKEND), check=False)

    assert out.returncode == 2
    assert "usage:" in out.stdout


def test_a_second_message_does_not_swallow_the_first(peer):
    """Whole lines only. Two messages written in one `sendall` must come back as
    two, because half a JSON object silently dropped is a target frame that
    never happened."""
    client, server = peer
    server.sendall(
        (json.dumps({"type": rr.OBSERVATION_TYPE, "state": [1]}) + "\n"
         + json.dumps({"type": rr.OBSERVATION_TYPE, "state": [2]}) + "\n").encode()
    )

    first = client.recv(2.0)
    second = client.recv(2.0)

    assert first["state"] == [1]
    assert second["state"] == [2]


def test_a_closed_peer_reads_as_no_message_rather_than_a_traceback(peer):
    client, server = peer
    server.close()

    assert client.recv(2.0) is None


def test_sending_to_a_closed_ingest_names_what_holds_the_arms(peer):
    """The server has the arms; they stop where its own staleness gate leaves
    them. Saying so is the difference between a refusal and a mystery."""
    client, server = peer
    server.close()
    # The first write after the peer goes away is often absorbed by the kernel;
    # the RST arrives on the next one. Both are the same failure.
    with pytest.raises(SystemExit) as excinfo:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            client.send({"type": rr.ACTION_TYPE})

    assert "closed mid-run" in str(excinfo.value)
