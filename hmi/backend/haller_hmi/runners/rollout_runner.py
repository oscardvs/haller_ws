# hmi/backend/haller_hmi/runners/rollout_runner.py
"""Detached child that runs a trained policy's INFERENCE and streams targets.

**This child owns the policy. It never owns the servo bus.** Ruled by the
contract's rollout addendum ("rollout: the child owns the policy, never the
bus", 2026-08-27), which closed the shape the kit ships: the kit's rollout
child opens the serial port itself, pre-syncs the goal registers, energises
the servos and hands the arm to `lerobot-rollout`. Two things follow from that
shape and both are unacceptable here:

* `/estop` walks every motor IN-PROCESS. A child holding the bus means the
  server has nothing to talk to, and a SIGSEGV or a SIGKILL in that child
  leaves the arm energised on its last goal until someone reaches the bench
  PSU. (2026-08-21: an overloaded shoulder aborted a bulk torque-disable
  mid-sweep and left four joints stiff. That is the incident this rule exists
  for.)
* A policy driving outside the commit chain gets NO collision guard, NO
  workspace floor and NO rate cap. A freshly-trained ACT policy is *less*
  trustworthy than a human hand, not more.

So: load the checkpoint, run inference, and stream target joint angles to the
server over a loopback socket. The server keeps the bus and commits those
targets through the same chain every other input already goes through — LPF ->
per-tick rate cap -> clamp -> collision guard -> workspace floors -> E-STOP.
This is the teleop architecture with a different leader. The Quest is a leader
that streams targets; so is the policy. Latency is not the objection: the
kit's own rollout ran at 4.8 Hz against a 30 Hz target, so a loopback hop is
nowhere near the binding constraint.

**Degrees on every joint including the gripper, on every message, and the unit
is declared ONCE at handshake.** The VR converter emits a `[0, 1]` gripper and
the session scales it onto the calibrated degree range; a policy emits what
the dataset's `action` column held, and that column is degrees
(`haller_joint_calibration.state_unit == "deg"`, gripper range
[-9.969465635276324, 100.26761414789407] on the real bimanual dataset). Sent
through the `[0, 1]` path a legitimate 88.1 deg clamps to 1.0 and opens fully
and a 0.5 deg command becomes half open — every gripper command collapsing to
one of two values, silently, in the direction of dropping the object.
Normalising HERE was rejected on purpose: this process does not own the bus,
so it would need the calibrated range shipped over the wire or re-read from
disk, and two copies that drift are wrong in the same direction as the bug
being fixed, with two conversions where the honest path has zero.

**THE SERVER SIDE EXISTS, AND THE WIRE IS DEMONSTRATED — IN TESTS.**
`policy_ingest` (the wire) and `policy_bridge` (observe, submit, bus_conflict)
landed 2026-08-29 in `7be71ac`, and every spelling below — inbound and
outbound — plus the endpoint is the one they read, frozen by the integrator's
ruling of 2026-08-27. Not a second copy that agrees by inspection:
`tests/test_policy_bridge.py` drives THIS FILE'S OWN `IngestClient` against the
real ingest over a real loopback socket, through the server's own lifespan —
the handshake, the observation stream, a NaN frame refused, a malformed frame
that no longer wedges the operator paths. The rate floor was settled the same
way and earlier: `safety.MIN_RATE_FRACTION` landed 2026-08-27 and is imported
directly.

**What that still is not is a policy.** No checkpoint has been loaded through
`_load_policy`, no `select_action` has run, and no target this file produced
has reached a servo: the tests supply the arms and the observations, and
`_rollout` — the one function here with no test — remains the part standing on
argument alone. `MIN_RATE_FRACTION` was proven by republishing `0.5` and
watching the resolver follow, because two names agreeing is not evidence that
one reads the other; the equivalent proof for the loop below is a rollout
somebody watched. See the acceptance criteria in
`docs/port/trackb-lab-contract.md`.

When nothing is listening this child REFUSES, loudly, naming the endpoint. It
does not fall back to driving the arm, and it does not stand up a server of its
own.

Nothing here is imported by the serving process, and `lerobot`/`torch` appear
only inside `_rollout`'s call tree. That is what lets the plan, the handshake,
the message builder, the refusals and the rate gate be tested at all: the tests
run under the serving venv (lerobot 0.5.1), which cannot import 0.6.1's rollout
stack and must never be asked to.

    python -m haller_hmi.runners.rollout_runner SPEC.json
    python -m haller_hmi.runners.rollout_runner SPEC.json --dry-run
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

from ..lab.lease import port_holders
from ..lab.runs import MAX_ROLLOUT_DURATION_S
from ..lab.schema import RigSpec
from ..safety import MIN_RATE_FRACTION
from ._common import load_spec, run_guarded

__all__ = [
    "ACK_TYPE",
    "ACTION_TYPE",
    "ACTION_UNIT",
    "CONNECT_TIMEOUT_S",
    "DEFAULT_INGEST_URL",
    "HANDSHAKE_TIMEOUT_S",
    "HELLO_TYPE",
    "INGEST_URL_ENV",
    "MAX_ROLLOUT_DURATION_S",
    "MIN_RATE_FRACTION",
    "OBSERVATION_TYPE",
    "RATE_ALERT_UNDER_S",
    "RATE_SIDECAR",
    "RATE_WARMUP_TICKS",
    "REFUSED_TYPE",
    "IngestClient",
    "RateWatch",
    "action_from_vector",
    "action_message",
    "build_plan",
    "control_rate_refusal",
    "decode_observation",
    "foreign_port_holders",
    "gate_control_rate",
    "hello_message",
    "ingest_url",
    "main",
    "parse_ingest",
    "rate_floor_fraction",
    "rate_floor_hz",
    "resolve_rig",
]

# ---- the wire ------------------------------------------------------------

#: Outbound message types. FROZEN with Track A (haller-ws-d7) 2026-08-27 —
#: changing either spelling is a message to the integrator, not an edit.
HELLO_TYPE = "policy_hello"
ACTION_TYPE = "policy_action"

#: The unit every joint is sent in, gripper included, declared once in the
#: handshake as a property of the SOURCE. There is deliberately no per-message
#: `unit` field: a per-message unit is a per-message chance to disagree with
#: the declaration, and the addendum makes a contradicting one a REFUSAL rather
#: than a conversion — which is the server's check to make, not this side's.
ACTION_UNIT = "deg"

#: Inbound message types. **FROZEN by the integrator's ruling 2026-08-27** —
#: these were this child's named guesses, and Track A adopts them rather than
#: publishing a second set for this side to follow. Changing either half is now
#: a message to the integrator, exactly like the outbound pair above.
#:
#: Left as guesses they would have been worse than wrong: had Track A happened
#: to spell them the same way, the file would still have read "placeholder"
#: while the system depended on it, and no later reader could tell a settled
#: contract from an unclaimed coincidence.
#:
#: An unrecognised inbound type is ignored; a handshake that never gets
#: acknowledged is a refusal naming the endpoint.
ACK_TYPE = "policy_hello_ack"
REFUSED_TYPE = "policy_refused"
OBSERVATION_TYPE = "policy_observation"

#: Where the server listens for a policy source. Newline-delimited JSON over a
#: loopback TCP socket, chosen because it is the only transport both
#: interpreters already speak: the lab venv has no WebSocket client, and adding
#: one is a new third-party dependency in the venv that runs the GPU.
#: **FROZEN by the same ruling** — Track A listens here. Until `9360e8b` this
#: was the only occurrence of the port in the whole tree, which is what made it
#: readable as settled when it was not.
DEFAULT_INGEST_URL = "tcp://127.0.0.1:8781"

#: Environment override, for a box running the HMI somewhere else.
INGEST_URL_ENV = "HALLER_POLICY_INGEST"

#: Seconds to wait for the TCP connect and for the handshake acknowledgement.
#: Both are loopback, so anything approaching these numbers means the ingest is
#: not answering rather than that it is slow.
CONNECT_TIMEOUT_S = 5.0
HANDSHAKE_TIMEOUT_S = 10.0

# ---- the rate gate -------------------------------------------------------

#: How long the measured rate may sit under the floor mid-run before it is
#: called out. Two seconds at a legitimate 30 Hz is 60 ticks — long enough that
#: one slow inference step is not an alert, short enough that the operator hears
#: about it while the arm is still in the first move.
RATE_ALERT_UNDER_S = 2.0

#: Inference cycles run before anything is streamed, purely to measure the
#: achieved rate. Nothing is sent during them, so the arm does not move while
#: the gate is deciding — measure first, then commit.
RATE_WARMUP_TICKS = 20


#: Where the two rate numbers land. `lab/runs.write_result` carries exactly
#: four keys and is shared with `load()`, so it is not the place to add fields;
#: this sidecar sits beside it in the run directory and the same numbers are
#: also appended to `metrics.jsonl`, which the runs page already tails.
RATE_SIDECAR = "rollout.json"


# ---- the plan ------------------------------------------------------------

def rate_floor_fraction() -> float:
    """Track A's published floor, read directly.

    This was a name-probe with a local fallback until `MIN_RATE_FRACTION`
    landed (2026-08-27, safety.py:33). The fallback is gone deliberately rather
    than left as belt-and-braces: before publication a missing constant was
    normal, but AFTER it a lookup that cannot find it is a rename, a move or a
    typo — never "normal" — and swallowing that would hide the exact drift the
    probe existed to prevent. A warning was the softer option and is not enough:
    it still resolves to a number and still runs the arm. Absence is now an
    ImportError at module load, which is the loudest and earliest it can be.

    Imported at module scope safely because `safety.py` is stdlib-only — enum,
    math, dataclasses. `haller_hmi.tick` owns the rate MEASUREMENT and therefore
    reaches `arm.py` and therefore lerobot, which is why the constant lives in
    safety.py and why this must never follow it somewhere heavier.
    """
    return MIN_RATE_FRACTION


def rate_floor_hz(declared_hz: float) -> float:
    """The lowest measured rate a rollout may start at."""
    return float(declared_hz) * rate_floor_fraction()


def ingest_url(spec: dict) -> str:
    """Where to stream targets: the spec, the environment, then the default.

    Spec first so one run can be pointed elsewhere without touching the
    server's environment, which is shared by every other run it launches.
    """
    value = spec.get("ingest_url")
    if value is None:
        value = os.environ.get(INGEST_URL_ENV)
    if value is None:
        value = DEFAULT_INGEST_URL
    return str(value).strip()


def parse_ingest(url: str) -> tuple[str, int]:
    """`(host, port)` from an ingest URL, or a refusal naming what is wrong.

    Only `tcp://` is spoken here — see `DEFAULT_INGEST_URL` for why. An
    unsupported scheme is refused by NAME rather than coerced, because the
    coercion nobody notices is the one that connects to the wrong thing.
    """
    if not url:
        raise SystemExit(
            "no policy ingest endpoint configured: set spec['ingest_url'] or "
            f"${INGEST_URL_ENV} (default {DEFAULT_INGEST_URL})."
        )
    parts = urlparse(url)
    if parts.scheme != "tcp":
        raise SystemExit(
            f"unsupported policy ingest scheme {parts.scheme!r} in {url} — this "
            "child speaks newline-delimited JSON over a loopback TCP socket "
            "(tcp://host:port). Track A owns the ingest; if it is not TCP, say "
            "so and this side changes."
        )
    if not parts.hostname or not parts.port:
        raise SystemExit(f"policy ingest URL needs a host and a port: {url}")
    return parts.hostname, int(parts.port)


def resolve_rig(spec: dict) -> RigSpec:
    """The arm layout the policy's action vector is in.

    Two sources, in order, and no third: `spec["action_names"]` (the dataset's
    own `action` column names, which is what a caller that already read the
    metadata should pass), else the dataset's `meta/info.json` for
    `spec["repo_id"]`. Derived from the dataset, never configured — same rule
    as `lab/schema.RigSpec`, and reusing that class rather than re-deriving the
    side/gripper split here is what keeps one spelling of it.
    """
    names = spec.get("action_names")
    if names:
        return RigSpec.from_info(
            {"features": {"observation.state": {"names": list(names)}}}
        )

    repo_id = spec.get("repo_id")
    if not repo_id:
        raise SystemExit(
            "cannot tell which joints the policy's action vector holds: the "
            "spec carries neither 'action_names' nor 'repo_id'."
        )
    # Imported HERE rather than at module scope: `lab/catalog` pulls in numpy
    # and the grader for a job this needs on one cold path, and `--dry-run`
    # with explicit action_names must not pay for it.
    from ..lab.catalog import dataset_root

    info_path = Path(dataset_root(str(repo_id))) / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text())
    except Exception as e:
        # The PATH is the useful half of this refusal — a repo_id that resolved
        # to the wrong root reads identically to one whose metadata is missing.
        raise SystemExit(f"cannot read {info_path}: {e}") from e
    return RigSpec.from_info(info)


def _plain(raw: str, side: str) -> str:
    """A raw column name as the plain joint name the commit chain uses.

    `left_shoulder_pan` and `shoulder_pan.pos` both become `shoulder_pan`:
    Haller's recorder writes the first spelling and the kit's datasets the
    second, and `human_teleop.status()["goal_deg"]` is keyed by neither — it is
    keyed by the bare joint.
    """
    base = raw.removesuffix(".pos")
    prefix = f"{side}_"
    if side and base.startswith(prefix):
        base = base[len(prefix):]
    return base


def action_from_vector(values, rig: RigSpec, *, side: str = "") -> dict:
    """The policy's action vector as `{side: {joint: degrees}}`.

    The SAME plain joint -> float per side that `human_teleop.status()`
    already publishes as `goal_deg`, so it lands on the commit chain as any
    other leader's target does.

    **An absent side has no key at all.** Not an empty dict: a rig with one arm
    sends one key, and a `{"right": {}}` on a solo rig is a shape that reads as
    "the right arm was commanded to nothing", which is a different and wrong
    claim.

    `side` names the arm for an UNPREFIXED dataset (`RigSpec.rig == "solo"`,
    `ArmSpec.side == ""`), where the columns cannot say which arm they were
    recorded from. It is refused rather than guessed: the two arms are 40 cm
    apart, and a rollout aimed at the wrong one is a collision, not a typo.
    """
    out: dict[str, dict[str, float]] = {}
    for arm in rig.arms:
        key = arm.side or side
        if not key:
            raise SystemExit(
                f"this dataset's columns carry no side prefix (rig {rig.rig!r}), "
                "so the spec must say which arm the policy drives: set "
                "spec['side'] to 'left' or 'right'."
            )
        joints = {
            _plain(name, arm.side): float(values[idx])
            for name, idx in zip(arm.joint_names, arm.joint_idx, strict=True)
        }
        if arm.gripper_name is not None and arm.gripper_idx is not None:
            joints[_plain(arm.gripper_name, arm.side)] = float(values[arm.gripper_idx])
        out[key] = joints
    return out


def build_plan(spec: dict) -> dict:
    """Everything the run is about to do, validated. Refusals are SystemExit.

    Built BEFORE any socket is opened and before any checkpoint is read, so
    `--dry-run` prints exactly what a real run would do without doing any of
    it, and so a bad spec fails on the spec rather than three minutes into a
    CUDA context.
    """
    duration_s = float(spec.get("duration_s") or 0.0)
    if duration_s <= 0:
        raise SystemExit("duration_s must be > 0 — a rollout is always bounded.")
    if duration_s > MAX_ROLLOUT_DURATION_S:
        raise SystemExit(
            f"duration_s {duration_s:g} exceeds the "
            f"{MAX_ROLLOUT_DURATION_S:g} s ceiling: "
            "a policy loop started from a browser button must not be able to run "
            "until someone notices."
        )

    policy_path = str(spec.get("policy_path") or "")
    if not policy_path:
        raise SystemExit("spec['policy_path'] is required: no checkpoint to load.")
    if not Path(policy_path).exists():
        raise SystemExit(f"no checkpoint at {policy_path}")

    url = ingest_url(spec)
    host, port = parse_ingest(url)

    rig = resolve_rig(spec)
    side = str(spec.get("side") or "")
    # Built once here so an unprefixed rig with no `side` is refused now, on the
    # spec, rather than on the first action message with the arm already live.
    joints = action_from_vector([0.0] * rig.dim, rig, side=side)

    declared = float(spec.get("control_hz") or spec.get("fps") or 30.0)
    if declared <= 0:
        raise SystemExit("control_hz must be > 0 — it is what the rate gate is against.")

    return {
        "run_id": spec.get("run_id") or "",
        "policy_path": policy_path,
        "device": str(spec.get("device", "cuda")),
        "task": str(spec.get("task") or ""),
        "robot_type": str(spec.get("robot_type") or ""),
        "ingest_url": url,
        "ingest_host": host,
        "ingest_port": port,
        "duration_s": duration_s,
        "unit": ACTION_UNIT,
        "control_hz_declared": declared,
        "control_hz_floor": rate_floor_hz(declared),
        "control_hz_fraction": rate_floor_fraction(),
        "allow_slow": bool(spec.get("allow_slow")),
        "rig": rig.rig,
        "sides": sorted(joints),
        "joints": {k: sorted(v) for k, v in joints.items()},
        # Optional. Only used to name a NON-server process holding the bus; the
        # server's own answer at handshake is what actually refuses.
        "port": str(spec.get("port") or ""),
    }


# ---- the messages --------------------------------------------------------

def hello_message(plan: dict) -> dict:
    """The handshake. FROZEN: exactly these five keys.

    The unit is declared HERE and nowhere else — a source declaring degrees
    gets no gripper special-case downstream. `control_hz_declared` is the
    spelling Track A publishes for the recorder; there is deliberately no
    second one.
    """
    return {
        "type": HELLO_TYPE,
        "run_id": plan["run_id"],
        "unit": plan["unit"],
        "control_hz_declared": plan["control_hz_declared"],
        "rig": plan["rig"],
    }


def action_message(run_id: str, seq: int, action: dict,
                   t_ms: int | None = None) -> dict:
    """One target frame. FROZEN: exactly these five keys.

    No per-message `unit` field — see `ACTION_UNIT`. `t_ms` is wall clock, the
    same clock the frozen example carries, so a frame's age can be compared
    against anything else the server timestamps.
    """
    return {
        "type": ACTION_TYPE,
        "seq": int(seq),
        "t_ms": int(time.time() * 1000) if t_ms is None else int(t_ms),
        "run_id": run_id,
        "action": action,
    }


def decode_observation(msg: dict) -> dict | None:
    """One inbound observation, or None if this is not one.

    **Track A owns this shape and has not published it.** What is decoded here
    is the minimum inference needs — the state vector, whatever camera frames
    came with it, and the sequence/timestamp to attribute a stall to the right
    side — and it is kept in one small function so that when the real shape
    lands, this is the only thing that moves.

    Images are left as their encoded payloads: decoding them needs numpy and an
    image codec, which belong in `_rollout`'s heavy call tree, not in a function
    the tests reach.
    """
    if not isinstance(msg, dict) or msg.get("type") != OBSERVATION_TYPE:
        return None
    state = msg.get("state")
    if state is None:
        state = (msg.get("observation") or {}).get("state")
    if state is None:
        return None
    return {
        "state": [float(v) for v in state],
        "images": dict(msg.get("images") or {}),
        "seq": int(msg.get("seq") or 0),
        "t_ms": int(msg.get("t_ms") or 0),
    }


# ---- the socket ----------------------------------------------------------

class IngestClient:
    """Newline-delimited JSON over one loopback TCP socket.

    Deliberately blocking and deliberately tiny. The child has exactly one peer
    and one job, and every failure mode it has — nothing listening, no
    acknowledgement, a refusal from the server — has to arrive as a sentence
    naming the endpoint rather than as a traceback.
    """

    def __init__(self, host: str, port: int, url: str = "") -> None:
        self.host = host
        self.port = port
        self.url = url or f"tcp://{host}:{port}"
        self.server_pid: int | None = None
        self._sock: socket.socket | None = None
        self._buf = b""

    def connect(self, timeout: float = CONNECT_TIMEOUT_S) -> None:
        try:
            sock = socket.create_connection((self.host, self.port), timeout=timeout)
        except OSError as e:
            raise SystemExit(
                f"no policy ingest at {self.url}: {e}. The server owns the bus "
                "and this child only streams targets at it, so there is nothing "
                "to fall back to — start the HMI (or point "
                f"${INGEST_URL_ENV} at it) and try again."
            ) from e
        # Nagle would coalesce small target frames on a link whose whole purpose
        # is to deliver them one per tick.
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock

    def send(self, msg: dict) -> None:
        if self._sock is None:
            raise SystemExit(f"policy ingest {self.url} is not connected")
        try:
            self._sock.sendall((json.dumps(msg) + "\n").encode())
        except OSError as e:
            raise SystemExit(
                f"policy ingest {self.url} closed mid-run: {e}. The server has "
                "the arms; they stop where its own staleness gate leaves them."
            ) from e

    def recv(self, timeout: float) -> dict | None:
        """One message, or None on timeout / a closed peer.

        Whole lines only: a partial read is buffered rather than parsed, because
        half a JSON object silently dropped is a target frame that never
        happened.
        """
        if self._sock is None:
            return None
        deadline = time.monotonic() + timeout
        while True:
            line, sep, rest = self._buf.partition(b"\n")
            if sep:
                self._buf = rest
                text = line.strip()
                if not text:
                    continue
                try:
                    return json.loads(text)
                except ValueError:
                    # One malformed line must not end the run: the server is
                    # holding the arms, and a parse error here is not a reason
                    # to stop telling it where to put them.
                    continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self._sock.settimeout(remaining)
            try:
                chunk = self._sock.recv(65536)
            except OSError:
                # A read timeout and a peer that went away are the same answer
                # here — no message — and `socket.timeout` IS `TimeoutError` IS
                # an `OSError` since 3.10, so one arm covers both. The caller's
                # own deadline is what turns either into a named refusal.
                return None
            if not chunk:
                return None
            self._buf += chunk

    def handshake(self, hello: dict, timeout: float = HANDSHAKE_TIMEOUT_S) -> dict:
        """Declare the source, and refuse on the SERVER's answer.

        This child cannot see the recorder or the teleop session, so it does not
        try: `lab/lease.bus_conflict` runs in the process that can, and its
        sentence comes back here as the refusal. That is the whole preflight —
        an episode open, a teleop session driving, or a foreign holder of the
        bus are all one question, asked of the only process that can answer it.
        """
        self.send(hello)
        deadline = time.monotonic() + timeout
        while True:
            msg = self.recv(max(0.0, deadline - time.monotonic()))
            if msg is None:
                raise SystemExit(
                    f"policy ingest {self.url} accepted the connection but never "
                    f"acknowledged the handshake ({timeout:g} s). Expected a "
                    f"{ACK_TYPE!r}; Track A owns that side."
                )
            mtype = msg.get("type")
            if mtype == REFUSED_TYPE:
                raise SystemExit(
                    str(msg.get("detail") or "the server refused this rollout")
                )
            if mtype == ACK_TYPE:
                if not msg.get("ok", True):
                    raise SystemExit(
                        str(msg.get("detail") or "the server refused this rollout")
                    )
                pid = msg.get("server_pid")
                self.server_pid = int(pid) if pid else None
                return msg

    def close(self) -> None:
        """Closing IS the goodbye.

        There is no `policy_bye` in the frozen shape and this side does not
        invent one: the server already treats a source that stopped streaming
        the way it treats a teleop client that stopped streaming, and a second
        end-of-stream signal is a second thing that can disagree with the first.
        """
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def foreign_port_holders(device: str, server_pid: int | None) -> list[str]:
    """Bus holders that are NOT this child and NOT the server.

    `lab/lease.port_holders` already skips the caller's own pid — but the
    caller here is the CHILD, so the server's own fd would show up, and the
    server holding the bus is the NORMAL, REQUIRED state under this
    architecture. Refusing on it would refuse every rollout forever and look
    like a hardware fault.

    So the server's pid comes from the handshake acknowledgement and is filtered
    out here. **When the server did not declare one, nothing is reported**: a
    holder list this child cannot attribute is not evidence of a conflict, and
    the server's own `bus_conflict` answer is authoritative anyway.
    """
    if not device or server_pid is None:
        return []
    prefix = f"pid {server_pid}:"
    return [h for h in port_holders(device) if not h.startswith(prefix)]


# ---- the rate gate -------------------------------------------------------

def control_rate_refusal(declared_hz: float, measured_hz: float, *,
                         override: bool = False,
                         fraction: float | None = None) -> str | None:
    """Why this rate must not start a rollout, or None. Pure.

    `override` does not change the numbers, only whether they stop the run —
    which is why `gate_control_rate` stamps them either way.
    """
    frac = rate_floor_fraction() if fraction is None else float(fraction)
    floor = float(declared_hz) * frac
    if measured_hz >= floor:
        return None
    if override:
        return None
    return (
        f"measured control rate {measured_hz:.2f} Hz is below "
        f"{floor:.2f} Hz ({frac:.0%} of the declared {declared_hz:.2f} Hz). "
        "A policy trained at the declared rate and executed at this one is a "
        "different dynamical system, not a slow rollout — its action deltas "
        "are sized for a step that long. Set spec['allow_slow'] to watch it "
        "anyway."
    )


def gate_control_rate(run_dir: str | Path, declared_hz: float, measured_hz: float,
                      *, override: bool = False, source: str = "warmup") -> dict:
    """Stamp both rates into the run record, THEN refuse if the rate is short.

    One function rather than two because the stamping is the part a caller
    forgets. The kit's failure was not that a 4.8 Hz run happened — it was that
    "success" was reported with that number attached nowhere. So the numbers are
    written before the decision is taken, on the refusal path as well as the
    override path.
    """
    refusal = control_rate_refusal(declared_hz, measured_hz, override=override)
    record = _rate_record(declared_hz, measured_hz, override=override,
                          source=source, refused=refusal or "")
    _stamp_rate(run_dir, record)
    if refusal:
        raise SystemExit(refusal)
    return record


def _rate_record(declared_hz: float, measured_hz: float, *, override: bool,
                 source: str, refused: str = "") -> dict:
    """The two key spellings Track A publishes for the recorder, plus what they
    were judged against. Do not invent a second spelling of either."""
    return {
        "control_hz_declared": float(declared_hz),
        "control_hz_measured": float(measured_hz),
        "control_hz_floor": rate_floor_hz(declared_hz),
        "control_hz_fraction": rate_floor_fraction(),
        "control_hz_override": bool(override),
        "control_hz_source": source,
        "refused": refused,
    }


def _stamp_rate(run_dir: str | Path, record: dict) -> None:
    """`rollout.json` beside `result.json`, and one row in `metrics.jsonl`.

    Both, because they answer different questions: the sidecar is the run's
    final word on the rate it achieved, and the metrics row puts it on the
    timeline the runs page already tails without a new route to read it.
    Neither may take the run down — a stamp that fails is a missing number, and
    an arm is moving.
    """
    base = Path(run_dir)
    try:
        (base / RATE_SIDECAR).write_text(json.dumps(record, indent=2) + "\n")
    except OSError:
        pass
    try:
        row = {"kind": "rate"}
        row.update(record)
        with open(base / "metrics.jsonl", "a", buffering=1) as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError:
        pass


class RateWatch:
    """Mid-run rate alarm: under the floor for more than `under_s`, say so.

    An ALERT, not an abort. The start gate is what refuses; once the arm is
    moving, the server's own staleness handling is what decides a source has
    stopped being usable, and a second authority racing it is how a healthy
    run gets torn down mid-move.

    The rate is measured over a trailing window rather than from the last
    interval, because one slow inference step at 30 Hz is a 0.5 Hz instantaneous
    reading and not a fact about the run.
    """

    def __init__(self, floor_hz: float, *, under_s: float = RATE_ALERT_UNDER_S) -> None:
        self.floor_hz = float(floor_hz)
        self.under_s = float(under_s)
        self._ticks: deque[float] = deque()
        self._under_since: float | None = None
        self._alerted = False

    def tick(self, now: float | None = None) -> str | None:
        """Record one control cycle. Returns an alert sentence, once per slump."""
        t = time.monotonic() if now is None else float(now)
        self._ticks.append(t)
        # Keep a window twice the alert period: enough to measure over, bounded
        # so an hour-long run does not accumulate an hour of timestamps.
        cutoff = t - 2.0 * self.under_s
        while len(self._ticks) > 2 and self._ticks[0] < cutoff:
            self._ticks.popleft()
        rate = self.rate(t)
        if rate is None:
            return None
        if rate >= self.floor_hz:
            self._under_since = None
            self._alerted = False
            return None
        if self._under_since is None:
            self._under_since = t
            return None
        if self._alerted or (t - self._under_since) <= self.under_s:
            return None
        self._alerted = True
        return (f"control rate {rate:.2f} Hz has been under {self.floor_hz:.2f} Hz "
                f"for {t - self._under_since:.1f} s")

    def rate(self, now: float | None = None) -> float | None:
        """Hz over the trailing window, or None before there are two ticks."""
        if len(self._ticks) < 2:
            return None
        span = self._ticks[-1] - self._ticks[0]
        if span <= 0:
            return None
        return (len(self._ticks) - 1) / span


# ---- the run ------------------------------------------------------------

def _load_policy(plan: dict):
    """`(policy, preprocessor, postprocessor)` from a checkpoint directory.

    Mirrors lerobot 0.6.1's own `rollout/context.py` and
    `rollout/inference/sync.py`: config -> policy class -> weights ->
    pre/post-processor pipelines. Reusing lerobot's own pipeline rather than
    calling the model directly is what makes the returned action DEGREES — the
    postprocessor is where un-normalisation happens, and a hand-rolled forward
    pass would emit normalised numbers that look plausible and are not.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(plan["policy_path"])
    cfg.pretrained_path = plan["policy_path"]
    policy = get_policy_class(cfg.type).from_pretrained(plan["policy_path"], config=cfg)
    policy = policy.to(plan["device"])
    policy.eval()
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, pretrained_path=plan["policy_path"]
    )
    return policy, preprocessor, postprocessor


def _infer(policy, preprocessor, postprocessor, obs: dict, plan: dict):
    """One inference step, in the order lerobot's sync engine runs it.

    The returned list is indexed by the DATASET'S `action` column order —
    lerobot's own `make_robot_action` reads `ds_features["action"]["names"][i]`
    for element `i` of exactly this tensor. That is what makes it safe to hand
    straight to `action_from_vector`, whose indices come from the same names.
    """
    import numpy as np
    import torch
    from lerobot.policies.utils import prepare_observation_for_inference

    frame: dict = {"observation.state": np.asarray(obs["state"], dtype=np.float32)}
    for key, payload in (obs.get("images") or {}).items():
        frame[f"observation.images.{key}"] = _decode_image(payload)
    with torch.inference_mode():
        prepared = prepare_observation_for_inference(
            frame, torch.device(plan["device"]), plan["task"] or None,
            # The robot type belongs to the DATASET, not to this file: it is
            # what the checkpoint was trained against, and a literal here would
            # be a second copy of it that cannot be right for every rig.
            plan["robot_type"] or None,
        )
        action = postprocessor(policy.select_action(preprocessor(prepared)))
    return action.squeeze(0).cpu().tolist()


def _decode_image(payload):
    """One camera frame from the wire, as (H, W, 3) RGB.

    Decoded through PIL rather than OpenCV ON PURPOSE. `cv2.imdecode` hands
    back BGR whatever the file held, so anyone reading this would have to know
    which order the encoder assumed to know whether a swap belongs here — and a
    channel order that is reasoned about rather than pinned is exactly how the
    mast camera's magenta decode happened. PIL's `convert("RGB")` yields RGB
    from any input, which is the order lerobot's image pipeline expects, so
    there is no swap to get backwards.
    """
    import base64
    from io import BytesIO

    import numpy as np
    from PIL import Image

    if isinstance(payload, list):
        return np.asarray(payload, dtype=np.uint8)
    raw = base64.b64decode(payload)
    return np.asarray(Image.open(BytesIO(raw)).convert("RGB"), dtype=np.uint8)


def _rollout(run_dir: Path, spec: dict) -> None:
    """Connect, declare, measure, then stream targets until the clock runs out.

    The one function here with no test, for the same reason `train_runner._train`
    has none: it needs lerobot 0.6.1, a checkpoint, a GPU and a server holding
    two arms. Everything it decides is decided in a function above that does
    have one.

    **`build_plan` is called HERE rather than by `main`**, so every refusal it
    raises happens inside `run_guarded` and lands in `result.json` as `failed`.
    Raised outside it, a refused rollout would leave no `result.json` at all and
    `lab/runs.load()` would report it as `died` — a crash, which is not what a
    duration of zero is.
    """
    plan = build_plan(spec)
    client = IngestClient(plan["ingest_host"], plan["ingest_port"], plan["ingest_url"])
    # Resolved a second time because `build_plan` uses it only to validate. On
    # the `repo_id` path that is one more `info.json` read at startup, which is
    # cheaper than handing a mutable RigSpec around through the plan dict.
    rig = resolve_rig(spec)
    side = str(spec.get("side") or "")
    print(f"policy ingest: {plan['ingest_url']}", flush=True)
    client.connect()
    try:
        ack = client.handshake(hello_message(plan))
        print(f"handshake acknowledged: {json.dumps(ack)}", flush=True)

        holders = foreign_port_holders(plan["port"], client.server_pid)
        if holders:
            raise SystemExit(
                f"cannot start a rollout: {plan['port']} is open by a process "
                "that is neither the HMI nor this one:\n  " + "\n  ".join(holders)
                + "\nStop it first — two processes on one Feetech bus corrupt "
                  "each other's packets."
            )

        print(f"loading {plan['policy_path']} on {plan['device']}", flush=True)
        policy, pre, post = _load_policy(plan)

        # WARMUP: full inference cycles with nothing sent. The rate is measured
        # before the arm can move on it, so a refusal costs no motion at all.
        warm = RateWatch(plan["control_hz_floor"])
        for _ in range(RATE_WARMUP_TICKS):
            obs = _next_observation(client, plan)
            _infer(policy, pre, post, obs, plan)
            warm.tick()
        measured = warm.rate() or 0.0
        gate_control_rate(run_dir, plan["control_hz_declared"], measured,
                          override=plan["allow_slow"], source="warmup")
        print(f"control rate: {measured:.2f} Hz measured, "
              f"{plan['control_hz_declared']:.2f} Hz declared", flush=True)

        watch = RateWatch(plan["control_hz_floor"])
        period = 1.0 / plan["control_hz_declared"]
        deadline = time.monotonic() + plan["duration_s"]
        seq = 0
        print("streaming targets — the server is committing them to the arms",
              flush=True)
        while time.monotonic() < deadline:
            cycle = time.monotonic()
            obs = _next_observation(client, plan)
            values = _infer(policy, pre, post, obs, plan)
            client.send(action_message(plan["run_id"], seq,
                                       action_from_vector(values, rig, side=side)))
            seq += 1
            alert = watch.tick()
            if alert:
                print(f"RATE ALERT: {alert}", flush=True)
            # Pace to the declared rate. A policy running FASTER than it was
            # trained is the same class of mismatch as one running slower.
            sleep = period - (time.monotonic() - cycle)
            if sleep > 0:
                time.sleep(sleep)
        # Stamped, not gated: the run is over, so there is nothing left to
        # refuse — and `control_hz_override` must keep saying what the OPERATOR
        # asked for rather than being set true to stop this call raising.
        _stamp_rate(run_dir, _rate_record(
            plan["control_hz_declared"], watch.rate() or 0.0,
            override=plan["allow_slow"], source="run"))
        print(f"done: {seq} targets in {plan['duration_s']:g} s", flush=True)
    finally:
        client.close()


def _next_observation(client: IngestClient, plan: dict) -> dict:
    """The next observation from the server, or a refusal naming the stall.

    A rollout with no observation has nothing to infer from, and inventing one
    (the last frame, zeros) would drive the arm off a state that never existed.
    """
    deadline = time.monotonic() + max(1.0, 5.0 / plan["control_hz_declared"])
    while time.monotonic() < deadline:
        msg = client.recv(max(0.0, deadline - time.monotonic()))
        if msg is None:
            break
        obs = decode_observation(msg)
        if obs is not None:
            return obs
        if msg.get("type") == REFUSED_TYPE:
            raise SystemExit(str(msg.get("detail") or "the server stopped the rollout"))
    raise SystemExit(
        f"no {OBSERVATION_TYPE!r} from {client.url} — the policy has nothing to "
        "infer from. Track A owns the observation stream."
    )


def main() -> int:
    spec, dry_run = load_spec(sys.argv[1:])
    run_dir = Path(spec["run_dir"])

    if dry_run:
        # Every check `build_plan` makes has run. Nothing here imports lerobot
        # or torch, opens a socket or reads a checkpoint — this is the path that
        # answers "what exactly would you do" from a box with the arms powered
        # down and the HMI stopped.
        plan = build_plan(spec)
        print("rollout plan: " + json.dumps(plan, sort_keys=True))
        print("handshake: " + json.dumps(hello_message(plan), sort_keys=True))
        print("dry run: no socket was opened and no target was sent.")
        return 0

    return run_guarded(run_dir, lambda: _rollout(run_dir, spec))


if __name__ == "__main__":
    raise SystemExit(main())
