# hmi/backend/haller_hmi/policy_ingest.py
"""Server side of the policy wire: the rollout child streams, we commit.

The other half of `runners/rollout_runner.py`. The ruled architecture is that
**the child owns the POLICY and never the bus** — it runs inference in its own
venv (lerobot 0.6.1, torch) and streams target degrees here, and this process
commits them through the same chain as every other input: LPF, rate cap, clamp,
collision guard, workspace floors, E-STOP. Handing the bus to the child was
considered and CLOSED, because it would mean `/estop` cannot drop torque during
a rollout, which is the exact trade the port's central decision refused.

The safety argument runs the other way from the intuitive one: a freshly-trained
policy is LESS trustworthy than a practised hand, so it should get MORE of the
commit chain, not less.

STDLIB ONLY. This module runs in the serving process, which is the teleop
latency path — `import lerobot` and `import torch` are banned here exactly as
they are in `lab/`. Everything heavy lives in the child.

## The wire

Newline-delimited JSON over one loopback TCP socket, because that is what the
child already speaks (`rollout_runner.IngestClient`).

    child -> server   policy_hello        five frozen keys, then nothing until answered
    server -> child   policy_hello_ack    {ok, server_pid}   -- or --
    server -> child   policy_refused      {detail}
    server -> child   policy_observation  state + images, one per inference tick
    child -> server   policy_action       five frozen keys, {side: {joint: deg}}

Every spelling below is a MODULE CONSTANT rather than an inline literal, and
that is load-bearing rather than tidy. The two halves live in different venvs
and cannot import each other — `lab/` and this module are banned from the
child's lerobot world, and the child is banned from this process — so the only
instrument that can pin the two sides equal is a source grep, and a grep needs a
name to anchor on. Track B pins `"haller_rate"` the same way.

**These four spellings were the CHILD's named guesses and are adopted here by
the integrator's ruling of 2026-08-27, as a decision rather than a
coincidence.** That distinction is the `MIN_RATE_FRACTION` lesson: two names
agreeing is not evidence that one reads the other, and had this side simply
happened to spell them the same way, the child's file would still have said
"placeholder" while the system depended on it — with no later reader able to
tell a settled contract from an unclaimed coincidence.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

#: Where the child dials. Must equal `rollout_runner.DEFAULT_INGEST_URL`, and
#: nothing but a source grep can enforce that — see the module docstring.
INGEST_HOST = "127.0.0.1"
INGEST_PORT = 8781
INGEST_URL = f"tcp://{INGEST_HOST}:{INGEST_PORT}"

#: LOOPBACK ONLY, and never configurable to anything else. This socket commits
#: motion to real arms with no authentication of any kind; the gate that makes
#: that acceptable is that the kernel will not route a packet to it from off the
#: box. `api/gate.py::require_local` guards the HTTP surface for the same
#: reason, and this is the same decision one layer down.
_BIND_HOST = "127.0.0.1"

#: Inbound message types — the child's outbound pair. FROZEN with Track B.
HELLO_TYPE = "policy_hello"
ACTION_TYPE = "policy_action"

#: Outbound message types — the child's inbound trio. FROZEN by the
#: integrator's ruling; these were the child's named guesses and this side
#: adopts them. `rollout_runner.py:126-132` is the other end of each.
ACK_TYPE = "policy_hello_ack"
REFUSED_TYPE = "policy_refused"
OBSERVATION_TYPE = "policy_observation"

#: The unit every joint arrives in, gripper included, declared ONCE in the
#: handshake as a property of the source. A hello declaring anything else is
#: REFUSED, never converted: a conversion here would be this process guessing
#: what a policy meant, and the one thing worse than refusing a rollout is
#: running one whose numbers mean something other than what they say.
ACTION_UNIT = "deg"

#: Exactly the keys `rollout_runner.hello_message` promises. A hello missing one
#: is refused by name rather than defaulted — the child builds this dict from a
#: frozen literal, so a missing key is a version skew between the two halves and
#: not an operator error.
HELLO_KEYS = ("type", "run_id", "unit", "control_hz_declared", "rig")

#: Exactly the keys `rollout_runner.action_message` promises.
ACTION_KEYS = ("type", "seq", "t_ms", "run_id", "action")

#: How long a connected child may sit without completing the handshake. It has
#: nothing to compute first — `hello_message` is a dict literal — so a peer that
#: connects and says nothing is a peer that is not the rollout child.
HANDSHAKE_TIMEOUT_S = 10.0


class PolicyRefusal(Exception):
    """A reason this rollout must not run, in a sentence fit for an operator.

    Carried rather than returned because every refusal path must both answer
    the child and close the socket, and a return value can be dropped.
    """


def _canonical_state_names(sides: list[tuple[str, str]],
                           joint_order: Callable[[str], list[str]]) -> list[str]:
    """`observation.state`'s column names, in the recorder's own order.

    THE COUPLING WORTH KNOWING ABOUT. `hello_message` carries five keys and the
    joint order is not one of them, so the server cannot be TOLD which order the
    policy expects. Both sides instead derive it from the same rule: sides left
    then right, joints in `SO101_JOINT_ORDER`, named `{side}_{joint}` — which is
    exactly `recorder._state_names()`, because the policy was trained on a
    dataset this recorder wrote.

    That holds for every dataset recorded here and is UNVERIFIED for one
    recorded anywhere else, where a different column order would feed the policy
    a permuted state vector and drive the arm somewhere confidently wrong. So
    the observation carries `state_names` beside `state`, making the wire
    self-describing even though nothing on the child reads it yet. Escalated:
    the honest fix is a hello that declares the order and a server that refuses
    a mismatch, and that is a change to a frozen shape, not something to invent
    here.
    """
    names: list[str] = []
    for side, arm_id in sides:
        names += [f"{side}_{j}" for j in joint_order(arm_id)]
    return names


class PolicyIngest:
    """Accepts ONE rollout child at a time and commits what it streams.

    Everything that decides whether a rollout may run is injected as a zero-arg
    callable resolved AT HANDSHAKE TIME, not at construction: the answer to "is
    an episode open, is teleop driving, does anything else hold the bus" is a
    fact about the moment the child asks, and this object is built once at
    startup. Same reasoning as `lab/routes.py`'s injected getters.

    `submit` is the commit chain's front door. It receives one action and
    returns what the chain did with it, per joint — which is the only thing that
    can distinguish a policy action that was COMMITTED from one that was merely
    RECEIVED (acceptance criterion C3).
    """

    def __init__(
        self,
        *,
        bus_conflict: Callable[[], str | None],
        submit: Callable[[str, int, dict], dict],
        observe: Callable[[], dict | None],
        on_session_end: Callable[[], None] | None = None,
        host: str = _BIND_HOST,
        port: int = INGEST_PORT,
    ) -> None:
        self._bus_conflict = bus_conflict
        self._submit = submit
        self._observe = observe
        self._on_session_end = on_session_end
        self._host = host
        self._port = port
        self._server: asyncio.Server | None = None
        #: The run_id currently admitted, or None. One at a time: two policies
        #: streaming into one commit chain is the same defect as two leaders,
        #: and the arm would follow whichever frame arrived last.
        self.active_run_id: str | None = None
        #: Counters that exist to make an ACCEPTANCE observable. C0: a run that
        #: merely failed to refuse is indistinguishable from a child that never
        #: connected, an ingest that dropped the message, or a listener that
        #: accepted bytes nobody parsed. These are the positive artefacts.
        self.actions_received = 0
        self.actions_committed = 0
        self.observations_sent = 0
        self.last_commit: dict | None = None
        self.last_refusal: str | None = None

    # ---- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_client, self._host, self._port)
        logger.info("policy ingest listening on tcp://%s:%d",
                    self._host, self._port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self.active_run_id = None

    @property
    def port(self) -> int:
        """The bound port — the real one, which matters when it was 0."""
        if self._server is None or not self._server.sockets:
            return self._port
        return int(self._server.sockets[0].getsockname()[1])

    # ---- the protocol ----------------------------------------------------

    def ack_message(self) -> dict:
        """The handshake acknowledgement. Exactly three keys.

        `server_pid` IS NOT OPTIONAL, and it is the one key here whose absence
        is silent. The child filters its own bus-holder walk against it, because
        this server holding `/dev/ttyACM0` is the NORMAL required state and must
        never read as a conflict — so with no pid, `foreign_port_holders`
        returns empty ("no foreign holders") and the check passes for every
        process on the bus (`rollout_runner.py:620`, `:868`). No error, no
        warning: a check that cannot fire in either direction, guarding the one
        thing that corrupts Feetech packets.

        Note the asymmetry that makes this the dangerous one. `ok` absent
        defaults true and `detail` absent costs a sentence — both degrade toward
        MORE refusal or less information. `server_pid` degrades toward LESS
        refusal.
        """
        return {"type": ACK_TYPE, "ok": True, "server_pid": os.getpid()}

    @staticmethod
    def refusal_message(detail: str) -> dict:
        """A refusal the operator can act on. The sentence is the payload.

        `detail` reaches the operator as the child's exit message
        (`rollout_runner.py:579`), so it names the CONDITION rather than the
        code path. An absent detail degrades to a generic sentence on the child,
        which costs the reason — so this never sends one empty.
        """
        return {"type": REFUSED_TYPE,
                "detail": detail or "the server refused this rollout"}

    def check_hello(self, msg: dict) -> str:
        """Validate a hello and return its `run_id`, or raise `PolicyRefusal`.

        Shape first, then policy. A hello that does not parse is a version skew
        between two halves that cannot import each other, and saying so by name
        is the only debugging aid either side gets.
        """
        missing = [k for k in HELLO_KEYS if k not in msg]
        if missing:
            raise PolicyRefusal(
                f"malformed {HELLO_TYPE}: missing {', '.join(sorted(missing))}. "
                f"The child builds this from a frozen literal, so this is a "
                f"version skew between the runner and the server, not a "
                f"configuration error.")

        unit = str(msg.get("unit") or "")
        if unit != ACTION_UNIT:
            # REFUSED, never converted. The unit is declared once as a property
            # of the source, so a source declaring something else is a source
            # this server cannot interpret — and guessing would mean committing
            # numbers whose meaning we inferred.
            raise PolicyRefusal(
                f"this rollout declares its actions in {unit!r}; the commit "
                f"chain takes {ACTION_UNIT!r} on every joint, gripper included. "
                f"Refused rather than converted: a unit this server had to "
                f"guess at is a unit it could guess wrong.")

        if self.active_run_id is not None:
            raise PolicyRefusal(
                f"a rollout is already streaming (run {self.active_run_id}). "
                f"Two policies on one commit chain is two leaders, and the arm "
                f"follows whichever frame arrived last.")

        # LAST, because it is the only check that walks /proc, and because the
        # three cheap answers above are facts about the message rather than
        # about the rig. Resolved NOW rather than at construction: whether an
        # episode is open is a fact about this instant.
        conflict = self._bus_conflict()
        if conflict:
            raise PolicyRefusal(conflict)

        return str(msg.get("run_id") or "")

    def decode_action(self, msg: dict) -> tuple[int, dict]:
        """One action frame as `(seq, {side: {joint: deg}})`.

        Raises `PolicyRefusal` on anything malformed. A target frame that half
        parses is a frame that moves some joints and not others, which is a
        pose the policy never asked for and nothing downstream could recognise
        as wrong.
        """
        missing = [k for k in ACTION_KEYS if k not in msg]
        if missing:
            raise PolicyRefusal(
                f"malformed {ACTION_TYPE}: missing {', '.join(sorted(missing))}")
        if msg.get("run_id") != self.active_run_id:
            raise PolicyRefusal(
                f"action from run {msg.get('run_id')!r} but the admitted run is "
                f"{self.active_run_id!r}")
        action = msg.get("action")
        if not isinstance(action, dict) or not action:
            raise PolicyRefusal(
                f"{ACTION_TYPE} carries no targets. An absent side has no key "
                f"at all; an empty one claims the arm was commanded to nothing.")
        out: dict[str, dict[str, float]] = {}
        for side, joints in action.items():
            if not isinstance(joints, dict) or not joints:
                raise PolicyRefusal(
                    f"{ACTION_TYPE} side {side!r} carries no joints")
            try:
                out[str(side)] = {str(j): float(v) for j, v in joints.items()}
            except (TypeError, ValueError) as e:
                raise PolicyRefusal(
                    f"{ACTION_TYPE} side {side!r} has a non-numeric target: {e}"
                ) from e
        return int(msg.get("seq") or 0), out

    def observation_message(self, seq: int, payload: dict) -> dict:
        """One observation for the child to infer from.

        `state_names` rides along beside `state` even though nothing on the
        child reads it yet — see `_canonical_state_names` for why the order is
        derived rather than declared, and why a self-describing wire is the
        cheapest thing available until the hello can carry it.
        """
        return {
            "type": OBSERVATION_TYPE,
            "seq": int(seq),
            "t_ms": int(time.time() * 1000),
            "state": [float(v) for v in payload.get("state", [])],
            "state_names": list(payload.get("state_names", [])),
            "images": dict(payload.get("images") or {}),
        }

    # ---- the connection --------------------------------------------------

    async def _handle_client(self, reader: asyncio.StreamReader,
                             writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        run_id: str | None = None
        try:
            hello = await self._read_message(reader, HANDSHAKE_TIMEOUT_S)
            if hello is None or hello.get("type") != HELLO_TYPE:
                await self._send(writer, self.refusal_message(
                    f"expected a {HELLO_TYPE} first; got "
                    f"{(hello or {}).get('type')!r}"))
                return
            try:
                run_id = self.check_hello(hello)
            except PolicyRefusal as e:
                self.last_refusal = str(e)
                logger.info("policy ingest REFUSED %s: %s", peer, e)
                await self._send(writer, self.refusal_message(str(e)))
                return

            self.active_run_id = run_id
            self.last_refusal = None
            await self._send(writer, self.ack_message())
            logger.info("policy ingest ADMITTED run %s from %s (server pid %d)",
                        run_id, peer, os.getpid())
            await self._stream(reader, writer,
                               float(hello.get("control_hz_declared") or 0.0))
        except (ConnectionError, asyncio.IncompleteReadError):
            logger.info("policy ingest: run %s disconnected", run_id)
        except Exception:
            logger.exception("policy ingest: session failed")
        finally:
            # Closing IS the goodbye, on both sides. There is no `policy_bye` in
            # the frozen shape and this side does not invent one: the server
            # already treats a source that stopped streaming the way it treats a
            # teleop client that stopped, and a second end-of-stream signal is a
            # second thing that can disagree with the first.
            if self.active_run_id == run_id:
                self.active_run_id = None
                if self._on_session_end is not None:
                    try:
                        self._on_session_end()
                    except Exception:
                        logger.exception("policy ingest: session teardown failed")
            writer.close()

    async def _stream(self, reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter, declared_hz: float) -> None:
        """Observations out, actions in, until the child stops or is revoked.

        Observations are paced on a CLOCK at the child's declared rate, not
        emitted one-per-action. The child's warmup runs `RATE_WARMUP_TICKS` full
        inference cycles with nothing sent — deliberately, so its rate is
        measured before the arm can move on it — so a server that only spoke
        when spoken to would deadlock before the first action ever existed.

        The rate is the child's DECLARED one because that is the cadence it
        paces itself at; supplying faster would queue observations it will never
        read in time, and supplying slower would starve the inference loop and
        make this side the reason its measured rate fails its own gate.
        """
        obs_seq = 0
        period = 1.0 / declared_hz if declared_hz > 0 else 0.05
        next_obs = 0.0
        while self.active_run_id is not None:
            now = time.monotonic()
            if now >= next_obs:
                payload = self._observe()
                if payload is not None:
                    await self._send(writer,
                                     self.observation_message(obs_seq, payload))
                    self.observations_sent += 1
                    obs_seq += 1
                next_obs = now + period
            # Poll well inside the observation period, so an action is committed
            # promptly rather than sitting until the next frame is due.
            wait = max(0.001, min(period / 4.0, next_obs - time.monotonic()))
            msg = await self._read_message(reader, timeout=wait)
            if msg is None:
                continue
            if msg.get("type") != ACTION_TYPE:
                continue
            self.actions_received += 1
            try:
                seq, action = self.decode_action(msg)
            except PolicyRefusal as e:
                # A malformed frame refuses the RUN rather than being skipped.
                # Skipping would leave a policy driving on a subset of its own
                # targets, which is a different policy than the one under test.
                self.last_refusal = str(e)
                await self._send(writer, self.refusal_message(str(e)))
                self.active_run_id = None
                return
            outcome = self._submit(self.active_run_id or "", seq, action)
            self.actions_committed += 1
            self.last_commit = outcome

    # ---- framing ---------------------------------------------------------

    async def _read_message(self, reader: asyncio.StreamReader,
                            timeout: float) -> dict | None:
        """One newline-delimited JSON object, or None on TIMEOUT.

        Whole lines only: half a JSON object silently dropped is a target frame
        that never happened.

        **EOF RAISES; it does not return None.** A timeout means "nothing yet,
        keep waiting" and EOF means "the peer is gone, stop", and collapsing
        them is how this loop span at 100% of a core after the child exited —
        `readline()` on a closed socket returns immediately and forever, so a
        `continue` on that answer is an unbounded tight loop on the teleop
        latency path. Caught while writing the round-trip tests, which hung.

        The child's own `recv` makes the same collapse deliberately and gets
        away with it because every caller there holds a deadline. This side's
        stream loop has none — it runs until the run ends — so the distinction
        has to live here.
        """
        try:
            line = await asyncio.wait_for(reader.readline(), timeout)
        except (TimeoutError, asyncio.IncompleteReadError) as e:
            if isinstance(e, asyncio.IncompleteReadError):
                raise ConnectionResetError("policy child closed the stream") from e
            return None
        if not line:
            raise ConnectionResetError("policy child closed the stream")
        text = line.strip()
        if not text:
            return None
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("policy ingest: undecodable line (%d bytes)", len(text))
            return None
        return msg if isinstance(msg, dict) else None

    @staticmethod
    async def _send(writer: asyncio.StreamWriter, msg: dict[str, Any]) -> None:
        writer.write((json.dumps(msg) + "\n").encode())
        await writer.drain()

    # ---- observable state ------------------------------------------------

    def status(self) -> dict:
        """What a route can report about the ingest, for C0.

        Every field here is a POSITIVE artefact: something that cannot be true
        unless the thing actually happened. `actions_committed` in particular
        cannot advance without `submit` having returned, so it distinguishes a
        run that was accepted from a run that merely was not refused.
        """
        return {
            "listening": self._server is not None,
            "url": f"tcp://{self._host}:{self.port}",
            "active_run_id": self.active_run_id,
            "observations_sent": self.observations_sent,
            "actions_received": self.actions_received,
            "actions_committed": self.actions_committed,
            "last_commit": self.last_commit,
            "last_refusal": self.last_refusal,
        }
