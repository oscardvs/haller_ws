# hmi/backend/tests/test_policy_ingest.py
"""The server half of the policy wire, asserted from Track B's criteria.

Written against `docs/port/trackb-lab-contract.md`'s addendum — the acceptance
criteria Track B wrote BLIND, before this module existed. That timing is the
whole point of them and it cannot be recovered, so these tests are written from
the criteria's own words rather than from what the implementation happens to do.

C0 is the one that shapes the file: **an acceptance must be observable, and it
is not the absence of a refusal.** So no test here passes by not-erroring. Each
names a positive artefact that cannot exist unless the thing happened.

The round-trip tests drive the CHILD'S OWN `IngestClient` against this server
rather than a hand-rolled socket. That is the only thing that can establish the
two halves fit: a fake client would be built from this module's assumptions,
which is the blind spot the criteria were written early to avoid.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from haller_hmi.policy_ingest import (
    ACK_TYPE,
    ACTION_TYPE,
    ACTION_UNIT,
    HELLO_TYPE,
    INGEST_PORT,
    INGEST_URL,
    OBSERVATION_TYPE,
    REFUSED_TYPE,
    PolicyIngest,
    PolicyRefusal,
)


def _hello(run_id="r1", unit=ACTION_UNIT, **over):
    msg = {"type": HELLO_TYPE, "run_id": run_id, "unit": unit,
           "control_hz_declared": 30.0, "rig": "bimanual"}
    msg.update(over)
    return msg


def _action(run_id="r1", seq=0, action=None):
    return {"type": ACTION_TYPE, "seq": seq, "t_ms": 1, "run_id": run_id,
            "action": action if action is not None
            else {"left": {"shoulder_pan": 10.0}}}


def _ingest(*, conflict=None, submit=None, observe=None):
    commits: list[tuple] = []

    def _submit(run_id, seq, act):
        commits.append((run_id, seq, act))
        return {"seq": seq, "committed": act}

    ing = PolicyIngest(
        bus_conflict=lambda: conflict,
        submit=submit or _submit,
        observe=observe or (lambda: None),
        port=0,
    )
    ing.commits = commits          # test handle, not part of the interface
    return ing


# --- the ack, and the key whose absence is silent ------------------------

def test_the_ack_carries_the_server_pid():
    """The one key here that degrades toward LESS refusal.

    The child filters its own bus-holder walk against this pid, because THIS
    server holding /dev/ttyACM0 is the normal required state. Omit it and
    `foreign_port_holders` returns empty — "no foreign holders" — so the check
    passes for every process on the bus, with no error and no warning. A check
    that cannot fire in either direction, guarding the one thing that corrupts
    Feetech packets.

    Asserted as the REAL pid rather than merely present: a placeholder would
    satisfy `"server_pid" in ack` and filter out the wrong process, which is the
    same dead check wearing a passing test.
    """
    ack = _ingest().ack_message()
    assert ack["type"] == ACK_TYPE
    assert ack["ok"] is True
    assert ack["server_pid"] == os.getpid()


def test_a_refusal_always_carries_a_sentence():
    """`detail` reaches the operator as the child's exit message. An absent one
    degrades to a generic sentence, which costs them the reason."""
    ing = _ingest()
    assert ing.refusal_message("an episode is being recorded")["detail"] \
        == "an episode is being recorded"
    assert ing.refusal_message("")["detail"]           # never empty
    assert ing.refusal_message("")["type"] == REFUSED_TYPE


# --- C1: the handshake is answered, not merely survived ------------------

def test_a_hello_missing_a_frozen_key_is_refused_by_name():
    """The child builds the hello from a dict literal, so a missing key is a
    version skew between two halves that cannot import each other. Naming it is
    the only debugging aid either side gets."""
    ing = _ingest()
    for key in ("run_id", "unit", "control_hz_declared", "rig"):
        msg = _hello()
        del msg[key]
        with pytest.raises(PolicyRefusal, match=key):
            ing.check_hello(msg)


def test_a_unit_this_server_did_not_expect_is_REFUSED_not_converted():
    """The unit is declared once as a property of the source.

    Converting would mean committing numbers whose meaning this process
    inferred. Radians silently treated as degrees is a 57x error into a rate
    cap, which is the shape that reads as a hardware fault.
    """
    ing = _ingest()
    with pytest.raises(PolicyRefusal, match="[Rr]efused rather than converted"):
        ing.check_hello(_hello(unit="rad"))


def test_the_bus_conflict_sentence_is_passed_through_verbatim():
    """`lease.bus_conflict` composes the sentence and this is its first caller.

    Verbatim matters: the sentence names WHICH condition — an episode open, a
    teleop session driving, a foreign holder — and a server that rewrote it
    would be a second place for that wording to drift from the checks.
    """
    sentence = ("cannot start a rollout: an episode is being recorded into "
                "oscardvs/x. Stop the recording first.")
    ing = _ingest(conflict=sentence)
    with pytest.raises(PolicyRefusal) as e:
        ing.check_hello(_hello())
    assert str(e.value) == sentence


def test_the_conflict_check_is_resolved_at_HANDSHAKE_not_at_construction():
    """Whether an episode is open is a fact about the instant the child asks.

    Pointed at the state that tells the two apart: an ingest built while the rig
    was clean, asked after a take opened. A construction-time read passes here
    and admits a policy into an open episode.
    """
    state = {"conflict": None}
    ing = PolicyIngest(bus_conflict=lambda: state["conflict"],
                       submit=lambda *a: {}, observe=lambda: None, port=0)
    assert ing.check_hello(_hello()) == "r1"          # clean at build time

    state["conflict"] = "cannot start a rollout: an episode is being recorded"
    ing.active_run_id = None
    with pytest.raises(PolicyRefusal, match="episode is being recorded"):
        ing.check_hello(_hello())


def test_a_second_rollout_is_refused_while_one_streams():
    """Two policies on one commit chain is two leaders, and the arm follows
    whichever frame arrived last — the same defect the teleop side refuses."""
    ing = _ingest()
    ing.check_hello(_hello(run_id="first"))
    ing.active_run_id = "first"
    with pytest.raises(PolicyRefusal, match="already streaming"):
        ing.check_hello(_hello(run_id="second"))


# --- actions -------------------------------------------------------------

def test_an_action_from_a_run_that_was_never_admitted_is_refused():
    ing = _ingest()
    ing.active_run_id = "admitted"
    with pytest.raises(PolicyRefusal, match="admitted run"):
        ing.decode_action(_action(run_id="someone-else"))


def test_a_half_parsed_action_is_refused_rather_than_partly_applied():
    """A frame that half parses moves some joints and not others — a pose the
    policy never asked for, which nothing downstream could recognise as wrong."""
    ing = _ingest()
    ing.active_run_id = "r1"
    with pytest.raises(PolicyRefusal, match="non-numeric"):
        ing.decode_action(_action(action={"left": {"shoulder_pan": "sideways"}}))
    with pytest.raises(PolicyRefusal, match="no joints"):
        ing.decode_action(_action(action={"left": {}}))
    with pytest.raises(PolicyRefusal, match="no targets"):
        ing.decode_action(_action(action={}))


def test_an_absent_side_is_an_absent_KEY_and_that_is_legal():
    """A solo rig sends one key. `{"right": {}}` would read as "the right arm
    was commanded to nothing", which is a different and wrong claim — and is
    refused above."""
    ing = _ingest()
    ing.active_run_id = "r1"
    _, act = ing.decode_action(_action(action={"left": {"shoulder_pan": 1.0}}))
    assert act == {"left": {"shoulder_pan": 1.0}}
    assert "right" not in act


# --- C0: an acceptance is observable ------------------------------------

def test_the_counters_distinguish_accepted_from_merely_not_refused():
    """C0. A run that failed to refuse could be a child that never connected, an
    ingest that dropped the message, or a listener accepting bytes nobody
    parsed. `actions_committed` cannot advance without `submit` having
    RETURNED, so it is the artefact that separates them."""
    ing = _ingest()
    st = ing.status()
    assert st["actions_committed"] == 0
    assert st["active_run_id"] is None
    assert st["last_commit"] is None


def test_the_url_matches_the_one_the_child_dials():
    """The two halves cannot import each other — different venvs, and `lab/` is
    banned from lerobot — so a source grep is the only instrument that can pin
    them equal, and a grep needs a NAME to anchor on.

    This asserts the value; Track B's tripwire asserts the two spellings match
    across the tree. Both are needed: this one would pass if both sides moved
    together to something wrong, and theirs would pass if this side never used
    its own constant.
    """
    assert INGEST_PORT == 8781
    assert INGEST_URL == "tcp://127.0.0.1:8781"


# --- the two halves, over a real socket ---------------------------------
#
# The child's OWN client, not a hand-rolled one. A fake built from this module's
# assumptions is the blind spot Track B's criteria were written early to avoid.

def _client(port):
    from haller_hmi.runners.rollout_runner import IngestClient
    return IngestClient("127.0.0.1", port)


async def _serve(ing):
    await ing.start()
    return ing.port


async def test_the_child_completes_the_handshake_and_ACTS_on_the_ack():
    """C1. Not "a connection opened" — TCP accept proves a listener, not a
    reader. The evidence is that the child PARSED the ack: `IngestClient`
    stores `server_pid` off it, and its bus-holder filter reads that field.
    """
    ing = _ingest()
    port = await _serve(ing)
    client = _client(port)
    try:
        await asyncio.to_thread(client.connect)
        ack = await asyncio.to_thread(client.handshake, _hello())
        assert ack["type"] == ACK_TYPE
        # THE assertion: the child took the pid off the wire, which a counted
        # ack could not produce.
        assert client.server_pid == os.getpid()
        assert ing.active_run_id == "r1"
    finally:
        client.close()
        await ing.stop()


async def test_a_refused_handshake_reaches_the_child_as_its_exit_sentence():
    """C1's other half, and C2's shape: the refusal must survive the wire with
    the condition still named in it."""
    sentence = ("cannot start a rollout: a teleop session is driving the arms. "
                "Stop it before handing them to a policy.")
    ing = _ingest(conflict=sentence)
    port = await _serve(ing)
    client = _client(port)
    try:
        await asyncio.to_thread(client.connect)
        with pytest.raises(SystemExit) as e:
            await asyncio.to_thread(client.handshake, _hello())
        assert str(e.value) == sentence          # verbatim, not paraphrased
        assert ing.active_run_id is None
        assert ing.last_refusal == sentence
    finally:
        client.close()
        await ing.stop()


async def test_the_child_receives_an_observation_it_can_DECODE():
    """The child's own `decode_observation` is the reader, so it is the judge.

    A test asserting the server's dict has a `state` key would pass against a
    shape the child returns None for — which is the two-halves-fit question
    answered by one half.
    """
    from haller_hmi.runners.rollout_runner import decode_observation

    payload = {"state": [1.0, 2.0, 3.0],
               "state_names": ["left_shoulder_pan", "left_elbow_flex",
                               "left_gripper"],
               "images": {}}
    ing = _ingest(observe=lambda: payload)
    port = await _serve(ing)
    client = _client(port)
    try:
        await asyncio.to_thread(client.connect)
        await asyncio.to_thread(client.handshake, _hello())
        msg = await asyncio.to_thread(client.recv, 5.0)
        assert msg is not None and msg["type"] == OBSERVATION_TYPE
        decoded = decode_observation(msg)
        assert decoded is not None, "the child could not decode our observation"
        assert decoded["state"] == [1.0, 2.0, 3.0]
    finally:
        client.close()
        await ing.stop()


async def test_N_actions_sent_are_N_actions_committed():
    """C3's third clause: sending N must move the arm N times, not once.

    A single accepted frame is consistent with a handshake that captured one
    message and stalled — which is exactly what a naive read-once server does,
    and it looks identical to success from the launch route.
    """
    from haller_hmi.runners.rollout_runner import action_message

    ing = _ingest()
    port = await _serve(ing)
    client = _client(port)
    try:
        await asyncio.to_thread(client.connect)
        await asyncio.to_thread(client.handshake, _hello())
        for seq in range(5):
            await asyncio.to_thread(
                client.send,
                action_message("r1", seq, {"left": {"shoulder_pan": float(seq)}}))
        for _ in range(100):
            if ing.actions_committed >= 5:
                break
            await asyncio.sleep(0.02)
        assert ing.actions_committed == 5, ing.status()
        assert [c[1] for c in ing.commits] == [0, 1, 2, 3, 4]
    finally:
        client.close()
        await ing.stop()
