# Integrator follow-ups — the kit port

Cross-track items the integrator (`haller-ws-13`) owes, or must chase once another
track lands its half. Not a backlog: everything here is blocked on a specific event,
and dies when that event happens. Add the trigger, not just the task.

## Open

- **Delete `episodesTotal()` and its 5 tests from `lib/vrTeleop.ts`.**
  *Trigger:* Track A's `episode_index` lands in `GET /record/status`.
  *Owner once triggered:* Track D (`haller-ws-1a`), who flagged it.
  It exists only to paper over lerobot buffering ten episodes' metadata in RAM before
  writing `meta/episodes.jsonl` — the HUD counter stalls at 7 while the operator banks
  their tenth take. A real index from the recorder makes the guess unnecessary, and a
  guess left next to a fact is how the two drift.

- **Mount the record routes in `server.py`.**
  *Trigger:* Track A reports the exact bodies for `POST /record/arm`, `/record/roll`,
  `/record/stop {save, rearm}`.
  `server.py` is the integrator's precisely because both backend tracks need routes
  there.

- **Mount Track B's Lab router in `server.py`, then delete `routes_data.py`.**
  *Trigger:* Track B reports `build_router(...)`'s signature and its four
  compatibility paths answer with the old shapes.
  The factory must take ZERO-ARG CALLABLES (`get_cameras=lambda: cameras`, …), not
  values: `server.py` mounts routers at import time but builds `cameras`/`recorder`
  inside `lifespan`, so a router closing over the values captures `None` and 503s
  forever. This bit the 08-22 unification; do not rediscover it.

- **`/estop` must revoke the rollout lease, and the lease must be mounted.**
  *Trigger:* Track B lands `lease.py` and the streaming-inference child.
  Ruled 2026-08-27: the rollout child owns the POLICY, never the bus. It streams
  actions to the server over loopback and they commit through the same chain as every
  other input — LPF, rate cap, clamp, collision guard, workspace floors, E-STOP.
  Handing the bus to a child was considered and CLOSED: it would mean `/estop` cannot
  drop torque during a rollout, the exact trade the port's central decision refused.
  The safety win is the other direction anyway — a freshly-trained policy is less
  trustworthy than a practised hand, so it should get MORE of the commit chain, not
  less. Fallback if streaming inference proves unworkable: rollout stays CLI-only with
  the HMI stopped, which is what the kit does today. Only after measuring.

- **Rewrite the A/X take protocol in the operator docs.**
  *Trigger:* Track A's `/record/arm|roll|stop` land and are committed.
  `hmi/QUICKSTART-QUEST.md` (~121-147, 298) and `docs/setup/dataset-collection.md`
  (~53-54) still document "hold A/X to start a take, hold again to stop-and-save" plus a
  two-way save/discard prompt. Wrong in three ways now: ARM vs ROLL, keep/redo rather
  than save/discard, and every decision returning to ARMED. Deliberately NOT done yet —
  a doc should describe the protocol that is committed, and documenting one whose server
  half is still in Verify is how it becomes wrong in a fourth way.

- **Reconcile `RecordStatus` once every track's fields are in.**
  *Trigger:* Tracks A and C both land.
  The type was already missing `auto_scored`, `success`, `success_frames` — fields
  `recorder.py::status()` returns TODAY — before this port added any. Worth one pass to
  confirm the type finally matches the payload rather than merely growing.

## Standing rules that came out of rulings

- **The message that only matters when something has gone wrong is the one that was
  not legible.** Two HUD strings were found clipped off the in-headset panel, both in
  branches that only render in a degraded state — which is exactly why neither was ever
  seen, since the happy path fits. `B/Y = E-STOP` had been invisible since before this
  port; `acquiring 1.2s (no tracking)` cut off mid-word, so a hand that stopped tracking
  during the countdown reported it in a sentence the panel truncated, and the operator's
  natural read was "broken" rather than "move your hand". Canvas text does not wrap, so
  every `fillText` site is pinned against its column width in tests (monospace advances
  at 0.6 em, so it is arithmetic). The PAINTER trims, not the caller: an ellipsis reads
  as truncation, a mid-glyph clip reads as a rendering fault. Does NOT apply to the
  desktop cockpit — DOM text wraps, so the failure mode does not exist there.

- **The surface that OWNS a fact publishes it; the other reads it.** Ruled after Track C
  asked Track A for the fps-refusal threshold instead of picking its own. A UI that
  invents its own copy of a number is how a dashboard ends up disagreeing with the
  system it monitors.
- **A route two shipped surfaces already call is not a new route.** `POST /record/stop`
  grew `rearm` for the headset state machine; it is OPTIONAL and defaults to false, so
  `{save}` alone still means stop-save-idle. Required would have broken the cockpit stop
  button and RecordPopover silently, since desktop callers have no idea the headset grew
  states.

## Closed

- ~~Plan doc's phase numbering disagreed with the track briefs~~ — pinned to the doc
  (`633bbdb`), briefs are track-local.
- ~~"No git write operations" contradicted the delegated-commit instruction~~ —
  delegation made explicit (`633bbdb`).
- ~~The end-of-take modal exception to invariant 5 was undocumented~~ — recorded with
  its two lapse conditions (`34b732f`).
- ~~`pos_reach_limit` brief said the kit used 0.25 m~~ — that was the `ClutchPoseMapper`
  default and the DK1's; the SO-101's is 0.15 ("smaller arm, smaller wall"). Moved to
  0.15, which is the only number here with 46 episodes behind it.
