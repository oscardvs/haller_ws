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

## Known-bad data

- **`local/haller_pick_the_red_cube_and_place_it_in_the_box` must never be trained on.**
  2 sim episodes / 997 frames. Already suspect (one arm never moved, per the grader), and
  now definite: its gripper column was recorded through the compressed mapping below, so
  it is squeezed into 0..63.6 with a dead band over the lower half of the trigger. Not a
  baseline, not a fixture, not a smoke-test target for anything that reads the gripper.

## Ways a test can be shaped like the code instead of the claim

Each produced a green suite over a real defect: the test matched the shape of the
implementation rather than the shape of the promise.

1. **A claim about a SEQUENCE cannot be pinned by tests about single transitions.** Ten
   takes without leaving ARMED passed trivially on a one-cycle test. The test walks the
   same control flow the code does, so it adopts the code's ordering assumptions.
2. **A fixture can encode the INTENDED contract while production encodes something else,
   and nothing compares the two.** The gripper fixture `(0.0, 100.0)` was RIGHT — lerobot
   hardcodes the gripper to `RANGE_0_100` regardless of `use_degrees`
   (`so_follower.py:59`). `_load_joint_limits` is what is wrong: it tick-centres every
   motor without checking `norm_mode` (`arm.py:337-348`), handing a percent channel a
   symmetric degrees window. The test passed because it tested the fixture's world, which
   was correct — while production's was not. **Nothing anywhere pins
   `_load_joint_limits`'s output against what lerobot actually expects per `norm_mode`.**
   That missing contract test is the real gap.
3. **A per-point assertion cannot see a dead zone; only a sweep can.** Every gripper test
   checked a command produced *some* sane value. None checked the mapping was a bijection
   onto the jaw's travel, so half a dead trigger was invisible.

**The common mechanism:** the test inherits the code's blind spot, because it was written
from the code. **The counter-discipline:** write the assertion from the CLAIM, in the
claim's own terms, before reading the implementation — a sweep for a mapping, an
end-to-end for a sequence, and for a contract, a test that compares the two real
components rather than either against a fixture.

4. **An absence assertion in a teardown-on-success path is VACUOUS BY CONSTRUCTION.**
   `not.toHaveBeenCalled()` reads identically whether the code under test ran or not, and
   if the success path tears down the thing being observed, the assertion holds however
   the code behaves. Three of eight E-STOP tests were vacuous on first writing; only
   mutation found them, in a harness built the day before. The three causes are each
   worth knowing: the request RESOLVED, so teardown stopped the loop before a second post
   was possible (fix: make it hang — the only state where a held button can post twice);
   the throttle clock was still 0 on a fresh session, so the throttle was never in the way
   (fix: publish a frame first, then act inside the window); and the component was
   REMOUNTED between the two presses, handing itself a fresh ref, so a permanently-latched
   guard would have passed too (fix: stay on the same instance).
   **The mutation pass is the mechanism. There is no reading-based substitute**, and
   nobody should claim they will notice next time.

**Also from that pass:** where two guards each independently prevent an outcome, no
single-mutation test can isolate either — the test pins the PROPERTY, not a mechanism, and
its comment must say so. A comment that reads as though it pinned one of them is the
checklist-asserting-a-contradicted-number failure again.

**Decision rule, from getting the fixture call wrong:** prefer the test that catches the
OBSERVED failure over the more elegant one. Elegance is not a coverage argument.

### Retracted, 2026-08-27 — "the impossible fixture"

An earlier version of this file claimed `gripper: (0.0, 100.0)` was a window the loader
could never produce, and generalised that into "an impossible fixture invents a world
where the bug cannot exist". **That was wrong and is withdrawn.** The window was correct
and production was not. On re-checking, the other flagged fixtures
(`shoulder_lift: (-100, 0)`, `elbow_flex: (0, 100)`, `elbow_flex: (0, 40)` in
`test_so101_ik.py`) are legitimate too: they are arbitrary INPUT to `SO101DecoupledIK`,
whose contract is to honour whatever limits it is handed, and asymmetry is the point of
two of them. So the lesson had **zero** real instances and is removed rather than
softened. The defect it was attached to is real and unchanged; only the diagnosis moved.

## Standing rules that came out of rulings

- **The symmetry rule holds only for fixtures standing in for loader OUTPUT.**
  `_load_joint_limits` centres on `(range_min + range_max)/2`, so every window it emits
  is symmetric about zero — but that says nothing about fixtures which are deliberately
  arbitrary INPUT to a component that accepts arbitrary limits (`SO101DecoupledIK` is
  contractually required to honour whatever it is handed). A scanner without that scope
  qualifier flags three legitimate fixtures on its first run, and a check that cries wolf
  is disabled by week two. Encode the scope or do not build it.

- **The message that only matters when something has gone wrong is the one that was
  not legible.** Two HUD strings were found clipped off the in-headset panel, both in
  branches that only render in a degraded state — which is exactly why neither was ever
  seen, since the happy path fits. `B/Y = E-STOP` had been invisible since before this
  port; `acquiring 1.2s (no tracking)` cut off mid-word, so a hand that stopped tracking
  during the countdown reported it in a sentence the panel truncated. That second one is
  the worse kind and the distinction is worth keeping: it did not merely hide
  information, it SUBSTITUTED A WRONG CONCLUSION — a stalled countdown with no legible
  reason reads as "this is broken", which is an answer, and the wrong one. Hunt the
  degraded-state strings that are the ONLY route to a diagnosis first. Canvas text does not wrap, so
  every `fillText` site is pinned against its column width in tests (monospace advances
  at 0.6 em, so it is arithmetic). The PAINTER trims, not the caller: an ellipsis reads
  as truncation, a mid-glyph clip reads as a rendering fault. Does NOT apply to the
  desktop cockpit — DOM text wraps, so the failure mode does not exist there.

- **Name the unit at the site, not in a docstring three files away.** Two dimension
  bugs surfaced in one phase, which is a pattern rather than luck: this codebase carries
  degrees, radians, ticks, percent and normalised [0,1] on adjacent surfaces, and a
  float carries none of that with it. Any constant crossing a module boundary is
  unit-suspect until proven. Both instances were live: lerobot pins the gripper to
  RANGE_0_100 regardless of `use_degrees`, so `read_joints_deg()["gripper"]` is a
  PERCENTAGE compared against a DEGREES window — preflight would have dropped a healthy
  arm limp, hardest right after a calibration, because the sweep ends at the jaw's open
  stop. And the kit's `last_limit_pressure` is RADIANS where Haller's is DEGREES, so a
  ported `max(pressure, 0.35)` sat below the 0.5 deg haptic dead zone and would have
  SILENCED the buzz at the exact pose it was added to raise.

- **A fixture that is not merely simplified but IMPOSSIBLE is worse than no fixture.**
  The gripper defect survived because a test hardcoded `gripper: (0.0, 100.0)` — a
  window `_load_joint_limits` can never produce. It converted an untested path into an
  apparently-tested one. Sweep fixtures for values the real loader could not emit.

- **Anything that drops torque goes through `_release_torque_per_motor`.** lerobot's
  bulk `disable_torque()` raises on the first refusal and leaves the rest energised —
  the 2026-08-21 incident verbatim, arriving by a new road. An arm with a latched
  shoulder alarm ends up HALF LIMP while the report prints "TORQUE STILL ENABLED".

- **A copied fact is only wrong once someone changes it elsewhere — and until then,
  the happy path fits.** Third instance of the legibility class, one level up. Track D's
  HUD carried `declared * 0.9` with a comment asserting it matched the recorder's gate.
  That comment was TRUE when written, which is what makes the class nasty: it passes
  review, it passes tests, and it rots on somebody else's commit. The failure mode is
  the sharp one — the rate warning and the 409 refusal telling the operator two
  different stories about one take, with no way to tell which is lying. Reading the
  published value is not sufficient on its own: the ABSENT case must not silently revert
  to a hardcoded copy, so both paths get pinned.

- **Preserving a distinction through the type and discarding it before the operator sees
  it is worse than never typing it.** It buys the reviewer's confidence and spends none
  of it on the operator. Found when the `drops` reducer, correctly typed nested, returned
  a bare key at the last step — so a camera named for a side and an arm named for a side
  collapse to one confident wrong answer.

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
