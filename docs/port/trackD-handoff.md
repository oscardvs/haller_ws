# Track D handoff — the in-headset VR client

Written 2026-08-27 on `feat/kit-port` by `haller-ws-1a`, at the integrator's request,
for whoever picks Track D up next. Track D is the in-headset client: the start gate,
the HUD, the controller bindings and their tests.

**Everything here is what you cannot reconstruct from the code.** The code says what it
does; this says why, what expires, and where I was wrong. Commit messages carry the rest —
`c085997` `8443748` `d8b7159` `a60b08a` `c63f719` `fb2d27d` `ae21a3b`, in that order, and
they are written to be read.

## State

Complete and **blocked on Track A's `/record/arm|roll|stop`**, which the integrator mounts
in `server.py` once A reports the bodies. Baselines to protect:

- **Track D's own five suites: 191 passed / 5 files**, measured 2026-08-27 at `fc3b6c5`
  on a clean tree — `vrTeleop`, `vrTeleopProtocol`, `vrTeleopRecord`, `vrTeleopXRLoop`,
  `humanTeleopClient`. **That is the number that is a property of THIS track.** The tree
  total was 396 when this line was first written and is **426 / 18 files at `fc3b6c5`**;
  the 30 in between are Track C's work landing, not a delta to hunt. Baseline when Track D
  started was 186. **Re-pin a total against the commit you measured it at, or the next
  session chases somebody else's landing.**
- **`npx tsc --noEmit` clean.** No CI typecheck exists — run it by hand, every time.
- **`npx eslint` — 10 errors + 1 warning in `VRTeleopPanel.tsx`, all PRE-EXISTING.** Nine
  `Cannot access refs during render` (the file's deliberate ref-mirroring idiom), one
  `setState synchronously within an effect`, one `<img>` warning on the MJPEG tile. Do not
  chase them and do not let them hide a new one: the check is that the count is still 11.

Ownership, unchanged: `components/VRTeleopPanel.tsx`, `lib/vrTeleop.ts`,
`lib/humanTeleopClient.ts`, `app/teleop/vr/**`, `__tests__/vrTeleop*.test.ts(x)`,
`__tests__/humanTeleopClient.test.ts`, and this file plus `headset-checklist.md`.
**Not** `lib/api.ts` (Track C), **not** `components/cockpit/**` or `lab/**`, **not** anything
under `hmi/backend/`, **never** `tests/equivalence/**` or `~/vr-teleop-kit`.

## The one thing that EXPIRES — read this first

**V13 is not "the item that needs no routes". It is the item that needs the routes to still
be MISSING.**

I had it filed the first way for two days and it is wrong. V13 tests the client's fallback
against a backend that has no `/record/arm` — which is every backend right now and none of
them after Track A lands. Its window closes rather than opens.

I did it (`ae21a3b`) the moment I noticed, and both halves are pinned headlessly, so **this
is no longer a trap for you**. It is written down because the shape recurs: a sequencing
fact that expires reads like a scheduling detail and is invisible once it has expired. If
you find yourself unable to reproduce a fallback path, this is why.

What V13 still needs from a real run: that an unmounted FastAPI route genuinely answers 404,
and that nothing lands on disk before ROLL. The tests assert the client makes no call; only
a run can see the directory.

## The remaining order — do not reshuffle it

1. **The sim walk of V3–V13**, once the routes are mounted. Every item in
   `headset-checklist.md` carries a note saying which half vitest already holds and what the
   run adds, so this is a short list of backend assertions, not a rediscovery.
2. ~~Retire the `episodesTotal()` fallbacks.~~ **DONE at `ccd79d6`**, once the routes
   landed at `9360e8b` and `episode_index` went real. What it actually took, and why it
   was not the two-line deletion the follow-up described, is under **The index and the
   count** below. Do not re-open it; do read that section before touching either number.

3. **V11 goes FIRST in the hardware session.** Not last. The invariant-5 modal exception
   rests on it, and a lapse found at the start of the evening is a design change with time
   to make it; found at midnight it is a wasted session with new servos on the bench.

## Decisions of record you would otherwise re-litigate

### ARMED is the resting state, not a step

`IDLE → ARMED → ROLLING → prompt`, all on A/X hold. Every decision returns to **ARMED**,
because in a session whose point is banking 46 takes the next take is always the expected
next thing, and ARMED writes nothing so sitting in it costs nothing. IDLE is what you are in
before you first arm and after you leave. There is deliberately **no stand-down gesture in
the headset**: exiting VR stops teleop, which invalidates an armed gate on the backend's own
rule, so leaving disarms without a command.

`L stick click = keep` (save, arm next index) · `R stick click = redo` (bin, same index) ·
`A/X hold = keep rolling`. The two stand-down combinations the protocol allows
(`keep_stop`, `drop`) are desktop buttons only.

### Why there is no hold variant, and the invariant-5 exception

An earlier draft put "save and go again" on a **left-stick hold**. `haller-ws-d7` caught that
this is the trained in-session home gesture (invariant 5) — the same physical action with the
same dwell, separated only by modal state — and that the consequences are asymmetric:
banking a take you did not mean to bank is one reject mark in a file that already carries 11
of 46, while **asking for home and silently not getting it is the direction that hurts**.

Making "go again" a mode rather than a per-decision gesture removed the collision. Both
clicks keep the binding they have had since the 08-22 unification, so nothing new breaks.

The modal exception was **granted by the integrator on two conditions**, and both are pinned
as tests so it cannot lapse quietly:

- the refusal is **felt and seen** — the 0.2 / 60 ms tick `resetArms` already uses for a
  refused reset, plus `home refused mid-take — pick first` on the HUD;
- the prompt is **bounded and modal**, so the exception cannot leak into normal driving.

If either stops being true the exception lapses and the prompt's stick binding comes out.
That is not my call or yours — it goes back to the integrator.

### The index and the count — two facts, and the HUD held them in one field

**Resolved at `ccd79d6` (`haller-ws-6e`, 2026-08-27). The follow-up said "delete
`episodesTotal()` and its five tests"; that was wrong twice over and is now corrected in
`integrator-followups.md` too.**

`episode_index` is the gate's index for **the take in hand**. `episodesTotal()` is the
**size of the dataset**. They agree for as long as nothing has been pruned and every index
is a 0-based sequential offset into the dataset — which is the whole happy path, and is
exactly why one field looked like enough. **Two numbers that coincide on the happy path are
not the same fact.**

The retirement was not two lines because `RecorderHudLike.episodes` had **three** readers,
not two, and the third was reached through the field rather than through `episodesTotal`,
so a grep for the function never found it:

- `takeNaming()` and the `● REC` row read it as an **INDEX**;
- the idle menu's `hold A/X to ARM  (N in dataset)` reads it as a **COUNT**.

And on a gated backend `episode_index` is **null exactly while idle** — the recorder sets it
at ARM (`recorder.py:842`) and clears it at STOP (`:990`, `:1016`). So the one reader that
wanted a count was the one reader guaranteed a null. Retiring the fallback without splitting
the field would have silently dropped the dataset count off the idle menu.

Split into `episodeIndex` and `datasetCount`. **Renamed rather than reused**, for the reason
`record_rate_gate` became `record_rate_tolerance` instead of keeping its name: a different
meaning gets a different name so a stale reader **breaks at the type** instead of plausibly
painting the wrong number. It did break — four sites, all in tests, all loud, none silent.

`episodesTotal()` **stays**, with its five tests (`__tests__/vrTeleopProtocol.test.ts:385-415`
— not in `lib/vrTeleop.ts`, whatever the follow-up said), `DatasetTally` and
`refreshEpisodes`. All five pin the COUNT, none the fallback, so all five survived. The
lerobot RAM buffering it papers over is a fact about `meta/episodes.jsonl`: **`N in dataset`
still stalls at 7 while the operator banks their tenth, on both surfaces, gate or no gate.**

One visible consequence, and it is the retirement working rather than a regression: an
ungated backend now reads `◆ ARMED take 3`, not `ep 34`. `take 3` is true of this page-load;
`ep 34` off a count was right only by coincidence, and V13's own wording said `take N` all
along.

### The toasts read the ask back instead of the outcome

**Fixed at `ccd79d6`.** Track A's contract extension made this reachable: a `{save, rearm}`
whose save lands and whose **re-arm is refused** is a **200** carrying `state:"idle"` and an
`invalidated_reason` — deliberately not a 409, because a 409 would report a banked take as a
failure. **So a 200 stopped being evidence that the ask was honoured**, and `stopAct` was
building its toast from `act.rearm`.

The result would have been `· armed for the next` printed over a gate that was **down**, at
the one moment the operator is deciding whether to keep driving. The 250 ms poll reconciles
the state and the HUD paints `GATE DROPPED — <reason>` a quarter-second later, so the client
self-corrected — but **the toast is what gets read at the decision**, and it substituted a
wrong conclusion rather than withholding one. Same class as the clipped
`acquiring 1.2s  (no tracking)`, one surface over.

It reads `st.state` under a **server** gate and the ask under a **local** one — where `rearm`
was never sent and the re-arm is this page's own, so reading `state` there would invent a
refusal that did not happen. The copy leads with `saved`: **a refused re-arm must not read as
a lost take.**

The other half of that extension needs nothing from us: a deliberate stand-down from ARMED
lands idle with the reason **null**, so the HUD correctly says nothing. An operator act is
not a fault.

### 404 vs 409 on the arm probe

**404/405 is "this backend has no gate". 409 is the gate WORKING** — a colliding camera key,
a measured rate under the floor. Swallowing one as the other arms locally against a recorder
that has just refused, which is the worst of both. Broadening `isMissingRoute` to any
`ApiError` fails a test that exists for exactly this.

It matters most **the day the routes exist**, because that is the day 409s start arriving.
Do not "simplify" it before then.

### U9 — episode_index is not durable, and Track D is not exposed

After `delete_episodes`, every index above a removed episode shifts down, so `episode_index`
is a live counter and not a key; Track A stamps a per-frame `episode_uid` as the durable one.
**I audited Track D and there is no exposure — do not re-audit it.** The panel persists
exactly five things and none is an index: view id, tile size, wrist pivot, HUD anchor, solo
arm. `episode_index` is read per-poll and painted. `DatasetTally` lives in a ref that dies
with the page load.

## Standing rules this track produced — keep enforcing them

**A degraded-state string is often the ONLY route to a diagnosis.** Two were being clipped
off the HUD before anyone noticed, because the happy path fits and they only render when
something has already gone wrong. `B/Y = E-STOP` was cut off entirely — the one binding whose
purpose is to be findable when things go wrong, invisible on the device it is meant to be
read from. Worse: `acquiring 1.2s  (no tracking)` cut mid-word did not merely hide
information, it **substituted a wrong conclusion** — a stalled countdown with no legible
reason reads as "this is broken", not as "your hand left the volume".

Every row is now pinned against its column's width in
`__tests__/vrTeleopRecord.test.ts` (monospace advances at 0.6 em, so it is arithmetic).
`STATUS_MAX_CHARS` 39 / `MENU_MAX_CHARS` 36 / `MENU_TITLE_MAX_CHARS` 32 are computed from the
1024 px canvas, the 563 px status clip and the menu box's 392 px. Caller data is trimmed by
the painter — an ellipsis reads as truncation, a mid-glyph clip reads as a rendering fault —
with the box clip as backstop.

**A copied fact rots on somebody else's commit.** I had `declared * 0.9` hardcoded under a
comment claiming it matched the backend. True when written; it would have silently stopped
being true the first time the gate was tuned, and then the RATE warning and the 409 refusal
would tell the operator two different stories about one take. It reads
`recordRateGate(status)` now. The owner of a fact publishes it; everyone else reads it.

**A caveat the operator can only have seen once is a caveat they do not have.** The HUD
carried `(local gate)` for as long as it was true; the desktop carried only a toast, gone in
seconds. Same class as the clips, one level up — not unreadable, just not persistent.

## How to test things that "need a headset" — they mostly do not

`__tests__/vrTeleopXRLoop.test.tsx` drives the **real** XR animation loop headless.
`requestTeleopSession` reads `navigator.xr` and nothing else, so stubbing that one property
hands the panel a session whose `requestAnimationFrame` the test owns: frame timestamps
become an argument and a 0.8 s stick hold is four function calls. Haptics come back through
the fake gamepad's actuator. No production code changed to make this possible.

jsdom has no canvas 2D context, so the HUD does not paint in that file — painting is pinned
separately against a `fillText`-recording stub in `vrTeleopRecord.test.ts`. The E-STOP, the
A/X ladder, the modal stick handling and the local-gate fallback are all reachable this way.
Grab-to-move and the precision modifier are **deliberately not tested** — the ladder and both
safety paths are covered and the rest would be tests for their own sake. That was a
considered decision, twice, and endorsed.

### The discipline that file demands

**An absence assertion in a teardown-on-success path is vacuous by construction.** This bit
me twice in two days and it defeats review: `not.toHaveBeenCalled()` reads identically
whether the code under test ran or not, so there is nothing on the page to notice.

Three of the eight E-STOP tests were untestable as first written, in a harness I had built
the day before, and only the mutation pass found it:

- `fireEstop` tears the session down once `/estop` resolves, which stops the loop — so "one
  press, one POST" held even with **both** the rising-edge check and the in-flight guard
  removed. Fixed by making the request hang, the only state in which a held button can post
  twice.
- The display-rate test pressed on a fresh session where the throttle clock was still zero,
  so the throttle was never in the way. Fixed by publishing one frame first.
- The in-flight test **remounted the panel** between presses, handing itself a fresh ref — it
  would have passed with the guard permanently latched. Remounting between steps looks like
  hygiene. Fixed by re-entering the session on the same instance.

**So: mutate every new absence assertion.** Break the guard it claims to protect, watch the
test go red, restore and `diff` to confirm byte-identical. Do not rely on noticing — I did
not, and I had just written the rule down. This repo has paid for it once before, in the
Phase 0 strict-xfail bodies where every assertion inside the body was unfalsifiable.

One honest note in the same file: the one-POST test pins the **property**, not a mechanism.
The edge check and the in-flight guard each prevent the second post alone, so no single
mutation isolates either. The comment says so. Do not let it read as though it pins the edge
check.

## Things I got wrong, so you do not trust the wrong parts

- The gesture map's first draft had the invariant-5 collision. `haller-ws-d7` caught it.
- The implementation spec's menu copy was too wide and its tuning row count was off by one.
  A subagent caught both by measuring what I had eyeballed.
- Both HUD clips were found by the integrator asking me to pin strings I had not, not by me
  looking at the HUD. Neither would have survived a visual review, because the strings only
  render in a degraded state.
- `V8` in `headset-checklist.md` claimed its 0.2 / 60 ms cue was "the weakest in the table".
  It is not — the dropped gate's 0.15 / 50 ms is lighter. The tests had both extremes right
  the whole time; the document was the only thing carrying the error.
- `V10` asserted a claim about a **sequence** while every test I had was about a single
  transition. One arm/roll/keep cycle passes trivially. It is now ten full cycles.
  Generalise: **a claim about a sequence cannot be pinned by tests about single transitions.**

## Reporting

Report to the integrator (`haller-ws-57` at time of writing), not to sibling tracks, except
for protocol shapes — those go direct to whoever owns the server side, with the outcome
relayed to the integrator.

**Flag `tests/equivalence/**` modifications every time**, even though a narrow exception has
been granted there before and the tree is clean today. You cannot see the integrator's
exception list, so silence is the wrong default. A stale flag costs one message; a missed
breach costs the oracle that judges the port.
