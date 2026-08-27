# Track C — the Lab frontend, handoff

Written 2026-08-27 by `haller-ws-fd`; revised that evening by `haller-ws-95`
from `haller-ws-f3`'s successor brief, after §2, §3, §4 and the suite figure
were all found stale. Track C is **complete and holding**.

Everything below is what a successor **cannot reconstruct from the code or the
git log**. What the code already says is deliberately not repeated here.

State at handoff: 11 commits, `tsc --noEmit` 0, eslint clean, production build
clean, `/lab/compare` prerenders.

**Track C's territory is 235 tests across 13 files, 235/235 measured IN
ISOLATION** at `b908cf6` with `hmi/frontend/**` clean.

The figure this document carried until 2026-08-27 evening was **150 across
six**, and it was wrong in the way this port keeps finding: it was a fact
about **which suites someone once chose to name**, not a fact about the track.
Those same six now read 157. Adding `labRuns.test.tsx` — 20 tests, and the
suite guarding all five defects the live-backend pass found — reads 177. Both
are subsets wearing the track's name. A successor who runs the short command
and sees 157 will believe they have run Track C.

The territory is every frontend suite except Track D's five, so:

```
npx vitest run __tests__/labClient.test.ts __tests__/labCharts.test.ts \
  __tests__/labReview.test.tsx __tests__/labRuns.test.tsx \
  __tests__/cockpitTabs.test.tsx __tests__/api.test.ts \
  __tests__/cockpitLib.test.ts __tests__/calibration.test.ts \
  __tests__/CalibrationWizard.test.tsx __tests__/DeadManIndicator.test.tsx \
  __tests__/EStopButton.test.tsx __tests__/JointSlider.test.tsx \
  __tests__/teleopPresets.test.ts
```

**It reconciles, and the reconciliation is the point:** Track D's five
(`vrTeleop`, `vrTeleopProtocol`, `vrTeleopRecord`, `vrTeleopXRLoop`,
`humanTeleopClient`) measure **191**, and 235 + 191 = **426** across 13 + 5 =
**18 files** — the full-suite baseline exactly. The two territories partition
the suite with nothing double-counted and nothing orphaned. If your isolated
number and your neighbour's stop summing to the tree's, one of you has
annexed a file.

**A full-suite total is only worth writing down beside the `git status` that
was true when it was taken.** The 426 above holds because
`hmi/frontend/**` was clean; four sessions share one working tree, so a count
taken while another track is mid-write is a fact about the tree at that instant
and nothing more. Measured across one afternoon at the same HEAD it read 396,
then 395/1, then 396 three times, then 376/20 — that last one was Track D
editing `__tests__/vrTeleopXRLoop.test.tsx`, which was the only file failing
and was dirty in `git status` at the time. The commit alone does not pin it;
the commit AND the tree state do.

So before chasing any frontend failure: `git status --short -- hmi/frontend/`.
If the failing file is dirty and is not yours, it is someone typing. Confirm
your own scope with the isolated run above rather than re-running the whole
suite and hoping.

---

## 1. The live-backend recipe

**This is the most valuable thing in this document.** It is how every claim
about the Review surface was verified, and it is not written down anywhere
else. It supersedes any mock.

```bash
# 1. the real backend, in sim, with the /lab router mounted
cd hmi/backend
HALLER_HMI_CONFIG=/home/odesha/haller_ws/hmi/backend/config.bimanual-sim.yaml \
MUJOCO_GL=egl \
~/venvs/haller-hmi/bin/python -m uvicorn haller_hmi.server:app \
    --host 127.0.0.1 --port 8021 --log-level warning

# 2. the frontend, BUILT — see the trap below
cd hmi/frontend
NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:8021 npx next build
npx next start -p 3993
```

Three things that cost hours to learn:

- **`HALLER_HMI_CONFIG`, not `HALLER_CONFIG`.** The wrong name is not an error;
  it silently falls through to the default config and the server tries to open
  `/dev/haller_arm_uart`, fails to connect to real hardware, and exits 3. The
  error message is about a serial port and says nothing about config.
- **`next dev` renders but never hydrates here.** The Turbopack HMR websocket
  fails with `ERR_INVALID_HTTP_RESPONSE`, so no `useEffect` runs, no data is
  fetched, and every panel sits on its loading state. The page *screenshots*
  correctly, which is what makes it expensive — it reads as a broken backend.
  Always `next build` + `next start`. It also type-checks and catches
  Suspense-boundary errors that dev does not.
- **Screenshots must be written under `/home/odesha/haller_ws/.playwright-mcp/`.**
  The Playwright MCP refuses paths outside the repo root; that directory is
  gitignored (`51844b9`).
- **`.next/` is shared exactly like the git index.** One tree, one build
  output. `next build` mints a new `BUILD_ID` and replaces the chunk
  directory, so building while another session is serving from the same tree
  can leave THEIR running server pointing at chunk paths that no longer exist.
  The page then renders and its JavaScript 404s — which looks identical to the
  `next dev` non-hydration trap above, and will send whoever hits it to the
  wrong cause. So: `ss -ltnp | grep next-server` BEFORE building, not just
  before choosing a port. If someone else is serving, coordinate or wait.
  (Observed 2026-08-27: two sessions on :3993, second `next start` refused with
  EADDRINUSE — which is the safe failure. The unsafe one is the build that
  precedes it and succeeds silently.)

**There is a ready-made isolated rig, and it is the answer to the trap above:**
`/home/odesha/haller-trackC-scratch/wt`, a detached worktree with a **real
`node_modules`** (~849 MB, a copy not a symlink — Turbopack rejects a symlinked
`node_modules` pointing outside the project root) and **its own `.next`**. So
it can `next build` + `next start` while other sessions keep serving from the
shared tree. It is pinned at `80984b6`, which is still frontend-identical to
HEAD — `git diff --name-only 80984b6..HEAD -- hmi/frontend` is EMPTY, and a
path-scoped diff is a stronger answer to "did the frontend move" than
re-running the suite. Standing ruling: **each session removes only its OWN
worktree.** A stale registration costs one line of `git worktree list`;
removing a live one costs another session its verification run mid-flight.

The two real datasets under `~/robot-data/lerobot/local/` are the point of
using a real backend: one is 6-channel single-arm with one camera key, the
other 12-channel bimanual with three. **Anything that silently assumes a rig
passes against one and fails against the other.** Render both.

---

## 2. The `RecordStatus` reconcile — one member is a rewrite, not a confirm

Re-measured 2026-08-27 evening against **committed `b908cf6`** — not the
working copy, and not reasoned about. See the trap at the end of this section
for why those are two different mistakes.

```
GET /record/status returns, today:
  recording, repo_id, task, episode_frames, skipped_frames, started_at,
  last_error, auto_scored, success, success_frames, drops, fps_declared,
  fps_measured, record_rate_tolerance, alerts

vs lib/api.ts::RecordStatus
  on the wire but NOT in the type:  none
  in the type but not on the wire:  state, episode_index, invalidated_reason
```

**Five of the original eight have landed.** `fps_measured`, `fps_declared`,
`drops` and `alerts` are on the wire; `record_rate_gate` is gone from BOTH
halves and `record_rate_tolerance` replaces it (see §4). The three that remain
are Track A's, they are **optional**, and they arrive with 2d — so the type is
correct against today's backend and against A's.

**But the top-level key sets matching is not the reconcile finishing.** One
MEMBER type is wrong today:

```
alerts[] — recorder.py::_rate_alerts() emits:
  level, code, source, measured_hz, fps, tolerance, held_s, message

vs lib/api.ts::RecordAlert declares:
  code, detail?, since?
                          overlap: code — ONE key of eight
```

`detail` and `since` have never existed on the wire; the operator-facing
sentence is `message` and the duration is `held_s`. Both phantoms are
optional, so it type-checks perfectly against a backend that has never sent
either.

**`RecordAlert` has no consumer** — `grep` finds only its own declaration —
which is exactly why it has survived. It cannot render `undefined` until
something reads it, and the first thing that does will reach for `detail`,
the only text-shaped field the type offers, get `undefined` on every alert,
and draw an empty warning row. A defect that is unobservable now and certain
on first use is worse than one that is merely wrong now, because whoever
trips it will be debugging their own new code.

**The answer already exists eight lines away.** `lib/telemetry.ts:50` declares
the same producer as `{level, code, message, source}` and
`AlertsPopover.tsx:37-58` renders it and works. So the fix is not "reconcile
the two" — it is **the working one is right, make the other match it.**

**One non-finding, recorded so it is not "fixed" into a regression:**
`RecordStatus.started_at` is `number | null` and that is CORRECT.
`recorder.py:861` stamps `time.time()`, a float. The identically-named field
on the runs surface was an ISO string (`runs.py:179 _now()`) and typing it
`number` cost every timestamp cell — so the pattern-match off that defect
breaks this one, in the direction that type-checks. **A field with the same
name on a different producer is a different question.**

**Do not answer the remaining question by reading A's code and reasoning** —
this type was already wrong twice this port, both times in ways that
type-check:

- `state` was nearly `"rolling"`; A caught it. It must be `"recording"` so
  that `recording === (state === "recording")` keeps holding for every call
  site that already reads the boolean.
- `drops` was nearly a flat `Record<string, number>`. It is nested per camera
  and per arm, because the arms are `left`/`right` and nothing stops a camera
  being named for a side — one key, two meanings, and the panel reports a
  confident wrong number.

### The diff script, and the two ways it lies

`/home/odesha/haller-trackC-scratch/tools/typediff.py` parses a TS type out of
`lib/api.ts` or `lib/lab.ts` and compares a LIVE payload key-by-key AND
type-by-type. It found the `tags` defect. Beside it, `cdp.py` drives an
isolated headless chromium over CDP — needed because **the Playwright MCP
browser profile is contended** and refuses outright with `Browser is already
in use ... use --isolated`.

Whether you use it or regenerate the four-liner, **both known failure modes
produce confident nonsense rather than an error:**

- **A one-line `export type X = { ... };` is not matched by the block regex,
  so it falls through to the NEXT type in the file** and compares your payload
  against the wrong type's fields. Observed live: `LogPage` is one line, so the
  log payload was diffed against `Checkpoint` and reported three required
  fields missing and two unexpected. Check that the names printed belong to the
  type you asked for.
- **A nested dict flattens into the top-level key set.** `"drops": {"cameras":
  ..., "arms": ...}` spans two lines, so a `^\s*"(\w+)":` regex hoists `arms`
  into the wire keys and reports a field the route has never sent at top level.
  Hit on the re-measure above; `arms` is not a wire key.
- It cannot resolve union aliases, so `RunKind` and `RunStatus` flag as
  `declared RunKind | wire string` on every row. Harmless noise.

**And measure the PRODUCER from `git show HEAD:<path>`, not the working copy.**
On this tree the file is very likely dirty under another track. Reading
`recorder.py::status()` off disk during Track A's 2d returns `state`,
`episode_index` and `invalidated_reason` — fields that are in that session's
editor buffer and in no commit. A payload read off someone else's working copy
is a fact about their editor and it arrives wearing the authority of a
measurement.

---

## 3. What is NOT verified — do not round this up

**Review is verified against the real backend and the real 707 MB dataset**:
the player, the packed-mp4 seek, the `to − 1/fps` clamp, the trace and gripper
charts, marking, filters, the keyboard triage.

**Train and Compare have since been verified against the real backend too**,
and this section said the opposite until 2026-08-27 evening. It told a
successor they were mock-only and sent them to go do a pass that `5a196a5`
had already done — so the continuity document was the thing most likely to
waste the next session's day. The doc was even edited AFTER that pass
(`a39b85b`) without this paragraph being touched. **A handoff's stalest
sentence is the one describing work that finished after it was written.**

What that pass actually cost, since "expect to find something" turned out to
be an understatement — Oscar's real 60k-step ACT run
(`train-20260826-213350`, 313 metric rows, 13 checkpoints, a 2.2 MB log),
rendered in a real browser, produced **five defects, none of which threw and
all of which type-checked**: ISO-string timestamps typed `number` (every AT
cell and every elapsed read `—`), thirteen checkpoints all rendering
`pretrained_model`, `step: null` on lerobot's `last` badging every row
`latest`, a refused metrics request blanking the whole Compare pane, and the
`MAX_KEYS` cap no client could see (closed `f7b862c` + `b644a60`).

### The boundary — this is the part that goes stale, so it is explicit

VERIFIED against the real backend: run-list rows, run detail spec/argv/exit/
repo, the metric grid, the log tail, the checkpoint list, Compare's legend,
curves and refusal path, timestamps and elapsed, tags.

NOT verified, ranked:

1. **No RUNNING run has ever been rendered.** Every run in the store is
   terminal. So nothing has exercised the 1 s poll loop, the log tail growing
   through opaque offsets, the metric stream resuming, `now`-based elapsed
   ticking, `onChanged` firing on a status transition, or the STOP button.
   `RunDetail` has a whole branch keyed on `status === "running"` that has
   never executed against a backend. **This is the biggest gap, bigger than
   the rollout one.**
2. **No `kind:"rollout"` DETAIL pane.** A rollout run WAS rendered in the run
   list and in Compare's legend ("no metrics logged yet, so out of the shared
   set") — the kit's own rollout directory is a fine catalog entry. What was
   never opened is its detail pane. Note the distinction: "none has ever been
   accepted" is true of a run the LAUNCHER produced, not of the catalog.
3. **Compare has never overlaid two real curves.** Only one run in the store
   logs metrics, so the shared-key intersection, batch merge, refusal path and
   legend were exercised — but the thing Compare exists for was not.
4. **No mutating path was exercised**: `DELETE /lab/runs/{id}` and
   `POST /{id}/stop`. Deliberate (see the COPY rule in §6), but the buttons
   have never been pressed.
5. **`TrainLauncher` has never submitted.** Form → `POST /lab/runs/train` is
   unverified end to end.
6. **The rate readout was verified with STUBBED fps.** No session was running,
   so `fps_declared`/`fps_measured` were injected. The TOLERANCE was genuine —
   read from the wire and proven by moving it — so the band logic is verified
   and the real measured-rate plumbing is not.
7. **Review was not re-verified** by the session that did Train and Compare.
   It rests on the earlier pass against the real 707 MB dataset. Also: the
   "render BOTH datasets" rule above is a REVIEW rule — Train and Compare are
   dataset-shape-agnostic, so only one was run there.

The mock lived in a session scratchpad and is **gone**. Do not rebuild it —
the real backend is strictly better, and a mock that agrees with the types
rather than the server is what hid all five defects.

---

## 4. The rate band — a FLOOR became a TOLERANCE, and the reading half changed

**This section described the removed behaviour as current until 2026-08-27
evening**, which is the worse kind of stale: a doc naming a missing symbol is
self-evidently wrong, and one describing a live-looking behaviour is not.

**Dead, all at `80984b6`:** `RECORD_RATE_GATE_FALLBACK`, `recordRateGate`,
`recordRateOk`, and the wire key `record_rate_gate`. None of the three survive
anywhere in `hmi/frontend/**` — the only trace is a historical comment at
`__tests__/api.test.ts:150`.

**Live:** `recordRateTolerance(status)` (`lib/api.ts:412`) returns the
published `record_rate_tolerance` or **`null`, with no fallback number**, and
`recordRateFaithful(status)` (`lib/api.ts:464`) answers the two-sided
`|measured − declared| / declared <= tol`, `null` when unanswerable. The
source is `FPS_FAITHFUL_FRACTION` in the recorder.

**There is deliberately no fallback, and the general form is the useful part.**
`0.9` read as a TOLERANCE means ±90% — a band no real rate can fall outside —
so the warning would not have become wrong, it would have stopped existing.
**Ask not whether a fallback is allowed, but what happens when it is WRONG.**
A stale compare cap is self-correcting: too high and the backend refuses in
words the pane displays, too low and the request is merely split more finely.
A wrong rate band shows the operator a wrong number with nothing to contradict
it. Same rule, opposite verdict — and that is why the two look inconsistent.

**The old trap SURVIVES and is sharper now.** The bound is a ratio against the
**declared** rate, so it only means what it says while `fps_declared` is
exactly the `fps` written into `info.json`. A ±0.5% band against a wrong
declared rate is worse than the old 90% floor was, because it is tighter — the
same divergence now fires constantly instead of never. Check the *referent*,
not just the number.

**`RATE_DECIMALS = 2` (`lib/api.ts:457`) is a CADENCE-COUPLED CONSTANT.** With
`d` decimals there is a rate outside the tolerance that still renders as the
declared one whenever `fps < 10^(2-d)`. At `d=1` a 5 Hz session shows a refused
5.025 as "5.0" beside a declared 5 — the readout glows as a warning while
showing two numbers that look equal, which is how an operator learns to
disbelieve the warning. At `d=0` it predicts the `RATE 30/30 fps` defect the
headset track had already hit, which is what makes it an instrument rather than
a guess. 10 Hz and below are reachable today via `POST /teleop/human/start
{hz}`. The plan's house rule about tick-counted constants was written for
per-tick motion limits; **a display resolution is the same class**, and a sweep
for that class would never have thought to look at a `toFixed`.

---

## 5. Invariants that are load-bearing and easy to break silently

Each of these was a decision, not a preference. Changing one is a decision too.

- **The best/final table reads RAW points, never smoothed ones.** A smoothed
  minimum is a value the run never reached, and "best loss 0.071" has to be a
  number that actually happened. The compare page's smoothing slider moves the
  curve, not the figures under it. `components/lab/ComparePane.tsx`.
- **The eval split is never recomputed in the browser.** It comes from
  `GET /lab/datasets/split`. Two implementations of "which episodes does the
  trainer not see" drift, and when they do the `val` badges lie about which
  demonstrations the policy has already learned — the one error on this
  surface that cannot be spotted by looking at it.
- **`sliceFor()` has NO fallback and returns null.** Guessing that an episode
  starts at 0 opens episode 6 at second 0 of an 82-second file and plays five
  other takes under the right label. Silence is the only safe answer to a
  missing slice.
- **The clamp is `to − 1/fps`, not an epsilon.** `to_timestamp` is EXCLUSIVE;
  the last real frame is one frame period before it. Track B bisected this on
  the real files: at 30 fps a 0.01 s epsilon lands *past* the last frame, and
  the last episode of every file — 7 of 46 — shows a blank frame instead of the
  take. `fps` is a REQUIRED prop on `EpisodePlayer` so an assumed 30 cannot
  compile.
- **Gripper guides come from `trace.gripper[].closed_below/open_above`.** Those
  are the exact floats `grade.py` graded with, index-aligned to the column.
  There is deliberately no second path — a `gripperGuides()` helper reading the
  same pair off `arms[]` was deleted for being one. When the backend sends no
  thresholds, draw NONE: an invented threshold looks like a measurement.
  Measured: 40.0/70.0 on the kit dataset, 34.13/67.20 on the degree-calibrated
  bimanual one.
- **`episode_index`, never `index`.** LeRobot's v3.0 parquet carries both and
  `index` is the GLOBAL FRAME index — episode 1's first three frames are
  `episode_index [1,1,1]`, `frame_index [0,1,2]`, `index [855,856,857]`. A
  field called `index` here reads correctly and means something else.
- **Every episode shows `Ep 4` beside a muted `idx 3`.** 1-based in
  conversation, 0-based in storage; both on screen so a disagreement is a
  visible artefact rather than a silent one.
- **Sorting and filtering are server-side.** A 70-seed campaign is thousands of
  episodes and the list is paged; a browser sort orders one page and calls it
  the answer. The shift-range selection follows the VISIBLE order for the same
  reason — a range walking numeric indices behaves correctly on a list sorted
  by index and betrays the operator the first time they sort by duration.
- **`PaneBoundary` is not decoration.** A subordinate widget must not be able
  to unmount the workspace. It caught two unrelated failures an hour apart —
  one a sloppy test fixture, one a genuine contract mismatch — and in both
  cases cost a chart instead of the Data tab. Do not remove it because
  "nothing throws any more".

---

## 6. Testing conventions

- **Mutation-check every test that guards an absence.** A confirm gate is an
  absence assertion in disguise and goes vacuous silently. Break the thing,
  confirm the test goes red, restore. One of my own anchor tests was found
  vacuous this way — it now documents what it *cannot* detect and why
  (`__tests__/labReview.test.tsx`, "keeps ranging from a live anchor").
- `routeFetch()` in `__tests__/cockpitTabs.test.tsx` is the house pattern:
  route by path fragment, collect calls, assert POST bodies verbatim.
- jsdom has no `scrollIntoView`, `ResizeObserver` or `IntersectionObserver`.
  Stub in the test file, not in the component — the guard would be dead code
  in every browser that matters.

### Measuring, on a live tree — five ways the harness lied

**A harness that cannot observe the event reports the same thing as an event
that never happened.** Every one of these was hit here, and each read as a
finding about the code:

- **Restart before reporting.** A backend that imported a module seconds
  before a colleague added a line (one message away from accusing Track A of
  skipping a migration step); a frontend built before the change under test,
  reporting that the fix did not work; `:8021` still serving routes that no
  longer existed.
- **A page-installed `fetch` hook is destroyed by navigation** and then reports
  **zero requests** — which reads identically to "the code never fired".
- **`cmd && echo "clean"` fires on EXIT STATUS, not on empty output.**
  `git status` exits 0 either way, so it reported clean unconditionally: true
  by luck every previous time, false the moment it mattered. The same shape
  bites in a pipeline — `tsc --noEmit | tail -5; echo $?` reads **`tail`'s**
  status and prints 0 through any number of type errors. Two sessions hit that
  one independently within an hour. Count with `| wc -l`, or let the output
  itself be the evidence.
- **`innerText` is what the CSS renders, not what the code wrote.**
  `text-transform` made a chip reading `baseline` come back as `BASELINE`,
  twice, one step short of a false regression filed against a landed fix.
  **Any assertion on rendered text is case-insensitive or it is testing the
  stylesheet.**

### Prove a published value is READ by MOVING it

`8` agreeing with a fallback of `8` cannot distinguish a live read from a copy.
Republish the value and watch the consumer follow: `compare.MAX_KEYS` patched
to 3 through a wrapper module on `PYTHONPATH` — **never editing another track's
file** — took requests from `[8,2]` to `[3,3,3,1]`; the same method took the
tolerance `0.005 → 0.2 → absent` and watched the band follow. Then mutate to
confirm the METHOD: replacing the read with the fallback must fail ONLY the
moved-value test.

### The run store is a COPY, never a link — and that is load-bearing

The runs surface owns `DELETE /lab/runs/{id}` and a delete button, and Oscar's
only real ACT run is **7 GB with no backup**. A hardlink or a symlink would put
it behind that button. The checkpoint weights are omitted because
`_checkpoint_wire` sends `{step, path, has_model}` only — provably invisible to
every route the frontend calls — which is how 7 GB became 3 MB without becoming
a fixture. **Any probe of a surface owning a mutating call gets a copy.**
Rebuild it the same way.

---

## 7. Git, on a shared tree

One working tree, one `.git`, four sessions. Anything any session stages sits
in the same index every other session commits from.

```
git add <paths>            # in the same breath as the commit, never before
git commit -F <msgfile> -- <explicit paths>
git show --stat HEAD       # confirm exactly the files you meant
```

**Never `git stash` on this tree.** It empties other tracks' working trees.

---

## 8. Territory

`hmi/frontend/**` **except** the in-headset VR client, which is Track D's:
`components/VRTeleopPanel.tsx`, `lib/vrTeleop.ts`, `lib/humanTeleopClient.ts`,
`app/teleop/vr/**`, and the `vrTeleop*` tests.

`lib/api.ts` is Track C's. Track D requests additions rather than editing it —
that is what stopped `recordStop`'s new `rearm` field from silently becoming
required and breaking two shipped desktop surfaces that call it as `{save}`.

`~/vr-teleop-kit` and `tests/equivalence/**` are read-only to everyone.
